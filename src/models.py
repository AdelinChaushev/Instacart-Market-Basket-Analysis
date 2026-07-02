"""
Model training and cross-validated evaluation for the Instacart reorder classifier.

The evaluation metric (mean per-user F1 on thresholded predictions) is not a
standard sklearn scorer, and the decision threshold must itself be tuned per
fold. This module provides a GroupKFold evaluation that, for each fold, fits an
estimator on the training portion, predicts probabilities on the held-out
portion, tunes the threshold on those held-out predictions, and records the
resulting F1. Grouping is by user id so no user appears in both the training and
validation portion of any fold.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.base import clone
import json
import pickle
from pathlib import Path
from evaluation import mean_per_user_f1, search_threshold


def groupkfold_evaluate(estimator, X, y, groups, n_splits=5, threshold_grid=None):
    """
    Evaluate an estimator with GroupKFold, tuning the threshold within each fold.

    For each fold: clone and fit the estimator on the training portion, predict
    positive-class probabilities on the held-out portion, search the threshold
    that maximises mean per-user F1 on that held-out portion, and record the F1
    and the chosen threshold. Grouping by user id prevents a user's rows from
    appearing in both training and validation of a fold.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Must implement fit and predict_proba. Cloned per fold so the input is
        left unfitted.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : array-like of {0, 1}
        Labels.
    groups : array-like
        Group id (user id) per row, used by GroupKFold.
    n_splits : int
        Number of folds.
    threshold_grid : array-like of float, optional
        Threshold grid passed to the per-fold search.

    Returns
    -------
    dict
        fold_f1 : list of per-fold mean per-user F1
        fold_threshold : list of per-fold chosen thresholds
        mean_f1 : mean of fold_f1
        std_f1 : std of fold_f1
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    groups = np.asarray(groups)

    gkf = GroupKFold(n_splits=n_splits)
    fold_f1 = []
    fold_threshold = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y[train_idx])

        val_proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        search = search_threshold(
            user_ids=groups[val_idx],
            y_true=y[val_idx],
            y_score=val_proba,
            grid=threshold_grid,
        )
        fold_f1.append(search["best_f1"])
        fold_threshold.append(search["best_threshold"])

    fold_f1 = np.array(fold_f1)
    return {
        "fold_f1": fold_f1.tolist(),
        "fold_threshold": fold_threshold,
        "mean_f1": float(fold_f1.mean()),
        "std_f1": float(fold_f1.std()),
    }


def tune_lightgbm(X, y, groups, param_distributions=None, n_iter=20, n_splits_search=3,
                   n_estimators=1000, early_stopping_rounds=50, random_state=42):
    """
    Tune LightGBM hyperparameters with RandomizedSearchCV on AUC, then evaluate
    the winning configuration with GroupKFold using the real per-user F1 metric.

    AUC is used for the search itself because it is a standard scorer RandomizedSearchCV
    accepts natively and gives a stable read on overall ranking quality independent of
    threshold. The task metric (mean per-user F1, with its own per-fold threshold) is
    not searchable this way (it needs user ids and a threshold search at scoring time),
    so it is applied once, after search, to the winning configuration via
    groupkfold_evaluate. This mirrors using a proven library metric to drive search while
    reporting the task-appropriate metric as the final number.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like of {0, 1}
        Labels.
    groups : array-like
        User id per row, used for GroupKFold in both the search and the final evaluation.
    param_distributions : dict, optional
        Hyperparameter grid for RandomizedSearchCV. A reasonable default is used if None.
    n_iter : int
        Number of random hyperparameter combinations to try.
    n_splits_search : int
        Number of folds used during the AUC-based search (kept small to control runtime).
    n_estimators : int
        Max boosting rounds; early stopping typically halts well before this.
    early_stopping_rounds : int
        Patience for early stopping during the final per-fold fits.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        best_params : dict of the winning hyperparameters
        search_auc : best cross-validated AUC from the search
        f1_eval : dict, output of groupkfold_evaluate on the winning configuration
            (fold_f1, fold_threshold, mean_f1, std_f1)
    """
    if param_distributions is None:
        param_distributions = {
            "num_leaves": [15, 31, 63, 127],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "min_child_samples": [10, 20, 50, 100],
            "reg_lambda": [0.0, 0.1, 1.0, 5.0],
            "reg_alpha": [0.0, 0.1, 1.0],
        }

    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    groups = np.asarray(groups)

    # The search phase uses a capped n_estimators (not the full final value) with
    # no early stopping wired in, since RandomizedSearchCV does not pass a per-fold
    # validation set to fit(). Running the full n_estimators on every one of
    # n_iter * n_splits_search fits is the dominant cost of this function; capping
    # it here keeps the search a relative ranking of configurations, not a fully
    # trained ensemble, and is the fix for the slow search runtime.
    search_n_estimators = min(n_estimators, 300)
    base = lgb.LGBMClassifier(
        n_estimators=search_n_estimators, random_state=random_state, verbosity=-1,
        n_jobs=1,  # avoid nested parallelism contention with RandomizedSearchCV's n_jobs
    )
    gkf_search = GroupKFold(n_splits=n_splits_search)

    search = RandomizedSearchCV(
        base, param_distributions, n_iter=n_iter, scoring="roc_auc",
        cv=gkf_search.split(X, y, groups), n_jobs=-1, random_state=random_state,
    )
    search.fit(X, y)

    best_model = lgb.LGBMClassifier(
        n_estimators=n_estimators, random_state=random_state, verbosity=-1,
        **search.best_params_,
    )
    f1_eval = groupkfold_evaluate(best_model, X, y, groups, n_splits=5)

    return {
        "best_params": search.best_params_,
        "search_auc": float(search.best_score_),
        "f1_eval": f1_eval,
    }


def save_model_artifacts(model_dir, model, model_kind, feature_cols, threshold, cv_results,
                          extra=None):
    """
    Persist a trained model and its metadata to a versioned directory.

    Saves four artifacts: the fitted model in a format appropriate to its kind,
    the exact feature column list and order used for training, the tuned
    decision threshold, and the full cross-validation results (per-fold F1 and
    thresholds, not just the mean), so later hypothesis tests have the raw
    per-fold numbers to work with.

    Parameters
    ----------
    model_dir : str or Path
        Directory to save into, created if it does not exist. Convention:
        models/<name>_v<n>/, e.g. models/lgbm_v1/.
    model : fitted estimator
        The trained model. Saved via its native format for LightGBM
        (model.txt), pickle otherwise (model.pkl).
    model_kind : str
        One of "lightgbm", "sklearn", "torch". Determines the save format.
    feature_cols : list of str
        Exact feature column names and order used to train this model. Critical
        for safe reuse: feeding columns in a different order or set later would
        silently corrupt predictions without raising an error.
    threshold : float
        The tuned decision threshold (tau*) for this model.
    cv_results : dict
        Output of groupkfold_evaluate (or equivalent), containing fold_f1,
        fold_threshold, mean_f1, std_f1. Saved in full, not just the mean, since
        later model-comparison hypothesis tests need the per-fold values.
    extra : dict, optional
        Any additional metadata to save alongside (e.g. best_params, search_auc).

    Returns
    -------
    Path
        The model_dir that was written to.
    """


    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_kind == "lightgbm":
        model.booster_.save_model(str(model_dir / "model.txt"))
    elif model_kind == "torch":
        import torch
        torch.save(model.state_dict(), model_dir / "model.pt")
    else:  # sklearn-compatible, includes Pipelines
        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)

    with open(model_dir / "features.json", "w") as f:
        json.dump({"feature_cols": list(feature_cols)}, f, indent=2)

    with open(model_dir / "threshold.json", "w") as f:
        json.dump({"threshold": float(threshold)}, f, indent=2)

    cv_results_clean = {
        "fold_f1": cv_results.get("fold_f1"),
        "fold_threshold": cv_results.get("fold_threshold"),
        "mean_f1": cv_results.get("mean_f1"),
        "std_f1": cv_results.get("std_f1"),
    }
    if extra:
        cv_results_clean["extra"] = extra

    with open(model_dir / "cv_scores.json", "w") as f:
        json.dump(cv_results_clean, f, indent=2)

    print(f"Saved model artifacts to {model_dir}")
    print(f"  model_kind: {model_kind}")
    print(f"  features: {len(feature_cols)}")
    print(f"  threshold: {threshold:.4f}")
    print(f"  mean_f1: {cv_results_clean['mean_f1']:.4f}")

    return model_dir


def load_model_artifacts(model_dir, model_kind):
    """
    Reload a model and its metadata saved by save_model_artifacts.

    Parameters
    ----------
    model_dir : str or Path
        Directory written by save_model_artifacts (e.g. models/lgbm_v1/).
    model_kind : str
        One of "lightgbm", "sklearn", "torch", matching how it was saved.
        (torch requires the caller to pass a model instance to load_state_dict
        into; only "lightgbm" and "sklearn" reload the full object directly.)

    Returns
    -------
    dict
        model : the reloaded model (Booster for lightgbm, estimator for sklearn)
        feature_cols : list of feature names in training order
        threshold : the saved decision threshold
        cv_scores : the saved cross-validation results dict
    """


    model_dir = Path(model_dir)

    if model_kind == "lightgbm":
        model = lgb.Booster(model_file=str(model_dir / "model.txt"))
    elif model_kind == "sklearn":
        with open(model_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"load_model_artifacts supports 'lightgbm'/'sklearn'; got {model_kind}")

    with open(model_dir / "features.json") as f:
        feature_cols = json.load(f)["feature_cols"]
    with open(model_dir / "threshold.json") as f:
        threshold = json.load(f)["threshold"]
    with open(model_dir / "cv_scores.json") as f:
        cv_scores = json.load(f)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "cv_scores": cv_scores,
    }