

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


    search_n_estimators = min(n_estimators, 300)
    base = lgb.LGBMClassifier(
        n_estimators=search_n_estimators, random_state=random_state, verbosity=-1,
        n_jobs=1,  
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