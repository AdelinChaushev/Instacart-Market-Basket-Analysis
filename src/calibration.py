"""
Probability calibration diagnostics and correction for the Instacart classifier.

Threshold tuning and the tau* = F1*/2 result both assume the model's scores are
genuine probabilities. This module diagnoses whether that holds (reliability
diagram plus the Brier score and its three-term decomposition) and, if not,
applies isotonic calibration and measures the effect on both the Brier score and
mean per-user F1.

Leakage note: the isotonic calibrator is fit and evaluated on disjoint data.
Fitting the calibrator on the same predictions used to measure its effect would
let it see the answers, inflating the apparent improvement. The cross-validated
routine below fits the model on the training portion, the calibrator on a
separate calibration portion, and measures everything on an untouched validation
portion.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression

from evaluation import mean_per_user_f1, search_threshold


def reliability_curve(y_true, y_score, n_bins=10):
    """
    Compute a reliability curve: mean predicted probability vs observed frequency per bin.

    Predictions are grouped into equal-width probability bins. For each bin, the
    mean predicted probability and the actual fraction of positives are returned.
    A perfectly calibrated model has these equal in every bin (points on the
    diagonal).

    Parameters
    ----------
    y_true : array-like of {0, 1}
        True labels.
    y_score : array-like of float
        Predicted probabilities.
    n_bins : int
        Number of equal-width bins over [0, 1].

    Returns
    -------
    dict
        bin_mean_pred : mean predicted probability per non-empty bin
        bin_obs_freq : observed positive frequency per non-empty bin
        bin_count : number of samples per non-empty bin
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_score, edges[1:-1]), 0, n_bins - 1)

    mean_pred, obs_freq, counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(y_score[mask].mean())
        obs_freq.append(y_true[mask].mean())
        counts.append(int(mask.sum()))

    return {
        "bin_mean_pred": np.array(mean_pred),
        "bin_obs_freq": np.array(obs_freq),
        "bin_count": np.array(counts),
    }


def brier_decomposition(y_true, y_score, n_bins=10):
    """
    Compute the Brier score and its reliability / resolution / uncertainty decomposition.

    The Brier score is the mean squared error between predicted probability and
    outcome. Murphy's decomposition splits it as:

        Brier = reliability - resolution + uncertainty

    where reliability is the calibration error (bin predicted vs bin observed,
    smaller is better), resolution is how far bin outcomes deviate from the base
    rate (larger is better), and uncertainty is the irreducible base-rate term
    rho * (1 - rho).

    Parameters
    ----------
    y_true : array-like of {0, 1}
        True labels.
    y_score : array-like of float
        Predicted probabilities.
    n_bins : int
        Number of bins for the decomposition.

    Returns
    -------
    dict
        brier : the Brier score computed directly
        reliability : calibration error term (want small)
        resolution : resolution term (want large)
        uncertainty : irreducible term rho * (1 - rho)
        brier_from_decomp : reliability - resolution + uncertainty (should match brier)
    """
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)

    brier = np.mean((y_score - y_true) ** 2)
    rho = y_true.mean()
    uncertainty = rho * (1.0 - rho)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_score, edges[1:-1]), 0, n_bins - 1)

    n = len(y_true)
    reliability = 0.0
    resolution = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        nb = mask.sum()
        if nb == 0:
            continue
        pbar = y_score[mask].mean()   # mean prediction in bin
        ybar = y_true[mask].mean()    # observed frequency in bin
        reliability += nb * (pbar - ybar) ** 2
        resolution += nb * (ybar - rho) ** 2
    reliability /= n
    resolution /= n

    return {
        "brier": float(brier),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_from_decomp": float(reliability - resolution + uncertainty),
    }


def evaluate_calibration_cv(estimator, X, y, groups, n_splits=5, n_bins=10,
                             threshold_grid=None, random_state=42):
    """
    Cross-validated calibration analysis with leakage-safe isotonic correction.

    For each GroupKFold split, the training portion is further divided (by group)
    into a model-fitting part and a calibration-fitting part. The model is fit on
    the first, an isotonic calibrator is fit on the model's predictions over the
    second, and both the raw and calibrated models are evaluated on the untouched
    validation fold: Brier decomposition, reliability curve, and mean per-user F1
    at each model's own tuned threshold.

    Parameters
    ----------
    estimator : sklearn-compatible estimator with predict_proba
    X : pd.DataFrame
    y : array-like of {0, 1}
    groups : array-like
        User ids, used so no user crosses model / calibration / validation.
    n_splits : int
        Number of outer GroupKFold splits.
    n_bins : int
        Bins for reliability and Brier decomposition.
    threshold_grid : array-like of float, optional
        Threshold grid for per-fold F1 tuning.
    random_state : int
        Seed for the inner model/calibration group split.

    Returns
    -------
    dict
        raw_brier, cal_brier : lists of Brier dicts per fold (raw / calibrated)
        raw_f1, cal_f1 : lists of tuned per-user F1 per fold (raw / calibrated)
        raw_threshold, cal_threshold : lists of tuned thresholds per fold
        raw_reliability, cal_reliability : lists of reliability curves (last fold usable for plotting)
        mean_raw_f1, mean_cal_f1 : means across folds
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    groups = np.asarray(groups)
    rng = np.random.default_rng(random_state)

    gkf = GroupKFold(n_splits=n_splits)

    out = {
        "raw_brier": [], "cal_brier": [],
        "raw_f1": [], "cal_f1": [],
        "raw_threshold": [], "cal_threshold": [],
        "raw_reliability": [], "cal_reliability": [],
    }

    for train_idx, val_idx in gkf.split(X, y, groups):
        # Split the training portion by group into model-fit and calibration-fit.
        train_groups = np.unique(groups[train_idx])
        rng.shuffle(train_groups)
        cut = int(0.75 * len(train_groups))
        model_groups = set(train_groups[:cut])
        calib_groups = set(train_groups[cut:])

        model_mask = np.array([g in model_groups for g in groups[train_idx]])
        calib_mask = ~model_mask

        Xtr, ytr = X.iloc[train_idx[model_mask]], y[train_idx[model_mask]]
        Xcal, ycal = X.iloc[train_idx[calib_mask]], y[train_idx[calib_mask]]
        Xval, yval = X.iloc[val_idx], y[val_idx]
        gval = groups[val_idx]

        model = clone(estimator)
        model.fit(Xtr, ytr)

        # Fit isotonic calibrator on the held-out calibration slice.
        cal_scores = model.predict_proba(Xcal)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal_scores, ycal)

        # Evaluate both on the untouched validation fold.
        val_raw = model.predict_proba(Xval)[:, 1]
        val_cal = iso.predict(val_raw)

        out["raw_brier"].append(brier_decomposition(yval, val_raw, n_bins))
        out["cal_brier"].append(brier_decomposition(yval, val_cal, n_bins))
        out["raw_reliability"].append(reliability_curve(yval, val_raw, n_bins))
        out["cal_reliability"].append(reliability_curve(yval, val_cal, n_bins))

        s_raw = search_threshold(gval, yval, val_raw, grid=threshold_grid)
        s_cal = search_threshold(gval, yval, val_cal, grid=threshold_grid)
        out["raw_f1"].append(s_raw["best_f1"])
        out["cal_f1"].append(s_cal["best_f1"])
        out["raw_threshold"].append(s_raw["best_threshold"])
        out["cal_threshold"].append(s_cal["best_threshold"])

    out["mean_raw_f1"] = float(np.mean(out["raw_f1"]))
    out["mean_cal_f1"] = float(np.mean(out["cal_f1"]))
    return out
