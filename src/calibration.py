
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression

from evaluation import mean_per_user_f1, search_threshold


def reliability_curve(y_true, y_score, n_bins=10):

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
        pbar = y_score[mask].mean()  
        ybar = y_true[mask].mean()   
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

       
        cal_scores = model.predict_proba(Xcal)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal_scores, ycal)

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
