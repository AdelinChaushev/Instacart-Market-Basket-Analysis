import numpy as np
import pandas as pd


def mean_per_user_f1(user_ids, y_true, y_pred):

    d = pd.DataFrame({
        "u": np.asarray(user_ids),
        "yt": np.asarray(y_true).astype(np.int8),
        "yp": np.asarray(y_pred).astype(np.int8),
    })
    d["tp"] = ((d["yp"] == 1) & (d["yt"] == 1)).astype(np.int32)
    d["fp"] = ((d["yp"] == 1) & (d["yt"] == 0)).astype(np.int32)
    d["fn"] = ((d["yp"] == 0) & (d["yt"] == 1)).astype(np.int32)

    g = d.groupby("u")[["tp", "fp", "fn"]].sum()
    tp, fp, fn = g["tp"].values, g["fp"].values, g["fn"].values

    denom = 2 * tp + fp + fn
    # denom == 0 means the user had no positives and none were predicted: F1 = 1.0
    f1 = np.where(denom == 0, 1.0, 2 * tp / np.where(denom == 0, 1, denom))
    return float(f1.mean())


def search_threshold(user_ids, y_true, y_score, grid=None):
   
    if grid is None:
        grid = np.round(np.arange(0.05, 0.50 + 1e-9, 0.01), 2)
    grid = np.asarray(grid)

    user_ids = np.asarray(user_ids)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    scores = np.empty(len(grid), dtype=float)
    for i, tau in enumerate(grid):
        y_pred = (y_score >= tau).astype(np.int8)
        scores[i] = mean_per_user_f1(user_ids, y_true, y_pred)

    best_idx = int(np.argmax(scores))
    return {
        "best_threshold": float(grid[best_idx]),
        "best_f1": float(scores[best_idx]),
        "grid": grid,
        "scores": scores,
    }
