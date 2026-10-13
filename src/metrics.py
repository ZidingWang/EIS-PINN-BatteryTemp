"""Regression metrics and tidy prediction tables."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics_c(y_true, y_pred) -> dict:
    """MAE / RMSE / R2 / MaxError in degrees Celsius."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MaxError": float(np.max(np.abs(y_true - y_pred))),
    }


def pred_frame(split, rows, batteries, socs, true_c, pred_c, fold=None) -> pd.DataFrame:
    """Build a long-format prediction table (one row per sample)."""
    df = pd.DataFrame({
        "split": split,
        "row_index": np.asarray(rows).ravel(),
        "excel_row": np.asarray(rows).ravel() + 2,  # +2: header + 1-based Excel rows
        "Battery": np.asarray(batteries).ravel(),
        "SOC": np.asarray(socs).ravel(),
        "T_true_C": np.asarray(true_c).ravel(),
        "T_pred_C": np.asarray(pred_c).ravel(),
    })
    df["T_error_C"] = df["T_pred_C"] - df["T_true_C"]
    if fold is not None:
        df["fold"] = fold
    return df
