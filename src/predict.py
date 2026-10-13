"""Inference: predict temperature from a model-ready feature table and compare
it against the label temperature.

Only the network's ``T_output`` branch is used for prediction, so the Arrhenius
parameters (which only enter the training-time physics loss) do not affect the
result here — the same architecture loads either the 628Ah or the 314Ah weights.
"""
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import ModelConfig, load_model_config
from src.data_loader import load_feature_columns, load_feature_table
from src.metrics import metrics_c
from src.model import build_pinn, predict_temp_c


def predict_on_table(df, feature_cols, imputer, scaler, weights_path, model_cfg=None) -> pd.DataFrame:
    """Predict on a model-ready table (columns: features + Battery/SOC/T_C/Rct).

    Returns a comparison table: Battery, SOC, T_true_C, T_pred_C, T_error_C.
    """
    model_cfg = model_cfg if model_cfg is not None else load_model_config(weights_path)

    X = scaler.transform(imputer.transform(df[feature_cols].to_numpy(np.float32))).astype(np.float32)
    rct = df[["Rct"]].to_numpy(np.float32)

    tf.keras.backend.clear_session()
    model = build_pinn(X.shape[1], model_cfg)
    model.load_weights(weights_path)

    pred_c = predict_temp_c(model, X, rct)
    out = pd.DataFrame({
        "Battery": df["Battery"].astype(int).to_numpy(),
        "SOC": df["SOC"].astype(int).to_numpy(),
        "T_true_C": df["T_C"].to_numpy(),
        "T_pred_C": np.round(pred_c, 3),
    })
    out["T_error_C"] = np.round(out["T_pred_C"] - out["T_true_C"], 3)
    return out


def predict_and_compare(
    data_file: str,
    sheet: str,
    feature_path: str,
    imputer_path: str,
    scaler_path: str,
    weights_path: str,
    model_cfg: ModelConfig = None,
) -> pd.DataFrame:
    """Convenience wrapper for an already-extracted ``Selected_Features`` sheet."""
    df = load_feature_table(data_file, sheet)
    feature_cols = load_feature_columns(feature_path)
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    return predict_on_table(df, feature_cols, imputer, scaler, weights_path, model_cfg)


def summarize(comparison: pd.DataFrame) -> dict:
    """Overall MAE / RMSE / R2 / MaxError for a comparison table."""
    return metrics_c(comparison["T_true_C"], comparison["T_pred_C"])
