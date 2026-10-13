"""Transfer learning: 628Ah PINN  ->  314Ah battery temperature estimation.

Loads a pre-trained 628Ah PINN, freezes everything except the final regression
layer, and fine-tunes that layer on a small (<=20%) stratified subset of the
314Ah data using a Huber data loss plus an Arrhenius physics loss.

Usage
-----
    python -m src.train_transfer --config configs/transfer_314Ah.yaml

Any YAML field can be overridden on the command line, e.g.::

    python -m src.train_transfer --config configs/transfer_314Ah.yaml --lam 1.0
"""
import argparse
import os

# Runtime env vars must be set before importing TensorFlow.
from src.utils import configure_runtime

configure_runtime()

import joblib
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

matplotlib.use("Agg")  # headless backend; must be set before importing pyplot
from matplotlib import pyplot as plt

from src.config import TransferConfig
from src.data_loader import (
    battery_soc_finetune_test_split,
    load_feature_columns,
    load_feature_table,
    print_temp_counts,
)
from src.config import save_model_config
from src.metrics import metrics_c, pred_frame
from src.model import (
    build_pinn,
    compile_pinn,
    fit_last_layer_ridge,
    freeze_all_but_last_regression_layer,
    make_optimizer,
    predict_temp_c,
)
from src.utils import enable_tf_determinism, set_seed
from tensorflow.keras import Model, layers


def parse_args() -> TransferConfig:
    p = argparse.ArgumentParser(description="628Ah -> 314Ah transfer learning")
    p.add_argument("--config", default="configs/transfer_314Ah.yaml")
    # Common overrides (optional).
    p.add_argument("--seed", type=int)
    p.add_argument("--lam", type=float)
    p.add_argument("--ft_lr", type=float)
    p.add_argument("--ft_epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--huber_delta", type=float)
    p.add_argument("--ft_method", choices=["finetune", "ridge"])
    args = p.parse_args()

    cfg = TransferConfig.from_yaml(args.config)
    for key in ["seed", "lam", "ft_lr", "ft_epochs", "batch_size", "huber_delta", "ft_method"]:
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)
    return cfg


def main() -> None:
    cfg = parse_args()
    enable_tf_determinism()
    set_seed(cfg.seed)

    for path in [cfg.pretrained_weights_path, cfg.imputer_path, cfg.scaler_path, cfg.feature_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing pre-trained 628Ah artifact: {path}")

    feature_cols = load_feature_columns(cfg.feature_path)
    imputer = joblib.load(cfg.imputer_path)
    scaler = joblib.load(cfg.scaler_path)

    # ---------------- data ----------------
    df = load_feature_table(cfg.target_path, cfg.target_sheet)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"314Ah is missing feature columns aligned with 628Ah, e.g. {missing[:10]}")

    X = scaler.transform(imputer.transform(df[feature_cols].to_numpy(np.float32))).astype(np.float32)
    yT = df[["T_K_model"]].to_numpy(np.float32)
    yR = df[["Rct"]].to_numpy(np.float32)
    temp_c = df["T_C"].to_numpy(np.float32)
    battery = df["Battery"].astype(int).to_numpy()
    soc = df["SOC"].astype(int).to_numpy()

    ft_idx, test_idx = battery_soc_finetune_test_split(battery, soc, cfg.ft_fraction, cfg.seed)
    X_ft, yT_ft, yR_ft = X[ft_idx], yT[ft_idx], yR[ft_idx]
    X_test, yT_test, yR_test = X[test_idx], yT[test_idx], yR[test_idx]

    print(f"314Ah samples: total={len(df)}, finetune={len(ft_idx)}, test={len(test_idx)}, "
          f"ft_fraction={len(ft_idx) / len(df):.3f}")
    print(f"Feature dim={len(feature_cols)}, Rct in features={'Rct' in feature_cols}")
    print_temp_counts("Finetune", temp_c[ft_idx])
    print_temp_counts("Test", temp_c[test_idx])

    # ---------------- source-only baseline ----------------
    tf.keras.backend.clear_session()
    set_seed(cfg.seed)
    model = build_pinn(X.shape[1], cfg.model)
    compile_pinn(model, make_optimizer(cfg.ft_lr), cfg.huber_delta, cfg.lam)
    model.load_weights(cfg.pretrained_weights_path)
    print(f"Loaded source 628Ah weights: {cfg.pretrained_weights_path}")

    source_test_pred = predict_temp_c(model, X_test, yR_test)
    test_true = yT_test.ravel() - 273.15
    source_test_m = metrics_c(test_true, source_test_pred)

    # ---------------- last-layer fine-tuning ----------------
    trainable_layer = freeze_all_but_last_regression_layer(model)
    if cfg.ft_method == "ridge":
        raw_dense = [l for l in model.layers if isinstance(l, layers.Dense)][-1]
        feature_model = Model(model.inputs, raw_dense.input)
        H_ft = feature_model.predict({"features": X_ft, "rct_true_in": yR_ft}, verbose=0).astype(np.float32)
        W, b = fit_last_layer_ridge(H_ft, yT_ft, cfg.ridge_alpha, cfg.model.tmin_k, cfg.model.tmax_k)
        raw_dense.set_weights([W, b])
    else:
        bs = min(cfg.batch_size, len(X_ft))
        steps = None
        if cfg.use_lr_schedule:
            steps_per_epoch = int(np.ceil(len(X_ft) / bs))
            steps = max(1, cfg.ft_epochs * steps_per_epoch)
        optimizer = make_optimizer(
            cfg.ft_lr, schedule_steps=steps,
            first_fraction=cfg.lr_first_fraction, m_mul=cfg.lr_m_mul, alpha=cfg.lr_alpha,
        )
        compile_pinn(model, optimizer, cfg.huber_delta, cfg.lam)
        model.fit(
            {"features": X_ft, "rct_true_in": yR_ft},
            [yT_ft, np.zeros_like(yT_ft, np.float32)],
            epochs=cfg.ft_epochs,
            batch_size=bs,
            shuffle=True,
            verbose=0,
        )

    ft_true = yT_ft.ravel() - 273.15
    ft_pred = predict_temp_c(model, X_ft, yR_ft)
    test_pred = predict_temp_c(model, X_test, yR_test)
    ft_m = metrics_c(ft_true, ft_pred)
    test_m = metrics_c(test_true, test_pred)

    print("\n========== 314Ah Transfer Results ==========")
    print(f"Source-only Test: MAE={source_test_m['MAE']:.3f}, RMSE={source_test_m['RMSE']:.3f}, R2={source_test_m['R2']:.4f}")
    print(f"Finetune Train:   MAE={ft_m['MAE']:.3f}, RMSE={ft_m['RMSE']:.3f}, R2={ft_m['R2']:.4f}, n={len(ft_true)}")
    print(f"Finetuned Test:   MAE={test_m['MAE']:.3f}, RMSE={test_m['RMSE']:.3f}, R2={test_m['R2']:.4f}, n={len(test_true)}")
    print("===========================================\n")

    export_results(cfg, model, feature_cols, trainable_layer,
                   ft_idx, test_idx, battery, soc,
                   ft_true, ft_pred, test_true, test_pred, source_test_pred,
                   source_test_m, ft_m, test_m)


def export_results(cfg, model, feature_cols, trainable_layer,
                   ft_idx, test_idx, battery, soc,
                   ft_true, ft_pred, test_true, test_pred, source_test_pred,
                   source_test_m, ft_m, test_m) -> None:
    out_dir = cfg.out_path
    os.makedirs(out_dir, exist_ok=True)

    pred_df = pd.concat(
        [
            pred_frame("Finetune", ft_idx, battery[ft_idx], soc[ft_idx], ft_true, ft_pred),
            pred_frame("Test_SourceOnly", test_idx, battery[test_idx], soc[test_idx], test_true, source_test_pred),
            pred_frame("Test_Finetuned", test_idx, battery[test_idx], soc[test_idx], test_true, test_pred),
        ],
        ignore_index=True,
    )
    metrics_df = pd.DataFrame(
        [
            {"split": "SourceOnly_Test", **source_test_m},
            {"split": "Finetune_Train", **ft_m},
            {"split": "Finetuned_Test", **test_m},
        ]
    )
    config_df = pd.DataFrame([{
        "source_weights": cfg.pretrained_weights_path,
        "target_file": cfg.target_path,
        "ft_fraction": cfg.ft_fraction,
        "ft_n": len(ft_idx),
        "test_n": len(test_idx),
        "seed": cfg.seed,
        "ft_method": cfg.ft_method,
        "ft_epochs": cfg.ft_epochs,
        "ft_lr": cfg.ft_lr,
        "batch_size": cfg.batch_size,
        "huber_delta": cfg.huber_delta,
        "lam": cfg.lam,
        "use_lr_schedule": cfg.use_lr_schedule,
        "arr_b": cfg.model.arr_b,
        "arr_intercept": cfg.model.arr_intercept,
        "trainable_layer": trainable_layer,
        "n_features": len(feature_cols),
    }])

    pred_xlsx = os.path.join(out_dir, "314Ah_transfer_predictions.xlsx")
    with pd.ExcelWriter(pred_xlsx, engine="openpyxl") as writer:
        pred_df.to_excel(writer, sheet_name="predictions", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        config_df.to_excel(writer, sheet_name="config", index=False)
        pd.DataFrame({"feature": feature_cols}).to_excel(writer, sheet_name="feature_columns", index=False)
    print(f"Saved predictions: {pred_xlsx}")

    ft_weights = cfg.finetuned_weights_path
    os.makedirs(os.path.dirname(ft_weights), exist_ok=True)
    model.save_weights(ft_weights)
    save_model_config(ft_weights, cfg.model)
    print(f"Saved finetuned weights: {ft_weights}")

    tmin_c, tmax_c = cfg.model.tmin_c, cfg.model.tmax_c
    fig_path = os.path.join(out_dir, "314Ah_transfer_scatter.png")
    plt.figure(figsize=(6.5, 6.5))
    plt.plot([tmin_c, tmax_c], [tmin_c, tmax_c], "--", color="gray", linewidth=1.2)
    plt.scatter(ft_true, ft_pred, s=70, facecolors="none", edgecolors="#2F6B3D", marker="s", label="Finetune")
    plt.scatter(test_true, test_pred, s=80, facecolors="none", edgecolors="#C25759", marker="o", label="Test")
    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.xlim(tmin_c, tmax_c)
    plt.ylim(tmin_c, tmax_c)
    plt.axis("equal")
    plt.grid(True, ls=":", lw=0.7, alpha=0.7)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
