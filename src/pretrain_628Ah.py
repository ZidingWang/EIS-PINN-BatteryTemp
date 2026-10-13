"""Pre-train the source 628Ah PINN model (5-fold CV + final fit).

Produces the artifacts consumed by ``train_transfer.py`` (under the run's
``outputs/pretrain_628Ah/`` folder; copy them into ``artifacts/``):
    weights/pinn_628Ah_pretrained.weights.h5
    preprocessors/imputer_628Ah.pkl
    preprocessors/scaler_628Ah.pkl
    preprocessors/feature_columns_628Ah.json

Usage
-----
    python -m src.pretrain_628Ah --config configs/pretrain_628Ah.yaml
"""
import argparse
import json
import os

from src.utils import configure_runtime

configure_runtime()

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from src.config import PretrainConfig
from src.data_loader import (
    battery_soc_groups,
    group_train_test_split,
    infer_feature_columns,
    load_feature_table,
)
from src.config import save_model_config
from src.metrics import metrics_c, pred_frame
from src.model import build_pinn, compile_pinn, make_optimizer
from src.utils import enable_tf_determinism, set_seed


def parse_args() -> PretrainConfig:
    p = argparse.ArgumentParser(description="Pre-train 628Ah source PINN")
    p.add_argument("--config", default="configs/pretrain_628Ah.yaml")
    args = p.parse_args()
    return PretrainConfig.from_yaml(args.config)


def scale_fit_transform(X_tr, X_va):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_tr = scaler.fit_transform(imputer.fit_transform(X_tr)).astype(np.float32)
    X_va = scaler.transform(imputer.transform(X_va)).astype(np.float32)
    return X_tr, X_va


def fit_model(model, cfg, X, rct, y, epochs, validation_data=None):
    callbacks = []
    if validation_data is not None:
        callbacks = [
            tf.keras.callbacks.EarlyStopping("val_loss", patience=cfg.patience, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr, verbose=0),
        ]
    return model.fit(
        {"features": X, "rct_true_in": rct},
        [y, np.zeros_like(y, np.float32)],
        validation_data=validation_data,
        epochs=epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=0,
    )


def new_model(cfg, input_dim, seed):
    tf.keras.backend.clear_session()
    set_seed(seed)
    model = build_pinn(input_dim, cfg.model)
    compile_pinn(model, make_optimizer(cfg.learning_rate), cfg.huber_delta, cfg.lam)
    return model


def main() -> None:
    cfg = parse_args()
    enable_tf_determinism()
    set_seed(cfg.seed)

    # ---------------- data ----------------
    df = load_feature_table(cfg.file_abs, cfg.sheet_name)
    feature_cols = infer_feature_columns(df)
    if "T_arrhenius_K" in feature_cols:
        raise ValueError("T_arrhenius_K must not enter model features.")

    X = df[feature_cols].to_numpy(np.float32)
    y_T = df[["T_K_model"]].to_numpy(np.float32)
    y_Rct = df[["Rct"]].to_numpy(np.float32)
    battery = df["Battery"].astype(int).to_numpy()
    soc = df["SOC"].astype(int).to_numpy()
    temp_c = df["T_C"].round(6).to_numpy()
    print(f"Data: X={X.shape}, loss=Huber(T)+{cfg.lam:g}*Arrhenius-log-MSE, "
          f"residual_blocks={cfg.model.use_residual_blocks}")

    # ---------------- fixed train/test split ----------------
    idx = np.arange(len(X))
    groups = battery_soc_groups(battery, soc)
    trainval_idx, test_idx = group_train_test_split(groups, cfg.test_size, cfg.seed)
    X_tv, X_test = X[trainval_idx], X[test_idx]
    yT_tv, yT_test = y_T[trainval_idx], y_T[test_idx]
    yR_tv, yR_test = y_Rct[trainval_idx], y_Rct[test_idx]
    bat_tv, bat_test = battery[trainval_idx], battery[test_idx]
    soc_tv, soc_test = soc[trainval_idx], soc[test_idx]
    groups_tv = groups[trainval_idx]
    print(f"TrainVal={len(X_tv)}, Test={len(X_test)}")

    # ---------------- 5-fold CV ----------------
    cv_rows, val_parts, best_epochs = [], [], []
    gkf = GroupKFold(n_splits=cfg.n_splits)
    for fold, (tr, va) in enumerate(gkf.split(X_tv, groups=groups_tv), 1):
        X_tr, X_va = scale_fit_transform(X_tv[tr], X_tv[va])
        model = new_model(cfg, X_tr.shape[1], cfg.seed + fold * 100)
        hist = fit_model(
            model, cfg, X_tr, yR_tv[tr], yT_tv[tr], cfg.max_epochs,
            validation_data=({"features": X_va, "rct_true_in": yR_tv[va]},
                             [yT_tv[va], np.zeros_like(yT_tv[va], np.float32)]),
        )
        best_epoch = int(np.argmin(hist.history["val_loss"]) + 1)
        pred = model.predict({"features": X_va, "rct_true_in": yR_tv[va]}, verbose=0)[0].ravel() - 273.15
        true = yT_tv[va].ravel() - 273.15
        m = metrics_c(true, pred)
        best_epochs.append(best_epoch)
        cv_rows.append({"fold": fold, "best_epoch": best_epoch, **m})
        val_parts.append(pred_frame("Validation", trainval_idx[va], bat_tv[va], soc_tv[va], true, pred, fold))
        print(f"[Fold {fold}] epoch={best_epoch} MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f}")

    val_df = pd.concat(val_parts, ignore_index=True)
    val_m = metrics_c(val_df["T_true_C"], val_df["T_pred_C"])
    cv_median_epochs = max(1, int(round(float(np.median(best_epochs)))))
    final_epochs = int(cfg.final_epochs_override) if cfg.final_epochs_override else cv_median_epochs
    print(f"CV OOF: MAE={val_m['MAE']:.3f}, RMSE={val_m['RMSE']:.3f}, R2={val_m['R2']:.4f}, "
          f"cv_median_epochs={cv_median_epochs}, final_epochs={final_epochs}")

    # ---------------- final fit on all train+val, one-time test ----------------
    final_imputer = SimpleImputer(strategy="median")
    final_scaler = RobustScaler()
    X_tv_s = final_scaler.fit_transform(final_imputer.fit_transform(X_tv)).astype(np.float32)
    X_test_s = final_scaler.transform(final_imputer.transform(X_test)).astype(np.float32)
    final_model = new_model(cfg, X_tv_s.shape[1], cfg.final_seed)
    fit_model(final_model, cfg, X_tv_s, yR_tv, yT_tv, final_epochs)

    train_true = yT_tv.ravel() - 273.15
    train_pred = final_model.predict({"features": X_tv_s, "rct_true_in": yR_tv}, verbose=0)[0].ravel() - 273.15
    test_true = yT_test.ravel() - 273.15
    test_pred = final_model.predict({"features": X_test_s, "rct_true_in": yR_test}, verbose=0)[0].ravel() - 273.15
    train_m, test_m = metrics_c(train_true, train_pred), metrics_c(test_true, test_pred)
    print("\nFinal metrics")
    for name, m in [("Train", train_m), ("Val", val_m), ("Test", test_m)]:
        print(f"{name}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, R2={m['R2']:.4f}, MaxErr={m['MaxError']:.3f}")

    # ---------------- export ----------------
    # Mirror the repository's artifacts/ layout so the files can be copied
    # straight into artifacts/preprocessors and artifacts/weights.
    out_dir = cfg.out_path
    prep_dir = os.path.join(out_dir, "preprocessors")
    wts_dir = os.path.join(out_dir, "weights")
    os.makedirs(prep_dir, exist_ok=True)
    os.makedirs(wts_dir, exist_ok=True)
    weights_path = os.path.join(wts_dir, "pinn_628Ah_pretrained.weights.h5")
    final_model.save_weights(weights_path)
    save_model_config(weights_path, cfg.model)
    joblib.dump(final_imputer, os.path.join(prep_dir, "imputer_628Ah.pkl"))
    joblib.dump(final_scaler, os.path.join(prep_dir, "scaler_628Ah.pkl"))
    with open(os.path.join(prep_dir, "feature_columns_628Ah.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    train_df = pred_frame("Train", trainval_idx, bat_tv, soc_tv, train_true, train_pred)
    test_df = pred_frame("Test", test_idx, bat_test, soc_test, test_true, test_pred)
    pred_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    metrics_df = pd.DataFrame([{"split": "Train", **train_m}, {"split": "Validation", **val_m}, {"split": "Test", **test_m}])

    pred_df.to_excel(os.path.join(out_dir, "T_true_pred_train_val_test_standard.xlsx"), sheet_name="long_with_soc", index=False)
    with pd.ExcelWriter(os.path.join(out_dir, "cv_results_standard.xlsx"), engine="openpyxl") as writer:
        pd.DataFrame(cv_rows).to_excel(writer, sheet_name="cv_folds", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        pd.DataFrame({"feature": feature_cols}).to_excel(writer, sheet_name="feature_columns", index=False)
    print(f"Saved outputs to: {out_dir}")
    print(f"Saved latest weights to: {weights_path}")


if __name__ == "__main__":
    main()
