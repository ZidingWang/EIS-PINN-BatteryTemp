"""Experiment configuration loaded from YAML files in ``configs/``.

Each script reads a single YAML file and turns it into a typed dataclass so the
rest of the code never touches raw dictionaries or hard-coded paths.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, fields
from typing import List

import yaml

# Repository root = parent folder of this ``src`` package.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(path: str) -> str:
    """Resolve a path from the YAML relative to the repository root."""
    return path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)


def _filter_known(cls, data: dict) -> dict:
    """Keep only keys that correspond to dataclass fields (ignore comments)."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


# Physical temperature window shared by the model output and the physics loss.
@dataclass
class ModelConfig:
    hidden_units: List[int] = field(default_factory=lambda: [128, 128, 128, 128, 128, 128])
    use_residual_blocks: bool = True
    tmin_c: float = -20.0
    tmax_c: float = 45.0
    # Arrhenius prior:  ln(1/Rct) = arr_intercept - arr_b * (1/T_kelvin)
    arr_b: float = 9558.6
    arr_intercept: float = 40.855

    @property
    def tmin_k(self) -> float:
        return self.tmin_c + 273.15

    @property
    def tmax_k(self) -> float:
        return self.tmax_c + 273.15


def model_config_path(weights_path: str) -> str:
    """Return the sidecar path used to persist the model configuration."""
    suffix = ".weights.h5"
    if weights_path.endswith(suffix):
        return weights_path[:-len(suffix)] + ".model.json"
    if weights_path.endswith(".h5"):
        return weights_path[:-3] + ".model.json"
    return weights_path + ".model.json"


def save_model_config(weights_path: str, model_cfg: ModelConfig) -> str:
    """Save architecture and Arrhenius parameters next to model weights."""
    path = model_config_path(weights_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": {
            "hidden_units": list(model_cfg.hidden_units),
            "use_residual_blocks": model_cfg.use_residual_blocks,
            "tmin_c": model_cfg.tmin_c,
            "tmax_c": model_cfg.tmax_c,
            "arr_b": model_cfg.arr_b,
            "arr_intercept": model_cfg.arr_intercept,
        }}, f, ensure_ascii=False, indent=2)
    return path


def load_model_config(weights_path: str) -> ModelConfig:
    """Load a persisted model configuration, falling back to defaults."""
    path = model_config_path(weights_path)
    if not os.path.exists(path):
        return ModelConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f) or {}
    return ModelConfig(**_filter_known(ModelConfig, raw.get("model", raw)))


@dataclass
class TransferConfig:
    """Configuration for last-layer transfer learning (628Ah -> 314Ah)."""
    # --- paths ---
    target_file: str = "data/eis_features_314Ah.xlsx"
    target_sheet: str = "Selected_Features"
    preprocessor_dir: str = "artifacts/preprocessors"
    weights_dir: str = "artifacts/weights"
    feature_name: str = "feature_columns_628Ah.json"
    imputer_name: str = "imputer_628Ah.pkl"
    scaler_name: str = "scaler_628Ah.pkl"
    pretrained_weights_name: str = "pinn_628Ah_pretrained.weights.h5"
    finetuned_weights_name: str = "pinn_628_to_314_finetuned.weights.h5"
    out_dir: str = "outputs/314Ah_transfer"
    # --- split ---
    seed: int = 25
    ft_fraction: float = 0.20
    # --- fine-tuning ---
    ft_method: str = "finetune"        # "finetune" (gradient) or "ridge"
    ft_epochs: int = 1200
    ft_lr: float = 3.5e-4
    batch_size: int = 4
    huber_delta: float = 8.0
    lam: float = 0.3                   # physics-loss weight
    ridge_alpha: float = 0.3
    # --- cosine-restart learning-rate schedule ---
    use_lr_schedule: bool = True
    lr_first_fraction: float = 0.25
    lr_m_mul: float = 0.8
    lr_alpha: float = 0.02
    # --- model ---
    model: ModelConfig = field(default_factory=ModelConfig)

    # resolved absolute paths -------------------------------------------------
    @property
    def feature_path(self) -> str:
        return _abs(os.path.join(self.preprocessor_dir, self.feature_name))

    @property
    def imputer_path(self) -> str:
        return _abs(os.path.join(self.preprocessor_dir, self.imputer_name))

    @property
    def scaler_path(self) -> str:
        return _abs(os.path.join(self.preprocessor_dir, self.scaler_name))

    @property
    def pretrained_weights_path(self) -> str:
        return _abs(os.path.join(self.weights_dir, self.pretrained_weights_name))

    @property
    def finetuned_weights_path(self) -> str:
        return _abs(os.path.join(self.weights_dir, self.finetuned_weights_name))

    @property
    def target_path(self) -> str:
        return _abs(self.target_file)

    @property
    def out_path(self) -> str:
        return _abs(self.out_dir)

    @classmethod
    def from_yaml(cls, path: str) -> "TransferConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        model = ModelConfig(**_filter_known(ModelConfig, raw.pop("model", {}) or {}))
        cfg = cls(**_filter_known(cls, raw))
        cfg.model = model
        # Clamp fine-tune fraction to a sensible range (<=20% labels).
        cfg.ft_fraction = min(max(cfg.ft_fraction, 0.10), 0.20)
        return cfg


@dataclass
class PretrainConfig:
    """Configuration for pre-training the source 628Ah PINN model."""
    file_path: str = "data/eis_features_628Ah.xlsx"
    sheet_name: str = "Selected_Features"
    out_dir: str = "outputs/pretrain_628Ah"
    seed: int = 42
    final_seed: int = 2024
    cv_seed: int = 2026
    test_size: float = 0.20
    n_splits: int = 5
    max_epochs: int = 800
    patience: int = 100
    lr_patience: int = 40
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    learning_rate: float = 1e-3
    batch_size: int = 8
    huber_delta: float = 0.5
    lam: float = 0.3
    final_epochs_override: int = 299
    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def file_abs(self) -> str:
        return _abs(self.file_path)

    @property
    def out_path(self) -> str:
        return _abs(self.out_dir)

    @classmethod
    def from_yaml(cls, path: str) -> "PretrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        model = ModelConfig(**_filter_known(ModelConfig, raw.pop("model", {}) or {}))
        cfg = cls(**_filter_known(cls, raw))
        cfg.model = model
        return cfg
