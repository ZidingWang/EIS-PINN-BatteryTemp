"""Read EIS feature spreadsheets and build train/test splits.

The Excel file must contain a sheet (default ``Selected_Features``) with at
least the columns ``Battery``, ``SOC``, ``Rct`` and a temperature column named
either ``T_C`` (Celsius) or ``T_K`` (Kelvin). Every other numeric column is
treated as an EIS / ECM / DRT feature.
"""
import json

import numpy as np
import pandas as pd

NON_FEATURE_COLS = {"Battery", "SOC", "T_C", "T_K", "T_K_model", "T_arrhenius_K"}


def add_temperature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``T_C`` (Celsius) and ``T_K_model`` (Kelvin) from a ``T_C``/``T_K`` column."""
    temp_col = "T_C" if "T_C" in df.columns else ("T_K" if "T_K" in df.columns else None)
    if temp_col is None:
        raise KeyError("Table must contain a temperature column (T_C or T_K).")
    if temp_col == "T_C" or float(df[temp_col].median()) < 150:
        df["T_C"] = df[temp_col]
        df["T_K_model"] = df["T_C"] + 273.15
    else:
        df["T_K_model"] = df[temp_col]
        df["T_C"] = df["T_K_model"] - 273.15
    return df


def load_feature_table(file_path: str, sheet_name: str) -> pd.DataFrame:
    """Load the spreadsheet, coerce to numeric and add ``T_C`` / ``T_K_model``."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = df.columns.astype(str).str.strip()

    temp_col = "T_C" if "T_C" in df.columns else ("T_K" if "T_K" in df.columns else None)
    base_cols = ["Battery", "SOC", temp_col, "Rct"] if temp_col else []
    if temp_col is None or any(c not in df.columns for c in base_cols):
        raise KeyError(
            "Required columns missing (Battery / SOC / Rct / T_C or T_K). "
            f"Found: {df.columns.tolist()}"
        )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=base_cols).reset_index(drop=True)
    return add_temperature_columns(df)


def infer_feature_columns(df: pd.DataFrame) -> list:
    """All numeric columns except labels/targets become model features."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def load_feature_columns(feature_path: str) -> list:
    with open(feature_path, "r", encoding="utf-8") as f:
        return json.load(f)


def battery_soc_groups(battery, soc) -> np.ndarray:
    """Build a stable group key for each Battery + SOC operating condition."""
    battery = np.asarray(battery)
    soc = np.asarray(soc)
    if len(battery) != len(soc):
        raise ValueError("Battery and SOC must have the same number of rows.")
    return np.asarray([f"{b}|{s}" for b, s in zip(battery, soc)], dtype=object)


def group_train_test_split(groups, test_size: float, seed: int):
    """Split rows by whole groups so no group occurs in both partitions."""
    groups = np.asarray(groups)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two Battery+SOC groups are required for a split.")

    n_test_groups = min(
        len(unique_groups) - 1,
        max(1, int(round(test_size * len(unique_groups)))),
    )
    rng = np.random.RandomState(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    test_groups = shuffled[:n_test_groups]
    test_mask = np.isin(groups, test_groups)
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def battery_soc_finetune_test_split(battery, soc, fraction: float, seed: int):
    """Select whole Battery + SOC groups for fine-tuning and testing."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1.")
    groups = battery_soc_groups(battery, soc)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two Battery+SOC groups are required for a split.")

    n_ft_groups = min(
        len(unique_groups) - 1,
        max(1, int(round(fraction * len(unique_groups)))),
    )
    rng = np.random.RandomState(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    ft_groups = shuffled[:n_ft_groups]
    ft_mask = np.isin(groups, ft_groups)
    return np.flatnonzero(ft_mask), np.flatnonzero(~ft_mask)


def print_temp_counts(name: str, temp_c) -> None:
    vc = pd.Series(np.asarray(temp_c).round(6)).value_counts().sort_index()
    print(f"\n{name} temperature counts:")
    print(vc.to_string())
