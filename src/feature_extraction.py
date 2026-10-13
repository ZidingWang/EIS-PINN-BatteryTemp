"""Feature extraction: select the model's input columns from a full EIS table.

The raw spreadsheets (``*_全部`` / "all columns") contain ~230 columns — the
full EIS spectrum (Imag/Mag/Phase/Real at every frequency) plus all ECM and DRT
parameters. The model only consumes the 52 columns listed in
``artifacts/preprocessors/feature_columns_628Ah.json``.

This module maps the raw columns onto those 52 names and returns a tidy
"model-ready" table. Frequency column names are matched **numerically**, so the
raw ``Imag_f118.55`` is correctly mapped to the model's ``Imag_f118.550``.

CLI
---
    python -m src.feature_extraction \
        --input data/eis_features_628Ah_full.xlsx \
        --output data/eis_features_628Ah.xlsx
"""
import argparse
import os
import re

import pandas as pd

from src.data_loader import add_temperature_columns, load_feature_columns

# Repository root (parent of this src/ package).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FEATURE_JSON = os.path.join(
    ROOT_DIR, "artifacts", "preprocessors", "feature_columns_628Ah.json"
)

# Frequency-style EIS column, e.g. "Imag_f118.55" / "Phase_f7.108".
_FREQ_RE = re.compile(r"^(Imag|Mag|Phase|Real)_f([0-9.]+)$")


def _normalize_name(name: str) -> str:
    """Canonicalize a column name so float formatting no longer matters.

    ``Imag_f118.55`` and ``Imag_f118.550`` both normalize to ``Imag_f118.55``.
    Non-frequency columns are returned unchanged.
    """
    m = _FREQ_RE.match(str(name))
    if not m:
        return str(name)
    return f"{m.group(1)}_f{float(m.group(2)):g}"


def build_column_map(full_columns, feature_cols):
    """Map each requested feature column to a column present in the raw table.

    Returns ``(colmap, missing)`` where ``colmap[feature] = raw_column``.
    """
    full_columns = [str(c) for c in full_columns]
    norm_lookup = {}
    for c in full_columns:
        norm_lookup.setdefault(_normalize_name(c), c)

    colmap, missing = {}, []
    for fc in feature_cols:
        if fc in full_columns:
            colmap[fc] = fc
        elif _normalize_name(fc) in norm_lookup:
            colmap[fc] = norm_lookup[_normalize_name(fc)]
        else:
            missing.append(fc)
    return colmap, missing


def extract_features(input_file: str, feature_cols, sheet=0) -> pd.DataFrame:
    """Read a full/raw EIS spreadsheet and return a model-ready feature table.

    Output columns: ``Battery``, ``SOC``, ``T_C``, ``T_K_model`` followed by the
    52 model feature columns (renamed to their canonical model names).
    """
    df = pd.read_excel(input_file, sheet_name=sheet)
    df.columns = df.columns.astype(str).str.strip()
    if "Rct" not in df.columns:
        raise KeyError("Raw table must contain Rct for the Arrhenius physics loss.")
    feature_cols = [fc for fc in feature_cols if fc != "Rct"]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    colmap, missing = build_column_map(list(df.columns), feature_cols)
    if missing:
        raise KeyError(
            f"{len(missing)} model feature columns are absent from the raw file, "
            f"e.g. {missing[:8]}"
        )

    # Feature block, renamed to the canonical model column names and ordered
    # exactly as the model / scaler expect.
    out = pd.DataFrame({fc: df[colmap[fc]] for fc in feature_cols})

    # Metadata + temperature labels (T_C / T_K_model added on the raw frame).
    df = add_temperature_columns(df)
    out.insert(0, "Battery", df["Battery"])
    out.insert(1, "SOC", df["SOC"])
    out["T_C"] = df["T_C"]
    out["T_K_model"] = df["T_K_model"]
    out["Rct"] = df["Rct"]

    base_cols = ["Battery", "SOC", "T_C", "Rct"]
    return out.dropna(subset=[c for c in base_cols if c in out.columns]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Extract the model's feature columns from a raw EIS table")
    p.add_argument("--input", required=True, help="raw/full EIS spreadsheet (all columns)")
    p.add_argument("--output", required=True, help="output .xlsx (Selected_Features sheet)")
    p.add_argument("--sheet", default=0, help="input sheet name or index (default: first sheet)")
    p.add_argument("--feature-json", default=DEFAULT_FEATURE_JSON, help="model feature-column list")
    args = p.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    feature_cols = load_feature_columns(args.feature_json)
    table = extract_features(args.input, feature_cols, sheet=sheet)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    table.to_excel(args.output, sheet_name="Selected_Features", index=False)
    print(f"Extracted {len(feature_cols)} feature columns for {len(table)} rows.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
