"""Two end-to-end demos: 628Ah (source model) and 314Ah (transfer model).

Pipeline for each cell:
  1. read a RAW sample with all ~230 EIS/ECM/DRT columns,
  2. EXTRACT the 52 model feature columns (src.feature_extraction), then
  3. PREDICT temperature and compare it against the label temperature.

Run from the repository root:
    python -m examples.demo_predict
"""
import os

from src.utils import configure_runtime

configure_runtime()

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd

from src.config import ROOT_DIR
from src.feature_extraction import extract_features, load_feature_columns
from src.predict import predict_on_table, summarize

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")

PREP = os.path.join(ROOT_DIR, "artifacts", "preprocessors")
WEIGHTS = os.path.join(ROOT_DIR, "artifacts", "weights")
SAMPLE_DIR = os.path.join(ROOT_DIR, "examples", "sample_data")
OUT_DIR = os.path.join(ROOT_DIR, "outputs", "examples")

# Shared preprocessors (fitted on the 628Ah source data).
FEATURE_JSON = os.path.join(PREP, "feature_columns_628Ah.json")
IMPUTER_PATH = os.path.join(PREP, "imputer_628Ah.pkl")
SCALER_PATH = os.path.join(PREP, "scaler_628Ah.pkl")

# Raw 10-row samples (all ~230 columns) are bundled in examples/sample_data/ so
# the demo runs without the full (git-ignored) datasets.
DEMOS = [
    {
        "name": "628Ah (source PINN)",
        "cell": "628Ah",
        "data": os.path.join(SAMPLE_DIR, "sample_628Ah.xlsx"),
        "weights": os.path.join(WEIGHTS, "pinn_628Ah_pretrained.weights.h5"),
        "color": "#2F6B3D",
    },
    {
        "name": "314Ah (transfer model)",
        "cell": "314Ah",
        "data": os.path.join(SAMPLE_DIR, "sample_314Ah.xlsx"),
        "weights": os.path.join(WEIGHTS, "pinn_628_to_314_finetuned.weights.h5"),
        "color": "#C25759",
    },
]


def run_demo(demo: dict, feature_cols, imputer, scaler) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"DEMO: {demo['name']}")
    print("=" * 70)
    if not os.path.exists(demo["data"]):
        print(f"[skipped] sample data not found: {demo['data']}")
        return None

    # Step 1+2: read raw columns and extract the model's feature columns.
    raw_ncols = pd.read_excel(demo["data"]).shape[1]
    table = extract_features(demo["data"], feature_cols, sheet=0)
    print(f"Feature extraction: {raw_ncols} raw columns -> {len(feature_cols)} model features "
          f"({len(table)} rows)")

    # Step 3: predict and compare against the label temperature.
    comp = predict_on_table(table, feature_cols, imputer, scaler, demo["weights"])
    print("\nInput -> predicted vs. label temperature:\n")
    print(comp.to_string(index=False))

    m = summarize(comp)
    print(f"\nMetrics on these {len(comp)} samples: MAE={m['MAE']:.3f} °C  "
          f"RMSE={m['RMSE']:.3f} °C  MaxError={m['MaxError']:.3f} °C")
    print("(Note: a tiny 10-row demo sample — not the held-out test set, so it is "
          "not directly comparable to the README test metrics.)")
    return comp


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    feature_cols = load_feature_columns(FEATURE_JSON)
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)

    plt.figure(figsize=(6.5, 6.5))
    lo, hi = -20, 45
    plt.plot([lo, hi], [lo, hi], "--", color="gray", lw=1.2, label="ideal")

    for demo in DEMOS:
        comp = run_demo(demo, feature_cols, imputer, scaler)
        if comp is None:
            continue
        csv_path = os.path.join(OUT_DIR, f"comparison_{demo['cell']}.csv")
        comp.to_csv(csv_path, index=False)
        print(f"Saved comparison: {csv_path}")
        m = summarize(comp)
        plt.scatter(comp["T_true_C"], comp["T_pred_C"], s=22, alpha=0.7,
                    edgecolors=demo["color"], facecolors="none",
                    label=f"{demo['name']}  (MAE={m['MAE']:.2f}°C)")

    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.axis("equal")
    plt.grid(True, ls=":", lw=0.7, alpha=0.7)
    plt.legend(frameon=True, fontsize=9)
    plt.title("Predicted vs. actual temperature")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "predicted_vs_actual_temperature.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\nSaved figure: {fig_path}")


if __name__ == "__main__":
    main()
