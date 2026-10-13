# EIS-PINN-BatteryTemp

Physics-informed neural network (PINN) for **lithium-ion battery temperature
estimation** from electrochemical impedance spectroscopy (EIS) features, with
**transfer learning** from a 628Ah source cell to a 314Ah target cell and an
**Arrhenius-based physics loss** on the charge-transfer resistance `Rct`.

---

## 1. Project Overview

A neural network predicts cell temperature `T` from EIS / ECM / DRT–derived
features. A second input feeds the measured charge-transfer resistance `Rct`
into an **Arrhenius physics loss**

```
ln(1 / Rct) = arr_intercept - arr_b / T_kelvin
```

so that the prediction is consistent with the physical temperature dependence
of `Rct`.

The workflow has two stages:

1. **Pre-train** a PINN on the 628Ah cell (`src/pretrain_628Ah.py`).
2. **Transfer** to the 314Ah cell (`src/train_transfer.py`): freeze every layer
   except the final regression head and fine-tune it on ≤ 20 % of the 314Ah
   labels, regularised by the 314Ah-specific Arrhenius prior.

**Results (held-out test sets):**

| Model | Cell (test set) | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|---|
| Source PINN (pre-trained) | 628Ah | **0.84** | 1.36 | **0.991** |
| Transfer — source-only (no fine-tune) | 314Ah | 14.66 | 17.66 | −0.51 |
| **Transfer — last-layer fine-tune + Arrhenius** | 314Ah | **1.36** | 1.67 | **0.986** |

The 628Ah row is the source model's own performance; the two 314Ah rows show the
target cell before and after transfer learning.

---

## 2. Environment

- Python 3.10
- TensorFlow 2.10.1 (deterministic mode; CPU or GPU)
- See [`requirements.txt`](requirements.txt) / [`environment.yml`](environment.yml)

```bash
# pip
pip install -r requirements.txt

# or conda
conda env create -f environment.yml
conda activate battery-pinn-transfer
```

---

## 3. File Structure

```text
.
├── README.md
├── requirements.txt / environment.yml
├── .gitignore / LICENSE / CITATION.cff
├── run.sh                      # one-command transfer run
│
├── configs/
│   ├── transfer_314Ah.yaml     # transfer-learning hyperparameters (paper best)
│   └── pretrain_628Ah.yaml     # source pre-training hyperparameters
│
├── src/
│   ├── config.py               # YAML -> typed dataclass config
│   ├── utils.py                # seeding & deterministic TF runtime
│   ├── metrics.py              # MAE/RMSE/R2 + prediction tables
│   ├── data_loader.py          # read spreadsheets, build splits
│   ├── feature_extraction.py   # raw (~230 cols) -> 52 model feature columns
│   ├── model.py                # PINN, Arrhenius loss, optimizer, freezing
│   ├── predict.py              # inference + compare against labels
│   ├── pretrain_628Ah.py       # stage 1: train the source model
│   └── train_transfer.py       # stage 2: transfer to 314Ah
│
├── examples/
│   ├── demo_predict.py         # extract features -> predict -> compare (both cells)
│   └── sample_data/            # raw 10-row samples (all columns), committed
│
├── artifacts/                  # committed (small, < 1.5 MB)
│   ├── preprocessors/          #   feature_columns / imputer / scaler
│   └── weights/                #   pretrained 628Ah + fine-tuned 314Ah weights
│
├── data/          (git-ignored) # input spreadsheets  -> see data/README.md
└── outputs/       (git-ignored) # predictions .xlsx + scatter .png per run
```

---

## 4. How to Run

Run from the **repository root** so that the `src` package is importable.

```bash
# Stage 2 — transfer learning to 314Ah (reproduces the paper result)
python -m src.train_transfer --config configs/transfer_314Ah.yaml
# or simply:
bash run.sh
```

```bash
# Stage 1 — pre-train the 628Ah source model (only needed to regenerate
# the artifacts; requires the 628Ah dataset, see data/README.md)
python -m src.pretrain_628Ah --config configs/pretrain_628Ah.yaml
```

```bash
# Demo — full pipeline on bundled raw samples, for BOTH cells:
#   raw (~230 cols)  ->  feature extraction (52 cols)  ->  predict  ->  compare
# Writes comparison CSVs + a scatter plot to outputs/examples/.
python -m examples.demo_predict
```

```bash
# Feature extraction only — turn a raw "all columns" spreadsheet into the
# 52-column Selected_Features table the model consumes.
python -m src.feature_extraction \
    --input  data/eis_features_628Ah_full.xlsx \
    --output data/eis_features_628Ah.xlsx
```

Any YAML field can be overridden on the command line:

```bash
python -m src.train_transfer --config configs/transfer_314Ah.yaml --lam 1.0 --ft_lr 5e-4
```

---

## 5. Input Data

Place the input spreadsheet in `data/`. It must contain a sheet named
`Selected_Features` with at least these columns:

| Column | Meaning |
|---|---|
| `Battery` | cell id |
| `SOC` | state of charge (%) |
| `T_C` **or** `T_K` | temperature (°C or K) — the prediction target |
| `Rct` | charge-transfer resistance (Ω), used by the physics loss |
| *(remaining numeric columns)* | EIS / ECM / DRT features |

The 314Ah feature columns must match the columns the 628Ah model was trained
on (stored in `artifacts/preprocessors/feature_columns_628Ah.json`). See
[`data/README.md`](data/README.md).

---

## 6. Outputs

Each run writes to `outputs/<experiment>/`:

- `*_predictions.xlsx` — per-sample predictions, metrics, and the run config
- `*_finetuned.weights.h5` — the fine-tuned model weights
- `*_scatter.png` — predicted vs. actual temperature scatter plot

---

## 7. Notes

- **Determinism.** `src/utils.configure_runtime()` enables single-threaded,
  deterministic TensorFlow so results are reproducible. A GPU is used
  automatically if one is available.
- **Random seed.** Default `seed: 25` controls both the data split and weight
  init. The reported numbers are for this seed; for a robustness study, sweep
  several seeds and report mean ± std.
- **Arrhenius parameters differ per cell.** The 314Ah prior
  (`arr_b=9558.6, arr_intercept=40.855`, i.e. `Ea=79.47 kJ/mol`) is fitted on
  the 314Ah `Rct` data. Using the wrong cell's prior markedly hurts accuracy —
  the prior must match the target cell.
- **Large files.** Datasets, model weights (`*.h5`) and `*.pkl` preprocessing
  objects are git-ignored. To share them, attach to a GitHub Release or a data
  repository (Zenodo / Figshare / Drive) and link here.
