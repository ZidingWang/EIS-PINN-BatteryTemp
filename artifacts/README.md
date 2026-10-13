# `artifacts/`

Model artifacts that the transfer pipeline consumes and produces. The files
here are small (< 1.5 MB total) and **are committed** so the repo runs out of
the box.

```text
artifacts/
├── preprocessors/
│   ├── feature_columns_628Ah.json   # ordered list of model input features
│   ├── imputer_628Ah.pkl            # fitted SimpleImputer (median)
│   └── scaler_628Ah.pkl             # fitted RobustScaler
└── weights/
    ├── pinn_628Ah_pretrained.weights.h5      # source 628Ah PINN (input)
    └── pinn_628_to_314_finetuned.weights.h5  # 314Ah fine-tuned head (output)
```

| File | Role |
|---|---|
| `preprocessors/feature_columns_628Ah.json` | feature column order expected by the model |
| `preprocessors/imputer_628Ah.pkl` | median imputer fitted on the 628Ah data |
| `preprocessors/scaler_628Ah.pkl` | robust scaler fitted on the 628Ah data |
| `weights/pinn_628Ah_pretrained.weights.h5` | **input** — pretrained source weights loaded by `train_transfer.py` |
| `weights/pinn_628_to_314_finetuned.weights.h5` | **output** — written by `train_transfer.py` after fine-tuning |

## Regenerating the source artifacts

Run the pre-training stage and copy its outputs here:

```bash
python -m src.pretrain_628Ah --config configs/pretrain_628Ah.yaml
cp outputs/pretrain_628Ah/preprocessors/* artifacts/preprocessors/
cp outputs/pretrain_628Ah/weights/*       artifacts/weights/
```

The transfer script raises `FileNotFoundError` if any preprocessor or the
pretrained weight file is missing.
