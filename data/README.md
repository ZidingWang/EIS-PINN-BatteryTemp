# `data/`

Input spreadsheets live here. **Raw datasets are not committed** to the
repository (see [`.gitignore`](../.gitignore)); only this description is.

## Expected files

| File | Used by | Notes |
|---|---|---|
| `eis_features_314Ah.xlsx` | `src/train_transfer.py` | 314Ah target data (already feature-extracted) |
| `eis_features_628Ah.xlsx` | `src/pretrain_628Ah.py` | 628Ah source data (already feature-extracted) |

> Have **raw** spreadsheets with all ~230 EIS/ECM/DRT columns instead? Convert
> them to the 52-column `Selected_Features` format first:
> ```bash
> python -m src.feature_extraction --input <raw>.xlsx --output data/eis_features_314Ah.xlsx
> ```

## Required format

Each file must contain a worksheet named **`Selected_Features`** with at least
the following columns:

- `Battery` — cell id
- `SOC` — state of charge (%)
- `T_C` **or** `T_K` — temperature (°C or Kelvin); this is the regression target
- `Rct` — charge-transfer resistance (Ω); also drives the Arrhenius physics loss
- one column per **EIS / ECM / DRT feature**

All cells must be numeric. Rows missing any of `Battery`, `SOC`, `Rct`, or the
temperature column are dropped automatically.

> The feature columns must match the columns the 628Ah model was trained on; the
> exact list is stored in `artifacts/preprocessors/feature_columns_628Ah.json`.

## Where to get the data

The battery datasets are not public in this repository. Request them from the
authors, or point `target_file` / `file_path` in the YAML config to your own
spreadsheet that follows the format above.
