# `examples/sample_data/`

Tiny **raw** samples (10 rows each, all ~230 EIS/ECM/DRT columns) used by
`examples/demo_predict.py`. They are committed so the demo runs without the full
(git-ignored) datasets.

| File | Rows | Columns | Source cell |
|---|---|---|---|
| `sample_628Ah.xlsx` | 10 | all (~233) | 628Ah |
| `sample_314Ah.xlsx` | 10 | all (~233) | 314Ah |

These are **raw** tables: the demo first runs `src.feature_extraction` to select
the 52 model feature columns, then predicts. They are *not* pre-extracted.
