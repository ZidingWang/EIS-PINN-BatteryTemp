# `outputs/`

Run artifacts are written here, one sub-folder per experiment. Only the demo
results (`examples/`) are committed; **everything else is git-ignored** and
regenerated on the next run.

| Sub-folder | Produced by | Contents | Committed? |
|---|---|---|---|
| `examples/` | `examples/demo_predict.py` | per-cell comparison `.csv`, predicted-vs-actual `.png` | ✅ yes |
| `314Ah_transfer/` | `src/train_transfer.py` | predictions `.xlsx`, scatter `.png` | no |
| `628Ah_pretrain/` | source model (628Ah) | predictions `.xlsx`, scatter `.png` | no |
| `pretrain_628Ah/` | `src/pretrain_628Ah.py` | CV results `.xlsx`, `preprocessors/`, `weights/` | no |

Headline test-set MAE: **628Ah = 0.84 °C**, **314Ah = 1.36 °C**.

(The fine-tuned model weights are saved to `artifacts/weights/`, not here.)
