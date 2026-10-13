"""Battery PINN transfer-learning package.

Modules:
    config        - dataclass-based experiment configuration (loaded from YAML).
    utils         - reproducibility / TensorFlow runtime helpers.
    metrics       - regression metrics and prediction tables.
    data_loader   - read EIS feature spreadsheets and build train/test splits.
    model         - PINN architecture, optimizer and physics loss.
"""
