"""Reproducibility and TensorFlow runtime helpers.

Import this module *before* TensorFlow whenever you need the deterministic
environment variables to take effect, e.g.::

    from src.utils import configure_runtime
    configure_runtime()          # sets env vars, then it is safe to use TF
"""
import os
import random


def configure_runtime(deterministic: bool = True) -> None:
    """Set environment variables for reproducible runs (call before TensorFlow).

    A GPU is used automatically if one is available. The single-threaded
    deterministic settings reproduce the exact numbers reported in the paper.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"


def enable_tf_determinism() -> None:
    """Force single-threaded deterministic ops on the active TF session."""
    import tensorflow as tf

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Older / newer TF versions may not expose every switch; ignore.
        pass


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow RNGs for reproducible results."""
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
