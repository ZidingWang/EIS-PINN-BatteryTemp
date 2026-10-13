"""PINN architecture, Arrhenius physics loss and optimizers.

The network maps EIS features to a battery temperature. A second ``rct_true_in``
input feeds the measured charge-transfer resistance into an Arrhenius physics
loss that regularises the prediction:

    physics residual = ln(1 / Rct) - (arr_intercept - arr_b / T_pred_K)

The temperature output is squashed into ``[tmin_k, tmax_k]`` with a sigmoid so
predictions stay physically plausible.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers

EPS = np.float32(1e-12)


def build_pinn(input_dim: int, model_cfg) -> Model:
    """Build the residual PINN. ``model_cfg`` is a ``config.ModelConfig``."""
    tmin_k = np.float32(model_cfg.tmin_k)
    tmax_k = np.float32(model_cfg.tmax_k)
    arr_b = np.float32(model_cfg.arr_b)
    arr_intercept = np.float32(model_cfg.arr_intercept)
    hidden_units = model_cfg.hidden_units

    features = Input(shape=(input_dim,), name="features")
    rct_true_in = Input(shape=(1,), name="rct_true_in")

    x = features
    if model_cfg.use_residual_blocks:
        x = layers.Dense(hidden_units[0], activation="relu")(x)
        for units in hidden_units[1:]:
            skip = x
            h = layers.Dense(units, activation="relu")(x)
            h = layers.Dense(units)(h)
            x = layers.ReLU()(layers.Add()([skip, h]))
    else:
        for units in hidden_units:
            x = layers.Dense(units, activation="relu")(x)

    t_raw = layers.Dense(1)(x)
    t_out = layers.Lambda(
        lambda z: tf.constant(tmin_k, tf.float32)
        + (tf.constant(tmax_k, tf.float32) - tf.constant(tmin_k, tf.float32)) * tf.nn.sigmoid(z),
        name="T_output",
    )(t_raw)

    def physics_loss(inputs):
        t_pred, rct = inputs
        rct = tf.maximum(tf.cast(rct, tf.float32), tf.constant(EPS, tf.float32))
        t_pred = tf.clip_by_value(tf.cast(t_pred, tf.float32), tmin_k, tmax_k)
        lhs = tf.math.log(1.0 / rct)
        rhs = tf.constant(arr_intercept, tf.float32) - tf.constant(arr_b, tf.float32) / t_pred
        return lhs - rhs

    phys_out = layers.Lambda(physics_loss, name="Phys_output")([t_out, rct_true_in])
    return Model([features, rct_true_in], [t_out, phys_out])


def make_optimizer(lr, *, schedule_steps=None, first_fraction=0.25, m_mul=0.8, alpha=0.02):
    """Adam optimizer; cosine-restart schedule when ``schedule_steps`` is given."""
    if schedule_steps is not None:
        schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=max(1, int(schedule_steps * first_fraction)),
            t_mul=2.0,
            m_mul=m_mul,
            alpha=alpha,
        )
        return tf.keras.optimizers.Adam(learning_rate=schedule)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def compile_pinn(model: Model, optimizer, huber_delta: float, lam: float) -> None:
    model.compile(
        optimizer=optimizer,
        loss={"T_output": tf.keras.losses.Huber(delta=huber_delta), "Phys_output": "mse"},
        loss_weights={"T_output": 1.0, "Phys_output": lam},
    )


def predict_temp_c(model: Model, X, rct):
    """Predict temperature in Celsius."""
    pred_k = model.predict({"features": X, "rct_true_in": rct}, verbose=0)[0].ravel()
    return pred_k - 273.15


def freeze_all_but_last_regression_layer(model: Model) -> str:
    """Freeze every layer except the final Dense(1) regression head."""
    for layer in model.layers:
        layer.trainable = False
    dense_layers = [layer for layer in model.layers if isinstance(layer, layers.Dense)]
    last_dense = dense_layers[-1]
    last_dense.trainable = True
    print(f"Trainable layer after freezing: {last_dense.name}")
    return last_dense.name


# --- optional closed-form last-layer solver (ft_method = "ridge") -------------
def kelvin_to_logit(y_k, tmin_k, tmax_k):
    p = (np.asarray(y_k).ravel() - tmin_k) / (tmax_k - tmin_k)
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p)).reshape(-1, 1).astype(np.float32)


def fit_last_layer_ridge(H, y_k, alpha, tmin_k, tmax_k):
    """Solve the frozen-feature -> logit regression in closed form."""
    A = np.c_[H, np.ones((len(H), 1), dtype=np.float32)]
    z = kelvin_to_logit(y_k, tmin_k, tmax_k)
    if alpha == 0:
        coef = np.linalg.lstsq(A, z, rcond=None)[0]
    else:
        reg = np.eye(A.shape[1], dtype=np.float32) * alpha
        reg[-1, -1] = 0.0
        coef = np.linalg.solve(A.T @ A + reg, A.T @ z)
    return coef[:-1].astype(np.float32), coef[-1].astype(np.float32)
