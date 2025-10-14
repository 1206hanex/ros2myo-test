import os, json, pickle
from typing import List, Union, Dict, Tuple
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def build_cnn_lstm(timesteps: int, channels: int, n_classes: int,
                   conv_filters=(32, 64), lstm_units=64, dropout=0.2):
    inp = tf.keras.Input(shape=(timesteps, channels), name="emg_seq")
    x = inp
    # 1D CNN stack (temporal convs)
    for f in conv_filters:
        x = tf.keras.layers.Conv1D(f, 5, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    # LSTM for sequence context
    x = tf.keras.layers.LSTM(lstm_units)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="cnn_lstm_emg")

def _standardize(X: np.ndarray, eps=1e-8) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # X: [N, T, C]
    mean = X.mean(axis=(0,1), keepdims=True)
    std  = X.std(axis=(0,1), keepdims=True)
    Xn = (X - mean) / (std + eps)
    return Xn, {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

def _to_windows(trials: List[np.ndarray], labels: List[Union[str, int]],
                timesteps: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    Xw, yw = [], []
    for arr, lab in zip(trials, labels):
        T = arr.shape[0]
        for s in range(0, max(1, T - timesteps + 1), step):
            win = arr[s:s+timesteps, :]
            if win.shape[0] == timesteps:
                Xw.append(win)
                yw.append(lab)
    return np.stack(Xw), np.asarray(yw)

def train_cnn_lstm(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Union[np.ndarray, List[Union[str, int]]],
    out_dir: str,
    *,
    sampling_rate: float = 200.0,
    window_sec: float = 0.2,          # 200 ms -> 40 steps at 200 Hz
    overlap: float = 0.5,
    epochs: int = 30,
    batch_size: int = 64,
    lstm_units: int = 64,
    conv_filters=(32, 64),
    learning_rate: float = 1e-3,
    random_state: int = 42,
    logger=print
) -> Tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    timesteps = max(1, int(round(window_sec * sampling_rate)))
    step = max(1, int(round(timesteps * (1.0 - overlap))))

    # 1) Accept raw trials [ (T_i, C) ... ] or already-windowed [N, T, C]
    if isinstance(X, list):  # raw trials
        C = X[0].shape[1]
        Xw, yw = _to_windows(X, y, timesteps=timesteps, step=step)
    else:
        X = np.asarray(X, dtype=np.float32)  # [N,T,C]
        C = X.shape[2]
        Xw, yw = X, np.asarray(y)

    # 2) Label encode
    le = LabelEncoder()
    y_enc = le.fit_transform(yw)
    n_classes = int(len(le.classes_))

    # 3) Standardize across dataset (mean/std per channel)
    Xw, norm_stats = _standardize(Xw)

    # 4) Train/val split
    X_tr, X_va, y_tr, y_va = train_test_split(
        Xw, y_enc, test_size=0.2, stratify=y_enc, random_state=random_state
    )

    # 5) Build + train
    model = build_cnn_lstm(timesteps=timesteps, channels=C, n_classes=n_classes,
                           conv_filters=conv_filters, lstm_units=lstm_units)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cb = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    ]
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=0)

    # 6) Report
    yhat = model.predict(X_va, verbose=0).argmax(axis=1)
    rep = classification_report(y_va, yhat, target_names=list(le.classes_))
    logger("\n" + rep)

    # 7) Save artifacts
    saved_model_dir = os.path.join(out_dir, "saved_model")
    model.save(saved_model_dir, include_optimizer=False)
    with open(os.path.join(out_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(out_dir, "norm_stats.npz"), "wb") as f:
        np.savez(f, **norm_stats)

    meta = {
        "timesteps": timesteps, "channels": C, "window_sec": window_sec, "overlap": overlap,
        "conv_filters": conv_filters, "lstm_units": lstm_units,
        "classes_": list(le.classes_)
    }
    with open(os.path.join(out_dir, "cnn_lstm_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return saved_model_dir, os.path.join(out_dir, "label_encoder.pkl"), os.path.join(out_dir, "norm_stats.npz")
