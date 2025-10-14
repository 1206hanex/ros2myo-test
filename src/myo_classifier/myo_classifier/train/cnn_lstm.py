# myo_classifier/train/cnn_lstm.py
import os, json, glob, pickle
from typing import List, Tuple, Union, Iterable, Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# ------------------------- CSV helpers (same behavior as RF/SVM) -------------------------

def _list_csvs(data_dir: str, pattern: str = "*.csv") -> List[str]:
    d = os.path.expanduser(data_dir)
    files = sorted(glob.glob(os.path.join(d, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {d} (pattern: {pattern}).")
    return files

def _pick_emg_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.lower().startswith("emg_")]
    if not cols:
        raise ValueError("No EMG columns found (expected headers like emg_1..emg_8).")
    def _key(c):
        try:
            return int(c.split("_", 1)[1])
        except Exception:
            return c
    return sorted(cols, key=_key)

def _trials_from_df(df: pd.DataFrame, emg_cols: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Returns (trials, labels). If multiple labels appear in one CSV, split into contiguous runs.
    """
    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column.")
    labels = df["label"].astype(str).to_numpy()
    emg = df[emg_cols].to_numpy(dtype=np.float32)

    trials, labs = [], []
    if len(np.unique(labels)) <= 1:
        if len(emg) > 0:
            trials.append(emg)
            labs.append(labels[0] if len(labels) else "unknown")
        return trials, labs

    change_idx = np.where(labels[1:] != labels[:-1])[0] + 1
    starts = np.r_[0, change_idx]
    ends   = np.r_[change_idx, len(labels)]
    for s, e in zip(starts, ends):
        seg = emg[s:e]
        if len(seg) >= 4:
            trials.append(seg)
            labs.append(labels[s])
    return trials, labs


# ------------------------- Windowing & normalization -------------------------

def _to_windows(trials: List[np.ndarray], labels: List[Union[str, int]],
                timesteps: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    trials: list of (T_i, C) arrays
    Returns Xw: [N, T, C], yw: [N]
    """
    Xw, yw = [], []
    for arr, lab in zip(trials, labels):
        T = arr.shape[0]
        for s in range(0, max(1, T - timesteps + 1), step):
            win = arr[s:s+timesteps, :]
            if win.shape[0] == timesteps:
                Xw.append(win)
                yw.append(lab)
    if not Xw:
        raise RuntimeError("No windows produced — check window_sec/overlap or trial lengths.")
    return np.stack(Xw).astype(np.float32), np.asarray(yw)

def _standardize(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Channel-wise standardization using dataset mean/std.
    X: [N, T, C]
    """
    mean = X.mean(axis=(0, 1), keepdims=True)
    std  = X.std(axis=(0, 1), keepdims=True)
    Xn = (X - mean) / (std + eps)
    return Xn.astype(np.float32), {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


# ------------------------- Model -------------------------

def build_cnn_lstm(timesteps: int, channels: int, n_classes: int,
                   conv_filters=(32, 64), lstm_units=64, dropout=0.2) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(timesteps, channels), name="emg_seq")
    x = inp
    for f in conv_filters:
        x = tf.keras.layers.Conv1D(f, 5, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LSTM(lstm_units)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="cnn_lstm_emg")


# ------------------------- Core training (can be called directly) -------------------------

def train_cnn_lstm(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Union[np.ndarray, List[Union[str, int]]],
    out_dir: str,
    *,
    sampling_rate: float = 200.0,
    window_sec: float = 0.2,
    overlap: float = 0.5,
    epochs: int = 30,
    batch_size: int = 64,
    lstm_units: int = 64,
    conv_filters=(32, 64),
    learning_rate: float = 1e-3,
    random_state: int = 42,
    logger=print
) -> Tuple[str, str, str]:
    """
    - If X is a list of raw trials [(T_i, C), ...], window → standardize → train.
    - If X is already windows [N, T, C], we standardize → train.
    Saves:
      - SavedModel at {out_dir}/saved_model
      - Label encoder at {out_dir}/label_encoder.pkl
      - Norm stats at {out_dir}/norm_stats.npz
      - Metadata JSON at {out_dir}/cnn_lstm_metadata.json
    Returns: (saved_model_dir, le_path, norm_stats_path)
    """
    os.makedirs(out_dir, exist_ok=True)
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    timesteps = max(1, int(round(window_sec * sampling_rate)))
    step = max(1, int(round(timesteps * (1.0 - overlap))))

    # Assemble windows
    if isinstance(X, list):
        C = int(X[0].shape[1])
        Xw, yw = _to_windows(X, y, timesteps=timesteps, step=step)
    else:
        X = np.asarray(X, dtype=np.float32)
        C = int(X.shape[2])
        Xw, yw = X, np.asarray(y)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(yw)
    n_classes = int(len(le.classes_))

    # Standardize
    Xw, norm_stats = _standardize(Xw)

    # Split
    X_tr, X_va, y_tr, y_va = train_test_split(
        Xw, y_enc, test_size=0.2, stratify=y_enc, random_state=random_state
    )

    # Build & train
    model = build_cnn_lstm(timesteps=timesteps, channels=C, n_classes=n_classes,
                           conv_filters=conv_filters, lstm_units=lstm_units)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cb = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=0),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, verbose=0)
    ]
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=0)

    # Report
    yhat = model.predict(X_va, verbose=0).argmax(axis=1)
    rep = classification_report(y_va, yhat, target_names=list(le.classes_))
    logger("\n" + rep)

    # Save artifacts
    saved_model_dir = os.path.join(out_dir, "saved_model")
    model.save(saved_model_dir, include_optimizer=False)

    le_path   = os.path.join(out_dir, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    norm_path = os.path.join(out_dir, "norm_stats.npz")
    np.savez(norm_path, **norm_stats)

    meta = {
        "timesteps": timesteps, "channels": C, "window_sec": window_sec, "overlap": overlap,
        "conv_filters": list(conv_filters), "lstm_units": lstm_units,
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "classes_": list(le.classes_)
    }
    with open(os.path.join(out_dir, "cnn_lstm_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return saved_model_dir, le_path, norm_path


# ------------------------- CSV directory wrapper for Manager -------------------------

def train(
    data_dir: str,
    out_dir: str,
    *,
    sampling_rate: float = 200.0,
    window_sec: float = 0.2,
    overlap: float = 0.5,
    epochs: int = 30,
    batch_size: int = 64,
    lstm_units: int = 64,
    conv_filters=(32, 64),
    learning_rate: float = 1e-3,
    random_state: int = 42,
    logger=print,
    **kwargs
):
    """
    Entry point for Manager._call_module_train().
    Reads ALL *.csv under data_dir (raw EMG + label), assembles trials → windows, and trains CNN-LSTM.
    Returns a dict with model_path (SavedModel directory) and a short detail string.
    """
    data_dir = os.path.expanduser(data_dir)
    out_dir  = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = _list_csvs(data_dir)
    all_trials, all_labels = [], []
    for f in files:
        df = pd.read_csv(f)
        emg_cols = _pick_emg_cols(df)
        trials, labs = _trials_from_df(df, emg_cols)
        all_trials.extend(trials)
        all_labels.extend(labs)

    if not all_trials:
        raise RuntimeError("No trials assembled from CSVs (check EMG columns and labels).")

    saved_model_dir, le_path, norm_path = train_cnn_lstm(
        X=all_trials, y=all_labels, out_dir=out_dir,
        sampling_rate=sampling_rate, window_sec=window_sec, overlap=overlap,
        epochs=epochs, batch_size=batch_size,
        lstm_units=lstm_units, conv_filters=conv_filters,
        learning_rate=learning_rate, random_state=random_state,
        logger=logger
    )

    logger(f"SavedModel: {saved_model_dir}\nLabelEncoder: {le_path}\nNormStats: {norm_path}")
    return {
        "model_path": saved_model_dir,  # Manager will surface this
        "metrics": {"note": "keras training summary printed; sklearn report for validation predictions printed to console."},
        "detail": "CNN_LSTM training completed (CSV directory loader)"
    }
