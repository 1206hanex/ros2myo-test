import os, glob, numpy as np, pandas as pd
from collections import Counter

def load_feature_csvs(data_dir: str):
    X_list, y_list = [], []
    for p in glob.glob(os.path.join(data_dir, "*.csv")):
        df = pd.read_csv(p)
        if "label" not in df.columns:
            continue
        feat_cols = [c for c in df.columns if c.startswith(("emg_rms_", "emg_mean_", "emg_var_"))]
        if not feat_cols:
            continue
        X_list.append(df[feat_cols].values)
        y_list.append(df["label"].values)
    if not X_list:
        return np.empty((0,)), []
    return np.vstack(X_list), np.concatenate(y_list).tolist()

def load_raw_sequences(data_dir: str, channels: int, seq_len: int, seq_stride: int):
    X_seq, y = [], []
    emg_cols = [f"emg_{i}" for i in range(channels)]
    for p in glob.glob(os.path.join(data_dir, "*.csv")):
        df = pd.read_csv(p)
        if not set(emg_cols).issubset(df.columns) or "label" not in df.columns:
            continue
        labels = df["label"].astype(str)
        label = labels.mode().iat[0] if not labels.mode().empty else str(labels.iloc[0])
        X = df[emg_cols].values.astype("float32")
        i = 0
        while i + seq_len <= X.shape[0]:
            X_seq.append(X[i:i+seq_len])
            y.append(label)
            i += seq_stride
    if not X_seq:
        return np.empty((0, seq_len, channels), dtype="float32"), []
    return np.stack(X_seq, axis=0), y
