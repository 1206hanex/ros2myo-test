# myo_classifier/train/svm.py
import os, json, pickle, glob
from typing import List, Tuple, Union, Iterable, Dict

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# ------------------------- Feature extraction -------------------------

def _zero_crossings(x: np.ndarray, thresh: float) -> int:
    if x.ndim != 1:
        x = np.ravel(x)
    s = np.sign(x)
    ds = np.diff(s)
    amp_jump = np.abs(np.diff(x))
    return int(np.sum((ds != 0) & (amp_jump > thresh)))

def _slope_sign_changes(x: np.ndarray, thresh: float) -> int:
    if x.ndim != 1:
        x = np.ravel(x)
    dx1 = np.diff(x)
    sign_change = (dx1[:-1] * dx1[1:]) < 0
    big_enough = (np.abs(dx1[:-1]) > thresh) | (np.abs(dx1[1:]) > thresh)
    return int(np.sum(sign_change & big_enough))

def _waveform_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))

def _extract_features_window(win: np.ndarray,
                             zc_thresh: float,
                             ssc_thresh: float) -> Tuple[np.ndarray, List[str]]:
    if win.ndim != 2:
        raise ValueError("Window must be 2-D [T, C].")
    T, C = win.shape
    feats: List[float] = []
    names: List[str] = []
    for ch in range(C):
        x = win[:, ch]
        mav = float(np.mean(np.abs(x)))
        wl = _waveform_length(x)
        rms = float(np.sqrt(np.mean(x ** 2)))
        zc = _zero_crossings(x, zc_thresh)
        ssc = _slope_sign_changes(x, ssc_thresh)
        mu = float(np.mean(x))
        med = float(np.median(x))
        sd = float(np.std(x, ddof=0))
        var = float(np.var(x, ddof=0))
        feats.extend([mav, wl, rms, zc, ssc, mu, med, sd, var])
        names.extend([
            f"ch{ch}_MAV", f"ch{ch}_WL", f"ch{ch}_RMS", f"ch{ch}_ZC", f"ch{ch}_SSC",
            f"ch{ch}_Mean", f"ch{ch}_Median", f"ch{ch}_Std", f"ch{ch}_Var"
        ])
    return np.asarray(feats, dtype=float), names

def _sliding_windows(arr: np.ndarray, window_len: int, step: int) -> Iterable[np.ndarray]:
    T = arr.shape[0]
    if T < window_len:
        return
    for start in range(0, T - window_len + 1, step):
        yield arr[start:start + window_len, :]

def _window_and_featurize(
    trials: List[np.ndarray],
    labels: List[Union[str, int]],
    sampling_rate: float,
    window_sec: float,
    overlap: float,
    zc_thresh: float,
    ssc_thresh: float,
    min_window_var: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    assert len(trials) == len(labels), "trials and labels length mismatch"
    window_len = max(1, int(round(window_sec * sampling_rate)))
    step = max(1, int(round(window_len * (1.0 - overlap))))
    if step < 1:
        step = 1

    X_list: List[np.ndarray] = []
    y_list: List[Union[str, int]] = []
    feat_names: List[str] = []

    for trial, lab in zip(trials, labels):
        if trial.ndim != 2:
            raise ValueError("Each trial must be (T, C).")
        for win in _sliding_windows(trial, window_len, step):
            if float(np.sum(np.var(win, axis=0))) < min_window_var:
                continue
            feats, names = _extract_features_window(win, zc_thresh=zc_thresh, ssc_thresh=ssc_thresh)
            if not feat_names:
                feat_names = names
            X_list.append(feats)
            y_list.append(lab)

    if not X_list:
        raise RuntimeError("No windows produced: check window_sec/overlap or input data.")
    X = np.vstack(X_list)
    y = np.asarray(y_list)
    return X, y, feat_names

# ------------------------- Core training -------------------------

def train_svm(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Union[np.ndarray, List[Union[str, int]]],
    out_dir: str,
    logger=print,
    *,
    # Preprocessing (used if raw trials detected)
    sampling_rate: float = 200.0,
    window_sec: float = 0.2,
    overlap: float = 0.5,
    zc_thresh: float = 0.01,
    ssc_thresh: float = 0.01,
    min_window_var: float = 1e-12,
    # SVM hyperparams
    kernel: str = "rbf",
    C: float = 10.0,
    gamma: Union[str, float] = "scale",
    class_weight: Union[None, str, Dict[int, float]] = "balanced",
    probability: bool = False,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    # Detect mode: raw trials vs features
    use_raw = isinstance(X, list)
    if use_raw:
        X_feat, y_feat, feat_names = _window_and_featurize(
            trials=X, labels=y,
            sampling_rate=sampling_rate, window_sec=window_sec, overlap=overlap,
            zc_thresh=zc_thresh, ssc_thresh=ssc_thresh, min_window_var=min_window_var
        )
    else:
        X_feat = np.asarray(X, dtype=float)
        y_feat = np.asarray(y)
        feat_names = [f"f{i}" for i in range(X_feat.shape[1])]

    # Encode labels & split
    le = LabelEncoder()
    y_enc = le.fit_transform(y_feat)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_feat, y_enc, test_size=test_size, stratify=y_enc, random_state=random_state
    )

    # Pipeline: Standardize -> SVM
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=kernel, C=C, gamma=gamma,
                    class_weight=class_weight, probability=probability,
                    random_state=random_state if kernel=="linear" else None))
    ])
    pipe.fit(X_tr, y_tr)

    # Report
    report = classification_report(y_va, pipe.predict(X_va), target_names=list(le.classes_))
    logger("\n" + report)

    # Save artifacts
    model_path = os.path.join(out_dir, "gesture_classifier.pkl")
    le_path    = os.path.join(out_dir, "label_encoder.pkl")
    meta_path  = os.path.join(out_dir, "svm_metadata.json")

    with open(model_path, "wb") as f: pickle.dump(pipe, f)
    with open(le_path, "wb") as f: pickle.dump(le, f)

    metadata: Dict[str, object] = {
        "feature_names": feat_names,
        "sampling_rate": sampling_rate if use_raw else None,
        "window_sec": window_sec if use_raw else None,
        "overlap": overlap if use_raw else None,
        "zc_thresh": zc_thresh if use_raw else None,
        "ssc_thresh": ssc_thresh if use_raw else None,
        "svm": {"kernel": kernel, "C": C, "gamma": gamma,
                "class_weight": class_weight, "probability": probability},
        "test_size": test_size,
        "classes_": list(le.classes_),
        "input_mode": "raw+features" if use_raw else "features_only"
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return model_path, le_path

# ------------------------- CSV directory wrapper for Manager -------------------------

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
        try: return int(c.split("_", 1)[1])
        except Exception: return c
    return sorted(cols, key=_key)

def _trials_from_df(df: pd.DataFrame, emg_cols: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column.")
    labels = df["label"].astype(str).to_numpy()
    emg = df[emg_cols].to_numpy(dtype=float)

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

def train(
    data_dir: str,
    out_dir: str,
    *,
    # window/feature params
    sampling_rate: float = 200.0,
    window_sec: float = 0.2,
    overlap: float = 0.5,
    zc_thresh: float = 0.01,
    ssc_thresh: float = 0.01,
    min_window_var: float = 1e-12,
    # SVM hyperparams
    kernel: str = "rbf",
    C: float = 10.0,
    gamma: Union[str, float] = "scale",
    class_weight: Union[None, str, Dict[int, float]] = "balanced",
    probability: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    # misc
    logger=print,
    **kwargs
):
    """
    Entry point for Manager._call_module_train().
    Reads ALL *.csv under data_dir (raw EMG + label), assembles trials, and trains SVM.
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
        raise RuntimeError("No trials assembled from CSVs (check columns and labels).")

    model_path, le_path = train_svm(
        X=all_trials, y=all_labels, out_dir=out_dir, logger=logger,
        sampling_rate=sampling_rate, window_sec=window_sec, overlap=overlap,
        zc_thresh=zc_thresh, ssc_thresh=ssc_thresh, min_window_var=min_window_var,
        kernel=kernel, C=C, gamma=gamma, class_weight=class_weight,
        probability=probability, test_size=test_size, random_state=random_state
    )

    logger(f"Saved model: {model_path}\nSaved labels: {le_path}")
    return {
        "model_path": model_path,
        "metrics": {"note": "sklearn classification_report was printed to console."},
        "detail": "SVM training completed (CSV directory loader)"
    }
