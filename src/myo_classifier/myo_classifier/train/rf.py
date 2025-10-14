# myo_classifier/train/rf.py
import os, json, pickle, glob
from typing import List, Tuple, Union, Iterable, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# ------------------------- Feature extraction -------------------------

def _zero_crossings(x: np.ndarray, thresh: float) -> int:
    """
    Count zero crossings with a small amplitude threshold to avoid noise-triggered flips.
    x: 1D array
    thresh: minimum jump between consecutive samples to count as a crossing
    """
    if x.ndim != 1:
        x = np.ravel(x)
    s = np.sign(x)
    ds = np.diff(s)
    # only count a crossing if the amplitude swing is meaningful
    amp_jump = np.abs(np.diff(x))
    return int(np.sum((ds != 0) & (amp_jump > thresh)))


def _slope_sign_changes(x: np.ndarray, thresh: float) -> int:
    """
    Count slope sign changes with a small threshold.
    Uses consecutive finite differences and checks sign flips with amplitude guard.
    """
    if x.ndim != 1:
        x = np.ravel(x)
    dx1 = np.diff(x)
    # A slope sign change occurs when consecutive slopes have different signs and
    # at least one of the involved steps exceeds the threshold.
    sign_change = (dx1[:-1] * dx1[1:]) < 0
    big_enough = (np.abs(dx1[:-1]) > thresh) | (np.abs(dx1[1:]) > thresh)
    return int(np.sum(sign_change & big_enough))


def _waveform_length(x: np.ndarray) -> float:
    """Sum of absolute successive differences."""
    return float(np.sum(np.abs(np.diff(x))))


def _extract_features_window(win: np.ndarray,
                             zc_thresh: float,
                             ssc_thresh: float) -> Tuple[np.ndarray, List[str]]:
    """
    Compute per-channel EMG features on a window: MAV, WL, RMS, ZC, SSC, Mean, Median, Std, Var.

    win: (window_len, n_channels)
    Returns: (feature_vector, feature_names)
    """
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
    """
    Yield contiguous windows with step (overlap = 1 - step/window_len).
    arr: (T, C)
    """
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
    """
    Convert raw EMG trials -> overlapping windows -> per-window features.

    trials: list of arrays, each (T_i, C)
    labels: list of scalar labels, one per trial
    sampling_rate: Hz
    window_sec: window size in seconds (e.g., 0.2 for 200 ms)
    overlap: fraction [0..0.9), e.g., 0.5 means 50% overlap
    zc_thresh/ssc_thresh: small amplitude guards
    min_window_var: drop windows if summed per-channel variance is below this threshold

    Returns: (X_features [N, F], y_labels [N], feature_names)
    """
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
        # Create windows
        for win in _sliding_windows(trial, window_len, step):
            # Optionally skip flat/noisy windows
            if float(np.sum(np.var(win, axis=0))) < min_window_var:
                continue
            feats, names = _extract_features_window(win, zc_thresh=zc_thresh, ssc_thresh=ssc_thresh)
            if not feat_names:
                feat_names = names  # capture once (same ordering for all windows)
            X_list.append(feats)
            y_list.append(lab)

    if not X_list:
        raise RuntimeError("No windows produced: check window_sec/overlap or input data.")
    X = np.vstack(X_list)            # [N, F]
    y = np.asarray(y_list)           # [N]
    return X, y, feat_names

def _list_csvs(data_dir: str, pattern: str = "*.csv"):
    d = os.path.expanduser(data_dir)
    files = sorted(glob.glob(os.path.join(d, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {d} (pattern: {pattern}).")
    return files

def _pick_emg_cols(df: pd.DataFrame):
    # Accept emg_0..emg_7 or emg_1..emg_8 (case-insensitive)
    cols = [c for c in df.columns if c.lower().startswith("emg_")]
    if not cols:
        raise ValueError("No EMG columns found (expected headers like emg_1..emg_8).")
    # keep column order stable (sort by numeric suffix if present)
    def _key(c):
        try:
            return int(c.split("_", 1)[1])
        except Exception:
            return c
    return sorted(cols, key=_key)

def _trials_from_df(df: pd.DataFrame, emg_cols):
    """
    Return (list_of_trials, list_of_labels).
    If multiple labels are present in one CSV, split into contiguous runs.
    """
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

    # segment by label change points
    change_idx = np.where(labels[1:] != labels[:-1])[0] + 1
    starts = np.r_[0, change_idx]
    ends   = np.r_[change_idx, len(labels)]
    for s, e in zip(starts, ends):
        seg = emg[s:e]
        if len(seg) >= 4:  # require a few samples
            trials.append(seg)
            labs.append(labels[s])
    return trials, labs

def train(
    data_dir: str,
    out_dir: str,
    *,
    # windowing/feature params (same defaults as train_rf)
    sampling_rate: float = 200.0,
    window_sec: float = 0.2,
    overlap: float = 0.5,
    zc_thresh: float = 0.01,
    ssc_thresh: float = 0.01,
    min_window_var: float = 1e-12,
    # RF hyperparams
    n_estimators: int = 150,
    random_state: int = 42,
    test_size: float = 0.2,
    # misc
    logger=print,
    **kwargs
):
    """
    Entry point for Manager._call_module_train().
    Reads ALL *.csv under data_dir (raw EMG + label), builds trials, and calls train_rf(...).
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

    model_path, le_path = train_rf(
        X=all_trials,
        y=all_labels,
        out_dir=out_dir,
        logger=logger,
        sampling_rate=sampling_rate,
        window_sec=window_sec,
        overlap=overlap,
        zc_thresh=zc_thresh,
        ssc_thresh=ssc_thresh,
        min_window_var=min_window_var,
        n_estimators=n_estimators,
        random_state=random_state,
        test_size=test_size,
    )

    logger(f"Saved model: {model_path}\nSaved labels: {le_path}")
    return {
        "model_path": model_path,
        "metrics": {"note": "sklearn classification_report was printed to console."},
        "detail": "RF training completed (CSV directory loader)"
    }


# ------------------------- Training entrypoint -------------------------

def train_rf(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Union[np.ndarray, List[Union[str, int]]],
    out_dir: str,
    logger=print,
    *,
    # Preprocessing defaults (used if we detect raw trials):
    sampling_rate: float = 200.0,
    window_sec: float = 0.2,
    overlap: float = 0.5,
    zc_thresh: float = 0.01,
    ssc_thresh: float = 0.01,
    min_window_var: float = 1e-12,
    # RF hyperparams:
    n_estimators: int = 150,
    random_state: int = 42,
    test_size: float = 0.2
) -> Tuple[str, str]:
    """
    Train a RandomForest EMG gesture classifier **with built-in preprocessing and features**.

    Two usage modes:
      1) If X is a 2-D array (N_samples, N_features) and len(y) == N_samples,
         we assume features are already computed and train directly.
      2) If X is a list of trials [ (T_i, C) ... ] and y is a list of labels (one per trial),
         we will: window the trials (window_sec, overlap) -> extract features -> train.

    Saves:
      - gesture_classifier.pkl  (sklearn RandomForestClassifier)
      - label_encoder.pkl       (sklearn LabelEncoder)
      - rf_metadata.json        (feature names, windowing params, etc.)

    Returns:
      (model_path, label_encoder_path)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- Detect input mode ----------------
    use_raw_pipeline = False
    feat_names: List[str] = []

    if isinstance(X, list):
        # list of trials -> raw pipeline
        use_raw_pipeline = True
    elif isinstance(X, np.ndarray):
        if X.ndim == 2 and hasattr(y, "__len__") and len(y) == X.shape[0]:
            use_raw_pipeline = False   # already features
        elif X.ndim == 2 and (not hasattr(y, "__len__") or len(y) in (0, 1)):
            # Treat as a single trial with one label
            X = [X]  # convert to list of one trial
            y = [y[0] if hasattr(y, "__len__") and len(y) == 1 else 0]
            use_raw_pipeline = True
        else:
            raise ValueError("Unsupported X/y shape combination.")
    else:
        raise ValueError("X must be a np.ndarray or list of np.ndarray trials.")

    # ---------------- Preprocess if needed ----------------
    if use_raw_pipeline:
        X_feat, y_feat, feat_names = _window_and_featurize(
            trials=X,  # type: ignore[arg-type]
            labels=y,  # type: ignore[arg-type]
            sampling_rate=sampling_rate,
            window_sec=window_sec,
            overlap=overlap,
            zc_thresh=zc_thresh,
            ssc_thresh=ssc_thresh,
            min_window_var=min_window_var
        )
    else:
        X_feat = np.asarray(X, dtype=float)
        y_feat = np.asarray(y)
        feat_names = [f"f{i}" for i in range(X_feat.shape[1])]

    # ---------------- Encode labels & split ----------------
    le = LabelEncoder()
    y_enc = le.fit_transform(y_feat)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_feat, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    # ---------------- Train RF ----------------
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_tr, y_tr)

    # ---------------- Report ----------------
    report = classification_report(y_va, clf.predict(X_va), target_names=list(le.classes_))
    logger("\n" + report)

    # ---------------- Persist artifacts ----------------
    model_path = os.path.join(out_dir, "gesture_classifier.pkl")
    le_path    = os.path.join(out_dir, "label_encoder.pkl")
    meta_path  = os.path.join(out_dir, "rf_metadata.json")

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    metadata: Dict[str, object] = {
        "feature_names": feat_names,
        "sampling_rate": sampling_rate if use_raw_pipeline else None,
        "window_sec": window_sec if use_raw_pipeline else None,
        "overlap": overlap if use_raw_pipeline else None,
        "zc_thresh": zc_thresh if use_raw_pipeline else None,
        "ssc_thresh": ssc_thresh if use_raw_pipeline else None,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "test_size": test_size,
        "classes_": list(le.classes_),
        "input_mode": "raw+features" if use_raw_pipeline else "features_only"
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return model_path, le_path
