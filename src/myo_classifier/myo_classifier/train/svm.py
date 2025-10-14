# myo_classifier/train/svm.py
import os, json, pickle
from typing import List, Tuple, Union, Iterable, Dict

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


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
    amp_jump = np.abs(np.diff(x))
    return int(np.sum((ds != 0) & (amp_jump > thresh)))


def _slope_sign_changes(x: np.ndarray, thresh: float) -> int:
    """
    Count slope sign changes with a small threshold using consecutive diffs.
    """
    if x.ndim != 1:
        x = np.ravel(x)
    dx1 = np.diff(x)
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


# ------------------------- Training entrypoint -------------------------

def train_svm(
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
    # SVM hyperparams:
    kernel: str = "rbf",          # "linear" | "rbf" | "poly" | "sigmoid"
    C: float = 10.0,              # a bit higher than 1.0 often helps with EMG features
    gamma: Union[str, float] = "scale",  # for RBF: "scale" or "auto" or float
    class_weight: Union[None, str, Dict[int, float]] = "balanced",
    probability: bool = False,    # set True if you need calibrated predict_proba (slower)
    test_size: float = 0.2
) -> Tuple[str, str]:
    """
    Train an SVM EMG gesture classifier **with built-in preprocessing and features**.

    Two usage modes:
      1) If X is (N_samples, N_features) and len(y) == N_samples,
         we assume features are already computed and train directly.
      2) If X is a list of trials [ (T_i, C) ... ] and y is a list of labels (one per trial),
         we will: window the trials (window_sec, overlap) -> extract features -> train.

    Saves:
      - gesture_classifier.pkl  (sklearn Pipeline: StandardScaler + SVC)
      - label_encoder.pkl       (sklearn LabelEncoder)
      - svm_metadata.json       (feature names, windowing params, SVM hyperparams)

    Returns:
      (model_path, label_encoder_path)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- Detect input mode ----------------
    use_raw_pipeline = False
    feat_names: List[str] = []

    if isinstance(X, list):
        use_raw_pipeline = True
    elif isinstance(X, np.ndarray):
        if X.ndim == 2 and hasattr(y, "__len__") and len(y) == X.shape[0]:
            use_raw_pipeline = False
        elif X.ndim == 2 and (not hasattr(y, "__len__") or len(y) in (0, 1)):
            # Treat as a single trial with one label
            X = [X]  # type: ignore[list-item]
            y = [y[0] if hasattr(y, "__len__") and len(y) == 1 else 0]  # type: ignore[index]
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
        X_feat, y_enc, test_size=test_size, stratify=y_enc, random_state=42
    )

    # ---------------- Build & train SVM pipeline ----------------
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=kernel, C=C, gamma=gamma,
                    class_weight=class_weight, probability=probability))
    ])
    pipe.fit(X_tr, y_tr)

    # ---------------- Report ----------------
    report = classification_report(y_va, pipe.predict(X_va), target_names=list(le.classes_))
    logger("\n" + report)

    # ---------------- Persist artifacts ----------------
    model_path = os.path.join(out_dir, "gesture_classifier.pkl")
    le_path    = os.path.join(out_dir, "label_encoder.pkl")
    meta_path  = os.path.join(out_dir, "svm_metadata.json")

    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    metadata: Dict[str, object] = {
        "feature_names": feat_names,
        "sampling_rate": sampling_rate if use_raw_pipeline else None,
        "window_sec": window_sec if use_raw_pipeline else None,
        "overlap": overlap if use_raw_pipeline else None,
        "zc_thresh": zc_thresh if use_raw_pipeline else None,
        "ssc_thresh": ssc_thresh if use_raw_pipeline else None,
        "svm": {
            "kernel": kernel,
            "C": C,
            "gamma": gamma,
            "class_weight": class_weight,
            "probability": probability
        },
        "test_size": test_size,
        "classes_": list(le.classes_),
        "input_mode": "raw+features" if use_raw_pipeline else "features_only"
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return model_path, le_path
