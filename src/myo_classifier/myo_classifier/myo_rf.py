#!/usr/bin/env python3
import os, json, pickle
from pathlib import Path
from collections import deque, Counter

import numpy as np
import rclpy
from rclpy.node import Node
from myo_msgs.msg import MyoMsg
from std_msgs.msg import String

# --------- Match training feature definitions exactly ---------
def _waveform_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))

def _zero_crossings(x: np.ndarray, thresh: float) -> int:
    # Training schema: count sign changes AND require amp jump > thresh
    if x.ndim != 1:
        x = np.ravel(x)
    s = np.sign(x)
    ds = np.diff(s)
    amp_jump = np.abs(np.diff(x))
    return int(np.sum((ds != 0) & (amp_jump > thresh)))

def _slope_sign_changes(x: np.ndarray, thresh: float) -> int:
    # Training schema: sign flip in consecutive slopes; either step > thresh
    if x.ndim != 1:
        x = np.ravel(x)
    dx1 = np.diff(x)
    sign_change = (dx1[:-1] * dx1[1:]) < 0
    big_enough = (np.abs(dx1[:-1]) > thresh) | (np.abs(dx1[1:]) > thresh)
    return int(np.sum(sign_change & big_enough))

class EMGClassifierRF(Node):
    def __init__(self):
        super().__init__('myo_rf')

        # ---------- Params ----------
        self.declare_parameter('artifacts_dir', str(Path.home() / 'runs'))
        self.declare_parameter('classifier_file', 'gesture_classifier.pkl')
        self.declare_parameter('label_encoder_file', 'label_encoder.pkl')
        self.declare_parameter('metadata_file', 'rf_metadata.json')
        # If window_size <= 0, we'll derive from metadata (window_sec * sampling_rate)
        self.declare_parameter('window_size', -1)
        # Voting to stabilize output
        self.declare_parameter('vote_k', 5)
        # Allow overriding thresholds if needed (else from metadata)
        self.declare_parameter('zc_thresh', None)
        self.declare_parameter('ssc_thresh', None)

        artifacts_dir = os.path.expanduser(self.get_parameter('artifacts_dir').value)
        clf_path = os.path.join(artifacts_dir, self.get_parameter('classifier_file').value)
        le_path  = os.path.join(artifacts_dir, self.get_parameter('label_encoder_file').value)
        meta_path= os.path.join(artifacts_dir, self.get_parameter('metadata_file').value)

        # ---------- Load artifacts ----------
        with open(clf_path, 'rb') as f:
            self.clf = pickle.load(f)
        with open(le_path, 'rb') as f:
            self.le = pickle.load(f)

        # Metadata (window + thresholds) — mirrors training defaults if missing
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except Exception as e:
                self.get_logger().warn(f'Could not read metadata {meta_path}: {e}')

        # window_len = round(window_sec * sampling_rate)
        sampling_rate = meta.get('sampling_rate', 200.0) or 200.0
        window_sec    = meta.get('window_sec', 0.2) or 0.2
        auto_W = int(round(float(sampling_rate) * float(window_sec)))

        # thresholds (with metadata defaults)
        meta_zc  = meta.get('zc_thresh', 0.01)
        meta_ssc = meta.get('ssc_thresh', 0.01)

        # Apply param overrides (if provided)
        param_W = int(self.get_parameter('window_size').value)
        self.W = auto_W if param_W <= 0 else param_W

        zc_param  = self.get_parameter('zc_thresh').value
        ssc_param = self.get_parameter('ssc_thresh').value
        self.zc_thresh  = float(meta_zc if zc_param in (None, '') else zc_param)
        self.ssc_thresh = float(meta_ssc if ssc_param in (None, '') else ssc_param)

        self.vote_k = int(self.get_parameter('vote_k').value)

        exp = getattr(self.clf, 'n_features_in_', None)
        self.get_logger().info(
            f'Loaded classifier ({clf_path}), expects {exp} feats | '
            f'classes={list(self.le.classes_)}'
        )
        self.get_logger().info(
            f'Metadata: sampling_rate={sampling_rate}, window_sec={window_sec} -> W={auto_W}, '
            f'zc_thresh={self.zc_thresh}, ssc_thresh={self.ssc_thresh}; Runtime W={self.W}'
        )

        # ---------- Buffers & I/O ----------
        self.emg_buf = deque(maxlen=self.W)
        self.pred_buf = deque(maxlen=self.vote_k)
        self.sub = self.create_subscription(MyoMsg, 'myo/data', self.cb, 10)
        self.pub_label = self.create_publisher(String, 'myo/pred', 10)

        # Optional sanity: warn if feature count mismatch
        if exp is not None and exp != 72:
            self.get_logger().warn(
                f'Classifier n_features_in_={exp} but runtime will emit 72; '
                f'ensure the model was trained with this 9x8 feature pack.'
            )

    # Exact training feature order:
    # [MAV, WL, RMS, ZC, SSC, Mean, Median, Std, Var] × 8 channels
    def _features_train_exact(self, Xw: np.ndarray) -> np.ndarray:
        feats = []
        for ch in range(Xw.shape[1]):
            x = Xw[:, ch]
            mav = float(np.mean(np.abs(x)))
            wl  = _waveform_length(x)
            rms = float(np.sqrt(np.mean(x ** 2)))
            zc  = _zero_crossings(x, self.zc_thresh)
            ssc = _slope_sign_changes(x, self.ssc_thresh)
            mu  = float(np.mean(x))
            med = float(np.median(x))
            sd  = float(np.std(x, ddof=0))
            var = float(np.var(x, ddof=0))
            feats.extend([mav, wl, rms, zc, ssc, mu, med, sd, var])
        return np.asarray(feats, dtype=float)

    def _predict(self, feats: np.ndarray):
        y = self.clf.predict(feats)[0]
        probs = None
        if hasattr(self.clf, 'predict_proba'):
            try:
                probs = self.clf.predict_proba(feats)[0]
            except Exception:
                probs = None
        return y, probs

    def _topk_str(self, probs, k=3):
        if probs is None:
            return ''
        idx = np.argsort(probs)[::-1][:k]
        items = [f'{self.le.inverse_transform([i])[0]}={probs[i]:.2f}' for i in idx]
        return ' | ' + ' '.join(items)

    def cb(self, msg: MyoMsg):
        if len(msg.emg_data) != 8:
            return

        self.emg_buf.append(np.array(msg.emg_data, dtype=float))
        if len(self.emg_buf) < self.W:
            return

        Xw = np.stack(self.emg_buf, axis=0)  # (W, 8)
        feats = self._features_train_exact(Xw).reshape(1, -1)

        try:
            y_pred, probs = self._predict(feats)
        except Exception as e:
            self.get_logger().error(f'Prediction failed: {e}')
            return

        try:
            label = self.le.inverse_transform([y_pred])[0]
        except Exception:
            label = str(y_pred)

        # Majority vote smoothing
        self.pred_buf.append(label)
        voted = Counter(self.pred_buf).most_common(1)[0][0]

        self.get_logger().info(f'Pred: {label} -> voted: {voted}{self._topk_str(probs)}')
        self.pub_label.publish(String(data=voted))

def main(args=None):
    rclpy.init(args=args)
    node = EMGClassifierRF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
