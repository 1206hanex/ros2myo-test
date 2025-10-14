#!/usr/bin/env python3
import os, json, pickle
from pathlib import Path
from collections import deque, Counter

import numpy as np
import rclpy
from rclpy.node import Node
from myo_msgs.msg import MyoMsg
from std_msgs.msg import String

# ----- Feature helpers (match training definitions exactly) -----
def _waveform_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))

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
    big_enough  = (np.abs(dx1[:-1]) > thresh) | (np.abs(dx1[1:]) > thresh)
    return int(np.sum(sign_change & big_enough))

class EMGClassifierSVM(Node):
    def __init__(self):
        super().__init__('myo_svm')

        # ---------- Params ----------
        self.declare_parameter('artifacts_dir', str(Path.home() / 'runs'))
        self.declare_parameter('classifier_file', 'gesture_classifier.pkl')
        self.declare_parameter('label_encoder_file', 'label_encoder.pkl')
        # Try SVM metadata first, then RF metadata as fallback
        self.declare_parameter('metadata_file', 'svm_metadata.json')
        self.declare_parameter('fallback_metadata_file', 'rf_metadata.json')

        # If <= 0, window is inferred from metadata (window_sec * sampling_rate)
        self.declare_parameter('window_size', -1)

        # Optional scaler (recommended for SVM)
        self.declare_parameter('scaler_file', 'scaler.pkl')

        # Allow overriding thresholds (else use metadata defaults)
        self.declare_parameter('zc_thresh', None)
        self.declare_parameter('ssc_thresh', None)

        # Voting window for smoothing output labels
        self.declare_parameter('vote_k', 5)

        artifacts_dir = os.path.expanduser(self.get_parameter('artifacts_dir').value)
        clf_path  = os.path.join(artifacts_dir, self.get_parameter('classifier_file').value)
        le_path   = os.path.join(artifacts_dir, self.get_parameter('label_encoder_file').value)
        meta_path = os.path.join(artifacts_dir, self.get_parameter('metadata_file').value)
        meta_fallback = os.path.join(artifacts_dir, self.get_parameter('fallback_metadata_file').value)
        sc_path   = os.path.join(artifacts_dir, self.get_parameter('scaler_file').value)

        # ---------- Load artifacts ----------
        with open(clf_path, 'rb') as f:
            self.clf = pickle.load(f)
        with open(le_path, 'rb') as f:
            self.le = pickle.load(f)

        # Scaler (optional but ideal for SVM)
        self.scaler = None
        if os.path.exists(sc_path):
            try:
                with open(sc_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.get_logger().info(f'Loaded scaler: {sc_path}')
            except Exception as e:
                self.get_logger().warn(f'Could not load scaler ({e}); continuing without.')

        # Metadata
        meta = {}
        meta_source = None
        for candidate in (meta_path, meta_fallback):
            if os.path.exists(candidate):
                try:
                    with open(candidate, 'r') as f:
                        meta = json.load(f)
                        meta_source = candidate
                        break
                except Exception as e:
                    self.get_logger().warn(f'Could not read metadata {candidate}: {e}')
        if meta_source is None:
            self.get_logger().warn('No metadata JSON found; falling back to defaults.')

        sampling_rate = meta.get('sampling_rate', 200.0) or 200.0
        window_sec    = meta.get('window_sec', 0.2) or 0.2
        auto_W = int(round(float(sampling_rate) * float(window_sec)))

        # thresholds
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
            f'Loaded SVM classifier: {clf_path} (expects {exp} features); '
            f'classes={list(self.le.classes_)}'
        )
        self.get_logger().info(
            f'Metadata source={meta_source} | sampling_rate={sampling_rate}, '
            f'window_sec={window_sec} -> W={auto_W}, zc_thresh={self.zc_thresh}, '
            f'ssc_thresh={self.ssc_thresh}; Runtime W={self.W}'
        )

        # ---------- Buffers & I/O ----------
        self.emg_buf = deque(maxlen=self.W)
        self.pred_buf = deque(maxlen=self.vote_k)
        self.sub = self.create_subscription(MyoMsg, 'myo/data', self.cb, 10)
        self.pub_label = self.create_publisher(String, 'myo/pred', 10)

        # Warn if feature count unexpected
        if exp is not None and exp != 72:
            self.get_logger().warn(
                f'SVM n_features_in_={exp} but runtime computes 72; ensure training used the same 9×8 pack.'
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

    def _predict_with_scores(self, feats: np.ndarray):
        # scale if available
        if self.scaler is not None:
            try:
                feats = self.scaler.transform(feats)
            except Exception as e:
                self.get_logger().warn(f'Scaler transform failed: {e}')
        # prediction
        y = self.clf.predict(feats)[0]

        # probabilities or decision scores (for top-k display)
        probs = None
        scores = None
        class_order = getattr(self.clf, 'classes_', None)
        if hasattr(self.clf, 'predict_proba'):
            try:
                probs = self.clf.predict_proba(feats)[0]  # aligned with clf.classes_
            except Exception:
                probs = None
        if probs is None and hasattr(self.clf, 'decision_function'):
            try:
                s = self.clf.decision_function(feats)[0]
                # If one-vs-one shape, we can't easily map pairs → classes; skip.
                if class_order is not None and s.shape[0] == len(class_order):
                    # Convert margins to a softmax-like distribution for ranking only
                    # (not calibrated probabilities).
                    s = s - np.max(s)
                    e = np.exp(s)
                    scores = e / np.sum(e)
                else:
                    scores = None
            except Exception:
                scores = None
        return y, probs, scores, class_order

    def _topk_str(self, probs, scores, class_order, k=3):
        vec = None
        tag = ''
        if probs is not None:
            vec, tag = probs, ''    # true probabilities
        elif scores is not None:
            vec, tag = scores, ' (scores)'
        else:
            return ''
        idx = np.argsort(vec)[::-1][:k]
        # Map indices via classifier.classes_ → label encoder (more robust than assuming 0..K-1)
        items = []
        for i in idx:
            enc_cls = int(class_order[i]) if class_order is not None else i
            name = self.le.inverse_transform([enc_cls])[0]
            items.append(f'{name}={vec[i]:.2f}')
        return ' | ' + ' '.join(items) + tag

    def cb(self, msg: MyoMsg):
        if len(msg.emg_data) != 8:
            return

        self.emg_buf.append(np.array(msg.emg_data, dtype=float))
        if len(self.emg_buf) < self.W:
            return

        Xw = np.stack(self.emg_buf, axis=0)  # (W,8)
        feats = self._features_train_exact(Xw).reshape(1, -1)

        try:
            y_pred, probs, scores, class_order = self._predict_with_scores(feats)
        except Exception as e:
            self.get_logger().error(f'Prediction failed: {e}')
            return

        try:
            label = self.le.inverse_transform([int(y_pred)])[0]
        except Exception:
            label = str(y_pred)

        # Majority vote smoothing
        self.pred_buf.append(label)
        voted = Counter(self.pred_buf).most_common(1)[0][0]

        self.get_logger().info(
            f'Pred: {label} -> voted: {voted}{self._topk_str(probs, scores, class_order)}'
        )
        self.pub_label.publish(String(data=voted))

def main(args=None):
    rclpy.init(args=args)
    node = EMGClassifierSVM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
