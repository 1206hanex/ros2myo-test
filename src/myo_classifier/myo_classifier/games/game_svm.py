#!/usr/bin/env python3
import os, json, pickle, time, shutil, subprocess
from pathlib import Path
from collections import deque, Counter

import numpy as np
import rclpy
from rclpy.node import Node
from myo_msgs.msg import MyoMsg
from std_msgs.msg import String

# --------- Training-matched feature helpers ---------
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


# --------- Key sender (pynput → xdotool → none) ---------
class KeySender:
    """
    send('enter'|'space'|'left'|'right') using:
    - pynput (preferred)
    - xdotool (fallback)
    - none (logs a warning)
    """
    def __init__(self, logger):
        self.logger = logger
        self.mode = 'none'
        self._kbd = None
        try:
            from pynput.keyboard import Controller, Key
            self._kbd = Controller()
            self._Key = Key
            self.mode = 'pynput'
            self.logger.info('KeySender: using pynput')
        except Exception:
            if shutil.which('xdotool'):
                self.mode = 'xdotool'
                self.logger.info('KeySender: using xdotool')
            else:
                self.logger.warn('KeySender: no pynput or xdotool found; keypresses disabled.')

    def send(self, canonical_name: str) -> bool:
        name = canonical_name.lower()
        if self.mode == 'pynput':
            key_map = {
                'enter': self._Key.enter,
                'space': self._Key.space,
                'left' : self._Key.left,
                'right': self._Key.right,
            }
            key = key_map.get(name)
            if key is None:
                return False
            try:
                self._kbd.press(key)
                self._kbd.release(key)
                return True
            except Exception:
                return False

        if self.mode == 'xdotool':
            x_map = {
                'enter': 'Return',
                'space': 'space',
                'left' : 'Left',
                'right': 'Right',
            }
            xk = x_map.get(name)
            if xk is None:
                return False
            try:
                subprocess.run(['xdotool', 'key', xk], check=False)
                return True
            except Exception:
                return False

        return False


class EMGClassifierSVM(Node):
    def __init__(self):
        super().__init__('myo_svm')

        # ---------- Params ----------
        self.declare_parameter('artifacts_dir', str(Path.home() / 'runs'))
        self.declare_parameter('classifier_file', 'gesture_classifier.pkl')
        self.declare_parameter('label_encoder_file', 'label_encoder.pkl')
        # prefer SVM metadata, fall back to RF if needed
        self.declare_parameter('metadata_file', 'svm_metadata.json')
        self.declare_parameter('fallback_metadata_file', 'rf_metadata.json')

        # If <= 0, infer window from metadata: round(window_sec * sampling_rate)
        self.declare_parameter('window_size', -1)

        # Feature thresholds (ZC/SSC) override; else from metadata (defaults 0.01)
        self.declare_parameter('zc_thresh', None)
        self.declare_parameter('ssc_thresh', None)

        # Optional scaler (recommended for SVM)
        self.declare_parameter('scaler_file', 'scaler.pkl')

        # Voting / smoothing
        self.declare_parameter('vote_k', 5)

        # Keypress behavior
        self.declare_parameter('send_keys', True)            # enable/disable key sending
        self.declare_parameter('min_interval_sec', 0.35)     # debounce between sends
        self.declare_parameter('min_confidence', 0.0)        # require this prob/score (0 disables)
        # Optional JSON to override mapping:
        # {"fist":"enter","open_hand":"space","wrist_left":"left","wrist_right":"right"}
        self.declare_parameter('keymap_json', '')

        # ---------- Load artifacts ----------
        adir = os.path.expanduser(self.get_parameter('artifacts_dir').value)
        clf_path  = os.path.join(adir, self.get_parameter('classifier_file').value)
        le_path   = os.path.join(adir, self.get_parameter('label_encoder_file').value)
        meta_path = os.path.join(adir, self.get_parameter('metadata_file').value)
        meta_fb   = os.path.join(adir, self.get_parameter('fallback_metadata_file').value)
        sc_path   = os.path.join(adir, self.get_parameter('scaler_file').value)

        with open(clf_path, 'rb') as f:
            self.clf = pickle.load(f)
        with open(le_path, 'rb') as f:
            self.le = pickle.load(f)

        # scaler is optional
        self.scaler = None
        if os.path.exists(sc_path):
            try:
                with open(sc_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.get_logger().info(f'Loaded scaler: {sc_path}')
            except Exception as e:
                self.get_logger().warn(f'Could not load scaler ({e}); continuing without.')

        # metadata (try SVM, then RF)
        meta = {}
        meta_source = None
        for candidate in (meta_path, meta_fb):
            if os.path.exists(candidate):
                try:
                    with open(candidate, 'r') as f:
                        meta = json.load(f)
                        meta_source = candidate
                        break
                except Exception as e:
                    self.get_logger().warn(f'Could not read metadata {candidate}: {e}')
        if meta_source is None:
            self.get_logger().warn('No metadata JSON found; using defaults.')

        sampling_rate = float(meta.get('sampling_rate', 200.0) or 200.0)
        window_sec    = float(meta.get('window_sec', 0.2) or 0.2)
        auto_W = int(round(sampling_rate * window_sec))

        meta_zc  = float(meta.get('zc_thresh', 0.01))
        meta_ssc = float(meta.get('ssc_thresh', 0.01))

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
        self.emg_buf  = deque(maxlen=self.W)
        self.pred_buf = deque(maxlen=self.vote_k)
        self.sub = self.create_subscription(MyoMsg, 'myo/data', self.cb, 10)
        self.pub_label = self.create_publisher(String, 'myo/pred', 10)

        # ---------- Key sending setup ----------
        self.send_keys = bool(self.get_parameter('send_keys').value)
        self.min_interval = float(self.get_parameter('min_interval_sec').value)
        self.min_conf = float(self.get_parameter('min_confidence').value)
        self._sender = KeySender(self.get_logger())
        self._last_sent_label = None
        self._last_sent_time = 0.0

        # Default mapping (can be overridden with keymap_json)
        self.gesture_to_key = {
            'fist': 'enter',
            'open_hand': 'space',
            'wrist_left': 'left',
            'wrist_right': 'right',
        }
        km_raw = self.get_parameter('keymap_json').value
        if isinstance(km_raw, str) and km_raw.strip():
            try:
                override = json.loads(km_raw)
                canon = {'enter', 'space', 'left', 'right'}
                cleaned = {}
                for k, v in override.items():
                    vv = str(v).lower()
                    if vv not in canon:
                        self.get_logger().warn(f'Ignoring unknown key "{v}" for gesture "{k}"')
                        continue
                    cleaned[str(k)] = vv
                self.gesture_to_key.update(cleaned)
                self.get_logger().info(f'Custom keymap applied: {self.gesture_to_key}')
            except Exception as e:
                self.get_logger().warn(f'Failed to parse keymap_json: {e}')

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
        # optional scaler
        if self.scaler is not None:
            try:
                feats = self.scaler.transform(feats)
            except Exception as e:
                self.get_logger().warn(f'Scaler transform failed: {e}')
        y = self.clf.predict(feats)[0]

        # probabilities or decision scores
        probs = None
        scores = None
        class_order = getattr(self.clf, 'classes_', None)
        if hasattr(self.clf, 'predict_proba'):
            try:
                probs = self.clf.predict_proba(feats)[0]
            except Exception:
                probs = None
        if probs is None and hasattr(self.clf, 'decision_function'):
            try:
                s = self.clf.decision_function(feats)[0]
                if class_order is not None and np.ndim(s) == 1 and len(s) == len(class_order):
                    # convert to a softmax-like distribution for ranking/gating (not calibrated)
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
            vec, tag = probs, ''
        elif scores is not None:
            vec, tag = scores, ' (scores)'
        else:
            return ''
        idx = np.argsort(vec)[::-1][:k]
        items = []
        for i in idx:
            enc_cls = int(class_order[i]) if class_order is not None else i
            name = self.le.inverse_transform([enc_cls])[0]
            items.append(f'{name}={vec[i]:.2f}')
        return ' | ' + ' '.join(items) + tag

    def _mapped_key_for_label(self, label: str):
        if label in self.gesture_to_key:
            return self.gesture_to_key[label]
        if label == 'left':
            return self.gesture_to_key.get('wrist_left')
        if label == 'right':
            return self.gesture_to_key.get('wrist_right')
        if label in ('open', 'openhand'):
            return self.gesture_to_key.get('open_hand')
        return None

    def _maybe_send_key(self, label: str, probs, scores, class_order):
        if not self.send_keys:
            return
        key = self._mapped_key_for_label(label)
        if not key:
            return

        # optional confidence gate
        if self.min_conf > 0.0:
            passed = False
            if probs is not None:
                try:
                    cls_index = int(self.le.transform([label])[0])
                    passed = float(probs[cls_index]) >= self.min_conf
                except Exception:
                    passed = False
            elif scores is not None and class_order is not None:
                try:
                    cls_index = int(self.le.transform([label])[0])
                    # find where this class sits in class_order
                    # class_order holds encoded class labels aligned to scores
                    order_index = np.where(class_order == cls_index)[0]
                    if len(order_index):
                        passed = float(scores[order_index[0]]) >= self.min_conf
                except Exception:
                    passed = False
            else:
                passed = True  # no scores available; ignore gate
            if not passed:
                return

        now = time.time()
        if label != self._last_sent_label or (now - self._last_sent_time) >= self.min_interval:
            ok = self._sender.send(key)
            if ok:
                self._last_sent_label = label
                self._last_sent_time = now
                self.get_logger().info(f'Sent key: {key} (for gesture: {label})')
            else:
                self.get_logger().warn(f'Failed to send key: {key}')

    def cb(self, msg: MyoMsg):
        # Myo has 8 EMG channels
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

        # Log + publish
        self.get_logger().info(f'Pred: {label} -> voted: {voted}{self._topk_str(probs, scores, class_order)}')
        self.pub_label.publish(String(data=voted))

        # Send key on voted label (debounced)
        self._maybe_send_key(voted, probs, scores, class_order)

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
