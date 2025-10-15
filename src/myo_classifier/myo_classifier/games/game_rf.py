#!/usr/bin/env python3
import os, json, pickle, time, shutil, subprocess
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
        # Try pynput
        try:
            from pynput.keyboard import Controller, Key
            self._kbd = Controller()
            self._Key = Key
            self.mode = 'pynput'
            self.logger.info('KeySender: using pynput')
        except Exception as e:
            # Try xdotool
            if shutil.which('xdotool'):
                self.mode = 'xdotool'
                self.logger.info('KeySender: using xdotool')
            else:
                self.logger.warn('KeySender: no pynput or xdotool found; keypresses will be disabled.')

    def send(self, canonical_name: str) -> bool:
        """
        canonical_name in {'enter','space','left','right'}
        """
        canonical_name = canonical_name.lower()
        if self.mode == 'pynput':
            key_map = {
                'enter': self._Key.enter,
                'space': self._Key.space,
                'left' : self._Key.left,
                'right': self._Key.right,
            }
            key = key_map.get(canonical_name)
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
            xk = x_map.get(canonical_name)
            if xk is None:
                return False
            try:
                subprocess.run(['xdotool', 'key', xk], check=False)
                return True
            except Exception:
                return False

        return False


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

        # Keypress behavior
        self.declare_parameter('send_keys', True)            # enable/disable key sending
        self.declare_parameter('min_interval_sec', 0.35)     # debounce between sends of different labels
        self.declare_parameter('min_confidence', 0.0)        # require this prob for mapped labels (0 disables)
        # Optional JSON param to override mapping: {"fist":"enter","open_hand":"space","wrist_left":"left","wrist_right":"right"}
        self.declare_parameter('keymap_json', '')

        artifacts_dir = os.path.expanduser(self.get_parameter('artifacts_dir').value)
        clf_path = os.path.join(artifacts_dir, self.get_parameter('classifier_file').value)
        le_path  = os.path.join(artifacts_dir, self.get_parameter('label_encoder_file').value)
        meta_path= os.path.join(artifacts_dir, self.get_parameter('metadata_file').value)

        # ---------- Load artifacts ----------
        with open(clf_path, 'rb') as f:
            self.clf = pickle.load(f)
        with open(le_path, 'rb') as f:
            self.le = pickle.load(f)

        # Metadata (window + thresholds)
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except Exception as e:
                self.get_logger().warn(f'Could not read metadata {meta_path}: {e}')

        sampling_rate = meta.get('sampling_rate', 200.0) or 200.0
        window_sec    = meta.get('window_sec', 0.2) or 0.2
        auto_W = int(round(float(sampling_rate) * float(window_sec)))

        meta_zc  = meta.get('zc_thresh', 0.01)
        meta_ssc = meta.get('ssc_thresh', 0.01)

        # Apply param overrides
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

        # ---------- Key sending setup ----------
        self.send_keys = bool(self.get_parameter('send_keys').value)
        self.min_interval = float(self.get_parameter('min_interval_sec').value)
        self.min_conf = float(self.get_parameter('min_confidence').value)
        self._sender = KeySender(self.get_logger())
        self._last_sent_label = None
        self._last_sent_time = 0.0

        # Default mapping (can be overridden with keymap_json param)
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
                # normalize values to canonical names
                canon = {'enter','space','left','right'}
                cleaned = {}
                for k,v in override.items():
                    vv = str(v).lower()
                    if vv not in canon:
                        self.get_logger().warn(f'Ignoring unknown key "{v}" for gesture "{k}"')
                        continue
                    cleaned[str(k)] = vv
                self.gesture_to_key.update(cleaned)
                self.get_logger().info(f'Custom keymap applied: {self.gesture_to_key}')
            except Exception as e:
                self.get_logger().warn(f'Failed to parse keymap_json: {e}')

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

    def _mapped_key_for_label(self, label: str):
        # Allow simple synonyms if your training used 'left'/'right'
        if label in self.gesture_to_key:
            return self.gesture_to_key[label]
        if label == 'left':
            return self.gesture_to_key.get('wrist_left')
        if label == 'right':
            return self.gesture_to_key.get('wrist_right')
        if label in ('open', 'openhand'):
            return self.gesture_to_key.get('open_hand')
        return None

    def _maybe_send_key(self, label: str, probs):
        if not self.send_keys:
            return
        key = self._mapped_key_for_label(label)
        if not key:
            return
        # Optional confidence gate (if proba available)
        if self.min_conf > 0.0 and probs is not None:
            # find probability for this label index
            try:
                cls_index = int(self.le.transform([label])[0])
                p = float(probs[cls_index])
                if p < self.min_conf:
                    return
            except Exception:
                pass
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

        # Log + publish
        self.get_logger().info(f'Pred: {label} -> voted: {voted}{self._topk_str(probs)}')
        self.pub_label.publish(String(data=voted))

        # Send key on voted label (debounced)
        self._maybe_send_key(voted, probs)

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
