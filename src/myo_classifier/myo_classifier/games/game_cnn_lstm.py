#!/usr/bin/env python3
import os, json, pickle, time, shutil, subprocess
from pathlib import Path
from collections import deque, Counter

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from myo_msgs.msg import MyoMsg
import tensorflow as tf

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
            if key is None: return False
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
            if xk is None: return False
            try:
                subprocess.run(['xdotool', 'key', xk], check=False)
                return True
            except Exception:
                return False
        return False


class EMGClassifierCNNLSTM(Node):
    def __init__(self):
        super().__init__('myo_cnn_lstm')

        # ---------- Params ----------
        self.declare_parameter('artifacts_dir', str(Path.home() / 'runs'))
        self.declare_parameter('model_dir', 'saved_model')                 # SavedModel dir OR single file (.keras/.h5)
        self.declare_parameter('label_encoder_file', 'label_encoder.pkl')  # file
        self.declare_parameter('norm_stats_file', 'norm_stats.npz')        # file
        self.declare_parameter('metadata_file', 'cnn_lstm_metadata.json')  # file

        # Smoothing
        self.declare_parameter('vote_k', 5)

        # Optional channel remap (if armband rotated); default identity
        self.declare_parameter('channel_order', [])  # e.g. [2,3,4,5,6,7,0,1]

        # Keypress behavior
        self.declare_parameter('send_keys', True)            # enable/disable key sending
        self.declare_parameter('min_interval_sec', 0.35)     # debounce between sends
        self.declare_parameter('min_confidence', 0.0)        # require this prob (0 disables)
        # Optional JSON to override mapping:
        # {"fist":"enter","open_hand":"space","wrist_left":"left","wrist_right":"right"}
        self.declare_parameter('keymap_json', '')

        adir = os.path.expanduser(self.get_parameter('artifacts_dir').value)
        model_path = os.path.join(adir, self.get_parameter('model_dir').value)
        le_path    = os.path.join(adir, self.get_parameter('label_encoder_file').value)
        norm_path  = os.path.join(adir, self.get_parameter('norm_stats_file').value)
        meta_path  = os.path.join(adir, self.get_parameter('metadata_file').value)

        self.vote_k      = int(self.get_parameter('vote_k').value)
        self.send_keys   = bool(self.get_parameter('send_keys').value)
        self.min_interval= float(self.get_parameter('min_interval_sec').value)
        self.min_conf    = float(self.get_parameter('min_confidence').value)

        # ---------- Load artifacts ----------
        # Model (SavedModel dir or single-file model)
        if os.path.isdir(model_path) or os.path.isfile(model_path):
            self.model = tf.keras.models.load_model(model_path, compile=False)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

        with open(le_path, 'rb') as f:
            self.le = pickle.load(f)

        if os.path.exists(norm_path):
            stats = np.load(norm_path)
            self.norm_mean = stats['mean']  # (1,1,C)
            self.norm_std  = stats['std']   # (1,1,C)
        else:
            self.norm_mean = self.norm_std = None
            self.get_logger().warn(f'No norm_stats at {norm_path}; running UNNORMALIZED.')

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Timesteps/channels from metadata/model
        self.timesteps = int(meta.get('timesteps', 40))
        self.channels  = int(meta.get('channels', 8))
        try:
            shp = self.model.inputs[0].shape  # (None, T, C)
            t_m, c_m = int(shp[1]), int(shp[2])
            if (t_m != self.timesteps) or (c_m != self.channels):
                self.get_logger().warn(f'Metadata (T={self.timesteps},C={self.channels}) '
                                       f'!= model input (T={t_m},C={c_m}); using model dims.')
                self.timesteps, self.channels = t_m, c_m
        except Exception:
            pass

        classes = list(meta.get('classes_', list(self.le.classes_)))
        self.get_logger().info(f'CNN-LSTM ready | T={self.timesteps}, C={self.channels} | classes={classes}')

        # Channel order
        order = self.get_parameter('channel_order').value
        if order:
            if len(order) != self.channels or sorted(map(int, order)) != list(range(self.channels)):
                raise ValueError(f'channel_order must be a permutation of [0..{self.channels-1}]')
            self.channel_order = list(map(int, order))
        else:
            self.channel_order = list(range(self.channels))
        if self.norm_mean is not None and self.norm_mean.shape[-1] != self.channels:
            self.get_logger().warn(
                f'Norm stats channels={self.norm_mean.shape[-1]} != model channels={self.channels}; skipping normalization.'
            )
            self.norm_mean = self.norm_std = None

        # I/O buffers
        self.emg_buf  = deque(maxlen=self.timesteps)
        self.pred_buf = deque(maxlen=self.vote_k)
        self.sub = self.create_subscription(MyoMsg, 'myo/data', self.cb, 10)
        self.pub_label = self.create_publisher(String, 'myo/pred', 10)

        # Key sending
        self._sender = KeySender(self.get_logger())
        self._last_sent_label = None
        self._last_sent_time = 0.0

        # Default key mapping (overridable)
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

    # ---------- Helpers ----------
    def _standardize(self, Xw: np.ndarray) -> np.ndarray:
        if self.norm_mean is None or self.norm_std is None:
            return Xw.astype(np.float32)
        mean_c = self.norm_mean.reshape(-1).astype(np.float32)
        std_c  = self.norm_std.reshape(-1).astype(np.float32)
        return ((Xw - mean_c) / (std_c + 1e-8)).astype(np.float32)

    def _topk_str(self, probs: np.ndarray, k: int = 3) -> str:
        if probs is None or probs.ndim != 1:
            return ''
        idx = np.argsort(probs)[::-1][:k]
        names = [self.le.inverse_transform([i])[0] for i in idx]
        return ' | ' + ' '.join(f'{n}={probs[i]:.2f}' for n, i in zip(names, idx))

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

    def _maybe_send_key(self, label: str, probs: np.ndarray | None):
        if not self.send_keys:
            return
        key = self._mapped_key_for_label(label)
        if not key:
            return

        # Optional confidence gate (uses softmax probabilities)
        if self.min_conf > 0.0 and probs is not None:
            try:
                cls_index = int(self.le.transform([label])[0])
                if float(probs[cls_index]) < self.min_conf:
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

    # ---------- Callback ----------
    def cb(self, msg: MyoMsg):
        if len(msg.emg_data) != self.channels:
            self.get_logger().warn(f'Dropped frame: got {len(msg.emg_data)} ch, expected {self.channels}')
            return

        x = np.array(msg.emg_data, dtype=np.float32)
        if self.channel_order != list(range(self.channels)):
            x = x[self.channel_order]

        self.emg_buf.append(x)
        if len(self.emg_buf) < self.timesteps:
            return

        Xw = np.stack(self.emg_buf, axis=0)   # (T,C)
        Xw = self._standardize(Xw)
        Xb = np.expand_dims(Xw, 0)            # (1,T,C)

        try:
            probs = self.model(Xb, training=False).numpy().squeeze()
            if probs.ndim == 0:
                probs = np.array([probs], dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
            return

        yhat = int(np.argmax(probs))
        try:
            label = self.le.inverse_transform([yhat])[0]
        except Exception:
            label = str(yhat)

        # Majority vote smoothing
        self.pred_buf.append(label)
        voted = Counter(self.pred_buf).most_common(1)[0][0]

        # Log + publish
        self.get_logger().info(f'Pred: {label} -> voted: {voted}{self._topk_str(probs)}')
        self.pub_label.publish(String(data=voted))

        # Send key on voted label (debounced / optional confidence)
        self._maybe_send_key(voted, probs)

def main(args=None):
    rclpy.init(args=args)
    node = EMGClassifierCNNLSTM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
