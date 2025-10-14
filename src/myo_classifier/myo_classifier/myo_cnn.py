#!/usr/bin/env python3
import os, json, pickle
from pathlib import Path
from collections import deque, Counter

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from myo_msgs.msg import MyoMsg

import tensorflow as tf

class EMGClassifierCNNLSTM(Node):
    def __init__(self):
        super().__init__('myo_cnn_lstm')

        # ---------- Params ----------
        self.declare_parameter('artifacts_dir', str(Path.home() / 'runs'))
        self.declare_parameter('model_dir', 'saved_model')                 # directory
        self.declare_parameter('label_encoder_file', 'label_encoder.pkl')  # file
        self.declare_parameter('norm_stats_file', 'norm_stats.npz')        # file
        self.declare_parameter('metadata_file', 'cnn_lstm_metadata.json')  # file
        self.declare_parameter('vote_k', 5)                                # smoothing
        # (FYI: timesteps is fixed by the trained model; we ignore any window_size overrides)

        artifacts_dir = os.path.expanduser(self.get_parameter('artifacts_dir').value)
        model_dir = os.path.join(artifacts_dir, self.get_parameter('model_dir').value)
        le_path   = os.path.join(artifacts_dir, self.get_parameter('label_encoder_file').value)
        norm_path = os.path.join(artifacts_dir, self.get_parameter('norm_stats_file').value)
        meta_path = os.path.join(artifacts_dir, self.get_parameter('metadata_file').value)
        self.vote_k = int(self.get_parameter('vote_k').value)

        # ---------- Load artifacts ----------
        # 1) Keras SavedModel
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"SavedModel directory not found: {model_dir}")
        self.model = tf.keras.models.load_model(model_dir, compile=False)

        # 2) Label encoder
        with open(le_path, 'rb') as f:
            self.le = pickle.load(f)

        # 3) Norm stats (dataset-level mean/std, shapes typically (1,1,C))
        if not os.path.exists(norm_path):
            self.get_logger().warn(f'No norm_stats found at {norm_path}; will run UNNORMALIZED (mismatch vs training).')
            self.norm_mean = None
            self.norm_std  = None
        else:
            stats = np.load(norm_path)
            self.norm_mean = stats['mean']  # expect shape (1,1,C)
            self.norm_std  = stats['std']   # expect shape (1,1,C)

        # 4) Metadata (timesteps/channels/classes, etc.)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.timesteps = int(meta.get('timesteps', 40))
        self.channels  = int(meta.get('channels', 8))
        classes = meta.get('classes_', list(self.le.classes_))

        # Sanity: SavedModel input must match (timesteps, channels)
        # Try to read Keras input shape if available
        try:
            # Keras usually exposes shape as (None, T, C)
            mdl_in = self.model.inputs[0].shape
            t_mdl = int(mdl_in[1])
            c_mdl = int(mdl_in[2])
            if t_mdl != self.timesteps or c_mdl != self.channels:
                self.get_logger().warn(
                    f'Metadata (T={self.timesteps},C={self.channels}) '
                    f'!= model input (T={t_mdl},C={c_mdl}); using model dims.'
                )
                self.timesteps, self.channels = t_mdl, c_mdl
        except Exception:
            pass

        self.get_logger().info(
            f'CNN-LSTM ready | model={model_dir} | T={self.timesteps}, C={self.channels} '
            f'| classes={list(classes)}'
        )

        # ---------- Buffers & I/O ----------
        self.emg_buf = deque(maxlen=self.timesteps)
        self.pred_buf = deque(maxlen=self.vote_k)
        self.sub = self.create_subscription(MyoMsg, 'myo/data', self.cb, 10)
        self.pub_label = self.create_publisher(String, 'myo/pred', 10)

        # Check norm stats channels
        if self.norm_mean is not None and self.norm_mean.shape[-1] != self.channels:
            self.get_logger().warn(
                f'Norm stats channel dim {self.norm_mean.shape[-1]} != model channels {self.channels}. '
                'Normalization will be skipped.'
            )
            self.norm_mean, self.norm_std = None, None

    def _standardize(self, Xw: np.ndarray) -> np.ndarray:
        """Channel-wise standardization using saved dataset mean/std.
        Xw: (T, C) -> returns (T, C) float32."""
        if self.norm_mean is None or self.norm_std is None:
            return Xw.astype(np.float32)
        # Saved shapes are (1,1,C); reshape to (C,) for broadcasting over (T,C)
        mean_c = self.norm_mean.reshape(-1).astype(np.float32)
        std_c  = self.norm_std.reshape(-1).astype(np.float32)
        return ((Xw - mean_c) / (std_c + 1e-8)).astype(np.float32)

    def _topk_str(self, probs: np.ndarray, k: int = 3) -> str:
        if probs is None or probs.ndim != 1:
            return ''
        idx = np.argsort(probs)[::-1][:k]
        names = [self.le.inverse_transform([i])[0] for i in idx]
        return ' | ' + ' '.join(f'{n}={probs[i]:.2f}' for n, i in zip(names, idx))

    def cb(self, msg: MyoMsg):
        # Enforce expected channel count
        if len(msg.emg_data) != self.channels:
            self.get_logger().warn(
                f'Dropped frame: got {len(msg.emg_data)} channels, expected {self.channels}.'
            )
            return

        self.emg_buf.append(np.array(msg.emg_data, dtype=np.float32))
        if len(self.emg_buf) < self.timesteps:
            return

        # (T,C) window
        Xw = np.stack(self.emg_buf, axis=0)  # shape (T,C)
        Xw = self._standardize(Xw)           # (T,C)
        Xb = np.expand_dims(Xw, axis=0)      # (1,T,C)

        try:
            # Prefer model(X) over predict() in TF2 eager
            logits = self.model(Xb, training=False).numpy().squeeze()  # (C_classes,)
            if logits.ndim == 0:
                logits = np.array([logits], dtype=np.float32)
            probs = logits  # model has softmax output per training schema
            yhat = int(np.argmax(probs))
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
            return

        try:
            label = self.le.inverse_transform([yhat])[0]
        except Exception:
            label = str(yhat)

        # Majority vote smoothing
        self.pred_buf.append(label)
        voted = Counter(self.pred_buf).most_common(1)[0][0]

        self.get_logger().info(f'Pred: {label} -> voted: {voted}{self._topk_str(probs)}')
        self.pub_label.publish(String(data=voted))

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
