#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import os, time, pickle
import numpy as np
from collections import deque
from ament_index_python.packages import get_package_share_directory

# ---- Keyboard control (pynput) ----
try:
    from pynput.keyboard import Controller, Key
    _KEYBOARD = Controller()
    _KEY_SUPPORT = True
except Exception as e:
    _KEYBOARD = None
    _KEY_SUPPORT = False
    _KEY_IMPORT_ERROR = e

# -----------------------
# Parameters / constants
# -----------------------
window_size = 20  # Must match training (number of time steps per window)
NUM_CHANNELS = 8

# Label map must match your training labels
gesture_labels = {
    0: "rest",
    1: "pinch",
    2: "fist",
    3: "extension",
    4: "left",
    5: "right",
}

PKG_NAME = "myo_space_trigger"
# Preferred: a single Pipeline (StandardScaler -> SVM)
PIPELINE_FNAME = "gesture_svm_pipeline.pkl"
# Fallback: separate scaler + model
SCALER_FNAME = "gesture_scaler.pkl"
MODEL_FNAME = "gesture_svm.pkl"


class EMG_SVM_Node(Node):
    def __init__(self):
        super().__init__("myo_svm")

        # Resolve paths
        share_dir = get_package_share_directory(PKG_NAME)
        pipeline_path = os.path.join(share_dir, PIPELINE_FNAME)
        scaler_path = os.path.join(share_dir, SCALER_FNAME)
        model_path = os.path.join(share_dir, MODEL_FNAME)

        # Try to load a single Pipeline first; else load scaler + model
        self.pipeline = None
        self.scaler = None
        self.model = None

        if os.path.exists(pipeline_path):
            with open(pipeline_path, "rb") as f:
                self.pipeline = pickle.load(f)
            self.get_logger().info(f"Loaded scikit-learn Pipeline: {pipeline_path}")
        else:
            if not (os.path.exists(scaler_path) and os.path.exists(model_path)):
                raise FileNotFoundError(
                    f"Could not find {PIPELINE_FNAME} or both {SCALER_FNAME} and {MODEL_FNAME} in {share_dir}"
                )
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.get_logger().info(f"Loaded scaler: {scaler_path} and SVM: {model_path}")

        # Buffer and sub
        self.emg_buffer = deque(maxlen=window_size)
        self.subscription = self.create_subscription(
            Float32MultiArray, "emg_raw", self.listener_callback, 10
        )

        # Trigger debounce/stability
        self.cooldown_sec = 0.6
        self.last_trigger_ts = 0.0
        self.consecutive_required = 2
        self._prev_pred = None
        self._streak = 0

        if not _KEY_SUPPORT:
            self.get_logger().warn(
                f"Keyboard control disabled (pynput import failed: {_KEY_IMPORT_ERROR}). "
                "Predictions will log but not press keys."
            )

        self.get_logger().info("SVM EMG Node started → listening on /emg_raw")

    # --------- feature extraction (same as RF version) ----------
    def _extract_features(self, window_TxC: np.ndarray) -> np.ndarray:
        """
        window_TxC: shape [T, C]
        Returns a flat feature vector: [rms(8), mean(8), var(8)] -> 24 dims if C=8
        """
        # Safety: ensure correct shape
        if window_TxC.ndim != 2 or window_TxC.shape[1] != NUM_CHANNELS:
            raise ValueError(f"Expected window shape [T,{NUM_CHANNELS}], got {window_TxC.shape}")

        rms = np.sqrt(np.mean(np.square(window_TxC), axis=0))
        mean = np.mean(window_TxC, axis=0)
        var = np.var(window_TxC, axis=0)
        feat = np.concatenate([rms, mean, var], axis=0).astype(np.float32)
        return feat  # shape (24,)

    # ---------------- keyboard helpers ----------------
    def _press_key(self, key):
        if not _KEY_SUPPORT:
            return
        try:
            _KEYBOARD.press(key)
            _KEYBOARD.release(key)
        except Exception as e:
            self.get_logger().warn(f"Failed to send keypress: {e}")

    def _maybe_trigger_action(self, prediction_id: int):
        now = time.time()

        # stability requirement
        if prediction_id == self._prev_pred:
            self._streak += 1
        else:
            self._streak = 1
            self._prev_pred = prediction_id

        if self._streak < self.consecutive_required:
            return

        if (now - self.last_trigger_ts) < self.cooldown_sec:
            return

        label = gesture_labels.get(prediction_id, f"unknown({prediction_id})")
        if label == "fist":
            self.get_logger().info("Action: SPACE (from 'fist')")
            self._press_key(Key.space)
            self.last_trigger_ts = now
        elif label == "pinch":
            self.get_logger().info("Action: ENTER (from 'pinch')")
            self._press_key(Key.enter)
            self.last_trigger_ts = now

    # ---------------- ROS callback ----------------
    def listener_callback(self, msg: Float32MultiArray):
        # Expect per-sample EMG vector of length 8
        if len(msg.data) != NUM_CHANNELS:
            self.get_logger().warn(f"Unexpected EMG vector size: {len(msg.data)} (expected {NUM_CHANNELS})")
            return

        self.emg_buffer.append(list(msg.data))

        if len(self.emg_buffer) < window_size:
            return

        # Prepare features
        window_np = np.array(self.emg_buffer, dtype=np.float32)  # [T, 8]
        feat = self._extract_features(window_np).reshape(1, -1)  # [1, F]

        # Inference
        try:
            if self.pipeline is not None:
                pred = int(self.pipeline.predict(feat)[0])
            else:
                # scaler + model
                feat_scaled = self.scaler.transform(feat) if self.scaler is not None else feat
                pred = int(self.model.predict(feat_scaled)[0])
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            return

        # Log and maybe act
        label = gesture_labels.get(pred, f"id:{pred}")
        self.get_logger().info(f"Predicted Gesture: {label}")
        self._maybe_trigger_action(pred)


def main(args=None):
    rclpy.init(args=args)
    node = EMG_SVM_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
