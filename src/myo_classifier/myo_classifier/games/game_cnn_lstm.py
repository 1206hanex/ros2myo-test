#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import os, time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
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
# Must match the time dimension your CNN+LSTM was trained with
window_size = 50

gesture_labels = {
    0: "rest",
    1: "pinch",
    2: "fist",
    3: "extension",
    4: "left",
    5: "right",
}

# If your model expects [batch, time, channels] set this True.
# If it expects [batch, channels, time], set False.
MODEL_TIME_MAJOR = True

# Optionally normalize EMG with training stats (mean, std) per channel
USE_NORM = True
NORM_FNAME = "emg_norm_stats.npz"      # keys: mean, std (shape: [8])

MODEL_FNAME = "gesture_cnn_lstm.pt"    # TorchScript or regular .pt (state_dict or scripted module)

class EMG_CNNLSTM_Node(Node):
    def __init__(self):
        super().__init__("myo_cnn_lstm")

        # Locate model + (optional) norm stats from your share dir
        pkg = 'myo_space_trigger'   # change if your package name differs
        share_dir = get_package_share_directory(pkg)

        model_path = os.path.join(share_dir, MODEL_FNAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Try to load as TorchScript first; if that fails, load a torch nn.Module via state_dict
        self.model = None
        try:
            self.model = torch.jit.load(model_path, map_location="cpu")
            self.get_logger().info("Loaded TorchScript CNN+LSTM model.")
        except Exception:
            # Fallback: user saved a plain state_dict; define a stub and load it (you must match arch)
            self.get_logger().warn("TorchScript load failed; attempting state_dict load. "
                                   "Ensure your architecture here matches training.")
            self.model = self._build_stub_architecture()
            sd = torch.load(model_path, map_location="cpu")
            # Many training scripts save {'model_state_dict': ..., ...}
            if isinstance(sd, dict) and 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            self.model.load_state_dict(sd)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Load normalization stats if present
        self.norm_mean = None
        self.norm_std = None
        if USE_NORM:
            norm_path = os.path.join(share_dir, NORM_FNAME)
            if os.path.exists(norm_path):
                try:
                    stats = np.load(norm_path)
                    self.norm_mean = stats['mean'].astype(np.float32)   # shape (8,)
                    self.norm_std  = stats['std'].astype(np.float32)    # shape (8,)
                    # guard against zeros
                    self.norm_std[self.norm_std == 0] = 1.0
                    self.get_logger().info(f"Loaded normalization stats from {norm_path}")
                except Exception as e:
                    self.get_logger().warn(f"Failed to load norm stats: {e}. Proceeding without.")
                    self.norm_mean = None
                    self.norm_std = None

        # Buffer and subscription
        self.emg_buffer = deque(maxlen=window_size)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'emg_raw',
            self.listener_callback,
            10
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

        self.get_logger().info("CNN+LSTM EMG Node started → listening on /emg_raw")

    # ---- OPTIONAL: replace with your exact model architecture if loading a state_dict ----
    def _build_stub_architecture(self):
        """
        Minimal stub CNN+LSTM that often matches common training setups:
        Conv1d over time (per channel), then LSTM over time, then FC.
        Update to exactly match your training code if you are loading a state_dict.
        """
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self, n_channels=8, n_classes=6, conv_out=32, lstm_hidden=64, lstm_layers=1):
                super().__init__()
                # If input is [B, T, C], we’ll transpose to [B, C, T] for Conv1d
                self.conv = nn.Sequential(
                    nn.Conv1d(n_channels, conv_out, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(conv_out, conv_out, kernel_size=5, padding=2),
                    nn.ReLU(),
                )
                self.lstm = nn.LSTM(
                    input_size=conv_out,
                    hidden_size=lstm_hidden,
                    num_layers=lstm_layers,
                    batch_first=True,
                    bidirectional=False
                )
                self.fc = nn.Linear(lstm_hidden, n_classes)

            def forward(self, x):
                # x: [B, T, C] or [B, C, T] depending on external flag
                if x.dim() != 3:
                    raise ValueError("Expected input shape [B, T, C] or [B, C, T]")
                if MODEL_TIME_MAJOR:
                    # [B, T, C] -> [B, C, T] for Conv1d
                    x = x.transpose(1, 2)
                # Conv over time
                x = self.conv(x)  # [B, C, T]
                # Back to time-major for LSTM: [B, T, C]
                x = x.transpose(1, 2)
                # LSTM over time
                x, _ = self.lstm(x)  # [B, T, H]
                # Use last timestep
                x = x[:, -1, :]      # [B, H]
                logits = self.fc(x)  # [B, n_classes]
                return logits

        return Model()

    # -------------- helpers --------------
    def _normalize(self, arr_TxC: np.ndarray) -> np.ndarray:
        if self.norm_mean is None or self.norm_std is None:
            return arr_TxC
        # Broadcast mean/std over time dimension
        return (arr_TxC - self.norm_mean) / self.norm_std

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

    # -------------- ROS callback --------------
    def listener_callback(self, msg: Float32MultiArray):
        # Expect 8-channel EMG per sample
        if len(msg.data) != 8:
            self.get_logger().warn(f"Unexpected EMG vector size: {len(msg.data)} (expected 8)")
            return

        self.emg_buffer.append(list(msg.data))

        if len(self.emg_buffer) < window_size:
            return

        # Prepare window: [T, C]
        window_np = np.array(self.emg_buffer, dtype=np.float32)  # shape [T, 8]
        if USE_NORM:
            window_np = self._normalize(window_np)

        # Convert to torch: expected either [B, T, C] or [B, C, T]
        if MODEL_TIME_MAJOR:
            x = torch.from_numpy(window_np).unsqueeze(0)         # [1, T, C]
        else:
            x = torch.from_numpy(window_np).transpose(0, 1).unsqueeze(0)  # [1, C, T]

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            pred = int(torch.argmax(probs, dim=-1).item())

        if pred in gesture_labels:
            self.get_logger().info(f"Predicted Gesture: {gesture_labels[pred]}")
        else:
            self.get_logger().info(f"Predicted Gesture ID: {pred}")

        self._maybe_trigger_action(pred)


def main(args=None):
    rclpy.init(args=args)
    node = EMG_CNNLSTM_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
