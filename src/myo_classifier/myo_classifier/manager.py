#!/usr/bin/env python3
import os, time, csv, threading, glob, pickle
from datetime import datetime
from collections import deque, Counter

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from ament_index_python.packages import get_package_share_directory
from myo_msgs.msg import MyoMsg
from myo_msgs.srv import AddGesture, TrainModel

from myo_classifier.train.data import load_feature_csvs, load_raw_sequences
from myo_classifier.train.rf import train_rf
from myo_classifier.train.cnn_lstm import train_cnn_lstm


class Manager(Node):
    def __init__(self):
        super().__init__('manager')

        # Parameters common to collection
        self.declare_parameter('feature_stride_sec', 0.20)   # how often to sample a window (features mode)
        self.declare_parameter('expect_emg_channels', 8)
        self.feature_stride_sec = float(self.get_parameter('feature_stride_sec').value)
        self.expect_emg_channels = int(self.get_parameter('expect_emg_channels').value)

        # State for collection
        self._collect_lock = threading.Lock()
        self._collect_active = False
        self._collect_mode_raw = False
        self._collect_label = None
        self._collect_window = 20
        self._collect_outdir = None
        self._collect_start_time = 0.0
        self._collect_end_time = 0.0
        self._emg_buf = deque(maxlen=512)       # rolling buffer of EMG samples (each is list[float])
        self._imu_buf = deque(maxlen=512)       # reserved if you add IMU later
        self._feature_rows = []                 # features rows when raw_mode=False
        self._raw_rows = []                     # raw rows (per-sample) when raw_mode=True
        self._last_feature_emit = 0.0

        # Subscribe to the streaming topic from your publisher
        self.sub = self.create_subscription(MyoMsg, 'myo/data', self._on_myo, 50)

        # Services
        self.srv_add = self.create_service(AddGesture, 'myo_classifier/add_gesture', self._srv_add_gesture)
        self.srv_train = self.create_service(TrainModel, 'myo_classifier/train_model', self._srv_train_model)

        self.get_logger().info('Manager ready: services {add_gesture, train_model}')

    # ===========================
    # Subscriber: data collection
    # ===========================
    def _on_myo(self, msg: MyoMsg):
        # Handles incoming MyoMsg from myo_stream.py.
        # - Buffers EMG and IMU with timestamps
        # - If collection is active:
        #   * raw mode: append per-sample row (EMG sample + latest IMU)
        #   * feature mode: every feature_stride_sec, emit features over a short window

        # Convert ROS time to float seconds (monotonic)
        t = float(msg.stamp.sec) + float(msg.stamp.nanosec) * 1e-9

        # --- Update rolling buffers (keep timestamp with each sample) ---
        if msg.emg_data:
            if len(msg.emg_data) != self.expect_emg_channels:
                # Warn once if channel mismatch
                self.get_logger().warn(
                    f'EMG channels ({len(msg.emg_data)}) != expect_emg_channels ({self.expect_emg_channels})'
                )
            self._emg_buf.append((t, list(msg.emg_data)))

        if msg.imu_data:
            # imu_data order from myo_stream.py:
            # [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, orient_x, orient_y, orient_z, orient_w]
            self._imu_buf.append((t, list(msg.imu_data)))

        # Nothing to do if not actively collecting
        if not self._collect_active:
            return

        # Pull the freshest IMU vector (or empty if none yet)
        latest_imu = self._latest_imu()

        # -----------------------
        # Raw mode: log per sample
        # -----------------------
        if self._collect_mode_raw:
            # Only log when we actually have an EMG sample in this callback
            if msg.emg_data:
                t_rel = t - self._collect_start_time
                row = {
                    "t": t_rel,
                    "label": self._collect_label,
                    "emg": list(msg.emg_data),
                    "imu": latest_imu,  # may be []
                }
                self._raw_rows.append(row)
            return

        # --------------------------------
        # Feature mode: periodic feature row
        # --------------------------------
        # throttle by feature_stride_sec
        if (t - self._last_feature_emit) < self.feature_stride_sec:
            return

        # Build a short EMG window: ~200 ms (works well for 200 Hz default)
        sr = msg.sampling_rate if msg.sampling_rate > 0 else 200.0
        win_sec = 0.20
        win_samples = max(1, int(sr * win_sec))

        # Gather last win_samples EMG frames (most recent at end)
        if len(self._emg_buf) == 0:
            return  # no EMG yet → skip

        # Slice the last N EMG frames and stack to array (N x C)
        emg_frames = [v for (_ts, v) in list(self._emg_buf)[-win_samples:]]
        # If early in buffer fill, window may be shorter; proceed anyway
        X = np.array(emg_frames, dtype=np.float32)  # shape ~ (n, C)

        # Simple channel-wise features: mean, std, rms
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        rms = np.sqrt((X**2).mean(axis=0))

        # IMU summary: just take freshest vector (already in latest_imu)
        # (You could also average over the last ~5 IMU samples, but this keeps it simple.)
        feature_row = {
            "t_mid": t - self._collect_start_time,
            "label": self._collect_label,
            "emg_mean": mean.tolist(),
            "emg_std": std.tolist(),
            "emg_rms": rms.tolist(),
            "imu": latest_imu,  # [] if none yet
        }
        self._feature_rows.append(feature_row)
        self._last_feature_emit = t

    # Helper: get freshest IMU vector (or [])
    def _latest_imu(self):
        return self._imu_buf[-1][1] if len(self._imu_buf) else []

    # ---------------------------
    # Service: AddGesture
    # ---------------------------
    def _srv_add_gesture(self, req: AddGesture.Request, resp: AddGesture.Response):
        return None

    # ---------------------------
    # Service: TrainModel
    # ---------------------------
    def _srv_train_model(self, req: TrainModel.Request, resp: TrainModel.Response):
        return None
        
def main():
    rclpy.init()
    node = Manager()
    try:
        # Multi-threaded so subscriber keeps running while services execute
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
