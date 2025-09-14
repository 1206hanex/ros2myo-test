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

    # ---------------------------
    # Subscriber: data collection
    # ---------------------------
    def _on_myo(self, msg: MyoMsg):
        if len(msg.emg_data) != self.expect_emg_channels:
            return

        # Store latest sample
        sample = [float(v) for v in msg.emg_data]
        self._emg_buf.append(sample)

        # If not actively collecting, nothing else to do
        if not self._collect_active:
            return

        now = time.monotonic()

        if self._collect_mode_raw:
            # Save raw per-sample rows: emg_0..emg_7,label
            row = sample + [self._collect_label]
            self._raw_rows.append(row)
        else:
            # Feature mode: sample a window every stride
            if len(self._emg_buf) >= self._collect_window:
                if (now - self._last_feature_emit) >= self.feature_stride_sec:
                    window = np.array(list(self._emg_buf)[-self._collect_window:], dtype=np.float32)
                    feats = self._features_from_window(window)   # len = 8*3
                    row = feats + [self._collect_label]
                    self._feature_rows.append(row)
                    self._last_feature_emit = now

    @staticmethod
    def _features_from_window(X: np.ndarray) -> list:
        # X shape: (W, C)
        rms  = np.sqrt(np.mean(X**2, axis=0)).tolist()
        mean = np.mean(X, axis=0).tolist()
        var  = np.var(X, axis=0).tolist()
        return rms + mean + var

    # ---------------------------
    # Service: AddGesture
    # ---------------------------
    def _srv_add_gesture(self, req: AddGesture.Request, resp: AddGesture.Response):
        # Validate/prepare
        if not req.gesture:
            resp.success = False; resp.message = 'gesture name required'; return resp
        os.makedirs(req.out_dir, exist_ok=True)
        self.get_logger().info(f"AddGesture: label={req.gesture} seconds={req.seconds} cycles={req.cycles} raw_mode={req.raw_mode}")

        # Optional countdown per cycle
        countdown = max(0, int(req.countdown_sec))

        # Do cycles
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(req.out_dir, f"{req.gesture}_{'raw' if req.raw_mode else 'fe'}_{ts}.csv")

        # Reset buffers/state
        with self._collect_lock:
            self._collect_mode_raw = bool(req.raw_mode)
            self._collect_label = req.gesture
            self._collect_window = max(2, int(req.window_size))
            self._collect_outdir = req.out_dir
            self._feature_rows.clear()
            self._raw_rows.clear()
            self._last_feature_emit = 0.0

        # Ensure executor can run subscriber callbacks while we "wait" in this service
        for cycle in range(req.cycles):
            if countdown:
                self.get_logger().info(f"Cycle {cycle+1}/{req.cycles}: starting in {countdown}s...")
                time.sleep(countdown)

            self.get_logger().info(f"Recording '{req.gesture}' for {req.seconds}s...")
            start = time.monotonic()
            with self._collect_lock:
                self._collect_active = True
                self._collect_start_time = start
                self._collect_end_time = start + req.seconds

            # Wait for the duration while callbacks accumulate data
            while time.monotonic() < (start + req.seconds):
                time.sleep(0.05)

            with self._collect_lock:
                self._collect_active = False

        # Write CSV
        try:
            if req.raw_mode:
                # Header: emg_0..emg_7,label
                header = [f"emg_{i}" for i in range(self.expect_emg_channels)] + ["label"]
                rows = self._raw_rows
            else:
                # Header: emg_rms_0..7, emg_mean_0..7, emg_var_0..7, label
                header = \
                    [f"emg_rms_{i}"  for i in range(self.expect_emg_channels)] + \
                    [f"emg_mean_{i}" for i in range(self.expect_emg_channels)] + \
                    [f"emg_var_{i}"  for i in range(self.expect_emg_channels)] + \
                    ["label"]
                rows = self._feature_rows

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

            resp.success = True
            resp.message = f"Recorded {len(rows)} rows to {csv_path}"
            resp.csv_path = csv_path
            self.get_logger().info(resp.message)
        except Exception as e:
            resp.success = False
            resp.message = f"Failed to write CSV: {e}"
        return resp

    # ---------------------------
    # Service: TrainModel
    # ---------------------------
    def _srv_train_model(self, req: TrainModel.Request, resp: TrainModel.Response):
        self.get_logger().info(f"TrainModel: type={req.model_type} data_dir={req.data_dir}")
        try:
            os.makedirs(req.out_dir, exist_ok=True)
            
            model_type = model_type.lower().strip()

            if model_type == "rf":
                X, y = load_feature_csvs(req.data_dir)
                if X.size == 0:
                    resp.success = False; resp.message = "No feature rows found."; return resp
                model_path, _ = train_rf(X, y, req.out_dir, logger=self.get_logger().info)
            elif model_type == "cnn_lstm":
                # hyperparams could be declared as ROS params if you want
                X, y = load_raw_sequences(req.data_dir, self.expect_emg_channels, seq_len=64, seq_stride=16)
                if X.shape[0] == 0:
                    resp.success = False; resp.message = "No raw sequences found."; return resp
                model_path = train_cnn_lstm(X, y, req.out_dir, logger=self.get_logger().info)
            else:
                resp.success = False; resp.message = "model_type must be 'rf' or 'cnn_lstm'"; return resp
            
            
        except Exception as e:
            resp.success = False
            resp.message = f"Training failed: {e}"
            return resp

    @staticmethod
    def _load_feature_csvs(data_dir: str):
        import pandas as pd
        X_list, y_list = [], []
        for p in glob.glob(os.path.join(data_dir, "*.csv")):
            try:
                df = pd.read_csv(p)
                if "label" not in df.columns: 
                    continue
                # Heuristic: consider files with feature-like columns as features CSVs
                feat_cols = [c for c in df.columns if c.startswith(("emg_rms_", "emg_mean_", "emg_var_"))]
                if len(feat_cols) == 0:
                    continue
                X_list.append(df[feat_cols].values)
                y_list.append(df["label"].values)
            except Exception:
                pass
        if not X_list:
            return np.empty((0,)), []
        X = np.vstack(X_list)
        y = np.concatenate(y_list).tolist()
        return X, y

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
