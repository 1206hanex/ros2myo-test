#!/usr/bin/env python3
import os, time, csv, threading, glob, pickle
from datetime import datetime
from collections import deque, Counter

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from ament_index_python.packages import get_package_share_directory
from myo_msgs.msg import MyoMsg
from myo_msgs.srv import AddGesture, TrainModel

from myo_classifier.train.data import load_feature_csvs, load_raw_sequences
from myo_classifier.train.rf import train_rf
# from myo_classifier.train.cnn_lstm import train_cnn_lstm


class Manager(Node):
    def __init__(self):
        super().__init__('manager')

        # Callback groups so sub + service can run concurrently
        self.cb_sub  = ReentrantCallbackGroup()
        self.cb_srv  = ReentrantCallbackGroup()

        # (optional but recommended) Sensor-data QoS for 200 Hz EMG
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=30,
        )

        # Subscribe (note the callback_group)
        self.sub = self.create_subscription(
            MyoMsg,
            'myo/data',
            self._on_myo,
            sensor_qos,                      # instead of plain "50"
            callback_group=self.cb_sub
        )

        # Services (note the callback_group)
        self.srv_add = self.create_service(
            AddGesture,
            'myo_classifier/add_gesture',
            self._srv_add_gesture,
            callback_group=self.cb_srv
        )

        # Parameters common to collection
        self.declare_parameter('feature_stride_sec', 0.20)   # how often to sample a window (features mode)
        self.declare_parameter('expect_emg_channels', 8)
        self.feature_stride_sec = float(self.get_parameter('feature_stride_sec').value)
        self.expect_emg_channels = int(self.get_parameter('expect_emg_channels').value)

        # Sliding window for feature extraction (EMG only)
        self._emg_window = deque(maxlen=200)

        # State for collection
        self._collect_lock = threading.Lock()
        self._collect_active = False
        self._collect_mode_raw = True
        self._collect_label = None
        self._collect_window = 20
        self._collect_outdir = None
        self._collect_start_time = 0.0
        self._collect_end_time = 0.0

        self._emg_buf = deque(maxlen=4096)       # rolling buffer of EMG samples (each is list[float])
        self._imu_buf = deque(maxlen=512)       # reserved if you add IMU later

        self._feature_rows = []                 # features rows when raw_mode=False
        self._raw_rows = []                     # raw rows (per-sample) when raw_mode=True

        self._last_feature_emit = 0.0

        # Simple telemetry
        self._emg_samples_total = 0
        self._imu_samples_total = 0

        # Subscribe to the streaming topic from your publisher
        #self.sub = self.create_subscription(MyoMsg, 'myo/data', self._on_myo, 50)

        # Services
        self.srv_add = self.create_service(AddGesture, 'myo_classifier/add_gesture', self._srv_add_gesture)
        self.srv_train = self.create_service(TrainModel, 'myo_classifier/train_model', self._srv_train_model)

        self.recorder_cli = self.create_client(AddGesture, '/myo_classifier/gesture_recorder')

        self.get_logger().info('Manager ready: services {add_gesture, train_model}')

    def _expand_path(self, p: str) -> str:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(p or "")))

    # ===========================
    # Subscriber: data collection
    # ===========================
    def _on_myo(self, msg: MyoMsg):
        """Handle incoming MyoMsg: maintain buffers; when collecting, append raw rows or feature rows."""
        # ROS time → float seconds
        t_ros = float(msg.stamp.sec) + 1e-9 * float(msg.stamp.nanosec)

        # ---- Update rolling buffers (always) ----
        if msg.emg_data:
            self._emg_samples_total += 1
            emg_vec = list(msg.emg_data)
            self._emg_buf.append((t_ros, emg_vec))
            # Keep a sliding EMG window for feature mode
            self._emg_window.append(emg_vec)

        if msg.imu_data:
            self._imu_samples_total += 1
            self._imu_buf.append((t_ros, list(msg.imu_data)))

        # Quick, low-noise visibility
        if self._emg_samples_total and (self._emg_samples_total % 400 == 0):
            self.get_logger().info(f'EMG frames seen: {self._emg_samples_total}')
        if self._imu_samples_total and (self._imu_samples_total % 50 == 0):
            self.get_logger().info(f'IMU frames seen: {self._imu_samples_total}')

        # ---- If not actively collecting, stop here ----
        if not self._collect_active:
            return

        # Get latest IMU snapshot (may be empty if none yet)
        latest_imu = self._imu_buf[-1][1] if self._imu_buf else []

        # Time relative to cycle start (use wall time so start/stop is consistent)
        t_rel = time.time() - self._collect_start_time

        # ---- RAW MODE: one row per EMG packet ----
        if self._collect_mode_raw and msg.emg_data:
            if (len(self._raw_rows) % 200) == 0:
                self.get_logger().info(f'[debug] collecting RAW… rows={len(self._raw_rows)}')
            self._raw_rows.append({
                "t": t_rel,
                "label": self._collect_label,
                "emg": list(msg.emg_data),
                "imu": latest_imu,
            })
            return

        # ---- FEATURE MODE: when window full, emit RMS/Mean/Var (+ optional throttle) ----
        # Respect window size set by service
        if self._emg_window.maxlen != self._collect_window:
            self._emg_window = deque(self._emg_window, maxlen=self._collect_window)

        if len(self._emg_window) < self._collect_window:
            return  # not enough EMG yet

        # Optional throttle (about 5 Hz)
        now = time.time()
        if (now - self._last_feature_emit) < self.feature_stride_sec:
            return

        import numpy as np
        X = np.asarray(self._emg_window, dtype=np.float32)  # shape: (W, C)
        # features per channel
        rms = np.sqrt((X ** 2).mean(axis=0)).tolist()
        mean = X.mean(axis=0).tolist()
        var = X.var(axis=0).tolist()

        self._feature_rows.append({
            "t_mid": t_rel,
            "label": self._collect_label,
            "emg_rms": rms,
            "emg_mean": mean,
            "emg_var": var,
            "imu": latest_imu,  # snapshot – you can add IMU features later if desired
     })
        self._last_feature_emit = now


    # Helper: get freshest IMU vector (or [])
    def _latest_imu(self):
        return self._imu_buf[-1][1] if len(self._imu_buf) else []
  
    # ---------------------------
    # Service: AddGesture
    # ---------------------------
    def _srv_add_gesture(self, req: AddGesture.Request, resp: AddGesture.Response):
        if not self.recorder_cli.wait_for_service(timeout_sec=2.0):
            resp.success = False
            resp.message = "Recorder service not available (/myo_classifier/recorder_add_gesture). Is gesture_recorder running?"
            resp.csv_path = ""
            return resp

        forward = AddGesture.Request()
        forward.gesture       = req.gesture
        forward.seconds       = req.seconds
        forward.cycles        = req.cycles
        forward.window_size   = req.window_size
        forward.raw_mode      = req.raw_mode
        forward.out_dir       = req.out_dir
        forward.countdown_sec = req.countdown_sec

        future = self.recorder_cli.call_async(forward)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result:
            resp.success = result.success
            resp.message = result.message
            resp.csv_path = result.csv_path
        else:
            resp.success = False
            resp.message = "Recorder call failed."
            resp.csv_path = ""
        return resp

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
