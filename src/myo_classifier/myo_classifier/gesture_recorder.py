#!/usr/bin/env python3
import os, time, csv
from datetime import datetime
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from myo_msgs.msg import MyoMsg
from myo_msgs.srv import AddGesture

class GestureRecorder(Node):
    def __init__(self):
        super().__init__('gesture_recorder')

        # Defaults configurable via YAML
        self.declare_parameter('default_seconds', 5)
        self.declare_parameter('default_cycles', 2)
        self.declare_parameter('default_window_size', 200)
        self.declare_parameter('default_raw_mode', True)
        self.declare_parameter('default_out_dir', 'myo_data')
        self.declare_parameter('default_countdown_sec', 3)
        self.declare_parameter('default_gestures', 'fist, open_hand, thumbs_up')

        self.default_seconds       = int(self.get_parameter('default_seconds').value)
        self.default_cycles        = int(self.get_parameter('default_cycles').value)
        self.default_window_size   = int(self.get_parameter('default_window_size').value)
        self.default_raw_mode      = bool(self.get_parameter('default_raw_mode').value)
        self.default_out_dir       = self._expand(self.get_parameter('default_out_dir').value)
        self.default_countdown_sec = int(self.get_parameter('default_countdown_sec').value)

        _g = self.get_parameter('default_gestures').value
        if isinstance(_g, (list, tuple)):
            self.default_gestures = [str(x).strip() for x in _g if str(x).strip()]
        else:
            self.default_gestures = [s.strip() for s in str(_g).replace(',', ' ').split() if s.strip()]

        self.expect_emg_channels = 8
        self.feature_stride_sec = 0.20

        # Buffers / state
        self._collect_active = False
        self._collect_mode_raw = True
        self._collect_label = None
        self._collect_window = self.default_window_size
        self._collect_start_time = 0.0
        self._emg_window = deque(maxlen=self._collect_window)
        self._imu_buf = deque(maxlen=512)
        self._raw_rows = []
        self._feature_rows = []
        self._last_feature_emit = 0.0

        # QoS + callback groups
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=30,
        )
        self.cb_sub = ReentrantCallbackGroup()
        self.cb_srv = ReentrantCallbackGroup()

        self.sub = self.create_subscription(
            MyoMsg, 'myo/data', self._on_myo, sensor_qos, callback_group=self.cb_sub
        )
        self.srv = self.create_service(
            AddGesture, 'myo_classifier/recorder_add_gesture',
            self._srv_add_gesture, callback_group=self.cb_srv
        )

        self.get_logger().info('GestureRecorder ready (service: /myo_classifier/recorder_add_gesture)')

    def _expand(self, p: str) -> str:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(p or "")))

    def _on_myo(self, msg: MyoMsg):
        if not self._collect_active:
            return
        # latest IMU snapshot
        if msg.imu_data:
            self._imu_buf.append(list(msg.imu_data))

        t_rel = time.time() - self._collect_start_time

        if self._collect_mode_raw:
            if msg.emg_data:
                self._raw_rows.append({
                    "t": t_rel,
                    "label": self._collect_label,
                    "emg": list(msg.emg_data),
                    "imu": (self._imu_buf[-1] if self._imu_buf else []),
                })
            return

        # feature mode (RMS, mean, var per channel)
        if self._emg_window.maxlen != self._collect_window:
            self._emg_window = deque(self._emg_window, maxlen=self._collect_window)

        if msg.emg_data:
            self._emg_window.append(list(msg.emg_data))

        if len(self._emg_window) < self._collect_window:
            return

        now = time.time()
        if (now - self._last_feature_emit) < self.feature_stride_sec:
            return

        import numpy as np
        X = np.asarray(self._emg_window, dtype=np.float32)
        rms  = (np.sqrt((X**2).mean(axis=0))).tolist()
        mean = X.mean(axis=0).tolist()
        var  = X.var(axis=0).tolist()

        self._feature_rows.append({
            "t_mid": t_rel,
            "label": self._collect_label,
            "emg_rms": rms,
            "emg_mean": mean,
            "emg_var": var,
            "imu": (self._imu_buf[-1] if self._imu_buf else []),
        })
        self._last_feature_emit = now

    def _srv_add_gesture(self, req: AddGesture.Request, resp: AddGesture.Response):
        # Resolve parameters (support empty/zero → use defaults)
        gestures = [g.strip() for g in (req.gesture or "").replace(',', ' ').split() if g.strip()]
        if not gestures:
            gestures = list(self.default_gestures)

        seconds     = req.seconds     if req.seconds     > 0 else self.default_seconds
        cycles      = req.cycles      if req.cycles      > 0 else self.default_cycles
        window_size = req.window_size if req.window_size > 0 else self.default_window_size
        raw_mode    = req.raw_mode if isinstance(req.raw_mode, bool) else self.default_raw_mode
        out_dir     = self._expand((req.out_dir or "") or self.default_out_dir)
        countdown   = req.countdown_sec if req.countdown_sec > 0 else self.default_countdown_sec

        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(out_dir, f'add_gesture_{stamp}.csv')

        C = self.expect_emg_channels
        emg_cols = [f'emg_{i+1}' for i in range(C)]
        imu_cols = ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','orient_x','orient_y','orient_z','orient_w']

        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            if raw_mode:
                header = ['t','label'] + emg_cols + imu_cols
            else:
                feat_cols = ([f'emg_rms_{i+1}' for i in range(C)] +
                             [f'emg_mean_{i+1}' for i in range(C)] +
                             [f'emg_var_{i+1}' for i in range(C)])
                header = ['t_mid','label'] + feat_cols + imu_cols
            w.writerow(header)

        try:
            for g in gestures:
                for rep in range(1, cycles+1):
                    if countdown > 0:
                        for k in range(countdown, 0, -1):
                            self.get_logger().info(f'[Countdown] {g} rep {rep}/{cycles} starts in {k}...')
                            time.sleep(1.0)

                    # arm
                    self._collect_label = g
                    self._collect_mode_raw = raw_mode
                    self._collect_window = int(window_size)
                    self._raw_rows.clear()
                    self._feature_rows.clear()
                    self._emg_window.clear()
                    self._collect_start_time = time.time()
                    self._collect_active = True
                    self._last_feature_emit = 0.0

                    self.get_logger().info(f'[Collect] {g} rep {rep}/{cycles} for {seconds}s...')
                    t0 = time.time()
                    while (time.time() - t0) < float(seconds):
                        time.sleep(0.02)

                    # disarm & snapshot
                    self._collect_active = False
                    rows_raw  = list(self._raw_rows)
                    rows_feat = list(self._feature_rows)
                    self._raw_rows.clear()
                    self._feature_rows.clear()

                    # append to CSV
                    with open(csv_path, 'a', newline='') as f:
                        w = csv.writer(f)
                        if raw_mode:
                            for r in rows_raw:
                                emg = (list(r.get('emg', [])) + [""]*C)[:C]
                                imu = (list(r.get('imu', [])) + [""]*10)[:10]
                                w.writerow([r.get('t',''), r.get('label', g)] + emg + imu)
                            self.get_logger().info(f'[Collect] Saved {len(rows_raw)} RAW samples for {g} rep {rep}/{cycles}')
                        else:
                            pad = lambda a: (list(a)+[""]*C)[:C]
                            for r in rows_feat:
                                imu = (list(r.get('imu', [])) + [""]*10)[:10]
                                w.writerow([r.get('t_mid',''), r.get('label', g)] +
                                           pad(r.get('emg_rms', [])) +
                                           pad(r.get('emg_mean', [])) +
                                           pad(r.get('emg_var', [])) + imu)
                            self.get_logger().info(f'[Collect] Saved {len(rows_feat)} FEATURE rows for {g} rep {rep}/{cycles}')

            resp.success = True
            resp.csv_path = csv_path
            resp.message = f"Recorder wrote CSV: {csv_path}"
            return resp
        except Exception as e:
            self.get_logger().error(f'Recorder error: {e}')
            resp.success = False
            resp.csv_path = csv_path
            resp.message = f'Error: {e}'
            return resp

def main():
    rclpy.init()
    node = GestureRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
