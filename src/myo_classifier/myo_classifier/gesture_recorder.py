#!/usr/bin/env python3
import os, time, csv, threading, sys
from datetime import datetime
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from myo_msgs.msg import MyoMsg

class GestureRecorder(Node):
    def __init__(self):
        super().__init__('gesture_recorder')

        # ---------- Parameters (YAML or CLI) ----------
        # gestures can be "fist, open_hand, thumbs_up" or ["fist","open_hand"]
        self.declare_parameter('gestures', 'rest, fist, open_hand, wrist_left, wrist_right')
        self.declare_parameter('seconds', 8)                 # per cycle
        self.declare_parameter('cycles', 2)                  # repetitions per gesture
        self.declare_parameter('window_size', 200)           # samples for features
        self.declare_parameter('raw_mode', True)             # True=raw rows, False=features
        self.declare_parameter('out_dir', '~/myo_data')      # where to save CSV
        self.declare_parameter('countdown_sec', 1)           # cue before each cycle
        self.declare_parameter('expect_emg_channels', 8)     # sanity for CSV shape
        self.declare_parameter('feature_stride_sec', 0.20)   # ~5 Hz feature emission
        self.declare_parameter('auto_exit', False)           # shutdown after done

        self.declare_parameter('prompt_enter', True)

        # Normalize params
        _g = self.get_parameter('gestures').value
        if isinstance(_g, (list, tuple)):
            self.gestures = [str(x).strip() for x in _g if str(x).strip()]
        else:
            self.gestures = [s.strip() for s in str(_g).replace(',', ' ').split() if s.strip()]

        self.seconds       = int(self.get_parameter('seconds').value)
        self.cycles        = int(self.get_parameter('cycles').value)
        self.window_size   = int(self.get_parameter('window_size').value)
        self.raw_mode      = bool(self.get_parameter('raw_mode').value)
        self.out_dir       = self._expand(self.get_parameter('out_dir').value)
        self.countdown_sec = int(self.get_parameter('countdown_sec').value)
        self.C             = int(self.get_parameter('expect_emg_channels').value)
        self.feature_stride_sec = float(self.get_parameter('feature_stride_sec').value)
        self.auto_exit     = bool(self.get_parameter('auto_exit').value)
        self.prompt_enter  = bool(self.get_parameter('prompt_enter').value)

        os.makedirs(self.out_dir, exist_ok=True)

        # ---------- State / buffers ----------
        self._lock = threading.Lock()
        self._collect_active = False
        self._collect_label = None
        self._collect_start_time = 0.0
        self._collect_window = self.window_size
        self._last_feature_emit = 0.0

        self._emg_window = deque(maxlen=self._collect_window)
        self._imu_buf = deque(maxlen=512)
        self._raw_rows = []
        self._feature_rows = []

        self._emg_samples_total = 0
        self._imu_samples_total = 0

        # ---------- I/O ----------
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.out_dir, f'add_gesture_{stamp}.csv')
        self._init_csv()

        # ---------- ROS I/O ----------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=30,
        )
        self.cb = ReentrantCallbackGroup()
        self.sub = self.create_subscription(MyoMsg, 'myo/data', self._on_myo, sensor_qos, callback_group=self.cb)

        # Orchestration thread (so subscriber can run freely)
        self._runner = threading.Thread(target=self._orchestrate, daemon=True)
        self._runner.start()

        self.get_logger().info(
            f'GestureRecorder ready. gestures={self.gestures} cycles={self.cycles} seconds={self.seconds} '
            f'raw_mode={self.raw_mode} window={self.window_size} out="{self.csv_path}" prompt_enter={self.prompt_enter}'
        )

    # ---------- utils ----------
    def _expand(self, p: str) -> str:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(p or "")))

    def _init_csv(self):
        emg_cols = [f'emg_{i+1}' for i in range(self.C)]
        imu_cols = ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','orient_x','orient_y','orient_z','orient_w']
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            if self.raw_mode:
                header = ['t','label'] + emg_cols + imu_cols
            else:
                feat_cols = ([f'emg_rms_{i+1}' for i in range(self.C)] +
                             [f'emg_mean_{i+1}' for i in range(self.C)] +
                             [f'emg_var_{i+1}' for i in range(self.C)])
                header = ['t_mid','label'] + feat_cols + imu_cols
            w.writerow(header)

    def _wait_for_enter(self, g: str, rep: int):
        if not self.prompt_enter:
            return
        try:
            if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                # use print/input instead of logger so the prompt appears cleanly
                print(f'\n[Ready] Prepare gesture "{g}" rep {rep}/{self.cycles}. '
                      f'Press ENTER to start recording...', flush=True)
                input()
            else:
                self.get_logger().warn('stdin is not a TTY; proceeding without Enter prompt.')
        except Exception as e:
            self.get_logger().warn(f'Enter prompt failed ({e}); proceeding without waiting.')


    # ---------- subscriber ----------
    def _on_myo(self, msg: MyoMsg):
        # basic telemetry
        if msg.emg_data:
            self._emg_samples_total += 1
        if msg.imu_data:
            self._imu_buf.append(list(msg.imu_data))
            self._imu_samples_total += 1

        # if not recording, keep a small live window but skip heavy work
        if not self._collect_active:
            if msg.emg_data and len(self._emg_window) < self._emg_window.maxlen:
                self._emg_window.append(list(msg.emg_data))
            return

        # during recording:
        latest_imu = (self._imu_buf[-1] if self._imu_buf else [])
        t_rel = time.time() - self._collect_start_time

        if self.raw_mode:
            if msg.emg_data:
                with self._lock:
                    self._raw_rows.append({
                        "t": t_rel,
                        "label": self._collect_label,
                        "emg": list(msg.emg_data),
                        "imu": latest_imu,
                    })
            return

        # feature mode
        if msg.emg_data:
            self._emg_window.append(list(msg.emg_data))

        # Ensure window matches latest setting
        if self._emg_window.maxlen != self._collect_window:
            self._emg_window = deque(self._emg_window, maxlen=self._collect_window)

        if len(self._emg_window) < self._collect_window:
            return

        now = time.time()
        if (now - self._last_feature_emit) < self.feature_stride_sec:
            return

        X = np.asarray(self._emg_window, dtype=np.float32)  # (W, C)
        rms  = np.sqrt((X**2).mean(axis=0)).tolist()
        mean = X.mean(axis=0).tolist()
        var  = X.var(axis=0).tolist()

        with self._lock:
            self._feature_rows.append({
                "t_mid": t_rel,
                "label": self._collect_label,
                "emg_rms": rms,
                "emg_mean": mean,
                "emg_var": var,
                "imu": latest_imu,
            })
        self._last_feature_emit = now

    # ---------- orchestrator thread ----------
    def _orchestrate(self):
        try:
            for g in self.gestures:
                for rep in range(1, self.cycles + 1):
                    self._wait_for_enter(g, rep)
                    # countdown
                    if self.countdown_sec > 0:
                        for k in range(self.countdown_sec, 0, -1):
                            self.get_logger().info(f'[Countdown] {g} rep {rep}/{self.cycles} starts in {k}...')
                            time.sleep(1.0)

                    # arm
                    with self._lock:
                        self._collect_label = g
                        self._collect_window = int(self.window_size)
                        self._raw_rows.clear()
                        self._feature_rows.clear()
                        self._emg_window.clear()
                        self._collect_start_time = time.time()
                        self._collect_active = True
                        self._last_feature_emit = 0.0

                    self.get_logger().info(f'[Collect] {g} rep {rep}/{self.cycles} for {self.seconds}s...')
                    t0 = time.time()
                    while (time.time() - t0) < float(self.seconds):
                        time.sleep(0.02)

                    # disarm & snapshot
                    with self._lock:
                        self._collect_active = False
                        rows_raw  = list(self._raw_rows)
                        rows_feat = list(self._feature_rows)
                        self._raw_rows.clear()
                        self._feature_rows.clear()

                    # append to CSV
                    with open(self.csv_path, 'a', newline='') as f:
                        w = csv.writer(f)
                        if self.raw_mode:
                            n = 0
                            for r in rows_raw:
                                emg = (list(r.get('emg', [])) + [""]*self.C)[:self.C]
                                imu = (list(r.get('imu', [])) + [""]*10)[:10]
                                w.writerow([r.get('t',''), r.get('label', g)] + emg + imu)
                                n += 1
                            self.get_logger().info(f'[Collect] Saved {n} RAW samples for {g} rep {rep}/{self.cycles}')
                        else:
                            n = 0
                            pad = lambda a: (list(a)+[""]*self.C)[:self.C]
                            for r in rows_feat:
                                imu = (list(r.get('imu', [])) + [""]*10)[:10]
                                w.writerow([r.get('t_mid',''), r.get('label', g)] +
                                           pad(r.get('emg_rms', [])) +
                                           pad(r.get('emg_mean', [])) +
                                           pad(r.get('emg_var', [])) + imu)
                                n += 1
                            self.get_logger().info(f'[Collect] Saved {n} FEATURE rows for {g} rep {rep}/{self.cycles}')
            self.get_logger().info(f'All done. CSV: {self.csv_path}')
        except Exception as e:
            self.get_logger().error(f'Recorder error: {e}')
        finally:
            if self.auto_exit:
                # allow logs to flush
                time.sleep(0.3)
                self.get_logger().info('Shutting down (auto_exit=true)')
                rclpy.shutdown()

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
