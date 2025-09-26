import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import pickle
from collections import deque
from ament_index_python.packages import get_package_share_directory
import os
import time

# ---- NEW: keyboard control (pynput) ----
try:
    from pynput.keyboard import Controller, Key
    _KEYBOARD = Controller()
    _KEY_SUPPORT = True
except Exception as e:
    _KEYBOARD = None
    _KEY_SUPPORT = False
    _KEY_IMPORT_ERROR = e

# Parameters
window_size = 20  # Must match training

gesture_labels = {
    0: "rest",
    1: "pinch",
    2: "fist",
    3: "extension",
    4: "left",
    5: "right",
}

class EMGClassifierNode(Node):
    def __init__(self):
        super().__init__('game_rf')

        # Load pre-trained Random Forest model
        model_path = os.path.join(get_package_share_directory('myo_space_trigger'), 'gesture_classifier.pkl')
        with open(model_path, 'rb') as f:
            self.clf = pickle.load(f)

        self.emg_buffer = deque(maxlen=window_size)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'emg_raw',
            self.listener_callback,
            10
        )

        # ---- NEW: trigger settings ----
        self.cooldown_sec = 0.6     # time after a trigger before another can fire
        self.last_trigger_ts = 0.0
        self.consecutive_required = 2  # require N identical predictions in a row to fire
        self._prev_pred = None
        self._streak = 0

        if not _KEY_SUPPORT:
            self.get_logger().warn(
                f"Keyboard control disabled (pynput import failed: {_KEY_IMPORT_ERROR}). "
                "Predictions will log but not press keys."
            )

        self.get_logger().info('EMG Classifier Node started, listening to /emg_raw')

    # feature extraction
    def extract_features(self, data):
        rms = np.sqrt(np.mean(np.square(data), axis=0)).tolist()
        mean = np.mean(data, axis=0).tolist()
        variance = np.var(data, axis=0).tolist()
        return rms + mean + variance

    def _press_key(self, key):
        """Press and release a key (best-effort)."""
        if not _KEY_SUPPORT:
            return
        try:
            _KEYBOARD.press(key)
            _KEYBOARD.release(key)
        except Exception as e:
            self.get_logger().warn(f"Failed to send keypress: {e}")

    def _maybe_trigger_action(self, prediction_id):
        """
        Debounced trigger: only fire if cooldown has passed and prediction is stable.
        """
        now = time.time()

        # Track streak of same predictions
        if prediction_id == self._prev_pred:
            self._streak += 1
        else:
            self._streak = 1
            self._prev_pred = prediction_id

        if self._streak < self.consecutive_required:
            return  # not stable yet

        if (now - self.last_trigger_ts) < self.cooldown_sec:
            return  # still cooling down

        # Map predictions to actions
        if prediction_id in gesture_labels:
            label = gesture_labels[prediction_id]
        else:
            label = f"unknown({prediction_id})"

        if label == "fist":
            self.get_logger().info("Action: SPACE (from 'fist')")
            self._press_key(Key.space)
            self.last_trigger_ts = now

        elif label == "pinch":
            self.get_logger().info("Action: ENTER (from 'pinch')")
            # Enter/Return key
            self._press_key(Key.enter)
            self.last_trigger_ts = now

        # You can add more mappings if needed:
        # elif label == "left":
        #     self._press_key(Key.left)
        # elif label == "right":
        #     self._press_key(Key.right)

    def listener_callback(self, msg):
        if len(msg.data) != 8:
            self.get_logger().warn('Unexpected EMG vector size')
            return

        self.emg_buffer.append(msg.data)

        if len(self.emg_buffer) == window_size:
            window_np = np.array(self.emg_buffer)
            features = self.extract_features(window_np)
            prediction = int(self.clf.predict([features])[0])

            # Log prediction (optional)
            if prediction in gesture_labels:
                self.get_logger().info(f'Predicted Gesture: {gesture_labels[prediction]}')
            else:
                self.get_logger().info(f'Predicted Gesture ID: {prediction}')

            # Try to trigger keys from predictions
            self._maybe_trigger_action(prediction)


def main(args=None):
    rclpy.init(args=args)
    node = EMGClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()