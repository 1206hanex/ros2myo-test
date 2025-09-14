import os
import pickle
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from myo_msgs.msg import MyoMsg
from std_msgs.msg import String
import tensorflow as tf

class CNNSeqClassifier(Node):
    def __init__(self):
        super().__init__('myo_cnn')

        self.declare_parameter('window_size', 40)
        self.declare_parameter('use_imu', False)
        self.declare_parameter('model_filename', 'gesture_cnn_lstm.h5')
        self.declare_parameter('norm_filename', 'cnn_norm_stats.npz')
        self.declare_parameter('label_encoder_filename', 'label_encoder.pkl')

        self.window_size = int(self.get_parameter('window_size').value)
        self.use_imu = bool(self.get_parameter('use_imu').value)

        pkg_share = get_package_share_directory('myo_gestures')
        self.model_path = os.path.join(pkg_share, self.get_parameter('model_filename').value)
        self.norm_path  = os.path.join(pkg_share, self.get_parameter('norm_filename').value)
        self.le_path    = os.path.join(pkg_share, self.get_parameter('label_encoder_filename').value)

        self.model = tf.keras.models.load_model(self.model_path)
        self.get_logger().info(f'Loaded CNN model: {self.model_path}')

        self.norm_mean, self.norm_std = None, None
        if os.path.exists(self.norm_path):
            stats = np.load(self.norm_path)
            self.norm_mean = stats.get('mean', None)
            self.norm_std  = stats.get('std', None)

        self.le = None
        if os.path.exists(self.le_path):
            with open(self.le_path, 'rb') as f:
                self.le = pickle.load(f)

        self.emg_buf = deque(maxlen=self.window_size)
        self.imu_buf = deque(maxlen=self.window_size)

        self.n_emg = None
        self.n_imu = None

        self.sub = self.create_subscription(MyoMsg, 'myo/data', self.cb, 10)
        self.pub_label = self.create_publisher(String, 'gesture/label', 10)

    def cb(self, msg: MyoMsg):
        if self.n_emg is None:
            self.n_emg = len(msg.emg_data)
        if self.use_imu and self.n_imu is None:
            self.n_imu = len(msg.imu_data)

        if len(msg.emg_data) == self.n_emg:
            self.emg_buf.append(msg.emg_data)
        else:
            return

        if self.use_imu:
            if len(msg.imu_data) == self.n_imu:
                self.imu_buf.append(msg.imu_data)
            else:
                return
        else:
            if len(self.imu_buf) < self.window_size:
                self.imu_buf.append([])

        if len(self.emg_buf) == self.window_size and (not self.use_imu or len(self.imu_buf) == self.window_size):
            X = self._assemble_window()
            X = self._normalize(X)
            X = X.astype(np.float32)[None, :, :]   # (1,T,C)
            probs = self.model.predict(X, verbose=0)
            pred = int(np.argmax(probs, axis=1)[0])
            if self.le:
                label = self.le.inverse_transform([pred])[0]
            else:
                label = f'class_{pred}'
            self.get_logger().info(f'Predicted: {label} (conf={float(np.max(probs)):.2f})')
            self.pub_label.publish(String(data=label))

    def _assemble_window(self):
        emg = np.array(self.emg_buf, dtype=np.float32)
        if self.use_imu and self.n_imu and self.n_imu > 0:
            imu = np.array(self.imu_buf, dtype=np.float32)
            return np.concatenate([emg, imu], axis=1)
        return emg

    def _normalize(self, X):
        eps = 1e-8
        if self.norm_mean is not None and self.norm_std is not None and \
           self.norm_mean.shape[0] == X.shape[1]:
            return (X - self.norm_mean[None, :]) / (self.norm_std[None, :] + eps)
        mean = X.mean(axis=0, keepdims=True)
        std  = X.std(axis=0, keepdims=True)
        return (X - mean) / (std + eps)

def main(args=None):
    rclpy.init(args=args)
    node = CNNSeqClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()