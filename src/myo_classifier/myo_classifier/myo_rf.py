import os
import pickle
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from myo_msgs.msg import MyoMsg
from std_msgs.msg import String

class EMGClassifierRF(Node):
    def __init__(self):
        super().__init__('myo_rf')

        self.declare_parameter('window_size', 20)
        self.window_size = int(self.get_parameter('window_size').value)

        # model artifacts in package share (adjust via params if you want)
        pkg_share = get_package_share_directory('myo_gestures')
        model_path = os.path.join(pkg_share, 'gesture_classifier.pkl')

        with open(model_path, 'rb') as f:
            self.clf = pickle.load(f)
        self.get_logger().info(f'Loaded RF model: {model_path}')

        self.emg_buf = deque(maxlen=self.window_size)
        self.sub = self.create_subscription(MyoMsg, 'myo/data', self.cb, 10)

        self.pub_label = self.create_publisher(String, 'gesture/label', 10)

        self.labels = {0:'rest', 1:'pinch', 2:'fist', 3:'extension', 4:'left', 5:'right'}

    def cb(self, msg: MyoMsg):
        if len(msg.emg_data) != 8:
            return
        self.emg_buf.append(msg.emg_data)
        if len(self.emg_buf) == self.window_size:
            X = np.array(self.emg_buf)  # (W,8)
            feats = self._feats(X)
            pred = int(self.clf.predict([feats])[0])
            label = self.labels.get(pred, f'UNK({pred})')
            self.get_logger().info(f'Predicted: {label}')
            self.pub_label.publish(String(data=label))

    def _feats(self, X):
        rms  = np.sqrt(np.mean(X**2, axis=0))
        mean = np.mean(X, axis=0)
        var  = np.var(X, axis=0)
        return np.hstack([rms, mean, var])

def main(args=None):
    rclpy.init(args=args)
    node = EMGClassifierRF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()