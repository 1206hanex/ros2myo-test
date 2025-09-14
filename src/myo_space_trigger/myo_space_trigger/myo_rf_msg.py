import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from myo_space_trigger.msg import MyoMsg
import numpy as np
import pickle
from collections import deque

from ament_index_python.packages import get_package_share_directory
import os


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
        super().__init__('myo_rf_msg')

        # Load pre-trained Random Forest model
        model_path = os.path.join(get_package_share_directory('myo_space_trigger'), 'gesture_classifier.pkl')
        with open(model_path, 'rb') as f:
            self.clf = pickle.load(f)

        self.emg_buffer = deque(maxlen=window_size)
        self.imu_buffer = deque(maxlen=window_size)
        # subscribe to myo/data topic and receive EMG data streams
        self.subscription = self.create_subscription(
            MyoMsg,
            'myo/data',
            self.listener_callback,
            10
        )
        self.get_logger().info('EMG Classifier Node started, listening to myo/data')

    # feature extraction
    def extract_features(self, emg_window, imu_window):
        emg_arr = np.array(emg_window)
        imu_arr = np.array(imu_window)
        
        # EMG values
        emg_rms = np.sqrt(np.mean(emg_arr**2, axis=0))
        emg_mean = np.mean(emg_arr, axis=0)
        emg_var = np.var(emg_arr, axis=0)
        
        # IMU values
        imu_rms = np.sqrt(np.mean(imu_arr**2, axis=0))
        imu_mean = np.mean(imu_arr, axis=0)
        imu_var = np.var(imu_arr, axis=0)
        
        # concatenate to one feature vector
        return np.hstack([emg_rms, emg_mean, emg_var, imu_rms, imu_mean, imu_var])

    def listener_callback(self, msg: MyoMsg):
        if len(msg.data) != 8:
            self.get_logger().warn('Unexpected EMG vector size')
            return

        # add new sample to sliding window
        self.emg_buffer.append(msg.data)
        self.imu_buffer.append(msg.data)

        # if buffer is full, perform classification
        if len(self.emg_buffer) == window_size:
            window_np = np.array(self.emg_buffer)
            features = self.extract_features(window_np)
            # model expects a 2D array, 
            prediction = self.clf.predict([features])[0]
            self.get_logger().info(f'Predicted Gesture: {gesture_labels[prediction]}')


# ROS2 boilerplate to run a node
def main(args=None):
    rclpy.init(args=args)
    node = EMGClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
