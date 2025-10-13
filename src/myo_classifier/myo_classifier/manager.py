#!/usr/bin/env python3
import os, time
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from myo_msgs.msg import MyoMsg
from myo_msgs.srv import TrainModel   # your existing training srv

class Manager(Node):
    def __init__(self):
        super().__init__('manager')

        # ------------- Callback groups -------------
        self.cb_mux  = ReentrantCallbackGroup()
        self.cb_srv  = ReentrantCallbackGroup()

        # ------------- Mux parameters -------------
        # ‘live’ routes /myo/data → /emg/selected
        # ‘replay’ routes /replay/data → /emg/selected
        # ‘off’ drops everything (classifier gets nothing)
        self.declare_parameter('mux.source', 'live')            # live | replay | off
        self.declare_parameter('mux.live_topic', '/myo/data')
        self.declare_parameter('mux.replay_topic', '/replay/data')
        self.declare_parameter('mux.output_topic', '/emg/selected')

        self.mux_source       = self.get_parameter('mux.source').get_parameter_value().string_value
        self.mux_live_topic   = self.get_parameter('mux.live_topic').get_parameter_value().string_value
        self.mux_replay_topic = self.get_parameter('mux.replay_topic').get_parameter_value().string_value
        self.mux_output_topic = self.get_parameter('mux.output_topic').get_parameter_value().string_value

        # Watch param changes so we can switch at runtime
        self.add_on_set_parameters_callback(self._on_set_params)

        # ------------- QoS -------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=30,
        )

        # ------------- Publisher (the single selected output) -------------
        self.pub_mux = self.create_publisher(MyoMsg, self.mux_output_topic, 10)

        # ------------- Inputs -------------
        self.sub_live = self.create_subscription(
            MyoMsg, self.mux_live_topic, self._on_live, sensor_qos, callback_group=self.cb_mux
        )
        self.sub_replay = self.create_subscription(
            MyoMsg, self.mux_replay_topic, self._on_replay, sensor_qos, callback_group=self.cb_mux
        )

        # ------------- TrainModel service (server) -------------
        # Keep your existing training implementation here. During training,
        # optionally set mux.source='off' to pause the classifier stream.
        self.srv_train = self.create_service(
            TrainModel, 'myo_classifier/train_model', self._srv_train_model, callback_group=self.cb_srv
        )

        self.get_logger().info(
            f'Manager up: mux source="{self.mux_source}", '
            f'live="{self.mux_live_topic}", replay="{self.mux_replay_topic}", out="{self.mux_output_topic}"'
        )

    # ------------------ Mux handlers ------------------
    def _on_live(self, msg: MyoMsg):
        if self.mux_source == 'live':
            self.pub_mux.publish(msg)

    def _on_replay(self, msg: MyoMsg):
        if self.mux_source == 'replay':
            self.pub_mux.publish(msg)

    def _on_set_params(self, params):
        for p in params:
            if p.name in ('mux.source', 'source', 'emg_source'):
                val = str(p.value).strip().lower()
                if val not in ('live', 'replay', 'off'):
                    self.get_logger().warn(f'Invalid mux.source="{val}" (use live|replay|off)')
                    return SetParametersResult(successful=False, reason='mux.source must be live|replay|off')
                self.mux_source = val
                self.get_logger().info(f'emg_mux source -> {self.mux_source}')
        return SetParametersResult(successful=True)

    # ------------------ TrainModel service ------------------
    def _srv_train_model(self, req: TrainModel.Request, resp: TrainModel.Response):
        """
        Example pattern:
        - Temporarily set mux to 'off' so classifier stops receiving data.
        - Run your training routine (load CSVs, train, write model file, etc).
        - Restore previous source.
        """
        prev = self.mux_source
        try:
            # Pause stream to classifier (optional)
            self._set_mux('off')

            # TODO: implement your training logic here using req fields
            # E.g., load data, train RF/CNN, save model, set resp fields.
            # Below is a placeholder:
            time.sleep(0.5)  # just to show where training work would happen
            resp.success = True
            resp.message = 'Training complete'
            resp.model_path = ''  # fill in your saved model path
            return resp
        except Exception as e:
            resp.success = False
            resp.message = f'Error during training: {e}'
            resp.model_path = ''
            return resp
        finally:
            self._set_mux(prev)

    def _set_mux(self, src: str):
        # local helper so you can call from code without ros2 param set
        if src not in ('live', 'replay', 'off'):
            self.get_logger().warn(f'ignored invalid mux source "{src}"')
            return
        self.mux_source = src
        self.get_logger().info(f'[manager] mux.source -> {src}')

def main():
    rclpy.init()
    node = Manager()
    try:
        # Reentrant + multithreaded so mux keeps running while train service executes
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
