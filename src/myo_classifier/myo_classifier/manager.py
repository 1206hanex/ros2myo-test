#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from myo_msgs.srv import AddGesture, TrainModel

class Manager(Node):
    def __init__(self):
        super().__init__('manager')

        # Reentrant group for services
        self.cb_srv = ReentrantCallbackGroup()

        # Manager’s AddGesture — forwards to recorder
        self.srv_add = self.create_service(
            AddGesture,
            'myo_classifier/add_gesture',
            self._srv_add_gesture,
            callback_group=self.cb_srv
        )

        # TrainModel service (stub/your impl)
        self.srv_train = self.create_service(
            TrainModel,
            'myo_classifier/train_model',
            self._srv_train_model,
            callback_group=self.cb_srv
        )

        # Client to the recorder service — make sure the name matches the recorder node!
        self.recorder_cli = self.create_client(AddGesture, '/myo_classifier/recorder_add_gesture')

        self.get_logger().info('Manager ready (forwarding to recorder)')

    def _srv_add_gesture(self, req, resp):
        if not self.recorder_cli.wait_for_service(timeout_sec=1.0):
            resp.success = False
            resp.message = "Recorder not available at /myo_classifier/recorder_add_gesture."
            resp.csv_path = ""
            return resp

        fwd = AddGesture.Request()
        fwd.gesture       = req.gesture
        fwd.seconds       = req.seconds
        fwd.cycles        = req.cycles
        fwd.window_size   = req.window_size
        fwd.raw_mode      = req.raw_mode
        fwd.out_dir       = req.out_dir
        fwd.countdown_sec = req.countdown_sec

        fut = self.recorder_cli.call_async(fwd)

        max_wait = max(5.0, float((req.seconds or 5) * (req.cycles or 1) + (req.countdown_sec or 0) + 5))
        rclpy.spin_until_future_complete(self, fut, timeout_sec=max_wait)

        res = fut.result()
        if res is None:
            resp.success = False
            resp.message = "Recorder call failed or timed out."
            resp.csv_path = ""
            return resp

        resp.success = res.success
        resp.message = res.message
        resp.csv_path = res.csv_path
        return resp

    def _srv_train_model(self, req, resp):
        # implement training reading CSVs from req or your cfg defaults
        return resp

def main():
    rclpy.init()
    node = Manager()
    try:
        exec = MultiThreadedExecutor(num_threads=2)
        exec.add_node(node)
        exec.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
