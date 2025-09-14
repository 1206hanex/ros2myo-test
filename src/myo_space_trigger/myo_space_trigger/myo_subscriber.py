import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

import pyautogui


class MyoSubscriber(Node):

    def __init__(self):
        super().__init__('myo_subscriber')
        self.subscription = self.create_subscription(
            Bool,
            'grasp_detected',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info("Myo subscriber started.")

    def listener_callback(self, msg):
        if msg.data:
            self.get_logger().info("Grasp detected")
            pyautogui.press('space')


def main(args=None):
    rclpy.init(args=args)
    node = MyoSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
