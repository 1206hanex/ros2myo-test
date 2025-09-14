import rclpy
from rclpy.node import Node

import asyncio
import struct
import numpy as np
import time
from collections import deque
from bleak import BleakClient
import pyautogui

# ROS2 message type (for grasp detection topic if needed)
from std_msgs.msg import Bool

# ===== MYO CONSTANTS =====
MYO_MAC = "EC:6D:69:B3:F8:73"  # Replace with your Myo MAC

UUID_CONTROL = "d5060401-a904-deb9-4748-2c7f4a124842"  # Enable streaming
UUID_EMG_DATA = "d5060105-a904-deb9-4748-2c7f4a124842"  # EMG notify

ENABLE_STREAMING_CMD = b'\x01\x03\x02\x01\x01\x00\x00'

# ===== EMG Processing =====
window_size = 5
emg_buffer = deque([0] * window_size, maxlen=window_size)
rms_threshold = 10.0
debounce_time = 0.5
last_keypress_time = 0


class MyoNode(Node):
    def __init__(self):
        super().__init__('myo_listener')
        self.get_logger().info("Myo Listener Node Started")

        # Publisher (optional, for use by other nodes)
        self.publisher_ = self.create_publisher(Bool, 'grasp_detected', 10)

        # Start async loop in background
        loop = asyncio.get_event_loop()
        loop.create_task(self.connect_myo())

    async def process_emg_data(self, sender, data):
        """Callback to process EMG data, compute RMS, and classify grasp with debounce."""
        global emg_buffer, rms_threshold, last_keypress_time

        if len(data) == 16:
            emg_values = struct.unpack('16b', data)
            channel_1_frame_1 = emg_values[0]
            channel_1_frame_2 = emg_values[8]

            emg_buffer.append(channel_1_frame_1)
            emg_buffer.append(channel_1_frame_2)

            rms_value = np.sqrt(np.mean(np.square(emg_buffer)))

            if rms_threshold is not None:
                if rms_value > rms_threshold:
                    current_time = time.time()
                    if current_time - last_keypress_time > debounce_time:
                        self.get_logger().info(f"Power Grasp Detected! (RMS: {rms_value:.2f}) - Pressing Space")
                        pyautogui.press('space')
                        self.publisher_.publish(Bool(data=True))
                        last_keypress_time = current_time
                else:
                    self.get_logger().info(f"Rest (RMS: {rms_value:.2f})")

    async def calibrate_threshold(self, client):
        """Prompt user to close fist and record grasp RMS value."""
        global rms_threshold

        input("Open your hand and relax. Press Enter when ready to calibrate...")
        print("Now, close your fist tightly for 5 seconds...")
        time.sleep(2)

        temp_buffer = []
        start_time = time.time()

        while time.time() - start_time < 5:
            temp_buffer.extend(list(emg_buffer))
            await asyncio.sleep(0.1)

        rms_threshold = np.sqrt(np.mean(np.square(temp_buffer))) * 0.95
        print(f"Calibration complete! Threshold set at RMS: {rms_threshold:.2f}")
        print("Running real-time classification...")

    async def connect_myo(self):
        while True:
            try:
                async with BleakClient(MYO_MAC) as client:
                    self.get_logger().info(f"Connected to Myo: {client.address}")

                    await client.write_gatt_char(UUID_CONTROL, ENABLE_STREAMING_CMD)
                    await client.start_notify(UUID_EMG_DATA, self.process_emg_data)

                    # Optional: calibrate threshold
                    # await self.calibrate_threshold(client)

                    while await client.is_connected:
                        await asyncio.sleep(0.1)

                    self.get_logger().warn("Lost connection. Reconnecting...")

            except Exception as e:
                self.get_logger().error(f"Connection error: {e}")
                await asyncio.sleep(5)


def main(args=None):
    rclpy.init(args=args)
    node = MyoNode()
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
