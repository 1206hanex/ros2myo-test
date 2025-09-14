import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import asyncio
import struct
import time
from bleak import BleakClient

# Myo MAC and UUIDs
MYO_MAC = "EC:6D:69:B3:F8:73"
UUID_CONTROL = "d5060401-a904-deb9-4748-2c7f4a124842"
UUID_EMG_STREAMS = [
    "d5060105-a904-deb9-4748-2c7f4a124842",
    "d5060405-a904-deb9-4748-2c7f4a124842",
]
ENABLE_STREAMING_CMD = b'\x01\x03\x02\x01\x01\x00\x00'


class MyoEMGPublisher(Node):
    def __init__(self):
        super().__init__('myo_stream')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'emg_raw', 10)
        self.get_logger().info("Myo EMG Publisher Node Started")

        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.connect_myo())

    async def connect_myo(self):
        while True:
            try:
                # attempt BLE connection
                async with BleakClient(MYO_MAC) as client:
                    self.get_logger().info(f"Connected to Myo: {client.address}")

                    # send command to start streaming
                    await client.write_gatt_char(UUID_CONTROL, ENABLE_STREAMING_CMD)
                    for uuid in UUID_EMG_STREAMS:
                        await client.start_notify(uuid, self.emg_callback)

                    self.get_logger().info("EMG streaming started. Publishing to /emg_raw")

                    # loop while connection is active
                    while client.is_connected:
                        await asyncio.sleep(0.1)

                    self.get_logger().warn("Disconnected from Myo. Reconnecting...")

            except Exception as e:
                self.get_logger().error(f"Connection error: {e}")
                await asyncio.sleep(5)

    def emg_callback(self, sender, data):
        # only process if there are exactly 16 bytes (16 channels)
        if len(data) == 16:
            # unpack into 16 signed integers
            emg_values = struct.unpack('16b', data)
            # read from the first 8 channels
            frame_1 = emg_values[:8]

            # prepare the ROS message to publish to /emg_raw
            msg = Float32MultiArray()
            msg.data = msg.data = [float(v) for v in frame_1] # convert the ints into floats
            self.publisher_.publish(msg) # publish the EMG frame to /emg_raw

            self.get_logger().debug(f"Published EMG: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = MyoEMGPublisher()
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
