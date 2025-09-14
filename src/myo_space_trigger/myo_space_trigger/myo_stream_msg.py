import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from myo_space_trigger.msg import MyoMsg

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
        super().__init__('myo_stream_msg')
        
        # declare parameters (set defaults or via .yaml config file)
        self.declare_parameter('device_id', '')
        self.declare_parameter('sampling_rate', 200.0)
        self.declare_parameter('emg_channels', [])
        self.declare_parameter('imu_channels', [])
        self.declare_parameter('uuid_control', [])
        self.declare_parameter('uuid_emg_streams', [])
        self.declare_parameter('enable_cmd', [])
        
        # read parameters
        self.device_id = self.get_parameter('device_id').value
        self.sampling_rate = self.get_parameter('sampling_rate').value
        self.emg_channels = self.get_parameter('emg_channels').value
        self.imu_channels = self.get_parameter('imu_channels').value
        self.UUID_CONTROL = self.get_parameter('uuid_control').value
        self.UUID_EMG_STREAMS = self.get_parameter('uuid_emg_streams').value
        cmd_list = self.get_parameter('enable_cmd').value
        
        self.ENABLE_STREAMING_CMD = bytes(cmd_list)
        
        
        self.publisher_ = self.create_publisher(MyoMsg, 'myo/data', 10)
        self.get_logger().info("Myo EMG Publisher Node Started")

        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.connect_myo())

    async def connect_myo(self):
        while True:
            try:
                # attempt BLE connection
                async with BleakClient(self.device_id) as client:
                    self.get_logger().info(f"Connected to Myo: {client.address}")

                    # send command to start streaming
                    await client.write_gatt_char(self.UUID_CONTROL, self.ENABLE_STREAMING_CMD)
                    for uuid in self.UUID_EMG_STREAMS:
                        await client.start_notify(uuid, self.emg_callback)

                    self.get_logger().info("EMG streaming started.")

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
            emg_values = struct.unpack('16b', data)[:8]
            emg_floats = [float(v) for v in emg_values]
            
            msg = MyoMsg()
            msg.stamp = self.get_clock().now().to_msg()
            msg.device_id = self.device_id
            msg.sampling_rate = float(self.sampling_rate)
            
            msg.emg_data = emg_floats
            msg.emg_channels = self.emg_channels
            
            # IMU fields
            msg.imu_data = []
            msg.imu_channels = self.imu_channels
            
            self.publisher_.publish(msg) # publish the EMG frame to topic
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
