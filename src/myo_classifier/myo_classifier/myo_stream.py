import asyncio
import struct

import rclpy
from rclpy.node import Node
from bleak import BleakClient

from myo_msgs.msg import MyoMsg
# from rclpy.parameter import ParameterDescriptor, ParameterType

class MyoEMGPublisher(Node):
    def __init__(self):
        super().__init__('myo_stream')

        # Parameters - added default values (Myo Armband)
        self.declare_parameter('device_id', 'EC:6D:69:B3:F8:73')
        self.declare_parameter('sampling_rate', 200.0)
        self.declare_parameter('emg_channels', ['emg_0','emg_1','emg_2','emg_3','emg_4','emg_5','emg_6','emg_7'])
        self.declare_parameter('imu_channels', ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 
                                                'orient_x', 'orient_y', 'orient_z', 'orient_w'])
        self.declare_parameter('uuid_control', 'd5060401-a904-deb9-4748-2c7f4a124842')
        self.declare_parameter('uuid_emg_streams', ['d5060105-a904-deb9-4748-2c7f4a124842',
                                                    'd5060405-a904-deb9-4748-2c7f4a124842',])
        self.declare_parameter('uuid_imu', 'd5060104-a904-deb9-4748-2c7f4a124842')
        self.declare_parameter('enable_cmd', [1, 3, 2, 1, 1, 0, 0])

        self.device_id        = self.get_parameter('device_id').value
        self.sampling_rate    = float(self.get_parameter('sampling_rate').value)
        self.emg_channels     = list(self.get_parameter('emg_channels').value)
        self.imu_channels     = list(self.get_parameter('imu_channels').value)
        self.UUID_CONTROL     = self.get_parameter('uuid_control').value
        self.UUID_EMG_STREAMS = list(self.get_parameter('uuid_emg_streams').value)
        self.UUID_IMU         = self.get_parameter('uuid_imu').value
        cmd_list              = list(self.get_parameter('enable_cmd').value)
        self.ENABLE_CMD       = bytes((int(x) & 0xFF) for x in cmd_list)

        if not self.device_id:
            self.get_logger().error('device_id parameter is required')
        if not self.UUID_CONTROL or not self.UUID_EMG_STREAMS:
            self.get_logger().error('UUIDs not set; check config')

        self.pub = self.create_publisher(MyoMsg, 'myo/data', 10)

        # Async BLE task
        #self._stop = False
        self.get_logger().info('MyoEMGPublisher started')
        self.get_logger().info(f'MAC Address: {self.device_id}')
        
        self.ble_task = asyncio.get_event_loop()
        self.ble_task.create_task(self._connect_loop())

    async def _connect_loop(self):
        while True:
            try:
                async with BleakClient(self.device_id) as client:
                    self.get_logger().info(f'Connected to Myo: {client.address}')
                    await client.write_gatt_char(self.UUID_CONTROL, self.ENABLE_CMD)

                    for uuid in self.UUID_EMG_STREAMS:
                        await client.start_notify(uuid, self._emg_cb)

                    if self.UUID_IMU:
                        try:
                            await client.start_notify(self.UUID_IMU, self._imu_cb)
                        except Exception as e:
                            self.get_logger().warn(f'IMU notify failed (continuing EMG-only): {e}')

                    while client.is_connected:
                        await asyncio.sleep(0.1)

                    self.get_logger().warn('Disconnected; retrying...')
            except Exception as e:
                self.get_logger().error(f'Connection error: {e}')
                await asyncio.sleep(5)

    def _emg_cb(self, _sender, data: bytes):
        # 16 bytes → two 8-ch frames; publish each as a MyoMsg
        if len(data) != 16:
            return
        vals = struct.unpack('16b', data)
        for offset in (0, 8):
            frame = [float(v) for v in vals[offset:offset+8]]
            msg = MyoMsg()
            msg.stamp = self.get_clock().now().to_msg()
            msg.device_id = self.device_id
            msg.sampling_rate = self.sampling_rate
            msg.emg_data = frame
            msg.emg_channels = self.emg_channels
            # leave IMU empty unless _imu_cb fills via another publisher pattern
            msg.imu_data = []
            msg.imu_channels = self.imu_channels
            self.pub.publish(msg)

    def _imu_cb(self, _sender, data: bytes):
        # Optional: if you want to publish combined EMG+IMU in one message,
        # maintain a small buffer and merge with latest EMG. For simplicity,
        # this example omits IMU packing. You can extend when ready.
        pass

    def destroy_node(self):
        self._stop = True
        try:
            self.ble_task.cancel()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MyoEMGPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()