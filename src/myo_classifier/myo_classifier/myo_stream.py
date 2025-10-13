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
        self.declare_parameter('device_id', "EC:6D:69:B3:F8:73")
        self.declare_parameter('sampling_rate', 200.0)
        self.declare_parameter('emg_channels', ['emg_0','emg_1','emg_2','emg_3','emg_4','emg_5','emg_6','emg_7'])
        self.declare_parameter('imu_channels', ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 
                                                'orient_x', 'orient_y', 'orient_z', 'orient_w'])
        self.declare_parameter('uuid_control', 'd5060401-a904-deb9-4748-2c7f4a124842')
        self.declare_parameter('uuid_emg_streams', ['d5060105-a904-deb9-4748-2c7f4a124842',
                                                    'd5060405-a904-deb9-4748-2c7f4a124842',])
        self.declare_parameter('uuid_imu', 'd5060402-a904-deb9-4748-2c7f4a124842')
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
        
        self._last_emg_frame = None
        self._last_emg_time = None

        # Async BLE task
        #self._stop = False
        self.get_logger().info('MyoEMGPublisher started')
        self.get_logger().info(f'MAC Address: {self.device_id}')
        
        self.ble_task = asyncio.get_event_loop()
        self.ble_task.create_task(self.connect_myo())
          
    async def connect_myo(self):
        while True:
            try:
                # attempt BLE connection
                async with BleakClient(self.device_id) as client:
                    self.get_logger().info(f"Connected to Myo: {client.address}")

                    # send command to start streaming
                    await client.write_gatt_char(self.UUID_CONTROL, self.ENABLE_CMD)
                    for uuid in self.UUID_EMG_STREAMS:
                        await client.start_notify(uuid, self.emg_callback)
                        self.get_logger().info(f'EMG notify successful!')
                        
                    if self.UUID_IMU:
                        try:
                            await client.start_notify(self.UUID_IMU, self.imu_callback)
                            self.get_logger().info(f'IMU notify successful!')
                        except Exception as e:
                            self.get_logger().warn(f'IMU notify failed (continuing EMG-only): {e}')

                    # loop while connection is active
                    while client.is_connected:
                        await asyncio.sleep(0.1)

                    self.get_logger().warn("Disconnected from Myo. Reconnecting...")
                    
            except Exception as e:
                self.get_logger().error(f"Connection error: {e}")
                await asyncio.sleep(5)

    def emg_callback(self, _sender, data: bytes):
        # 16 bytes → two 8-ch frames; publish each as a MyoMsg
        if len(data) != 16:
            return
        vals = struct.unpack('16b', data)
        for offset in (0, 8):
            frame = [float(v) for v in vals[offset:offset+8]]
            
            # remember latest EMG for IMU merge
            self._last_emg_frame = frame
            self._last_emg_time  = self.get_clock().now()
            
            msg = MyoMsg()
            msg.stamp = self.get_clock().now().to_msg()
            msg.device_id = self.device_id
            msg.sampling_rate = self.sampling_rate
            msg.emg_data = frame
            msg.emg_channels = self.emg_channels
            # leave IMU empty unless imu_callback fills via another publisher pattern
            msg.imu_data = []
            msg.imu_channels = self.imu_channels
            self.pub.publish(msg)

    def imu_callback(self, _sender, data: bytes):
        # Myo IMU packet is 20 bytes = 10 x int16 (little-endian):
        # [quat_x, quat_y, quat_z, quat_w, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        if len(data) != 20:
            return

        # Unpack and scale
        # Community-known scales (approx):
        #   quaternion: /16384.0 → ~unit quaternion
        #   accel: /2048.0 → g
        #   gyro: /16.0 → deg/s
        q_scale = 16384.0
        a_scale = 2048.0
        g_scale = 16.0

        qx, qy, qz, qw, ax, ay, az, gx, gy, gz = struct.unpack('<10h', data)

        qx_f = float(qx) / q_scale
        qy_f = float(qy) / q_scale
        qz_f = float(qz) / q_scale
        qw_f = float(qw) / q_scale

        ax_f = float(ax) / a_scale
        ay_f = float(ay) / a_scale
        az_f = float(az) / a_scale

        gx_f = float(gx) / g_scale
        gy_f = float(gy) / g_scale
        gz_f = float(gz) / g_scale

        # Order to match your imu_channels parameter:
        # ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','orient_x','orient_y','orient_z','orient_w']
        imu_vec = [ax_f, ay_f, az_f, gx_f, gy_f, gz_f, qx_f, qy_f, qz_f, qw_f]

        # Try to merge with the most recent EMG frame if it's fresh
        # (EMG at 200 Hz → 5 ms; allow up to ~50 ms tolerance)
        merged_emg = []
        if self._last_emg_frame is not None and self._last_emg_time is not None:
            now = self.get_clock().now()
            age = (now.nanoseconds - self._last_emg_time.nanoseconds) / 1e9
            if age <= 0.05:
                merged_emg = list(self._last_emg_frame)

        msg = MyoMsg()
        msg.stamp = self.get_clock().now().to_msg()
        msg.device_id = self.device_id
        msg.sampling_rate = self.sampling_rate

        # Fill whichever we have
        msg.emg_data = merged_emg
        msg.emg_channels = self.emg_channels
        msg.imu_data = imu_vec
        msg.imu_channels = self.imu_channels

        self.pub.publish(msg)

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