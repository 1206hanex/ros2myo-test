# Timestamp of this sample
builtin_interfaces/Time stamp

# EMG data: flat array of floats
float32[] emg_data
# Corresponding channel names
string[]   emg_channels

# IMU data: flat array of floats
float32[] imu_data
# Corresponding IMU labels
string[]   imu_channels

# Acquisition metadata
float32    sampling_rate
string     device_id