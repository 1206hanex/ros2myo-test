# myo_classifier/common/fe.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple

EMG_CHANNELS = [f"emg_{i}" for i in range(8)]
IMU_CHANNELS = ["quat_w","quat_x","quat_y","quat_z","accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"]

def scale_myo_imu(raw10: Tuple[int, ...]) -> Tuple[float, ...]:
    """
    Myo IMU: 10 * int16 => quat (w,x,y,z), accel (x,y,z), gyro (x,y,z)
    Scale to natural units.
    """
    qw, qx, qy, qz, ax, ay, az, gx, gy, gz = raw10
    qw = qw / 16384.0
    qx = qx / 16384.0
    qy = qy / 16384.0
    qz = qz / 16384.0
    ax = ax / 2048.0
    ay = ay / 2048.0
    az = az / 2048.0
    gx = gx / 16.0
    gy = gy / 16.0
    gz = gz / 16.0
    return (qw, qx, qy, qz, ax, ay, az, gx, gy, gz)

def extract_features(matrix_np: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
    """
    For a (T, C) window: RMS, Mean, Var for each channel (length C each).
    """
    rms  = np.sqrt(np.mean(np.square(matrix_np), axis=0)).tolist()
    mean = np.mean(matrix_np, axis=0).tolist()
    var  = np.var(matrix_np, axis=0).tolist()
    return rms, mean, var

def feature_columns(emg_ch: int = 8, include_imu: bool = False) -> List[str]:
    emg_cols = [f"emg_{i}" for i in range(emg_ch)]
    imu_cols = IMU_CHANNELS if include_imu else []
    cols = (
        [f"{c}_rms"  for c in emg_cols] + [f"{c}_rms"  for c in imu_cols] +
        [f"{c}_mean" for c in emg_cols] + [f"{c}_mean" for c in imu_cols] +
        [f"{c}_var"  for c in emg_cols] + [f"{c}_var"  for c in imu_cols] +
        ["label"]
    )
    return cols

def make_feature_row(
    emg_window: np.ndarray,
    imu_window: np.ndarray | None,
    label: str | int,
) -> List[float]:
    """
    Build one feature row:
      [emg_rms..., imu_rms..., emg_mean..., imu_mean..., emg_var..., imu_var..., label]
    If imu_window is None, omit IMU portions.
    """
    emg_rms, emg_mean, emg_var = extract_features(emg_window.astype(np.float32))
    if imu_window is not None:
        imu_rms, imu_mean, imu_var = extract_features(imu_window.astype(np.float32))
        return emg_rms + imu_rms + emg_mean + imu_mean + emg_var + imu_var + [label]
    else:
        return emg_rms + emg_mean + emg_var + [label]
