#!/usr/bin/env python3
import asyncio
import os
import csv
import time
import struct
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

import numpy as np
from bleak import BleakClient

# ======== USER SETUP ========
SUBJECT_ID = "S05"
BASE_DIR = "data"                     # root data folder
RUN_COUNT = 5                         # k-fold by runs
REPS_PER_GESTURE = 3                  # 3 reps per gesture per run
SAMPLE_RATE_HZ = 200                  # nominal Myo EMG rate

# Gesture map & order (you can change the block order per run if you like)
GESTURES = {
    "G0": "Fist",
    "G1": "Open hand",
    "G2": "Wrist left",
    "G3": "Wrist right",
}
# Example blocked order for run 1; rotate for other runs if desired
BLOCK_ORDER_BY_RUN = {
    1: ["G2", "G0", "G3", "G1"],
    2: ["G0", "G1", "G2", "G3"],
    3: ["G1", "G2", "G3", "G0"],
    4: ["G3", "G1", "G0", "G2"],
    5: ["G0", "G2", "G1", "G3"],
}

# Phase timing (seconds) per repetition
PHASE_IDLE = 3.0
PHASE_RAMP = 0.5
PHASE_HOLD = 5.0
PHASE_RELAX = 3.0
REP_TOTAL = PHASE_IDLE + PHASE_RAMP + PHASE_HOLD + PHASE_RELAX  # 11.5s

# ======== MYO CONFIG (same as your script) ========
MYO_MAC = "EC:6D:69:B3:F8:73"

UUID_CONTROL   = "d5060401-a904-deb9-4748-2c7f4a124842"  # enable streaming cmd
UUID_IMU_DATA  = "d5060402-a904-deb9-4748-2c7f4a124842"  # IMU notify
UUID_EMG_DATA  = "d5060105-a904-deb9-4748-2c7f4a124842"  # EMG notify (one of them)
UUID_EMG_STREAMS = [
    "d5060105-a904-deb9-4748-2c7f4a124842",
    "d5060405-a904-deb9-4748-2c7f4a124842",
]
ENABLE_STREAMING_CMD = b'\x01\x03\x02\x01\x01\x00\x00'

# ======== GLOBAL RECORDING STATE ========
current_trial_rows: Optional[List[List]] = None  # in-memory rows for the active trial CSV
current_meta: Dict[str, str] = {}
current_phase: str = "idle"
recording_active: bool = False

latest_imu: Optional[Tuple[float, ...]] = None  # (ax,ay,az,gx,gy,gz, qw,qx,qy,qz) when available

# ======== DATA STRUCTURES ========
RAW_HEADER = [
    "timestamp_ns",
    "subject_id", "run_idx", "gesture_id", "gesture_name", "rep_idx", "phase",
    "emg_0","emg_1","emg_2","emg_3","emg_4","emg_5","emg_6","emg_7",
    "accel_x","accel_y","accel_z",
    "gyro_x","gyro_y","gyro_z",
    "orient_w","orient_x","orient_y","orient_z",
]

MANIFEST_HEADER = [
    "subject_id","run_idx","trial_idx",
    "gesture_id","gesture_name","rep_idx",
    "raw_path","samples_hz","notes"
]

MIN_SAMPLES_FRACTION = 0.40   # accept >= 40% of nominal length
NOMINAL_SAMPLES = int(REP_TOTAL * SAMPLE_RATE_HZ)   # e.g., 11.5 * 200 = 2300
MIN_SAMPLES = int(NOMINAL_SAMPLES * MIN_SAMPLES_FRACTION)  # ~920

@dataclass
class ManifestRow:
    subject_id: str
    run_idx: int
    trial_idx: int
    gesture_id: str
    gesture_name: str
    rep_idx: int
    raw_path: str
    samples_hz: int
    notes: str = ""

# ======== HELPERS ========
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def phase_from_elapsed(elapsed: float) -> str:
    if elapsed < PHASE_IDLE:
        return "idle"
    elif elapsed < PHASE_IDLE + PHASE_RAMP:
        return "ramp"
    elif elapsed < PHASE_IDLE + PHASE_RAMP + PHASE_HOLD:
        return "hold"
    elif elapsed < REP_TOTAL:
        return "relax"
    else:
        return "done"

# ======== BLEAK CALLBACKS ========
def on_emg_notify(_sender: str, data: bytes):
    """
    Myo EMG packet (16 bytes typical per stream): 8 signed bytes (emg) + 8 signed bytes (second frame).
    In many setups, using the first 8 channels per notification is sufficient.
    """
    global current_trial_rows, recording_active, current_phase, latest_imu, current_meta
    if not recording_active or current_trial_rows is None:
        return

    if len(data) != 16:
        return

    # 16 signed bytes = two frames of 8 channels
    emg_vals = struct.unpack("16b", data)
    frame1 = emg_vals[:8]
    frame2 = emg_vals[8:]

    ts1 = time.time_ns()
    # approximate 1/sample_rate spacing for the second frame
    delta_ns = int(1e9 / SAMPLE_RATE_HZ)
    ts2 = ts1 + delta_ns

    if latest_imu is not None:
        ax, ay, az, gx, gy, gz, qw, qx, qy, qz = latest_imu
    else:
        ax = ay = az = gx = gy = gz = qw = qx = qy = qz = ""

    meta = (
        current_meta.get("subject_id",""),
        current_meta.get("run_idx",""),
        current_meta.get("gesture_id",""),
        current_meta.get("gesture_name",""),
        current_meta.get("rep_idx",""),
        current_phase,
    )

    # row(ts, meta..., emg8, imu10)
    current_trial_rows.append([ts1, *meta, *frame1, ax, ay, az, gx, gy, gz, qw, qx, qy, qz])
    current_trial_rows.append([ts2, *meta, *frame2, ax, ay, az, gx, gy, gz, qw, qx, qy, qz])

def on_imu_notify(_sender: str, data: bytes):
    """
    Myo IMU formats vary. Two common forms:
      - 20 bytes: 10 shorts (10h) (scaled accel/gyro/orient)
      - 18/20/other with a trailing status byte
    We'll try to parse 10h and then scale to floats lightly or just keep raw ints.
    """
    global latest_imu
    try:
        if len(data) == 20:
            vals = struct.unpack("10h", data)  # 10 short ints
            # Heuristic mapping: (ax, ay, az, gx, gy, gz, qw, qx, qy, qz) raw
            ax, ay, az, gx, gy, gz, qw, qx, qy, qz = vals
            latest_imu = (ax, ay, az, gx, gy, gz, qw, qx, qy, qz)
        elif len(data) == 18 or len(data) == 17:
            # Some firmwares pack differently; ignore or extend here if needed
            return
        else:
            return
    except Exception:
        # Silently ignore parse errors
        return

# ======== RECORDING CORE ========
async def record_single_repetition(gesture_id: str, rep_idx: int, run_idx: int, run_dir: str) -> str:
    """
    Records one repetition's raw stream into a CSV file.
    Returns the relative file path written (for the run manifest).
    """
    global current_trial_rows, recording_active, current_phase, current_meta

    gesture_name = GESTURES[gesture_id]
    filename = f"{gesture_id}_rep{rep_idx:02d}.csv"
    raw_rel_path = os.path.join(BASE_DIR, SUBJECT_ID, f"run{run_idx:02d}", filename)
    raw_abs_path = os.path.join(run_dir, filename)

    # Prepare in-memory buffer & meta
    current_trial_rows = []
    current_phase = "idle"
    current_meta = {
        "subject_id": SUBJECT_ID,
        "run_idx": str(run_idx),
        "gesture_id": gesture_id,
        "gesture_name": gesture_name,
        "rep_idx": str(rep_idx),
    }

    # Countdown/prompt (enter to start)
    input(f"\nRun {run_idx} | {gesture_name} (#{rep_idx})\n"
          f"Press ENTER when ready. Timing: idle {PHASE_IDLE}s → ramp {PHASE_RAMP}s → "
          f"hold {PHASE_HOLD}s → relax {PHASE_RELAX}s.")

    # Start timed phases
    start = time.monotonic()
    recording_active = True

    while True:
        elapsed = time.monotonic() - start
        ph = phase_from_elapsed(elapsed)
        if ph == "done":
            break
        current_phase = ph
        # small sleep to yield (callbacks are driving the sampling)
        await asyncio.sleep(0.01)

    recording_active = False

    # check if band is disconnected
    sample_count = len(current_trial_rows)
    if sample_count < MIN_SAMPLES:
        print(f"✗ Too few samples ({sample_count} < {MIN_SAMPLES}). Likely a drop/disconnect.")
        print("Retrying the SAME repetition; the file will be overwritten.\n")
        # reset state and retry recursively
        current_trial_rows = None
        current_meta = {}
        current_phase = "idle"
        return await record_single_repetition(gesture_id, rep_idx, run_idx, run_dir)

    # Write CSV to disk
    with open(raw_abs_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RAW_HEADER)
        w.writerows(current_trial_rows)

    print(f"Saved trial → {raw_rel_path}  ({len(current_trial_rows)} samples)")

    # Cleanup
    current_trial_rows = None
    current_meta = {}
    current_phase = "idle"

    return raw_rel_path

async def record_single_run(client: BleakClient, run_idx: int) -> None:
    """
    Records 20 trials = 4 gestures × 5 reps, in blocked order for this run.
    Writes: raw trial CSV files and a run-level manifest CSV with 20 rows.
    """
    subject_dir = os.path.join(BASE_DIR, SUBJECT_ID)
    run_dir = os.path.join(subject_dir, f"run{run_idx:02d}")
    ensure_dir(run_dir)

    order = BLOCK_ORDER_BY_RUN.get(run_idx, list(GESTURES.keys()))
    print(f"\n=== RUN {run_idx} | Blocked order: {order} ===")

    manifest_rows: List[ManifestRow] = []
    trial_idx = 1

    for g in order:
        for rep in range(1, REPS_PER_GESTURE + 1):
            rel_path = os.path.join(BASE_DIR, SUBJECT_ID, f"run{run_idx:02d}", f"{g}_rep{rep:02d}.csv")
            abs_path = os.path.join(run_dir, f"{g}_rep{rep:02d}.csv")

            # If you prefer to skip re-recording existing files during a retry:
            # if os.path.exists(abs_path):
            #     print(f"Skipping existing {rel_path}")
            #     # still add to manifest:
            #     manifest_rows.append(ManifestRow(SUBJECT_ID, run_idx, trial_idx, g, GESTURES[g], rep, rel_path, SAMPLE_RATE_HZ))
            #     trial_idx += 1
            #     continue

            # Record this repetition
            rel_path_written = await record_single_repetition(g, rep, run_idx, run_dir)

            manifest_rows.append(
                ManifestRow(
                    subject_id=SUBJECT_ID,
                    run_idx=run_idx,
                    trial_idx=trial_idx,
                    gesture_id=g,
                    gesture_name=GESTURES[g],
                    rep_idx=rep,
                    raw_path=rel_path_written,
                    samples_hz=SAMPLE_RATE_HZ,
                    notes=""
                )
            )
            trial_idx += 1

    # Write run-level manifest
    manifest_path = os.path.join(run_dir, f"trials_manifest_run{run_idx:02d}.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_HEADER)
        for row in manifest_rows:
            w.writerow([
                row.subject_id, row.run_idx, row.trial_idx,
                row.gesture_id, row.gesture_name, row.rep_idx,
                row.raw_path, row.samples_hz, row.notes
            ])

    print(f"\nRun {run_idx} complete. Manifest written → {manifest_path}")

# ======== MAIN BLEAK FLOW ========
async def connect_and_collect():
    async with BleakClient(MYO_MAC) as client:
        print(f"Connected to Myo: {client.address}")

        # Enable streaming
        await client.write_gatt_char(UUID_CONTROL, ENABLE_STREAMING_CMD)
        print("Streaming enabled")

        # Subscriptions
        # For EMG, subscribe to ONE stream to avoid duplicate/burst rows
        await client.start_notify(UUID_EMG_STREAMS[0], on_emg_notify)
        await client.start_notify(UUID_EMG_STREAMS[1], on_emg_notify)
        # If you find you need the second stream for stability, you can also subscribe to it.
        # await client.start_notify(UUID_EMG_STREAMS[1], on_emg_notify)

        await client.start_notify(UUID_IMU_DATA, on_imu_notify)

        # Per-run collection
        for run_idx in range(1, RUN_COUNT + 1):
            await record_single_run(client, run_idx)
            print("\nTake a rest. Press ENTER to proceed to the next run.")
            input()

        # Cleanup
        await client.stop_notify(UUID_EMG_STREAMS[0])
        # await client.stop_notify(UUID_EMG_STREAMS[1])
        await client.stop_notify(UUID_IMU_DATA)

        print("\nAll runs completed. Data saved under:", os.path.join(BASE_DIR, SUBJECT_ID))

if __name__ == "__main__":
    try:
        asyncio.run(connect_and_collect())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
