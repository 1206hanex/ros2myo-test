import asyncio
import struct
import time
import numpy as np
import pandas as pd
import pickle
import pyautogui
from collections import deque
from bleak import BleakClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Myo Bluetooth Address (Replace with your actual MAC)
MYO_MAC = "EC:6D:69:B3:F8:73"

# Myo UUIDs
UUID_CONTROL = "d5060401-a904-deb9-4748-2c7f4a124842"  # Enable streaming
UUID_EMG_DATA = "d5060105-a904-deb9-4748-2c7f4a124842"  # EMG notify
UUID_IMU_DATA = "d5060104-a904-deb9-4748-2c7f4a124842"  # IMU notify
UUID_EMG_STREAMS = [
    "d5060105-a904-deb9-4748-2c7f4a124842",
    "d5060405-a904-deb9-4748-2c7f4a124842",
]  # 2 EMG streams
# Commands to enable streaming
ENABLE_STREAMING_CMD = b'\x01\x03\x02\x01\x01\x00\x00'

# Data collection parameters
window_size = 20  # 200 samples per window
# gesture_labels = ["left", "right", "up", "down", "rest"]
GESTURE_TIME = 10

gesture_labels = {
    0: "rest",
    1: "pinch",
    2: "power",
    3: "extension",
    4: "left",
    5: "right",
}

training_cycles = 2
testing_cycles = 1

# Buffers for EMG and IMU data
emg_buffer = deque(maxlen=window_size)
imu_buffer = deque(maxlen=window_size)

# Data storage
training_data = []
testing_data = []

async def process_emg_data(sender, data):
    """Collect EMG data from Myo."""
    if len(data) == 16:
        emg_values = struct.unpack('16b', data)
        emg_buffer.append(emg_values[:8])  # First 8 values = EMG Frame 1

async def process_imu_data(sender, data):
    """Process and store IMU data dynamically based on received bytes."""
    if len(data) == 17:
        imu_values = struct.unpack('8hB', data)  # 8 shorts (16 bytes) + 1 extra byte
    elif len(data) == 20:
        imu_values = struct.unpack('10h', data)  # Standard 10 shorts (20 bytes)
    else:
        return

    imu_buffer.append(imu_values[:10])  # Store IMU data (only first 10 values)

async def record_gesture(client, gesture, cycles, is_training=True):
    """Record data for a specific gesture."""
    global training_data, testing_data

    for cycle in range(cycles):
        input(f"Perform **{gesture_labels[gesture].upper()}** for {GESTURE_TIME} sec. Press Enter when ready...")
        print(f"Recording {gesture_labels[gesture].upper()}...")

        collected_samples = []
        start_time = time.time()

        while time.time() - start_time < GESTURE_TIME:
            if len(emg_buffer) == window_size:# and len(imu_buffer) == window_size:
                emg_features = extract_features(np.array(emg_buffer))
                # imu_features = extract_features(np.array(imu_buffer))
                # collected_samples.append(emg_features + imu_features + [gesture])
                collected_samples.append(emg_features + [gesture])

            await asyncio.sleep(0.2)  # Allow proper data collection

        # Save data
        if is_training:
            training_data.extend(collected_samples)
        else:
            testing_data.extend(collected_samples)

        print(f"Completed {gesture_labels[gesture].upper()} - Cycle {cycle + 1}")

async def collect_data(client):
    """Guide the user to perform gestures and record data."""
    global training_data, testing_data

    # Collect training data (2 cycles per gesture)
    for gesture in gesture_labels.keys():
        await record_gesture(client, gesture, training_cycles, is_training=True)

    # Collect testing data (1 cycle per gesture)
    for gesture in gesture_labels:
        await record_gesture(client, gesture, testing_cycles, is_training=False)

    # Save to CSV
    save_to_csv(training_data, "training_data_fe.csv")
    save_to_csv(testing_data, "testing_data_fe.csv")
    print("Data collection complete.")

def extract_features(data):
    """Compute RMS, Mean, Variance for each channel."""
    rms = np.sqrt(np.mean(np.square(data), axis=0)).tolist()
    mean = np.mean(data, axis=0).tolist()
    variance = np.var(data, axis=0).tolist()
    return rms + mean + variance

def save_to_csv(data, filename):
    """Save collected data to CSV file."""
    # columns = [f"emg_rms_{i}" for i in range(8)] + \
    #           [f"imu_rms_{i}" for i in range(10)] + \
    #           [f"emg_mean_{i}" for i in range(8)] + \
    #           [f"imu_mean_{i}" for i in range(10)] + \
    #           [f"emg_var_{i}" for i in range(8)] + \
    #           [f"imu_var_{i}" for i in range(10)] + ["label"]
    columns = [f"emg_rms_{i}" for i in range(8)] + \
              [f"emg_mean_{i}" for i in range(8)] + \
              [f"emg_var_{i}" for i in range(8)] + ["label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)

def train_model():
    """Train a Random Forest classifier using collected training data."""
    df = pd.read_csv("training_data.csv")
    X = df.drop(columns=["label"])
    y = df["label"]

    # Train-Test Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Save Model
    with open("gesture_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    # Evaluate Model
    y_pred = clf.predict(X_val)
    print("Classification Report:")
    print(classification_report(y_val, y_pred))

async def deploy_model():
    """Deploy the trained model for real-time classification."""
    with open("gesture_classifier.pkl", "rb") as f:
        clf = pickle.load(f)

    async def classify_gesture(clf):
        """Predict gesture and simulate keyboard press."""
        while True:
            if len(emg_buffer) == window_size:  # No IMU buffer needed for EMG-based classification
                emg_features = extract_features(np.array(emg_buffer))
                features = np.array(emg_features).reshape(1, -1)

                prediction = clf.predict(features)[0]
                print(f"Predicted Gesture: {prediction}")

                # Simulate Keyboard Press (ignore rest)
                key_map = {"left": "left", "right": "right", "up": "up", "down": "down"}
                if prediction != "rest":
                    pyautogui.press(key_map[prediction])

            await asyncio.sleep(0.1)  # Prevent blocking

    # Directly await the classify_gesture() function
    await classify_gesture(clf)


async def connect_myo():
    async with BleakClient(MYO_MAC) as client:
        print(f"Connected to Myo: {client.address}")

        # Step 1: Enable EMG Streaming
        await client.write_gatt_char(UUID_CONTROL, ENABLE_STREAMING_CMD)
        print("EMG streaming enabled!")

        # Subscribe to EMG data
        for uuid in UUID_EMG_STREAMS:
            await client.start_notify(uuid, process_emg_data)

        # Collect data
        await collect_data(client)

        # Train model
        train_model()

        # Deploy model
        # await deploy_model()


# Run the async function
asyncio.run(connect_myo())
