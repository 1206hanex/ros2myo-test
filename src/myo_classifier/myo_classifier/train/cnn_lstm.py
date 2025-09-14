import os, numpy as np, pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def build_model(input_shape, num_classes: int):
    inp = keras.Input(shape=input_shape)  # (T, C)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train_cnn_lstm(X, y, out_dir: str, logger=print,
                   epochs=25, batch_size=64, val_split=0.2, shuffle_buf=2048):
    os.makedirs(out_dir, exist_ok=True)

    # Label encode
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    logger(f"Sequences: {X.shape}, classes={list(le.classes_)}")

    # Split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y_enc, test_size=val_split, random_state=42, stratify=y_enc)

    # Normalize (train stats)
    mean = X_tr.mean(axis=(0,1), keepdims=True)
    std  = X_tr.std(axis=(0,1), keepdims=True) + 1e-8
    X_tr = (X_tr - mean) / std
    X_va = (X_va - mean) / std

    # Datasets
    tr_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(shuffle_buf).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    va_ds = tf.data.Dataset.from_tensor_slices((X_va, y_va)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Model
    model = build_model(input_shape=X.shape[1:], num_classes=num_classes)
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    ]
    model.fit(tr_ds, validation_data=va_ds, epochs=epochs, callbacks=callbacks, verbose=2)

    # Save
    model_path = os.path.join(out_dir, "gesture_cnn_lstm.h5")
    model.save(model_path)
    with open(os.path.join(out_dir, "label_encoder.pkl"), "wb") as f: pickle.dump(le, f)
    np.savez(os.path.join(out_dir, "norm_stats.npz"),
             mean=mean.astype("float32"), std=std.astype("float32"))
    return model_path
