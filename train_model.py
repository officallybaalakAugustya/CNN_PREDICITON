"""
Model Training Script for Webcam CNN
====================================
This script trains the Convolutional Neural Network (CNN) architecture
used by `camera_cnn.py` and saves the trained weights to `model.keras`.

Supported Training Modes:
1. 'synthetic' (Default, Fast & Offline): Generates clean geometric shapes
   (circles, squares, triangles, plus-signs) directly on 64x64 grids.
   Requires NO internet connection and trains in under 15 seconds.
2. 'fashion_mnist': Downloads and trains on the Fashion-MNIST dataset
   (10 clothing categories) resized to 64x64.
3. 'mnist': Downloads and trains on handwritten digits (0-9) resized to 64x64.

Author: Antigravity Pair Programmer
"""

import os
import sys
import argparse
import numpy as np

# Suppress TensorFlow verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV is required. Install via: pip install opencv-python")
    sys.exit(1)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("[ERROR] TensorFlow is required. Install via: pip install tensorflow")
    sys.exit(1)


# Import architecture builder from camera_cnn if available, or define here
from camera_cnn import build_cnn_model, CNN_INPUT_SHAPE, MODEL_FILE, LABELS_FILE


def generate_synthetic_dataset(num_samples_per_class: int = 500) -> tuple:
    """
    Generates a synthetic dataset of 4 geometric shape classes:
      0: Circle
      1: Rectangle / Square
      2: Triangle
      3: Plus / Cross

    This guarantees fast, 100% offline training without downloading large datasets.
    """
    print("\nGenerating synthetic shape dataset (64x64 grayscale images)...")
    class_names = ["Circle", "Rectangle", "Triangle", "Cross"]
    num_classes = len(class_names)
    total_samples = num_samples_per_class * num_classes

    x_data = np.zeros((total_samples, 64, 64), dtype=np.uint8)
    y_data = np.zeros((total_samples,), dtype=np.int32)

    sample_idx = 0
    np.random.seed(42)

    for cls_idx in range(num_classes):
        for _ in range(num_samples_per_class):
            img = np.zeros((64, 64), dtype=np.uint8)
            # Random position and size
            center_x = np.random.randint(20, 44)
            center_y = np.random.randint(20, 44)
            radius = np.random.randint(8, 16)

            if cls_idx == 0:  # Circle
                cv2.circle(img, (center_x, center_y), radius, 255, -1)
            elif cls_idx == 1:  # Rectangle
                pt1 = (center_x - radius, center_y - radius)
                pt2 = (center_x + radius, center_y + radius)
                cv2.rectangle(img, pt1, pt2, 255, -1)
            elif cls_idx == 2:  # Triangle
                pt1 = (center_x, center_y - radius)
                pt2 = (center_x - radius, center_y + radius)
                pt3 = (center_x + radius, center_y + radius)
                pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(img, [pts], 255)
            elif cls_idx == 3:  # Cross
                thickness = np.random.randint(2, 5)
                cv2.line(img, (center_x - radius, center_y), (center_x + radius, center_y), 255, thickness)
                cv2.line(img, (center_x, center_y - radius), (center_x, center_y + radius), 255, thickness)

            # Add subtle random noise to improve CNN robustness
            noise = np.random.randint(0, 30, (64, 64), dtype=np.uint8)
            img = cv2.add(img, noise)

            x_data[sample_idx] = img
            y_data[sample_idx] = cls_idx
            sample_idx += 1

    # Shuffle dataset
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]

    # Split 80% train, 20% validation
    split = int(total_samples * 0.8)
    x_train, x_val = x_data[:split], x_data[split:]
    y_train, y_val = y_data[:split], y_data[split:]

    # Normalize to [0.0, 1.0] and add channel dimension
    x_train = np.expand_dims(x_train.astype(np.float32) / 255.0, axis=-1)
    x_val = np.expand_dims(x_val.astype(np.float32) / 255.0, axis=-1)

    print(f"Generated {x_train.shape[0]} training and {x_val.shape[0]} validation samples.")
    return (x_train, y_train), (x_val, y_val), class_names


def load_and_resize_keras_dataset(dataset_name: str = "fashion_mnist") -> tuple:
    """
    Loads MNIST or Fashion-MNIST from Keras and resizes 28x28 samples to 64x64.
    """
    if dataset_name == "fashion_mnist":
        print("\nLoading Fashion-MNIST dataset from Keras...")
        (x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
    else:
        print("\nLoading MNIST Digits dataset from Keras...")
        (x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
        class_names = [f"Digit_{i}" for i in range(10)]

    # Take a manageable subset for quick demonstration
    x_train_sub = x_train[:5000]
    y_train_sub = y_train[:5000]
    x_val_sub = x_val[:1000]
    y_val_sub = y_val[:1000]

    print(f"Resizing samples to 64x64...")
    def resize_batch(images):
        resized = [cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA) for img in images]
        return np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, axis=-1)

    x_train_64 = resize_batch(x_train_sub)
    x_val_64 = resize_batch(x_val_sub)

    return (x_train_64, y_train_sub), (x_val_64, y_val_sub), class_names


def train_and_save_model(mode: str = "synthetic", epochs: int = 5, batch_size: int = 32):
    """
    Trains the CNN model and saves `model.keras` and `labels.txt`.
    """
    print("=" * 60)
    print(f"           TRAINING CNN MODEL (Mode: {mode})")
    print("=" * 60)

    if mode in ("mnist", "fashion_mnist"):
        (x_train, y_train), (x_val, y_val), class_names = load_and_resize_keras_dataset(mode)
    else:
        (x_train, y_train), (x_val, y_val), class_names = generate_synthetic_dataset(num_samples_per_class=400)

    num_classes = len(class_names)
    print(f"\nTarget Classes ({num_classes}): {', '.join(class_names)}")

    # Build fresh model architecture matching camera_cnn
    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=num_classes)

    print("\nStarting Model Training...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate final validation accuracy
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}% (Loss: {val_loss:.4f})")

    # Save trained model to disk
    model.save(MODEL_FILE)
    print(f"[SUCCESS] Trained model saved to: '{MODEL_FILE}'")

    # Save class labels
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"[SUCCESS] Class labels saved to: '{LABELS_FILE}'")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("Now when you run `python camera_cnn.py`, it will load this trained")
    print(f"model and make real predictions using classes: {class_names}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN model for webcam inference.")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "fashion_mnist", "mnist"],
        default="synthetic",
        help="Dataset to train on (default: synthetic)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    args = parser.parse_args()

    train_and_save_model(mode=args.mode, epochs=args.epochs)
