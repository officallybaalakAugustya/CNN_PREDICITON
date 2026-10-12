"""
Webcam CNN Image Processing & Inference Pipeline
=================================================
This program captures an image from your computer's webcam,
converts it to genuine grayscale, preprocesses it into a normalized
4D tensor, and feeds it into a Convolutional Neural Network (CNN)
using TensorFlow/Keras.

Author: Antigravity Pair Programmer
"""

import os
import sys
import argparse
import numpy as np

# Verify OpenCV is installed
try:
    import cv2
except ImportError:
    print("\n[ERROR] OpenCV is not installed.")
    print("Please install it with: pip install opencv-python\n")
    sys.exit(1)

# Verify TensorFlow is installed
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow informational logs
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("\n[ERROR] TensorFlow is not installed.")
    print("Please install it with: pip install tensorflow\n")
    sys.exit(1)


# =====================================================================
# Configuration Parameters
# =====================================================================
CAPTURED_IMAGE_FILE = "captured_image.jpg"
MODEL_FILE = "model.keras"
LABELS_FILE = "labels.txt"
CNN_INPUT_SHAPE = (64, 64, 1)  # (Height, Width, Channels)
NUM_CLASSES = 10               # Default number of output categories


# =====================================================================
# 1. Camera Input and Photo Capture
# =====================================================================
def capture_image_from_webcam(camera_index: int = 0) -> np.ndarray:
    """
    Opens the computer's webcam, displays a live preview with instructions,
    and allows the user to capture a frame using the SPACE key or exit with Q.

    Args:
        camera_index: OpenCV device index (default 0 for built-in webcam).

    Returns:
        np.ndarray: Captured BGR frame, or None if the user quit.
    """
    print("\n" + "=" * 60)
    print(" 1. CAMERA INITIALIZATION & PREVIEW")
    print("=" * 60)
    print(f"Connecting to webcam (Device Index: {camera_index})...")

    # Initialize video capture. On Windows, cv2.CAP_DSHOW can provide faster startup
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"\n[ERROR] Unable to access webcam at device index {camera_index}.")
        print("Troubleshooting tips:")
        print("  - Make sure your webcam is plugged in and not in use by another app.")
        print("  - Check Windows camera privacy permissions (Settings > Privacy > Camera).")
        print("  - Try changing the camera index (e.g., camera_index = 1).\n")
        return None

    window_title = "Webcam Live Preview - Press [SPACE] to capture | [Q] to quit"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    print("\n>>> LIVE PREVIEW ACTIVE <<<")
    print("  - Press [SPACE] to capture a photo.")
    print("  - Press [Q] to cancel and quit.")

    captured_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[WARNING] Failed to grab frame from webcam. Retrying...")
                continue

            # Overlay user instructions on the live preview screen
            display_frame = frame.copy()
            overlay_h, overlay_w = display_frame.shape[:2]

            # Semi-transparent bottom banner for UI readability
            banner_height = 40
            sub_img = display_frame[overlay_h - banner_height:overlay_h, 0:overlay_w]
            black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
            blended = cv2.addWeighted(sub_img, 0.4, black_rect, 0.6, 1.0)
            display_frame[overlay_h - banner_height:overlay_h, 0:overlay_w] = blended

            cv2.putText(
                display_frame,
                "Press [SPACE] to Capture  |  [Q] to Quit",
                (20, overlay_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_title, display_frame)

            # Wait for key press (1 millisecond)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE key
                print("\n[INFO] SPACE key pressed! Capturing frame...")
                captured_frame = frame.copy()
                break
            elif key in (ord("q"), ord("Q"), 27):  # Q or ESC key
                print("\n[INFO] 'Q' pressed. Exiting without capture.")
                break

    finally:
        # Always release the camera resource and close the preview window
        cap.release()
        cv2.destroyWindow(window_title)
        print("[INFO] Camera preview closed and hardware resource released.")

    return captured_frame


# =====================================================================
# 2. Convert Image to Genuine Grayscale
# =====================================================================
def convert_to_grayscale(
    bgr_image: np.ndarray,
    output_path: str = CAPTURED_IMAGE_FILE,
    show_window: bool = True
) -> np.ndarray:
    """
    Converts a 3-channel BGR image into a genuine 1-channel grayscale image,
    saves it to disk, and displays it in a separate window.

    Args:
        bgr_image: The captured color frame.
        output_path: Filepath where the grayscale image will be saved.
        show_window: Whether to display the image in an OpenCV GUI window.

    Returns:
        np.ndarray: 2D grayscale image array (Height, Width).
    """
    print("\n" + "=" * 60)
    print(" 2. GRAYSCALE CONVERSION & SAVING")
    print("=" * 60)

    # Convert to grayscale using OpenCV standard color transform
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Verification: genuine grayscale has exactly 2 dimensions (Height, Width) and 1 channel
    print(f"Captured color image shape : {bgr_image.shape} (BGR, 3 channels)")
    print(f"Grayscale image shape      : {gray_image.shape} (Genuine Grayscale, 1 channel)")
    print(f"Grayscale array dimensions : {gray_image.ndim}D array")
    print(f"Pixel data type            : {gray_image.dtype}")

    # Save to disk
    try:
        success = cv2.imwrite(output_path, gray_image)
        if not success:
            raise IOError(f"cv2.imwrite returned False when saving to '{output_path}'.")
        print(f"[SUCCESS] Grayscale image saved successfully to '{output_path}'.")
    except Exception as e:
        print(f"[ERROR] Failed to save grayscale image: {e}")
        raise

    # Display the grayscale image in a separate window if enabled
    if show_window:
        display_title = "Captured Grayscale Image - Press any key to proceed"
        cv2.namedWindow(display_title, cv2.WINDOW_AUTOSIZE)

        # Add instruction text
        display_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            display_img,
            "Grayscale Image - Press ANY KEY to continue",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(display_title, display_img)
        print("Showing captured grayscale image. Press any key in the window to proceed...")
        cv2.waitKey(0)
        cv2.destroyWindow(display_title)

    return gray_image


# =====================================================================
# 3. Image Preprocessing for the CNN
# =====================================================================
def preprocess_image_for_cnn(
    image_input,
    target_size: tuple = (64, 64)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares a grayscale image for feeding into a Convolutional Neural Network:
    1. Loads or accepts the 2D grayscale array.
    2. Resizes it to target dimensions (width, height).
    3. Normalizes pixel values from [0, 255] to [0.0, 1.0].
    4. Expands dimensions to format (Batch_Size, Height, Width, Channels).

    Args:
        image_input: Either a numpy array or a filepath string.
        target_size: (width, height) tuple for resizing (default 64x64).

    Returns:
        tuple: (preprocessed_4d_tensor, resized_2d_image)
    """
    print("\n" + "=" * 60)
    print(" 3. IMAGE PREPROCESSING FOR CNN")
    print("=" * 60)

    # Load from filepath if a string is passed
    if isinstance(image_input, str):
        print(f"Loading grayscale image from: '{image_input}'")
        gray_image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            raise FileNotFoundError(f"Could not load image at '{image_input}'.")
    else:
        gray_image = image_input

    # Step A: Resize to CNN input dimensions (e.g. 64x64)
    # Note: cv2.resize expects (width, height)
    target_w, target_h = target_size
    resized_image = cv2.resize(gray_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    print(f"Step A - Resized image shape: {resized_image.shape} ({target_w}x{target_h} pixels)")

    # Step B: Normalize pixel intensities to range [0.0, 1.0]
    normalized_image = resized_image.astype(np.float32) / 255.0
    print(f"Step B - Normalized pixel values: min={normalized_image.min():.3f}, max={normalized_image.max():.3f}")

    # Step C: Add batch and channel dimensions
    # Shape becomes: (1, height, width, 1) -> (Batch, Height, Width, Channels)
    input_tensor = np.expand_dims(normalized_image, axis=(0, -1))
    print(f"Step C - Final input tensor shape: {input_tensor.shape}")
    print(f"         [Batch Size: {input_tensor.shape[0]}, "
          f"Height: {input_tensor.shape[1]}, "
          f"Width: {input_tensor.shape[2]}, "
          f"Channels: {input_tensor.shape[3]}]")

    return input_tensor, resized_image


# =====================================================================
# 4. CNN Model Architecture
# =====================================================================
def build_cnn_model(input_shape: tuple = CNN_INPUT_SHAPE, num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Builds a Convolutional Neural Network (CNN) architecture using TensorFlow/Keras.
    Features:
      - 3 Convolutional layers with ReLU activations and Max-Pooling.
      - Flatten layer to convert 2D feature maps to 1D vectors.
      - Dense hidden layer with Dropout regularization.
      - Dense output layer with Softmax activation for classification.

    Args:
        input_shape: Tensor input shape (Height, Width, Channels).
        num_classes: Number of target categories.

    Returns:
        keras.Model: Compiled Keras CNN model.
    """
    model = keras.Sequential([
        # Input layer specification
        keras.Input(shape=input_shape),

        # Conv Block 1
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

        # Conv Block 2
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name="conv2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

        # Conv Block 3
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", name="conv3"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),

        # Classification Head
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.Dropout(0.3, name="dropout"),
        layers.Dense(num_classes, activation="softmax", name="output_probabilities")
    ], name="Webcam_CNN_Classifier")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def load_or_initialize_model(model_path: str = MODEL_FILE) -> tuple[keras.Model, bool]:
    """
    Attempts to load a trained model from disk if it exists.
    If no trained model is found, creates an untrained model instance with random weights.

    Args:
        model_path: Path to the saved Keras model file.

    Returns:
        tuple: (model, is_trained: bool)
    """
    print("\n" + "=" * 60)
    print(" 4. CNN MODEL STATUS & ARCHITECTURE")
    print("=" * 60)

    if os.path.exists(model_path):
        print(f"[FOUND] Trained model detected at '{model_path}'. Loading model...")
        try:
            model = keras.models.load_model(model_path)
            print("[SUCCESS] Trained model loaded successfully.")
            return model, True
        except Exception as e:
            print(f"[WARNING] Could not load '{model_path}': {e}. Building new model instance.")

    # When no trained model file exists
    print(f"[STATUS] No trained model file found at '{model_path}'.")
    print("Creating an example CNN architecture with randomly initialized weights...")
    model = build_cnn_model(input_shape=CNN_INPUT_SHAPE, num_classes=NUM_CLASSES)
    return model, False


# =====================================================================
# 5. Prediction and Model Output Evaluation
# =====================================================================
def run_cnn_inference(
    model: keras.Model,
    input_tensor: np.ndarray,
    is_trained: bool,
    class_labels: list = None
):
    """
    Passes the preprocessed tensor through the CNN and prints results.
    Strictly explains that untrained random weights do NOT produce valid predictions.

    Args:
        model: Keras CNN model instance.
        input_tensor: 4D numpy array with shape (1, 64, 64, 1).
        is_trained: Boolean indicating whether weights are trained.
        class_labels: Optional list of class name strings.
    """
    print("\n" + "=" * 60)
    print(" 5. CNN PROCESSING & INFERENCE")
    print("=" * 60)

    # Pass the tensor through the model to verify pipeline execution
    raw_output = model.predict(input_tensor, verbose=0)
    predicted_class_index = int(np.argmax(raw_output[0]))
    confidence = float(raw_output[0][predicted_class_index]) * 100

    if is_trained:
        print("[RESULT] TRAINED MODEL INFERENCE COMPLETE:")
        label = class_labels[predicted_class_index] if (class_labels and predicted_class_index < len(class_labels)) else f"Class {predicted_class_index}"
        print(f"  - Predicted Category : {label}")
        print(f"  - Confidence Score   : {confidence:.2f}%")
        print("\nClass Probabilities:")
        for idx, prob in enumerate(raw_output[0]):
            name = class_labels[idx] if (class_labels and idx < len(class_labels)) else f"Class {idx}"
            bar = "#" * int(prob * 30)
            print(f"  [{idx:2d}] {name:<12}: {prob * 100:6.2f}% | {bar}")
    else:
        # Explicit, honest explanation: untrained random weights produce invalid predictions
        print("!" * 60)
        print("  NOTICE: UNTRAINED EXAMPLE MODEL")
        print("!" * 60)
        print("  The CNN architecture was successfully constructed and executed a forward")
        print("  pass on your webcam input tensor to verify shape and computational flow.")
        print()
        print("  HOWEVER: Because this model contains freshly initialized RANDOM WEIGHTS,")
        print("  it CANNOT make meaningful predictions. Any output class numbers produced")
        print("  are purely mathematical noise and MUST NOT be treated as real predictions.")
        print()
        print(f"  Raw forward pass output shape : {raw_output.shape}")
        print(f"  Untrained pseudo-argmax       : Class {predicted_class_index} (Confidence: {confidence:.2f}%)")
        print()
        print("  TO ENABLE MEANINGFUL PREDICTIONS:")
        print("    1. Train this CNN on your dataset (e.g., run `python train_model.py`).")
        print(f"    2. Save the trained weights to '{MODEL_FILE}'.")
        print("    3. Re-run this script; it will automatically load and use the trained weights.")
        print("!" * 60)


# =====================================================================
# 6. Display Processed Preprocessed Image
# =====================================================================
def display_processed_summary(
    resized_image: np.ndarray,
    input_tensor: np.ndarray,
    show_window: bool = True
):
    """
    Displays the preprocessed CNN input image enlarged for inspection,
    and prints the full dimension summary in the terminal.
    """
    print("\n" + "=" * 60)
    print(" 6. PROCESSED IMAGE SUMMARY & INSPECTION")
    print("=" * 60)
    print(f"  - Input dimensions (H x W)  : {resized_image.shape[0]} x {resized_image.shape[1]} pixels")
    print(f"  - Channels                  : 1 (Grayscale)")
    print(f"  - Final CNN Tensor Shape    : {input_tensor.shape}")
    print(f"  - Tensor Data Type          : {input_tensor.dtype}")
    print(f"  - Value Range (Min to Max)  : [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

    if show_window:
        # Scale the 64x64 image up to 256x256 using nearest-neighbor so it's easy to see
        preview_display = cv2.resize(resized_image, (256, 256), interpolation=cv2.INTER_NEAREST)
        display_title = "Preprocessed CNN Input (64x64 enlarged) - Press any key to finish"
        cv2.imshow(display_title, preview_display)
        print("\nDisplaying preprocessed 64x64 image preview window. Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =====================================================================
# 7. Main Workflow Function
# =====================================================================
def main():
    """
    Coordinates the complete webcam capture, grayscale conversion,
    preprocessing, and CNN inference workflow.
    """
    parser = argparse.ArgumentParser(description="Webcam CNN Image Processing and Inference Pipeline.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--image", type=str, default=None, help="Path to an existing image file (bypasses webcam)")
    parser.add_argument("--model", type=str, default=MODEL_FILE, help=f"Path to saved model file (default: {MODEL_FILE})")
    parser.add_argument("--labels", type=str, default=LABELS_FILE, help=f"Path to class labels file (default: {LABELS_FILE})")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows")
    parser.add_argument("--visualize", action="store_true", help="Launch visual demonstration of convolutions, pooling, and activations")
    parser.add_argument("--save-vis-dir", type=str, default="visualizations", help="Directory to save visualization figures")
    args = parser.parse_args()

    print("=" * 60)
    print("       WEBCAM CONVOLUTIONAL NEURAL NETWORK (CNN) PIPELINE")
    print("=" * 60)

    # Step 1: Capture image from webcam (or load provided image)
    if args.image:
        print(f"[INFO] Loading image directly from '{args.image}' (bypassing webcam)...")
        captured_frame = cv2.imread(args.image)
        if captured_frame is None:
            print(f"[ERROR] Could not load image from '{args.image}'.")
            return
    else:
        captured_frame = capture_image_from_webcam(camera_index=args.camera_index)
        if captured_frame is None:
            print("\nNo image was captured. Program terminated.")
            return

    # Step 2: Convert to grayscale and save
    gray_image = convert_to_grayscale(
        captured_frame,
        output_path=CAPTURED_IMAGE_FILE,
        show_window=not args.headless
    )

    # Step 3: Preprocess image for CNN
    input_tensor, resized_image = preprocess_image_for_cnn(
        gray_image,
        target_size=(CNN_INPUT_SHAPE[1], CNN_INPUT_SHAPE[0])  # (width, height) = (64, 64)
    )

    # Step 4: Load trained model or create CNN architecture
    model, is_trained = load_or_initialize_model(model_path=args.model)

    # Load class labels if available
    class_labels = None
    if os.path.exists(args.labels):
        try:
            with open(args.labels, "r", encoding="utf-8") as f:
                class_labels = [line.strip() for line in f if line.strip()]
            print(f"[INFO] Loaded {len(class_labels)} class labels from '{args.labels}'.")
        except Exception as e:
            print(f"[WARNING] Could not read '{args.labels}': {e}")

    # Print model architecture summary
    print("\nCNN Model Architecture Summary:")
    model.summary(line_length=60)

    # Step 5: Run prediction or report untrained status
    run_cnn_inference(model, input_tensor, is_trained=is_trained, class_labels=class_labels)

    # Step 6: Display processed image & tensor details
    display_processed_summary(resized_image, input_tensor, show_window=not args.headless)

    # Step 7: Visual Demonstration (Convolutions, Pooling, and Activations)
    if args.visualize:
        try:
            from cnn_visualization import run_all_visualizations
            run_all_visualizations(
                gray_image=gray_image,
                resized_input=resized_image,
                input_tensor=input_tensor,
                model=model if is_trained else None,
                save_dir=args.save_vis_dir,
                show_plots=not args.headless
            )
        except Exception as vis_err:
            print(f"[WARNING] Visualization encountered an error: {vis_err}")

    print("\n" + "=" * 60)
    print(" Pipeline execution finished successfully.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Program interrupted by user. Exiting cleanly...")
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as err:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {err}")
        cv2.destroyAllWindows()
        sys.exit(1)
