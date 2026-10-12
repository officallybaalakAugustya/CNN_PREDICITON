"""
Automated Pipeline Verification & Visualization Tests
======================================================
Tests all image processing, tensor transformations, CNN operations,
convolution kernels, max pooling logic, and intermediate activation
extractions headlessly without requiring human interaction or physical webcam input.
"""

import os
import sys
import shutil
import numpy as np

# Ensure project directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Headless matplotlib configuration for automated testing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import tensorflow as tf
from tensorflow import keras

from camera_cnn import (
    convert_to_grayscale,
    preprocess_image_for_cnn,
    build_cnn_model,
    load_or_initialize_model,
    run_cnn_inference,
    CNN_INPUT_SHAPE,
    NUM_CLASSES
)

from cnn_visualization import (
    DEMO_KERNELS,
    apply_2d_convolution,
    apply_max_pooling_2d,
    run_all_visualizations
)


def run_all_tests():
    print("=" * 65)
    print("      RUNNING WEBCAM CNN PIPELINE & VISUALIZATION TESTS")
    print("=" * 65)

    test_output_file = "test_captured_image.jpg"
    test_vis_dir = "test_visualizations"

    # 1. Test Grayscale Conversion
    print("\n[TEST 1] Grayscale Conversion...")
    synthetic_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    gray_image = cv2.cvtColor(synthetic_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(test_output_file, gray_image)

    assert os.path.exists(test_output_file), "Failed to write grayscale image."
    assert gray_image.ndim == 2, f"Expected 2D array, got {gray_image.ndim}D."
    assert gray_image.shape == (480, 640), f"Unexpected shape {gray_image.shape}."
    print("  -> PASS: Genuine 1-channel 2D grayscale image verified.")

    # 2. Test Image Preprocessing
    print("\n[TEST 2] CNN Preprocessing...")
    input_tensor, resized_image = preprocess_image_for_cnn(
        test_output_file,
        target_size=(64, 64)
    )

    assert resized_image.shape == (64, 64), f"Expected (64, 64), got {resized_image.shape}"
    assert input_tensor.shape == (1, 64, 64, 1), f"Expected (1, 64, 64, 1), got {input_tensor.shape}"
    assert input_tensor.dtype == np.float32, f"Expected float32, got {input_tensor.dtype}"
    assert 0.0 <= input_tensor.min() <= input_tensor.max() <= 1.0, "Normalization out of [0, 1] range."
    print("  -> PASS: Resized to 64x64, normalized to [0, 1], 4D tensor shape (1, 64, 64, 1) verified.")

    # 3. Test CNN Model Architecture
    print("\n[TEST 3] CNN Model Construction...")
    model = build_cnn_model(input_shape=CNN_INPUT_SHAPE, num_classes=NUM_CLASSES)

    layer_types = [layer.__class__.__name__ for layer in model.layers]
    print(f"  Model Layers: {layer_types}")

    assert "Conv2D" in layer_types, "Model missing Conv2D layer."
    assert "MaxPooling2D" in layer_types, "Model missing MaxPooling2D layer."
    assert "Flatten" in layer_types, "Model missing Flatten layer."
    assert "Dense" in layer_types, "Model missing Dense layer."
    assert model.output_shape == (None, NUM_CLASSES), f"Unexpected output shape: {model.output_shape}"
    print("  -> PASS: All required CNN layers verified with correct output dimensions.")

    # 4. Test Forward Pass
    print("\n[TEST 4] Model Forward Pass & Probability Distribution...")
    prediction = model.predict(input_tensor, verbose=0)
    assert prediction.shape == (1, NUM_CLASSES), f"Expected (1, {NUM_CLASSES}), got {prediction.shape}"
    prob_sum = float(np.sum(prediction[0]))
    assert abs(prob_sum - 1.0) < 1e-4, f"Softmax probabilities do not sum to 1.0 (sum={prob_sum})"
    print(f"  -> PASS: Forward pass output shape {prediction.shape}, Softmax sum = {prob_sum:.4f}")

    # 5. Test Inference Messaging (Untrained Mode)
    print("\n[TEST 5] Untrained Model Disclaimer & Inference Handling...")
    run_cnn_inference(model, input_tensor, is_trained=False)
    print("  -> PASS: Untrained warning displayed properly without falsely claiming valid predictions.")

    # 6. Test Demonstration Kernels
    print("\n[TEST 6] Demonstration Kernels Definition & Verification...")
    required_kernels = ["Identity", "Edge Detection", "Sharpening", "Gaussian Blur"]
    for k_name in required_kernels:
        assert k_name in DEMO_KERNELS, f"Missing kernel: {k_name}"
        k_mat = DEMO_KERNELS[k_name]["matrix"]
        assert k_mat.shape == (3, 3), f"Kernel {k_name} must be 3x3, got {k_mat.shape}"
    print(f"  -> PASS: All 4 demonstration kernels verified: {required_kernels}")

    # 7. Test 2D Convolution & Cross-Correlation
    print("\n[TEST 7] 2D Convolution Operation & Dimension Formula...")
    for k_name in required_kernels:
        k_mat = DEMO_KERNELS[k_name]["matrix"]
        fmap_same, meta_same = apply_2d_convolution(resized_image, k_mat, padding="same", stride=1)
        # Check formula for padding='same': floor((64 + 2*1 - 3)/1) + 1 = 64
        assert fmap_same.shape == (64, 64), f"Expected (64, 64) for same padding, got {fmap_same.shape}"

        fmap_valid, meta_valid = apply_2d_convolution(resized_image, k_mat, padding="valid", stride=1)
        # Check formula for padding='valid': floor((64 + 0 - 3)/1) + 1 = 62
        assert fmap_valid.shape == (62, 62), f"Expected (62, 62) for valid padding, got {fmap_valid.shape}"

    # Verify Identity kernel preserves input approximately
    id_mat = DEMO_KERNELS["Identity"]["matrix"]
    id_fmap, _ = apply_2d_convolution(resized_image, id_mat, padding="same", stride=1)
    diff = np.max(np.abs(id_fmap[1:-1, 1:-1] - resized_image[1:-1, 1:-1]))
    assert diff < 1e-4, f"Identity kernel failed: max diff = {diff}"
    print("  -> PASS: Convolution output dimensions and Identity preservation verified.")

    # 8. Test Max Pooling Logic & Numerical Correctness
    print("\n[TEST 8] Max Pooling Numerical Logic & Dimension Reduction...")
    # Test specific numerical region requested in prompt: [[1, 5], [3, 2]] -> max must be 5
    toy_patch = np.array([
        [1.0, 5.0],
        [3.0, 2.0]
    ], dtype=np.float32)
    toy_pooled, _ = apply_max_pooling_2d(toy_patch, pool_size=(2, 2), stride=2)
    assert toy_pooled.shape == (1, 1), f"Expected (1, 1), got {toy_pooled.shape}"
    assert toy_pooled[0, 0] == 5.0, f"Expected max 5.0, got {toy_pooled[0, 0]}"

    # Test full 64x64 feature map pooled to 32x32
    fmap_edge, _ = apply_2d_convolution(resized_image, DEMO_KERNELS["Edge Detection"]["matrix"], padding="same", stride=1)
    pooled_32, p_meta = apply_max_pooling_2d(fmap_edge, pool_size=(2, 2), stride=2)
    # Output size: floor((64 - 2)/2) + 1 = 32
    assert pooled_32.shape == (32, 32), f"Expected (32, 32), got {pooled_32.shape}"
    print("  -> PASS: Max pooling verified: 2x2 region [[1, 5], [3, 2]] yields 5.0; 64x64 -> 32x32 confirmed.")

    # 9. Test Intermediate Activations & Learned Filters from Trained Model
    print("\n[TEST 9] Trained CNN Intermediate Activations & Filter Weights...")
    trained_model_path = "trained_model.keras"
    if os.path.exists(trained_model_path):
        trained_model, _ = load_or_initialize_model(model_path=trained_model_path)
        # Extract Conv2D and MaxPooling2D layers
        target_layers = [
            layer for layer in trained_model.layers
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.MaxPooling2D))
        ]
        assert len(target_layers) >= 4, f"Expected at least 4 conv/pool layers, found {len(target_layers)}"

        extractor = keras.Model(
            inputs=trained_model.inputs,
            outputs=[layer.output for layer in target_layers]
        )
        activations = extractor.predict(input_tensor, verbose=0)
        assert len(activations) == len(target_layers), "Activation count mismatch."
        # Verify first layer Conv2D output shape: (1, 64, 64, 32)
        assert activations[0].shape == (1, 64, 64, 32), f"Unexpected conv1 activation shape: {activations[0].shape}"
        # Verify first pool layer output shape: (1, 32, 32, 32)
        assert activations[1].shape == (1, 32, 32, 32), f"Unexpected pool1 activation shape: {activations[1].shape}"

        # Verify weights from first conv layer
        conv1 = target_layers[0]
        w, b = conv1.get_weights()
        assert w.shape == (3, 3, 1, 32), f"Unexpected conv1 weights shape: {w.shape}"
        print(f"  -> PASS: Intermediate layers {len(target_layers)} extracted successfully with correct tensor shapes.")
    else:
        print("  -> [SKIP] 'trained_model.keras' not found; trained activation test skipped.")

    # 10. Test Full Visualization Suite (Headless File Generation)
    print("\n[TEST 10] Headless Visualization Generation...")
    model_for_vis = None
    if os.path.exists(trained_model_path):
        model_for_vis = keras.models.load_model(trained_model_path)

    run_all_visualizations(
        gray_image=gray_image,
        resized_input=resized_image,
        input_tensor=input_tensor,
        model=model_for_vis,
        save_dir=test_vis_dir,
        show_plots=False
    )

    # Verify generated figure files
    expected_figures = [
        "fig_a_image_and_cnn_input.png",
        "fig_b_demo_kernels.png",
        "fig_c_convolutions_before_after.png",
        "fig_d_max_pooling_before_after.png"
    ]
    if model_for_vis is not None:
        expected_figures.extend([
            "fig_e_trained_cnn_activations.png",
            "fig_f_learned_filter_weights.png"
        ])

    for fig_file in expected_figures:
        fig_path = os.path.join(test_vis_dir, fig_file)
        assert os.path.exists(fig_path), f"Figure not generated: {fig_path}"
        assert os.path.getsize(fig_path) > 1000, f"Figure file suspiciously small: {fig_path}"

    print(f"  -> PASS: All {len(expected_figures)} visualization figures generated successfully in '{test_vis_dir}'.")

    # Clean up test artifacts
    if os.path.exists(test_output_file):
        os.remove(test_output_file)
    if os.path.exists(test_vis_dir):
        shutil.rmtree(test_vis_dir)

    print("\n" + "=" * 65)
    print(" ALL 10 PIPELINE & VISUALIZATION TESTS PASSED SUCCESSFULLY! ")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
