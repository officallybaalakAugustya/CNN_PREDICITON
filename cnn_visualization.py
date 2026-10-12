"""
CNN Visual Demonstration and Activation Analysis Module
=========================================================
This module provides in-depth, educational visual demonstrations of how
a Convolutional Neural Network (CNN) processes images captured from a webcam.

Features:
1. Demonstration Kernels:
   - Identity, Edge Detection (Laplacian), Sharpening, Gaussian Blur.
   - Numerical matrix inspection, heatmaps, and functional explanations.
2. Convolution Operations (Before & After):
   - 2D Cross-correlation / convolution with padding and stride explanation.
   - Worked numerical dot-product example on image patches.
   - Min/max value tracking and dimension calculations.
3. Max-Pooling Operations (Before & After):
   - 2x2 pooling with stride 2.
   - Visual grid boundaries on feature maps.
   - Numerical walkthrough of local maximum selection.
4. Trained CNN Activations & Learned Filters:
   - Extracts and displays actual intermediate feature maps from Conv2D
     and MaxPooling2D layers of the trained model.
   - Inspects real learned filter weights and contrasts them with manual kernels.
5. Mathematical Foundations & LaTeX Formulas:
   - Output dimension formula: floor((W + 2P - K) / S) + 1.
   - Convolution sum and pooling equations.

Author: Antigravity Pair Programmer
"""

import os
import sys
import argparse
import numpy as np
import cv2

# Set matplotlib backend safely
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Suppress TensorFlow informational logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None


# =====================================================================
# 1. Demonstration Kernels Definition & Explanations
# =====================================================================
DEMO_KERNELS = {
    "Identity": {
        "matrix": np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]
        ], dtype=np.float32),
        "explanation": "Preserves the image structure approximately by passing the center pixel unchanged.",
        "role": "Baseline reference filter."
    },
    "Edge Detection": {
        "matrix": np.array([
            [-1.0, -1.0, -1.0],
            [-1.0,  8.0, -1.0],
            [-1.0, -1.0, -1.0]
        ], dtype=np.float32),
        "explanation": "Highlights intensity transitions by taking the difference between center and all 8 neighbors.",
        "role": "High-frequency edge and boundary extractor."
    },
    "Sharpening": {
        "matrix": np.array([
            [ 0.0, -1.0,  0.0],
            [-1.0,  5.0, -1.0],
            [ 0.0, -1.0,  0.0]
        ], dtype=np.float32),
        "explanation": "Emphasizes edges and fine details by amplifying the central pixel relative to orthogonal neighbors.",
        "role": "Detail enhancer and contrast booster."
    },
    "Gaussian Blur": {
        "matrix": (1.0 / 16.0) * np.array([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0]
        ], dtype=np.float32),
        "explanation": "Smooths the image and attenuates noise using a distance-weighted neighborhood average.",
        "role": "Low-frequency smoothing and noise filter."
    }
}


# =====================================================================
# 2. Mathematical Convolution and Correlation
# =====================================================================
def apply_2d_convolution(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: str = "same",
    stride: int = 1,
    flip_kernel: bool = False
) -> tuple[np.ndarray, dict]:
    """
    Applies a 2D convolution or cross-correlation on a single-channel image.

    Note on Convolution vs. Cross-Correlation:
    - Conventional mathematical convolution flips the kernel by 180 degrees before dot-product.
    - Deep learning libraries (TensorFlow Conv2D, PyTorch conv2d) and OpenCV filter2D
      perform cross-correlation (no kernel flip). Because CNN kernels are learned
      freely by backpropagation, cross-correlation is mathematically equivalent for optimization.

    Args:
        image: 2D array of shape (H, W).
        kernel: 2D array of shape (Kh, Kw).
        padding: 'same' (pads with zeros so H_out = H_in) or 'valid' (no padding).
        stride: Stride step size (default 1).
        flip_kernel: If True, mathematically flips the kernel (true convolution).

    Returns:
        tuple: (feature_map, metadata_dict)
    """
    h_in, w_in = image.shape
    kh, kw = kernel.shape

    active_kernel = np.flip(kernel) if flip_kernel else kernel.copy()

    if padding == "same":
        pad_top = kh // 2
        pad_bottom = kh - 1 - pad_top
        pad_left = kw // 2
        pad_right = kw - 1 - pad_left
        padded_img = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
        pad_val = pad_top
    else:
        padded_img = image
        pad_val = 0

    h_pad, w_pad = padded_img.shape
    h_out = int(np.floor((h_in + 2 * pad_val - kh) / stride)) + 1
    w_out = int(np.floor((w_in + 2 * pad_val - kw) / stride)) + 1

    feature_map = np.zeros((h_out, w_out), dtype=np.float32)

    # Perform sliding window calculation
    for i in range(h_out):
        for j in range(w_out):
            r_start = i * stride
            c_start = j * stride
            patch = padded_img[r_start:r_start + kh, c_start:c_start + kw]
            feature_map[i, j] = np.sum(patch * active_kernel)

    # Pick a sample central patch for a step-by-step worked example
    sample_r = h_in // 2
    sample_c = w_in // 2
    sample_patch = padded_img[sample_r:sample_r + kh, sample_c:sample_c + kw]
    products = sample_patch * active_kernel
    sample_output_val = float(np.sum(products))

    metadata = {
        "input_shape": (h_in, w_in),
        "kernel_shape": (kh, kw),
        "padding": padding,
        "padding_size": pad_val,
        "stride": stride,
        "output_shape": (h_out, w_out),
        "min_value": float(feature_map.min()),
        "max_value": float(feature_map.max()),
        "sample_patch": sample_patch,
        "sample_kernel": active_kernel,
        "sample_products": products,
        "sample_output": sample_output_val,
        "sample_coord": (sample_r, sample_c),
        "is_flipped": flip_kernel
    }

    return feature_map, metadata


# =====================================================================
# 3. Mathematical Max Pooling
# =====================================================================
def apply_max_pooling_2d(
    feature_map: np.ndarray,
    pool_size: tuple = (2, 2),
    stride: int = 2
) -> tuple[np.ndarray, dict]:
    """
    Applies 2D max pooling with specified pool size and stride.

    Args:
        feature_map: 2D array of shape (H, W).
        pool_size: (Ph, Pw) pooling window dimensions (default (2, 2)).
        stride: Stride step size (default 2).

    Returns:
        tuple: (pooled_map, metadata_dict)
    """
    h_in, w_in = feature_map.shape
    ph, pw = pool_size

    h_out = int(np.floor((h_in - ph) / stride)) + 1
    w_out = int(np.floor((w_in - pw) / stride)) + 1

    pooled_map = np.zeros((h_out, w_out), dtype=np.float32)

    for i in range(h_out):
        for j in range(w_out):
            r_start = i * stride
            c_start = j * stride
            window = feature_map[r_start:r_start + ph, c_start:c_start + pw]
            pooled_map[i, j] = np.max(window)

    # Pick a sample central pooling window for a worked example
    sample_i = h_out // 2
    sample_j = w_out // 2
    sample_window = feature_map[sample_i * stride:sample_i * stride + ph, sample_j * stride:sample_j * stride + pw]
    sample_max = float(np.max(sample_window))

    metadata = {
        "input_shape": (h_in, w_in),
        "pool_size": pool_size,
        "stride": stride,
        "output_shape": (h_out, w_out),
        "sample_window": sample_window,
        "sample_max": sample_max,
        "sample_coord": (sample_i, sample_j)
    }

    return pooled_map, metadata


# =====================================================================
# 4. Visualization Layouts (Matplotlib)
# =====================================================================

def plot_figure_a_inputs(
    original_gray: np.ndarray,
    resized_input: np.ndarray,
    save_path: str = None
):
    """
    Figure A: Displays the full captured grayscale image alongside the
    resized 64x64 CNN input tensor.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Visual Step A: Image Capture & CNN Tensor Resizing", fontsize=14, fontweight="bold")

    # Original Grayscale Image
    im0 = axes[0].imshow(original_gray, cmap="gray")
    axes[0].set_title(f"1. Original Grayscale Capture\nDimensions: {original_gray.shape[0]} x {original_gray.shape[1]} (uint8 [0, 255])")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Height (pixels)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Pixel Intensity [0, 255]")

    # Resized & Normalized CNN Input
    im1 = axes[1].imshow(resized_input, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"2. Resized CNN Input\nDimensions: {resized_input.shape[0]} x {resized_input.shape[1]} (float32 [0.0, 1.0])")
    axes[1].set_xlabel("Width (pixels)")
    axes[1].set_ylabel("Height (pixels)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Normalized Value [0.0, 1.0]")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVED] Figure A saved to: '{save_path}'")
    return fig


def plot_figure_b_kernels(save_path: str = None):
    """
    Figure B: Displays numerical matrices and heatmaps for all demonstration kernels.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Visual Step B: Demonstration Convolution Kernels (Numerical Values & Heatmaps)", fontsize=14, fontweight="bold")

    kernel_items = list(DEMO_KERNELS.items())

    for idx, (name, data) in enumerate(kernel_items):
        matrix = data["matrix"]

        # Top row: Numerical Matrix Representation
        ax_mat = axes[0, idx]
        ax_mat.axis("off")
        matrix_text = "\n".join(["[" + "  ".join([f"{val:6.2f}" for val in row]) + "]" for row in matrix])
        ax_mat.text(
            0.5, 0.5,
            f"Kernel: {name}\nDimensions: 3 x 3\n\n{matrix_text}\n\n{data['role']}",
            ha="center", va="center", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f7fa", edgecolor="#4a5568", linewidth=1.5)
        )
        ax_mat.set_title(f"{name} (Matrix)", fontsize=11, fontweight="bold")

        # Bottom row: Heatmap with cell values
        ax_hm = axes[1, idx]
        vbound = max(abs(matrix.min()), abs(matrix.max()), 1.0)
        im = ax_hm.imshow(matrix, cmap="coolwarm", vmin=-vbound, vmax=vbound)
        ax_hm.set_title(f"{name} (Heatmap)", fontsize=11, fontweight="bold")
        ax_hm.set_xticks(range(3))
        ax_hm.set_yticks(range(3))

        # Annotate exact numerical values inside each heatmap cell
        for r in range(3):
            for c in range(3):
                val = matrix[r, c]
                color = "white" if abs(val) > vbound * 0.5 else "black"
                ax_hm.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold", fontsize=10)

        fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVED] Figure B saved to: '{save_path}'")
    return fig


def plot_figure_c_convolutions(
    resized_input: np.ndarray,
    save_path: str = None
):
    """
    Figure C: Displays input image versus convolution feature map for each kernel,
    including output dimensions, min/max values, and a worked numerical example.
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle("Visual Step C: 2D Convolution Operation (Before vs. After)", fontsize=15, fontweight="bold")

    kernel_items = list(DEMO_KERNELS.items())

    for idx, (name, data) in enumerate(kernel_items):
        matrix = data["matrix"]
        fmap, meta = apply_2d_convolution(resized_input, matrix, padding="same", stride=1)

        # Col 0: Input Image (64x64)
        ax_in = axes[idx, 0]
        im_in = ax_in.imshow(resized_input, cmap="gray", vmin=0.0, vmax=1.0)
        ax_in.set_title(f"Input Image (64 x 64)\nKernel: {name}")
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        fig.colorbar(im_in, ax=ax_in, fraction=0.046, pad=0.04)

        # Col 1: Output Feature Map (64x64)
        ax_out = axes[idx, 1]
        im_out = ax_out.imshow(fmap, cmap="viridis")
        ax_out.set_title(f"Feature Map Output ({fmap.shape[0]} x {fmap.shape[1]})\nMin: {meta['min_value']:.2f} | Max: {meta['max_value']:.2f}")
        ax_out.set_xticks([])
        ax_out.set_yticks([])
        fig.colorbar(im_out, ax=ax_out, fraction=0.046, pad=0.04)

        # Col 2: Numerical Calculation & Worked Example
        ax_calc = axes[idx, 2]
        ax_calc.axis("off")

        p = meta["sample_patch"]
        k = meta["sample_kernel"]
        prod = meta["sample_products"]
        res = meta["sample_output"]

        explanation_text = (
            f"OPERATION: 2D Cross-Correlation\n"
            f"Padding: 'same' (p=1) | Stride: s=1\n"
            f"Output Size: floor((64 + 2(1) - 3)/1) + 1 = 64\n\n"
            f"WORKED 3x3 PATCH CALCULATION at center {meta['sample_coord']}:\n"
            f"Input Patch P:\n"
            f" [{p[0,0]:.2f}  {p[0,1]:.2f}  {p[0,2]:.2f}]\n"
            f" [{p[1,0]:.2f}  {p[1,1]:.2f}  {p[1,2]:.2f}]\n"
            f" [{p[2,0]:.2f}  {p[2,1]:.2f}  {p[2,2]:.2f}]\n\n"
            f"Kernel K:\n"
            f" [{k[0,0]:.2f}  {k[0,1]:.2f}  {k[0,2]:.2f}]\n"
            f" [{k[1,0]:.2f}  {k[1,1]:.2f}  {k[1,2]:.2f}]\n"
            f" [{k[2,0]:.2f}  {k[2,1]:.2f}  {k[2,2]:.2f}]\n\n"
            f"Output(i,j) = sum(P * K) = {res:.4f}\n"
            f"Role: {data['explanation']}"
        )
        ax_calc.text(
            0.05, 0.5, explanation_text,
            ha="left", va="center", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1")
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVED] Figure C saved to: '{save_path}'")
    return fig


def plot_figure_d_pooling(
    resized_input: np.ndarray,
    save_path: str = None
):
    """
    Figure D: Convolution Feature Map versus Max-Pooled Output.
    Includes visual pooling grid boundaries and numerical worked example.
    """
    # Use the Edge Detection feature map for dramatic visualization
    edge_kernel = DEMO_KERNELS["Edge Detection"]["matrix"]
    fmap, _ = apply_2d_convolution(resized_input, edge_kernel, padding="same", stride=1)
    pooled, pool_meta = apply_max_pooling_2d(fmap, pool_size=(2, 2), stride=2)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Visual Step D: Max Pooling Operation (Before vs. After)", fontsize=14, fontweight="bold")

    # 1. Feature Map Before Pooling (64x64)
    im0 = axes[0].imshow(fmap, cmap="viridis")
    axes[0].set_title(f"1. Before Max Pooling\nFeature Map: {fmap.shape[0]} x {fmap.shape[1]}")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Zoomed 8x8 Patch with 2x2 Pooling Grid Boundaries
    zoom_r, zoom_c = 28, 28
    zoom_patch = fmap[zoom_r:zoom_r + 8, zoom_c:zoom_c + 8]
    im1 = axes[1].imshow(zoom_patch, cmap="viridis")
    axes[1].set_title("2. Zoomed 8x8 Region\nShowing 2x2 Pooling Grids")
    axes[1].set_xlabel("Local X")
    axes[1].set_ylabel("Local Y")

    # Draw 2x2 grid lines over the zoomed region
    for r in range(0, 8, 2):
        for c in range(0, 8, 2):
            rect = patches.Rectangle((c - 0.5, r - 0.5), 2, 2, linewidth=1.5, edgecolor="red", facecolor="none")
            axes[1].add_patch(rect)
            # Find and mark local maximum
            cell = zoom_patch[r:r + 2, c:c + 2]
            max_idx = np.unravel_index(np.argmax(cell), cell.shape)
            axes[1].plot(c + max_idx[1], r + max_idx[0], "wo", markersize=4)

    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Pooled Feature Map (32x32)
    im2 = axes[2].imshow(pooled, cmap="viridis")
    axes[2].set_title(f"3. After Max Pooling (2x2, s=2)\nPooled Map: {pooled.shape[0]} x {pooled.shape[1]}")
    axes[2].set_xlabel("Width")
    axes[2].set_ylabel("Height")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Numerical Explanation & Worked Example
    axes[3].axis("off")
    sample_win = pool_meta["sample_window"]
    sample_m = pool_meta["sample_max"]

    explanation = (
        f"MAX POOLING MATHEMATICS:\n"
        f"Window: 2 x 2  |  Stride: 2\n"
        f"Output Size Formula:\n"
        f" Output = floor((64 - 2)/2) + 1 = 32\n\n"
        f"WORKED NUMERICAL EXAMPLE:\n"
        f"Example 2x2 Region:\n"
        f" [[ 1.0,  5.0 ],\n"
        f"  [ 3.0,  2.0 ]]\n"
        f" Max Value Selected = 5.0\n\n"
        f"ACTUAL REGION from Image:\n"
        f" [[ {sample_win[0,0]:.2f},  {sample_win[0,1]:.2f} ],\n"
        f"  [ {sample_win[1,0]:.2f},  {sample_win[1,1]:.2f} ]]\n"
        f" Max Value Selected = {sample_m:.2f}\n\n"
        f"PURPOSE:\n"
        f"- Downsamples spatial resolution (75% reduction).\n"
        f"- Retains strongest local activations.\n"
        f"- Introduces translation invariance."
    )
    axes[3].text(
        0.05, 0.5, explanation,
        ha="left", va="center", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1")
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVED] Figure D saved to: '{save_path}'")
    return fig


def plot_figure_e_trained_cnn_activations(
    model: keras.Model,
    input_tensor: np.ndarray,
    save_path: str = None
):
    """
    Figure E: Extracts and displays intermediate feature maps from all
    Conv2D and MaxPooling2D layers of the loaded trained CNN.
    """
    if model is None:
        print("[WARNING] No trained model provided for intermediate activation visualization.")
        return None

    # Identify Conv2D and MaxPooling2D layers
    target_layers = [
        layer for layer in model.layers
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.MaxPooling2D))
    ]

    if not target_layers:
        print("[WARNING] No Conv2D or MaxPooling2D layers found in model.")
        return None

    # Construct multi-output activation model
    extractor = keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for layer in target_layers]
    )
    activations = extractor.predict(input_tensor, verbose=0)

    # Number of channels to display per layer
    num_channels_to_display = 6
    num_layers = len(target_layers)

    fig, axes = plt.subplots(num_layers, num_channels_to_display + 1, figsize=(18, 2.8 * num_layers))
    fig.suptitle("Visual Step E: Actual Trained CNN Intermediate Activations (Intermediate Feature Maps)", fontsize=14, fontweight="bold")

    # If only 1 layer, ensure 2D array of axes
    if num_layers == 1:
        axes = np.expand_dims(axes, axis=0)

    for l_idx, (layer, act) in enumerate(zip(target_layers, activations)):
        layer_name = layer.name
        layer_type = layer.__class__.__name__
        act_tensor = act[0]  # Remove batch dimension: shape (H, W, Channels)
        h, w, total_channels = act_tensor.shape

        # First column: Layer information banner
        ax_info = axes[l_idx, 0]
        ax_info.axis("off")
        info_text = (
            f"Layer: {layer_name}\n"
            f"Type: {layer_type}\n"
            f"Shape: {h} x {w} x {total_channels}\n"
            f"Showing {num_channels_to_display}/{total_channels} channels"
        )
        ax_info.text(
            0.5, 0.5, info_text,
            ha="center", va="center", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#e2e8f0", edgecolor="#475569")
        )

        # Subsequent columns: Feature map channels
        for c_idx in range(num_channels_to_display):
            ax_ch = axes[l_idx, c_idx + 1]
            if c_idx < total_channels:
                ch_map = act_tensor[:, :, c_idx]
                im = ax_ch.imshow(ch_map, cmap="viridis")
                ax_ch.set_title(f"Ch {c_idx} ({h}x{w})", fontsize=8.5)
                ax_ch.set_xticks([])
                ax_ch.set_yticks([])
                fig.colorbar(im, ax=ax_ch, fraction=0.046, pad=0.04)
            else:
                ax_ch.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVED] Figure E saved to: '{save_path}'")
    return fig


def plot_figure_f_learned_filters(
    model: keras.Model,
    save_path: str = None
):
    """
    Figure F: Visualizes real learned filter weights from the trained CNN's
    first Conv2D layer as numerical heatmaps, contrasting them with manual demonstration kernels.
    """
    if model is None:
        return None

    # Find first Conv2D layer
    conv_layer = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            conv_layer = layer
            break

    if conv_layer is None:
        return None

    weights, biases = conv_layer.get_weights()  # shape (Kh, Kw, InChannels, OutChannels)
    kh, kw, in_c, out_c = weights.shape

    # Select up to 8 learned filters
    num_display = min(8, out_c)
    fig, axes = plt.subplots(2, num_display, figsize=(2.2 * num_display, 5))
    fig.suptitle(f"Visual Step F: Real Learned Filter Weights from Layer '{conv_layer.name}' (Kh={kh}, Kw={kw})", fontsize=13, fontweight="bold")

    for f_idx in range(num_display):
        # We assume 1 input channel (grayscale)
        filter_matrix = weights[:, :, 0, f_idx]

        # Top row: Learned Filter Heatmap
        ax_hm = axes[0, f_idx]
        vbound = max(abs(filter_matrix.min()), abs(filter_matrix.max()), 0.1)
        im = ax_hm.imshow(filter_matrix, cmap="coolwarm", vmin=-vbound, vmax=vbound)
        ax_hm.set_title(f"Learned Filter {f_idx}", fontsize=9, fontweight="bold")
        ax_hm.set_xticks(range(kw))
        ax_hm.set_yticks(range(kh))

        # Annotate values
        for r in range(kh):
            for c in range(kw):
                val = filter_matrix[r, c]
                color = "white" if abs(val) > vbound * 0.5 else "black"
                ax_hm.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7.5)

        # Bottom row: Numerical text
        ax_txt = axes[1, f_idx]
        ax_txt.axis("off")
        bias_val = biases[f_idx] if f_idx < len(biases) else 0.0
        txt = (
            f"Filter #{f_idx}\n"
            f"Bias: {bias_val:+.3f}\n"
            f"Min:  {filter_matrix.min():.2f}\n"
            f"Max:  {filter_matrix.max():.2f}\n\n"
            f"[LEARNED FILTER]\n"
            f"Optimized via SGD/\n"
            f"Backpropagation"
        )
        ax_txt.text(
            0.5, 0.5, txt,
            ha="center", va="center", fontsize=7.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1f5f9", edgecolor="#cbd5e1")
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVED] Figure F saved to: '{save_path}'")
    return fig


# =====================================================================
# 5. Core Pipeline Orchestrator
# =====================================================================
def run_all_visualizations(
    gray_image: np.ndarray,
    resized_input: np.ndarray,
    input_tensor: np.ndarray,
    model: keras.Model = None,
    save_dir: str = "visualizations",
    show_plots: bool = True
):
    """
    Coordinates and generates all visual demonstration figures:
      Figure A: Captured grayscale vs 64x64 CNN input.
      Figure B: Demonstration convolution kernels.
      Figure C: Convolution operation (before & after with worked examples).
      Figure D: Max pooling (before & after with 2x2 grid lines).
      Figure E: Intermediate activations from trained CNN layers.
      Figure F: Learned filter weights from trained Conv2D layer.
    """
    print("\n" + "=" * 65)
    print("      LAUNCHING CNN VISUAL DEMONSTRATION & ACTIVATION ANALYSIS")
    print("=" * 65)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"[INFO] Saving visualization figures to directory: '{save_dir}'")

    path_a = os.path.join(save_dir, "fig_a_image_and_cnn_input.png") if save_dir else None
    path_b = os.path.join(save_dir, "fig_b_demo_kernels.png") if save_dir else None
    path_c = os.path.join(save_dir, "fig_c_convolutions_before_after.png") if save_dir else None
    path_d = os.path.join(save_dir, "fig_d_max_pooling_before_after.png") if save_dir else None
    path_e = os.path.join(save_dir, "fig_e_trained_cnn_activations.png") if save_dir else None
    path_f = os.path.join(save_dir, "fig_f_learned_filter_weights.png") if save_dir else None

    # Step A: Original vs Resized CNN Input
    print("[1/5] Generating Figure A: Captured image & CNN input tensor...")
    plot_figure_a_inputs(gray_image, resized_input, save_path=path_a)

    # Step B: Demonstration Kernels
    print("[2/5] Generating Figure B: Demonstration kernels (matrices & heatmaps)...")
    plot_figure_b_kernels(save_path=path_b)

    # Step C: Convolution Before and After
    print("[3/5] Generating Figure C: Convolution operations & worked examples...")
    plot_figure_c_convolutions(resized_input, save_path=path_c)

    # Step D: Max Pooling Before and After
    print("[4/5] Generating Figure D: Max pooling operations & 2x2 grid overlay...")
    plot_figure_d_pooling(resized_input, save_path=path_d)

    # Step E & F: Trained CNN Activations & Learned Weights
    if model is not None:
        print("[5/5] Generating Figures E & F: Trained CNN layer activations & learned filters...")
        plot_figure_e_trained_cnn_activations(model, input_tensor, save_path=path_e)
        plot_figure_f_learned_filters(model, save_path=path_f)
    else:
        print("\n[NOTE] No trained CNN model provided.")
        print("  Figures E & F (intermediate activations & learned weights) were skipped.")
        print("  Demonstration kernels are displayed above to illustrate mathematical operations.")

    print("\n" + "=" * 65)
    print(" Visualization generation completed successfully.")
    print("=" * 65)

    if show_plots:
        print("Displaying interactive visualization figures. Close windows to continue...")
        plt.show()


# =====================================================================
# 6. Standalone CLI Entry Point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="CNN Visual Demonstration & Activation Analysis.")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file (e.g. captured_image.jpg)")
    parser.add_argument("--model", type=str, default="trained_model.keras", help="Path to trained Keras model file")
    parser.add_argument("--save-dir", type=str, default="visualizations", help="Directory to save figure PNGs")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index if capturing live")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows (saves to disk)")
    args = parser.parse_args()

    # Import helpers from camera_cnn
    from camera_cnn import capture_image_from_webcam, convert_to_grayscale, preprocess_image_for_cnn, load_or_initialize_model

    # Determine input image
    gray_image = None
    if args.image and os.path.exists(args.image):
        print(f"[INFO] Loading image from '{args.image}'...")
        loaded = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if loaded is not None:
            gray_image = loaded
        else:
            print(f"[WARNING] Could not read '{args.image}' as grayscale. Trying color read...")
            bgr = cv2.imread(args.image)
            if bgr is not None:
                gray_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # If no image provided, check default captured_image.jpg
    if gray_image is None and os.path.exists("captured_image.jpg"):
        print("[INFO] Found existing 'captured_image.jpg'. Loading for visualization...")
        gray_image = cv2.imread("captured_image.jpg", cv2.IMREAD_GRAYSCALE)

    # If still no image, capture from webcam if not headless
    if gray_image is None and not args.headless:
        print("[INFO] No existing image found. Opening webcam to capture frame...")
        captured = capture_image_from_webcam(camera_index=args.camera_index)
        if captured is not None:
            gray_image = convert_to_grayscale(captured, output_path="captured_image.jpg", show_window=False)

    # If still no image (e.g., headless or no webcam), generate synthetic pattern
    if gray_image is None:
        print("[INFO] Creating synthetic test pattern for demonstration...")
        gray_image = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(gray_image, (320, 240), 120, 255, -1)
        cv2.rectangle(gray_image, (150, 100), (250, 200), 180, -1)

    # Preprocess for CNN
    input_tensor, resized_input = preprocess_image_for_cnn(gray_image, target_size=(64, 64))

    # Load trained model if available
    model = None
    if args.model and os.path.exists(args.model):
        try:
            model, _ = load_or_initialize_model(model_path=args.model)
        except Exception as e:
            print(f"[WARNING] Could not load model from '{args.model}': {e}")
    elif os.path.exists("model.keras"):
        model, _ = load_or_initialize_model(model_path="model.keras")

    run_all_visualizations(
        gray_image=gray_image,
        resized_input=resized_input,
        input_tensor=input_tensor,
        model=model,
        save_dir=args.save_dir,
        show_plots=not args.headless
    )


if __name__ == "__main__":
    main()
