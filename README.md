# Webcam CNN Image Processing & Inference Project

A complete, end-to-end Python project that uses OpenCV to capture an image from your computer's webcam, converts it to genuine grayscale, preprocesses it into a normalized 4D tensor, and feeds it into a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

---

## Project Structure

```text
webcam_cnn_project/
│
├── camera_cnn.py          # Main application: webcam preview, capture, grayscale, preprocessing, CNN & --visualize
├── cnn_visualization.py   # Educational visualization engine: kernels, convolutions, pooling, activations & filters
├── train_model.py         # Model training script (supports synthetic shapes, MNIST, or Fashion-MNIST)
├── test_pipeline.py       # Automated unit & integration tests (runs headless without camera)
├── requirements.txt       # Python dependencies (OpenCV, NumPy, Matplotlib, TensorFlow)
├── visualizations/        # Generated figure exports (Figures A through F)
└── README.md              # Documentation and execution instructions
```

---

## Features

1. **Live Camera Input & Capture**:
   - Accesses computer webcam via OpenCV (`cv2.VideoCapture`).
   - Displays real-time live preview with on-screen HUD instructions.
   - Press **`SPACE`** to capture a frame.
   - Press **`Q`** or **`ESC`** to exit cleanly without capturing.
   - Automatically closes preview and releases camera hardware upon capture.

2. **Genuine Grayscale Conversion**:
   - Transforms color frame (`BGR`, 3 channels) into genuine 1-channel grayscale (`cv2.COLOR_BGR2GRAY`).
   - Confirms true single-channel 2D matrix structure (`ndim == 2`).
   - Automatically saves the image as `captured_image.jpg`.
   - Displays grayscale image in a dedicated window.

3. **CNN Preprocessing Pipeline**:
   - Resizes grayscale image to target CNN resolution ($64 \times 64$ pixels) using area interpolation.
   - Normalizes pixel values from $[0, 255]$ integer range to $[0.0, 1.0]$ floating-point range.
   - Expands batch and channel dimensions to shape `(1, 64, 64, 1)` (Channels-Last format).

4. **Convolutional Neural Network Architecture**:
   - **Input Layer**: `(64, 64, 1)`
   - **Conv2D Block 1**: 32 filters ($3 \times 3$), ReLU activation, `MaxPooling2D` ($2 \times 2$)
   - **Conv2D Block 2**: 64 filters ($3 \times 3$), ReLU activation, `MaxPooling2D` ($2 \times 2$)
   - **Conv2D Block 3**: 128 filters ($3 \times 3$), ReLU activation, `MaxPooling2D` ($2 \times 2$)
   - **Dense Head**: `Flatten` $\rightarrow$ `Dense(128, relu)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(num_classes, softmax)`

5. **Honest Evaluation & Prediction Handling**:
   - Checks if a trained model (`model.keras`) exists.
   - **If trained model exists**: loads weights and displays predicted class label, confidence percentage, and probability distribution.
   - **If no trained model exists**: clearly explains that predictions are unavailable until trained. Explicitly notes that random weights produce mathematical noise, not real predictions.

---

## Installation

### Prerequisites
- Python 3.9 through 3.13
- A functional webcam or external camera

### Setup Instructions
1. Open PowerShell or Command Prompt.
2. Navigate to the project directory:
   ```powershell
   cd C:\Users\Acer\.gemini\antigravity\scratch\webcam_cnn_project
   ```
3. Install required packages:
   ```powershell
   pip install -r requirements.txt
   ```

---

## How to Run

### Step 1: Run the Webcam CNN Program
```powershell
python camera_cnn.py
```
1. A live webcam window opens.
2. Frame your subject and press **`SPACE`** to take the photo (or **`Q`** to quit).
3. The captured genuine grayscale image appears in a preview window. Press any key to continue.
4. The terminal prints preprocessing metrics (resized dimensions, normalization range, 4D tensor shape).
5. The model processes the tensor and outputs status and predictions.
6. A preview window displays the preprocessed $64 \times 64$ image enlarged for visual inspection.

> [!NOTE]
> By default, `camera_cnn.py` looks for `model.keras`. If it does not exist, the program cleanly initializes the CNN architecture, tests the forward pass, and explains that predictions are unavailable until trained (without falsely treating random weights as predictions).

---

### Step 2: Test Real Predictions Immediately
A pre-trained model on geometric shapes (`Circle`, `Rectangle`, `Triangle`, `Cross`) is already included:
```powershell
python camera_cnn.py --model trained_model.keras --labels trained_labels.txt
```
Or simply copy it to make it the default:
```powershell
copy trained_model.keras model.keras
copy trained_labels.txt labels.txt
python camera_cnn.py
```

---

### Step 3: (Optional) Train the CNN on Your Own or Standard Datasets
To train fresh weights from scratch:
```powershell
# Fast, offline training on geometric shapes (Circles, Squares, Triangles, Crosses)
python train_model.py --mode synthetic --epochs 5

# Or train on Fashion-MNIST (10 clothing items)
python train_model.py --mode fashion_mnist --epochs 5

# Or train on MNIST (handwritten digits 0-9)
python train_model.py --mode mnist --epochs 5
```
This saves `model.keras` and `labels.txt`.

---

### Step 4: CNN Visual Demonstrations & Activation Analysis
To visually demonstrate how the CNN and its filters process your image, run with `--visualize`:

#### A. Run Visualizations on Live Webcam
```powershell
python camera_cnn.py --model trained_model.keras --labels trained_labels.txt --visualize
```
1. Captures a photo from your webcam via OpenCV.
2. Converts to grayscale and preprocesses the image to $64 \times 64$.
3. Evaluates CNN inference.
4. Generates and displays interactive Matplotlib visual demonstration figures:
   - **Figure A**: Captured grayscale image vs. resized $64 \times 64$ CNN input.
   - **Figure B**: Numerical matrices and heatmaps for 4 demonstration kernels (Identity, Edge Detection, Sharpening, Gaussian Blur).
   - **Figure C**: 2D Convolution operations before & after with worked patch dot-product calculations.
   - **Figure D**: Max pooling before & after with $2 \times 2$ grid overlays and numerical maximum selection.
   - **Figure E**: Actual intermediate feature maps from all `Conv2D` and `MaxPooling2D` layers of the trained model.
   - **Figure F**: Real learned filter weights extracted from the trained model's convolutional layers.

#### B. Run Visualizations Directly on `captured_image.jpg` (No Webcam Required)
```powershell
# Using camera_cnn.py:
python camera_cnn.py --image captured_image.jpg --model trained_model.keras --labels trained_labels.txt --visualize

# Or using the dedicated visualization script:
python cnn_visualization.py --image captured_image.jpg --model trained_model.keras
```

---

### CNN Mathematics & Operation Explanations

#### 1. Convolution / Cross-Correlation
$$\text{Output}(i,j) = \sum_{m} \sum_{n} \text{Input}(i+m, j+n) \times \text{Kernel}(m,n)$$
- **Output Size Formula**:
  $$\text{Output size} = \left\lfloor \frac{\text{Input size} + 2 \times \text{Padding} - \text{Kernel size}}{\text{Stride}} \right\rfloor + 1$$
- **Example**: Input size $64 \times 64$, Kernel $3 \times 3$, Padding $1$ (`same`), Stride $1$:
  $$\text{Output size} = \left\lfloor \frac{64 + 2(1) - 3}{1} \right\rfloor + 1 = 64$$
- **Cross-Correlation vs. Convolution**: Modern deep learning frameworks (TensorFlow, PyTorch) and OpenCV compute cross-correlation (without kernel flipping) because filter weights are learned freely through backpropagation.
- **Representation Change**: Convolutions extract spatial features (edges, corners, textures, shapes).

#### 2. Max Pooling
$$\text{Output}(i,j) = \max_{(m,n) \in \text{window}} \text{Input}(2i+m, 2j+n)$$
- **Output Size Formula**:
  $$\text{Output size} = \left\lfloor \frac{\text{Input size} - \text{Pool size}}{\text{Stride}} \right\rfloor + 1 = \left\lfloor \frac{64 - 2}{2} \right\rfloor + 1 = 32$$
- **Worked Example**: For a $2 \times 2$ region:
  $$\begin{bmatrix} 1 & 5 \\ 3 & 2 \end{bmatrix} \implies \max(1, 5, 3, 2) = 5$$
- **Representation Change**: Reduces spatial dimensions by 75%, retaining dominant feature responses while conferring translation invariance.

---

### Command-Line Arguments Reference
`camera_cnn.py` supports:
- `--camera-index <int>`: Select webcam index (default: 0).
- `--image <path>`: Process an existing image file instead of opening the webcam.
- `--model <path>`: Path to `.keras` model file (default: `model.keras`).
- `--labels <path>`: Path to `.txt` labels file (default: `labels.txt`).
- `--visualize`: Launch educational visual demonstration figures.
- `--save-vis-dir <dir>`: Directory where visualization PNGs are saved (default: `visualizations`).
- `--headless`: Run headlessly without opening GUI preview windows.

---

## Running Automated Pipeline & Visualization Tests
To verify all image processing, tensor manipulations, demonstration convolutions, pooling logic, and intermediate activation extractions:
```powershell
python test_pipeline.py
```
