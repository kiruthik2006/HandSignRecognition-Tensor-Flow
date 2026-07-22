# Hand Sign Recognition with TensorFlow

A machine learning project for recognizing hand gestures and signs using MediaPipe hand landmarks and TensorFlow neural networks. This project processes hand gesture images, extracts landmark data, and trains models for real-time gesture classification.

## Project Overview

This repository implements an end-to-end hand gesture recognition system that:
- Extracts hand landmarks from images using MediaPipe
- Processes landmark data into a training dataset
- Trains TensorFlow/Keras models for gesture classification
- Exports models to TensorFlow Lite format for deployment

## Directory Structure

```
.
├── Alpha+Gesture(F)/          # Alpha gesture dataset and related files
├── Model_stage/               # Model development and staging files
├── temp_localsend/            # Temporary files
├── gesture_landmarks.csv      # Processed hand landmark data (~60MB)
├── process.py                 # Data processing script
├── train_model.py             # Primary training script
├── trainer_v2.py              # Alternative training script
├── trainmodel2handto1.py       # Single/dual hand model trainer
├── npy_to_txt.py              # Utility to convert NumPy arrays to text
└── README.md                  # This file
```

## Key Files

### process.py
Extracts hand landmarks from gesture images and converts them to CSV format.

**Input:** Image dataset organized as `gestures/gesture_name/image.jpg`

**Output:** `gesture_landmarks.csv` containing 63 landmark coordinates (21 landmarks × 3 dimensions: x, y, z) and gesture labels

**Features:**
- Uses MediaPipe Hand solution for landmark detection
- Processes only the first detected hand per image
- Multi-threaded processing for performance (8 workers)

### train_model.py
Trains a neural network classifier on the extracted landmark data.

**Model Architecture:**
- Input layer: 63 features (hand landmarks)
- Dense layer: 128 units, ReLU activation
- Dropout: 30% regularization
- Dense layer: 64 units, ReLU activation
- Output layer: Softmax classification

**Outputs:**
- `gesture_model.tflite`: TensorFlow Lite model for deployment
- `labels.txt`: List of recognized gesture labels

**Training:**
- 30 epochs, batch size 16
- 80-20 train-test split
- Adam optimizer, categorical cross-entropy loss

### trainer_v2.py
Alternative training script with a different architecture for models with labeled handedness data.

**Input:** `HandLandmarks_Labeled_Corrected.csv` with columns: label, handedness, and 63 landmark coordinates

**Model Architecture:**
- Input layer: 63 features
- Dense layer: 64 units, ReLU activation
- Dense layer: 32 units, ReLU activation
- Output layer: Softmax classification

**Training:**
- 30 epochs with 10% validation split
- Sparse categorical cross-entropy loss

### Other Scripts

- **trainmodel2handto1.py**: Specialized training for single-hand to multi-hand models
- **npy_to_txt.py**: Converts NumPy array files (.npy) to text format

## Getting Started

### Prerequisites

```bash
pip install tensorflow
pip install pandas
pip install scikit-learn
pip install mediapipe
pip install opencv-python
pip install numpy
```

### Workflow

1. **Prepare Dataset:**
   - Organize gesture images in `gestures/` directory
   - Structure: `gestures/gesture_name/image_1.jpg`, `gestures/gesture_name/image_2.jpg`, etc.

2. **Extract Landmarks:**
   ```bash
   python process.py
   ```
   This generates `gesture_landmarks.csv`

3. **Train Model:**
   ```bash
   python train_model.py
   ```
   Or use the alternative trainer:
   ```bash
   python trainer_v2.py
   ```

4. **Use the Model:**
   - The generated `gesture_model.tflite` can be deployed on mobile or edge devices
   - `labels.txt` contains the mapping of model outputs to gesture names

## Data Format

The gesture landmarks CSV contains:
- First 63 columns: Hand landmark coordinates (21 landmarks × 3 dimensions)
  - X: Horizontal position (0-1 normalized)
  - Y: Vertical position (0-1 normalized)
  - Z: Depth (0-1 normalized)
- Last column: Gesture label (e.g., "palm", "fist", etc.)

Example:
```
0.123,0.456,0.789,0.234,...(60 more values),palm
0.111,0.222,0.333,0.444,...(60 more values),fist
```

## Model Output

### gesture_model.tflite
TensorFlow Lite model suitable for deployment on:
- Mobile devices (iOS, Android)
- Edge devices (Raspberry Pi, Jetson)
- Web browsers (with TensorFlow.js)

### labels.txt
Text file with one gesture label per line, ordered by model output class index.

## Notes

- The project supports multi-threaded processing for efficient landmark extraction
- Models are optimized for TensorFlow Lite deployment
- Hand landmark detection is performed using MediaPipe's pre-trained hand detector
- Only the first detected hand is processed per image

## Future Enhancements

- Real-time video gesture recognition
- Support for multi-hand recognition
- Model optimization for faster inference
- Deployment examples for mobile platforms
- Data augmentation techniques
- Confidence scoring for predictions

## License

This project is provided as-is for educational and research purposes.
