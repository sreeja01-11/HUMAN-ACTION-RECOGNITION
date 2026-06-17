# Real-Time Human Action Recognition (UCF101 & R3D-18)

This project implements a deep learning pipeline to recognize human actions from video sequences in real-time. It uses a 3D ResNet-18 (R3D-18) architecture trained on the UCF101 dataset, capable of identifying 101 different activities ranging from sports to daily household chores.

##  Project Features

* **Real-Time Dashboard:** A custom OpenCV-based interface that displays live webcam feed alongside prediction analytics.
* **Top-3 Predictions:** Visualizes the most likely actions with dynamic confidence bars.
* **Temporal Smoothing:** Uses a deque-based voting system to prevent flickering and ensure stable action detection.
* **Action History:** Keeps track of recently detected activities in a scrolling log.
* **High Performance:** Optimized for GPU inference with color-coded confidence indicators.
* **Preprocessing Pipeline:** Automatic frame sampling, resizing, and normalization for 3D temporal data.
* **Evaluation Pipeline:** Includes a dedicated evaluation script for measuring validation accuracy, F1-scores, and per-class performance.

---

##  Project Structure

```text
UCF101_ACTION_RECOGNITION/
├── notebook/
│   └── webcam.ipynb          # Kaggle/Jupyter notebook for training and fine-tuning
├── class_names.json          # Mapping of class indices to human-readable names
├── dashboard.py              # Main script to run real-time webcam inference
├── data_loader.py            # Logic for loading and splitting the UCF101 dataset
├── dataset.py                # PyTorch Dataset class for video processing
├── evaluate.py               # Script to generate F1-scores and accuracy reports
├── evaluation_results.json   # Validation metrics and class-wise evaluation results
├── preprocess.py             # Video processing (sampling, resizing, normalization)
├── r3d_model.py              # Model architecture definition (Torchvision R3D-18)
├── requirements.txt          # Python dependencies
└── .gitignore                # Ignore large files such as model weights
```

---

##  Workflow & Technical Implementation

### 1. Training (The Notebook Phase)

The model is trained using Transfer Learning. A pre-trained `r3d_18` model (trained on Kinetics-400) is fine-tuned on the UCF101 dataset by replacing the final classification layer with a 101-class output layer. Using pretrained spatiotemporal features significantly reduces training time and improves generalization across diverse human actions.

* **Input:** Video clips are sampled into 16-frame sequences.
* **Processing:** Frames are resized to 112x112 and normalized.
* **Optimization:** Trained using Adam optimizer with a Learning Rate Scheduler.

### 2. Inference (The Dashboard Phase)

When running `dashboard.py`:

* **Buffer Management:** The system captures live frames from the webcam and maintains a sliding window (buffer) of the last 16 frames.
* **Prediction:** Every 8 frames, the buffer is sent to the R3D model.
* **Smoothing:** The script uses a `SMOOTH_WINDOW` to look at recent predictions and picks the "Most Common" action to avoid erratic jumping between classes.
* **UI Rendering:** OpenCV draws a side panel showing FPS, device info (CPU/GPU), Top-3 bars, and a historical log.

### 3. Evaluation (The Validation Phase)

The project includes a dedicated evaluation pipeline through `evaluate.py`.

The evaluation script:

* Loads the trained model (`best_model.pth`).
* Runs inference on validation video clips.
* Calculates Overall Validation Accuracy.
* Generates Per-Class Precision, Recall, and F1-Scores.
* Computes Macro Average and Weighted Average F1 Scores.
* Identifies the Best 5 and Worst 5 Performing Action Classes.
* Exports detailed results to `evaluation_results.json` for further analysis.

---

##  Use Cases

* **Smart Surveillance:** Detecting suspicious activities or falls in public spaces.
* **Sports Analytics:** Identifying specific moves in gymnastics, cricket, or basketball.
* **Healthcare:** Monitoring patient movements or physical therapy exercises.
* **Human-Computer Interaction:** Using gestures/actions to control software interfaces.

---

##  Local Setup & Usage

### 1. Prerequisites

* Python 3.8 or higher.
* A Webcam.
* (Optional) NVIDIA GPU with CUDA for smoother real-time performance.

### 2. Installation

Clone your project folder to your local machine and install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Path Configuration

Open `dashboard.py` and update the paths to match your local directory where you saved the model and JSON file:

```python
MODEL_PATH = "best_model.pth"
CLASS_NAMES_PATH = "class_names.json"
```

### 4. Running the Dashboard

Execute the main dashboard script:

```bash
python dashboard.py
```

### 5. Running Evaluation

Execute the evaluation script:

```bash
python evaluate.py
```

### 6. Controls

**Q** or **Esc**: Quit the application.

**Confidence Colors:**

* 🟢 Green: High Confidence (>70%)
* 🟠 Orange: Moderate Confidence (40% - 70%)
* 🔴 Red: Low Confidence (<40%)

---

##  Evaluation Results

The trained R3D-18 model was evaluated on the UCF101 validation set containing **1,673 video clips**.

### Performance Metrics

| Metric              | Score           |
| ------------------- | --------------- |
| Validation Accuracy | **97.91%**      |
| Macro F1 Score      | **0.9817**      |
| Weighted F1 Score   | **0.9790**      |
| Correct Predictions | **1638 / 1673** |

## Note on Dataset and Model Weights

The UCF101 dataset and trained model weights (`best_model.pth`) are not included in this repository due to size constraints.

### Dataset Setup

To run training, evaluation, or real-time inference, users must first download the UCF101 dataset and organize it into the following structure:

```text
UCF101/
├── train/
├── val/
├── test/
├── train.csv
├── val.csv
└── test.csv
```

The dataset should contain all 101 action classes from the UCF101 benchmark.

### Model Weights

The trained model weights (`best_model.pth`) are not included because GitHub enforces a 100 MB file size limit, while the trained R3D-18 model exceeds this limit.

To reproduce the results:

1. Download and prepare the UCF101 dataset.
2. Run the training notebook (`webcam.ipynb`) to train the model.
3. Save the generated weights as `best_model.pth`.
4. Place `best_model.pth` in the project root directory.
5. Run `dashboard.py` for real-time inference or `evaluate.py` for validation metrics.

The complete evaluation summary is available in `evaluation_results.json`.


---

##  Technologies Used

* **Language:** Python
* **Framework:** PyTorch
* **Computer Vision:** OpenCV
* **Deep Learning Model:** R3D-18 (3D ResNet-18)
* **Dataset:** UCF101 (101 Action Classes)
* **Data Processing:** NumPy, Pandas
* **Evaluation:** Scikit-learn
* **Notebook Environment:** Jupyter Notebook / Kaggle

---

##  Key Highlights

* Real-Time Human Action Recognition
* Transfer Learning using R3D-18
* 101-Class Action Classification
* Live Webcam Inference
* Top-3 Prediction Visualization
* Temporal Prediction Smoothing
* GPU-Accelerated Inference
* Comprehensive Evaluation Pipeline
* Interactive OpenCV Dashboard