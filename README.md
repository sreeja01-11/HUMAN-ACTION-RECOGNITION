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
├── preprocess.py             # Video processing (sampling, resizing, normalization)
├── r3d_model.py              # Model architecture definition (Torchvision R3D-18)
├── requirements.txt          # Python dependencies
```

---

##  Workflow & Technical Implementation

### 1. Training (The Notebook Phase)

The model is trained using Transfer Learning. We take a pre-trained `r3d_18` model (trained on Kinetics-400) and replace the final fully connected layer to match the 101 classes of UCF101.

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

### Best Performing Classes

| Action Class   | F1 Score |
| -------------- | -------- |
| ApplyEyeMakeup | 1.00     |
| ApplyLipstick  | 1.00     |
| Archery        | 1.00     |
| BabyCrawling   | 1.00     |
| BaseballPitch  | 1.00     |

### Most Challenging Classes

| Action Class      | F1 Score |
| ----------------- | -------- |
| BasketballDunk    | 0.65     |
| Basketball        | 0.67     |
| RopeClimbing      | 0.90     |
| VolleyballSpiking | 0.90     |
| Punch             | 0.95     |

The lower scores for Basketball and BasketballDunk indicate that visually similar actions remain challenging to distinguish, highlighting the importance of temporal feature learning in video understanding.

> **Note:** The UCF101 dataset and pretrained model weights (`best_model.pth`) are not included in this repository due to size constraints. Users must download the dataset separately and train the model or provide their own trained weights before running evaluation.


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