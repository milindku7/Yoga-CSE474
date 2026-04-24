# 🧘 Yoga Pose Corrector

Welcome to your personal AI yoga instructor! [cite_start]This project uses computer vision and machine learning to analyze your yoga poses in real-time, providing instant feedback on your alignment and form. [cite: 1, 2]

---

## 🚀 Overview

[cite_start]This application leverages the **YOLO11-pose** model to detect 17 human keypoints and a custom-trained classifier to identify specific yoga poses. [cite: 1, 3] [cite_start]Beyond just identification, the system calculates joint angles and compares them against "ideal" reference data to offer specific, actionable corrections. [cite: 2]

### Key Features
* [cite_start]**Real-time Pose Classification**: Recognizes poses like Tree, Sitting, Wind, and Toe. [cite: 4]
* [cite_start]**Alignment Feedback**: Visual overlays (circles on joints) highlight where your form needs adjustment. [cite: 1, 2]
* [cite_start]**Voice/Text Hints**: Provides specific cues like "Straighten your left leg more" or "Lower your right arm." [cite: 2]
* [cite_start]**Performance Scoring**: Calculates a real-time "form score" from 0-100 based on your precision. [cite: 2]
* [cite_start]**96% Accuracy**: High-performance classification based on a feature vector of 43 distinct values. [cite: 4]

---

## 🛠️ Tech Stack

* [cite_start]**Deep Learning**: YOLO11 (Pose Estimation) [cite: 1]
* [cite_start]**Machine Learning**: Scikit-Learn (Pose Classification via Random Forest/SVM) [cite: 2, 5]
* [cite_start]**Backend**: Flask [cite: 1, 5]
* [cite_start]**Computer Vision**: OpenCV [cite: 1, 5]
* [cite_start]**Frontend**: HTML/JavaScript (via Flask templates) [cite: 1]

---

## 📂 Project Structure

```text
.
[cite_start]├── app.py                # Flask web server & real-time processing logic [cite: 1]
[cite_start]├── pose_corrector.py     # Logic for pose classification and correction hints [cite: 2]
[cite_start]├── extract_keypoints.py  # Script to convert dataset images into CSV features [cite: 3]
├── train_model.py        # (Implicit) Script to train the .joblib classifier
[cite_start]├── requirements.txt      # Project dependencies [cite: 5]
[cite_start]├── dataset/              # Folder containing pose subdirectories [cite: 3]
│   ├── tree/             # Training images for Tree pose
│   ├── sittingpose/      # Training images for Sitting pose
│   └── ...
[cite_start]└── training_report.txt   # Metrics from the latest model training [cite: 4]
```

---

## 📐 How It Works

### 1. Feature Extraction
The system extracts 43 features for every frame:
* [cite_start]**34 Normalized Coordinates**: (x, y) pairs for the 17 COCO keypoints, normalized by torso length to be scale-invariant. [cite: 3]
* [cite_start]**9 Joint Angles**: Calculated using the law of cosines at critical vertices like the elbow, knee, and hip. [cite: 2, 3]

The angle $\theta$ between three points $A, B$ (vertex), and $C$ is calculated as:

$$\theta = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}| |\vec{BC}|}\right)$$

### 2. Pose Correction Logic
[cite_start]The `PoseCorrector` compares your current joint angles against the `mean` and `std` (standard deviation) of the training dataset. [cite: 2] 
* [cite_start]**Green**: Perfect form (within 1.5 standard deviations). [cite: 2]
* [cite_start]**Yellow/Orange**: Minor deviation. [cite: 2]
* [cite_start]**Red**: Significant misalignment requiring immediate correction. [cite: 2]

---

## 🚦 Getting Started

### Prerequisites
Ensure you have Python 3.9+ installed. You will also need the `yolo11n-pose.pt` weights file in the root directory.

### Installation
1.  **Clone the repository** and navigate to the folder.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    [cite_start][cite: 5]

### Usage
1.  [cite_start]**Prepare Data**: Place your images in the `dataset/` folder, organized by pose name subfolders. [cite: 3]
2.  **Extract & Train**:
    ```bash
    python extract_keypoints.py
    # Run your training script to generate pose_classifier.joblib
    ```
    [cite_start][cite: 3]
3.  **Launch the App**:
    ```bash
    python app.py
    ```
    [cite_start][cite: 1]
4.  [cite_start]**Open in Browser**: Navigate to `http://localhost:5001`. [cite: 1]

---

## 📊 Performance
According to the latest `training_report.txt`, the model achieves:
* [cite_start]**CV Accuracy**: ~98.5% [cite: 4]
* [cite_start]**Test Accuracy**: 96% [cite: 4]
* [cite_start]**Top Features**: Left/Right ankle and knee Y-coordinates are the most critical for distinguishing these specific poses. [cite: 4]

> [cite_start]**Note**: For best results, ensure your entire body is visible in the frame and you are practicing in a well-lit environment. [cite: 3]














conda activate yoga 

pip install -r requirements.txt

python app.py

http://localhost:5000 ( or go into app.py and change the port if it says already in use)
