# 🧘 Yoga Pose Corrector

Welcome to your personal AI yoga instructor! This project uses computer vision and machine learning to analyze your yoga poses in real-time, providing instant feedback on your alignment and form. 

---

## 🚀 Overview

This application leverages the **YOLO11-pose** model to detect 17 human keypoints and a custom-trained classifier to identify specific yoga poses.  Beyond just identification, the system calculates joint angles and compares them against "ideal" reference data to offer specific, actionable corrections. 

### Key Features
* **Real-time Pose Classification**: Recognizes poses like Tree, Sitting, Wind, and Toe. 
* **Alignment Feedback**: Visual overlays (circles on joints) highlight where your form needs adjustment. 
* **Voice/Text Hints**: Provides specific cues like "Straighten your left leg more" or "Lower your right arm." 
* **Performance Scoring**: Calculates a real-time "form score" from 0-100 based on your precision. 
* **96% Accuracy**: High-performance classification based on a feature vector of 43 distinct values. 

---

## 🛠️ Tech Stack

* **Deep Learning**: YOLO11 (Pose Estimation) 
* **Machine Learning**: Scikit-Learn (Pose Classification via Random Forest/SVM) 
* **Backend**: Flask 
* **Computer Vision**: OpenCV 
* **Frontend**: HTML/JavaScript (via Flask templates) 

---

## 📂 Project Structure

```text
.
├── app.py                # Flask web server & real-time processing logic 
├── pose_corrector.py     # Logic for pose classification and correction hints 
├── extract_keypoints.py  # Script to convert dataset images into CSV features 
├── train_model.py        # (Implicit) Script to train the .joblib classifier
├── requirements.txt      # Project dependencies 
├── dataset/              # Folder containing pose subdirectories 
│   ├── tree/             # Training images for Tree pose
│   ├── sittingpose/      # Training images for Sitting pose
│   └── ...
└── training_report.txt   # Metrics from the latest model training 
```

---

## 📐 How It Works

### 1. Feature Extraction
The system extracts 43 features for every frame:
* **34 Normalized Coordinates**: (x, y) pairs for the 17 COCO keypoints, normalized by torso length to be scale-invariant. 
* **9 Joint Angles**: Calculated using the law of cosines at critical vertices like the elbow, knee, and hip. 

The angle $\theta$ between three points $A, B$ (vertex), and $C$ is calculated as:

$$\theta = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}| |\vec{BC}|}\right)$$

### 2. Pose Correction Logic
The `PoseCorrector` compares your current joint angles against the `mean` and `std` (standard deviation) of the training dataset.  
* **Green**: Perfect form (within 1.5 standard deviations). 
* **Yellow/Orange**: Minor deviation. 
* **Red**: Significant misalignment requiring immediate correction. 

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
    

### Usage
1.  **Prepare Data**: Place your images in the `dataset/` folder, organized by pose name subfolders. 
2.  **Extract & Train**:
    ```bash
    python extract_keypoints.py
    # Run your training script to generate pose_classifier.joblib
    ```
    
3.  **Launch the App**:
    ```bash
    python app.py
    ```
    
4.  **Open in Browser**: Navigate to `http://localhost:5001`. 

---

## 📊 Performance
According to the latest `training_report.txt`, the model achieves:
* **CV Accuracy**: ~98.5% 
* **Test Accuracy**: 96% 
* **Top Features**: Left/Right ankle and knee Y-coordinates are the most critical for distinguishing these specific poses. 

> **Note**: For best results, ensure your entire body is visible in the frame and you are practicing in a well-lit environment. 














conda activate yoga 

pip install -r requirements.txt

python app.py

http://localhost:5000 ( or go into app.py and change the port if it says already in use)
