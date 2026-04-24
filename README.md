# 🧘 Yoga Pose Corrector

## 🚀 Overview

This application leverages the **YOLO11-pose** model to detect 17 human keypoints and a custom-trained classifier to identify specific yoga poses.  Beyond just identification, the system calculates joint angles and compares them against "ideal" reference data to offer specific, actionable corrections. 

### Key Features
* **Real-time Pose Classification**: Recognizes poses like Tree, Sitting, Wind, and Toe. 
* **Alignment Feedback**: Visual overlays (circles on joints) highlight where your form needs adjustment. 
* **Voice/Text Hints**: Provides specific cues like "Straighten your left leg more" or "Lower your right arm." 
* **Performance Scoring**: Calculates a real-time "form score" from 0-100 based on your precision. 
* **96% Accuracy**: High-performance classification based on a feature vector of 43 distinct values. 

---

## 📐 How It Works

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

conda activate yoga 

pip install -r requirements.txt

python app.py

http://localhost:5000 ( or go into app.py and change the port if it says already in use)
