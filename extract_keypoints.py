"""
extract_keypoints.py — Extracts YOLO pose keypoints from dataset images.

Walks through dataset/ directory structure where each subfolder is a pose class.
For each image, runs YOLO pose estimation and saves the 17 keypoint coordinates
(normalized) as feature vectors, along with all joint angles.

Output: dataset_keypoints.csv
"""

import os
import csv
import sys
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────
# YOLO 17-Keypoint Layout (COCO format)
# 0: Nose           1: Left Eye       2: Right Eye
# 3: Left Ear       4: Right Ear      5: Left Shoulder
# 6: Right Shoulder 7: Left Elbow     8: Right Elbow
# 9: Left Wrist     10: Right Wrist   11: Left Hip
# 12: Right Hip     13: Left Knee     14: Right Knee
# 15: Left Ankle    16: Right Ankle
# ──────────────────────────────────────────────────────────────────────

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Angles to compute (triplet indices: point_a, vertex, point_b)
ANGLE_DEFINITIONS = {
    "left_elbow_angle":     (5, 7, 9),     # L.Shoulder → L.Elbow → L.Wrist
    "right_elbow_angle":    (6, 8, 10),    # R.Shoulder → R.Elbow → R.Wrist
    "left_shoulder_angle":  (7, 5, 11),    # L.Elbow → L.Shoulder → L.Hip
    "right_shoulder_angle": (8, 6, 12),    # R.Elbow → R.Shoulder → R.Hip
    "left_hip_angle":       (5, 11, 13),   # L.Shoulder → L.Hip → L.Knee
    "right_hip_angle":      (6, 12, 14),   # R.Shoulder → R.Hip → R.Knee
    "left_knee_angle":      (11, 13, 15),  # L.Hip → L.Knee → L.Ankle
    "right_knee_angle":     (12, 14, 16),  # R.Hip → R.Knee → R.Ankle
    "torso_inclination":    (0, 11, 13),   # Nose → L.Hip → L.Knee (approx torso tilt)
}


def calculate_angle(a, b, c):
    """Calculate angle at vertex b formed by points a-b-c (in degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def normalize_keypoints(keypoints):
    """
    Normalize keypoints to be translation/scale invariant.
    Centers on the midpoint of the hips, scales by torso length.
    """
    kp = keypoints.copy()
    
    # Center = midpoint of hips (11, 12)
    center = (kp[11] + kp[12]) / 2.0
    kp = kp - center
    
    # Scale by torso length (mid-hip to mid-shoulder)
    mid_shoulder = (kp[5] + kp[6]) / 2.0
    torso_length = np.linalg.norm(mid_shoulder) + 1e-8  # already centered so mid_hip is origin
    kp = kp / torso_length
    
    return kp


def extract_features(keypoints):
    """
    Extract a feature vector from raw keypoints:
      - 34 normalized (x,y) coordinates (17 keypoints × 2)
      - 9 joint angles
    Total: 43 features
    """
    # Check if enough keypoints are detected (non-zero)
    valid_count = np.sum(np.any(keypoints != 0, axis=1))
    if valid_count < 10:
        return None
    
    # Replace zero keypoints with midpoint estimates if possible
    kp = keypoints.copy()
    
    # Normalize
    norm_kp = normalize_keypoints(kp)
    
    # Flatten normalized keypoints
    flat_kp = norm_kp.flatten()  # 34 values
    
    # Compute angles
    angles = []
    for angle_name, (a_idx, b_idx, c_idx) in ANGLE_DEFINITIONS.items():
        a, b, c = kp[a_idx], kp[b_idx], kp[c_idx]
        # Only compute if all three points are detected
        if np.all(a != 0) and np.all(b != 0) and np.all(c != 0):
            angle = calculate_angle(a, b, c)
        else:
            angle = 0.0  # placeholder for undetected
        angles.append(angle)
    
    features = np.concatenate([flat_kp, np.array(angles)])
    return features


def get_header():
    """Generate CSV column headers."""
    headers = []
    for name in KEYPOINT_NAMES:
        headers.extend([f"{name}_x", f"{name}_y"])
    for angle_name in ANGLE_DEFINITIONS.keys():
        headers.append(angle_name)
    headers.append("pose_class")
    return headers


def main():
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    output_csv = os.path.join(os.path.dirname(__file__), "dataset_keypoints.csv")
    model_path = os.path.join(os.path.dirname(__file__), "yolo11n-pose.pt")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        sys.exit(1)
    
    model = YOLO(model_path)
    
    # Discover pose classes from subdirectories
    pose_classes = sorted([
        d for d in os.listdir(dataset_dir) 
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    
    if not pose_classes:
        print("Error: No pose class subdirectories found in dataset/")
        sys.exit(1)
    
    print(f"Found pose classes: {pose_classes}")
    
    headers = get_header()
    all_features = []
    
    for pose_class in pose_classes:
        class_dir = os.path.join(dataset_dir, pose_class)
        image_files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ])
        
        print(f"\nProcessing '{pose_class}': {len(image_files)} images")
        success_count = 0
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(class_dir, img_file)
            
            try:
                results = model(img_path, verbose=False)
                
                if (results[0].keypoints is not None and 
                    len(results[0].keypoints.xy) > 0 and
                    len(results[0].keypoints.xy[0]) > 0):
                    
                    keypoints = results[0].keypoints.xy[0].cpu().numpy()
                    features = extract_features(keypoints)
                    
                    if features is not None:
                        row = list(features) + [pose_class]
                        all_features.append(row)
                        success_count += 1
                        
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(image_files)}...")
        
        print(f"  Successfully extracted: {success_count}/{len(image_files)}")
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_features)
    
    print(f"\n✅ Saved {len(all_features)} feature vectors to {output_csv}")
    print(f"   Feature dimensions: {len(headers) - 1} features + 1 label")


if __name__ == "__main__":
    main()
