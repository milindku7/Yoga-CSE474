"""
pose_corrector.py — Real-time pose correction engine.

Provides feedback by comparing user's current angles against
the trained reference angles for each pose class.
"""

import json
import os
import numpy as np
import joblib


# ── YOLO COCO Keypoint indices ─────────────────────────────────────
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

ANGLE_DEFINITIONS = {
    "left_elbow_angle":     (5, 7, 9),
    "right_elbow_angle":    (6, 8, 10),
    "left_shoulder_angle":  (7, 5, 11),
    "right_shoulder_angle": (8, 6, 12),
    "left_hip_angle":       (5, 11, 13),
    "right_hip_angle":      (6, 12, 14),
    "left_knee_angle":      (11, 13, 15),
    "right_knee_angle":     (12, 14, 16),
    "torso_inclination":    (0, 11, 13),
}

# Human-readable correction instructions per angle
CORRECTION_HINTS = {
    "left_elbow_angle":     {"too_low": "Extend your left arm more",     "too_high": "Bend your left elbow more"},
    "right_elbow_angle":    {"too_low": "Extend your right arm more",    "too_high": "Bend your right elbow more"},
    "left_shoulder_angle":  {"too_low": "Raise your left arm higher",    "too_high": "Lower your left arm"},
    "right_shoulder_angle": {"too_low": "Raise your right arm higher",   "too_high": "Lower your right arm"},
    "left_hip_angle":       {"too_low": "Open your left hip more",       "too_high": "Close your left hip a bit"},
    "right_hip_angle":      {"too_low": "Open your right hip more",      "too_high": "Close your right hip a bit"},
    "left_knee_angle":      {"too_low": "Straighten your left leg more", "too_high": "Bend your left knee more"},
    "right_knee_angle":     {"too_low": "Straighten your right leg more","too_high": "Bend your right knee more"},
    "torso_inclination":    {"too_low": "Lean forward slightly",         "too_high": "Straighten your torso"},
}


def calculate_angle(a, b, c):
    """Calculate angle at vertex b formed by points a-b-c (degrees)."""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def normalize_keypoints(keypoints):
    """Normalize keypoints: center on hip midpoint, scale by torso length."""
    kp = keypoints.copy()
    center = (kp[11] + kp[12]) / 2.0
    kp = kp - center
    mid_shoulder = (kp[5] + kp[6]) / 2.0
    torso_length = np.linalg.norm(mid_shoulder) + 1e-8
    kp = kp / torso_length
    return kp


def extract_features(keypoints):
    """Extract 43-dim feature vector from 17 keypoints."""
    valid_count = np.sum(np.any(keypoints != 0, axis=1))
    if valid_count < 10:
        return None
    
    kp = keypoints.copy()
    norm_kp = normalize_keypoints(kp)
    flat_kp = norm_kp.flatten()
    
    angles = []
    for angle_name, (a_idx, b_idx, c_idx) in ANGLE_DEFINITIONS.items():
        a, b, c = kp[a_idx], kp[b_idx], kp[c_idx]
        if np.all(a != 0) and np.all(b != 0) and np.all(c != 0):
            angle = calculate_angle(a, b, c)
        else:
            angle = 0.0
        angles.append(angle)
    
    return np.concatenate([flat_kp, np.array(angles)])


class PoseCorrector:
    """
    Real-time pose correction engine.
    
    1. Classifies the user's current pose using the trained model
    2. Compares joint angles against reference values
    3. Generates prioritized correction feedback
    """
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(__file__)
        
        # Load classifier
        model_path = os.path.join(base_dir, "pose_classifier.joblib")
        encoder_path = os.path.join(base_dir, "label_encoder.joblib")
        ref_path = os.path.join(base_dir, "pose_references.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "pose_classifier.joblib not found. Run train_model.py first."
            )
        
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        
        # Load reference angles
        if os.path.exists(ref_path):
            with open(ref_path, 'r') as f:
                self.references = json.load(f)
        else:
            self.references = {}
        
        self.pose_classes = list(self.label_encoder.classes_)
        
        # Smoothing: keep history of predictions
        self.prediction_history = []
        self.history_size = 10
        
        # Smoothing: keep history of angles
        self.angle_history = {}
        self.angle_smooth_size = 5
    
    def classify_pose(self, keypoints, target_pose=None):
        """
        Classify the pose and return (class_name, confidence, corrections).
        
        Args:
            keypoints: numpy array of 17 keypoints
            target_pose: optional string — if set, corrections compare against
                         this pose's reference angles instead of auto-detected pose
        
        Returns:
            dict with keys:
                - pose: str (predicted class name)
                - confidence: float (0-1)
                - corrections: list of dicts with correction feedback
                - angles: dict of current angle values
                - score: float (0-100, overall pose accuracy)
                - target_pose: str (pose used for corrections)
        """
        features = extract_features(keypoints)
        if features is None:
            return None
        
        # Predict
        features_2d = features.reshape(1, -1)
        proba = self.model.predict_proba(features_2d)[0]
        pred_idx = np.argmax(proba)
        confidence = float(proba[pred_idx])
        pose_name = self.label_encoder.inverse_transform([pred_idx])[0]
        
        # Smooth predictions
        self.prediction_history.append(pose_name)
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Use majority vote for stability
        from collections import Counter
        vote_counts = Counter(self.prediction_history)
        smoothed_pose = vote_counts.most_common(1)[0][0]
        
        # Use target_pose for corrections if specified, otherwise use detected pose
        correction_pose = target_pose if (target_pose and target_pose in self.references) else smoothed_pose
        
        # Compute current angles
        current_angles = {}
        kp = keypoints.copy()
        for angle_name, (a_idx, b_idx, c_idx) in ANGLE_DEFINITIONS.items():
            a, b, c = kp[a_idx], kp[b_idx], kp[c_idx]
            if np.all(a != 0) and np.all(b != 0) and np.all(c != 0):
                angle = calculate_angle(a, b, c)
                
                # Smooth angles
                if angle_name not in self.angle_history:
                    self.angle_history[angle_name] = []
                self.angle_history[angle_name].append(angle)
                if len(self.angle_history[angle_name]) > self.angle_smooth_size:
                    self.angle_history[angle_name].pop(0)
                
                current_angles[angle_name] = float(np.mean(self.angle_history[angle_name]))
            else:
                current_angles[angle_name] = None
        
        # Generate corrections
        corrections = []
        total_score = 100.0
        
        if correction_pose in self.references:
            ref = self.references[correction_pose]
            
            for angle_name, current_val in current_angles.items():
                if current_val is None or angle_name not in ref:
                    continue
                
                ref_data = ref[angle_name]
                ref_mean = ref_data["mean"]
                ref_std = ref_data["std"]
                tolerance = max(ref_std * 1.5, 15.0)  # At least 15° tolerance
                
                deviation = current_val - ref_mean
                
                if abs(deviation) > tolerance:
                    severity = min(abs(deviation) / tolerance, 3.0)  # cap at 3x
                    
                    if deviation < 0:
                        hint = CORRECTION_HINTS.get(angle_name, {}).get("too_low", "Adjust angle")
                    else:
                        hint = CORRECTION_HINTS.get(angle_name, {}).get("too_high", "Adjust angle")
                    
                    corrections.append({
                        "angle": angle_name,
                        "current": round(current_val, 1),
                        "target": round(ref_mean, 1),
                        "deviation": round(deviation, 1),
                        "severity": round(severity, 2),
                        "hint": hint,
                        "keypoint_indices": list(ANGLE_DEFINITIONS[angle_name])
                    })
                    
                    # Deduct from score based on severity
                    total_score -= severity * 5
                else:
                    pass  # within tolerance
        
        # Sort corrections by severity (most urgent first)
        corrections.sort(key=lambda c: c["severity"], reverse=True)
        
        # Limit to top 3 corrections to avoid overwhelming the user
        top_corrections = corrections[:3]
        
        total_score = max(0, min(100, total_score))
        
        return {
            "pose": smoothed_pose,
            "target_pose": correction_pose,
            "confidence": round(confidence, 3),
            "corrections": top_corrections,
            "all_corrections": corrections,
            "angles": current_angles,
            "score": round(total_score, 1),
            "num_issues": len(corrections)
        }
    
    def get_correction_color(self, score):
        """Return BGR color based on pose accuracy score."""
        if score >= 80:
            return (0, 255, 0)      # Green — excellent
        elif score >= 60:
            return (0, 255, 255)    # Yellow — good but needs work
        elif score >= 40:
            return (0, 165, 255)    # Orange — needs improvement
        else:
            return (0, 0, 255)      # Red — poor form
    
    def get_score_label(self, score):
        """Return text label for the score."""
        if score >= 90:
            return "Perfect!"
        elif score >= 80:
            return "Great Form"
        elif score >= 60:
            return "Good - Minor Adjustments"
        elif score >= 40:
            return "Needs Improvement"
        else:
            return "Keep Trying"
