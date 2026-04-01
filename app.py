"""
app.py — Yoga Pose Corrector Web Application

Flask server that:
  1. Captures webcam frames via browser MediaDevices API
  2. Processes each frame with YOLO pose estimation 
  3. Classifies the pose using the trained model
  4. Generates correction feedback overlaid on the video
  5. Streams the annotated frames back to the browser
  
Run: python app.py
Open: http://localhost:5000
"""

import os
import sys
import cv2
import json
import base64
import numpy as np
from io import BytesIO
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

# Import our modules
from pose_corrector import PoseCorrector, extract_features, calculate_angle, ANGLE_DEFINITIONS

# ────────────────────────────────────────────────────────────────────
# Flask Setup
# ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load YOLO model
print("Loading YOLO pose model...")
yolo_model = YOLO(os.path.join(BASE_DIR, "yolo11n-pose.pt"))
print("YOLO model loaded.")

# Load pose corrector (if trained model exists)
corrector = None
try:
    corrector = PoseCorrector(BASE_DIR)
    print(f"Pose corrector loaded. Classes: {corrector.pose_classes}")
except FileNotFoundError as e:
    print(f"Warning: {e}")
    print("Running in detection-only mode (no pose classification).")


# ────────────────────────────────────────────────────────────────────
# Video Streaming (webcam capture)
# ────────────────────────────────────────────────────────────────────
camera = None

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera


def draw_correction_overlay(frame, result, keypoints):
    """Draw beautiful correction overlay on the frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay panel at top
    overlay = frame.copy()
    
    # ── Top status bar ──
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    pose_name = result["pose"].replace("_", " ").title()
    score = result["score"]
    confidence = result["confidence"]
    color = corrector.get_correction_color(score) if corrector else (0, 255, 0)
    label = corrector.get_score_label(score) if corrector else ""
    
    # Pose name
    cv2.putText(frame, f"Pose: {pose_name}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Score bar
    bar_x, bar_y = 20, 55
    bar_w, bar_h = 200, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * score / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    cv2.putText(frame, f"{score:.0f}%",
                (bar_x + bar_w + 10, bar_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Label
    cv2.putText(frame, label,
                (bar_x + bar_w + 60, bar_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Confidence
    cv2.putText(frame, f"Confidence: {confidence*100:.0f}%",
                (w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
    
    # ── Correction hints (right side panel) ──
    corrections = result["corrections"]
    if corrections:
        panel_w = 350
        panel_h = 40 + len(corrections) * 55
        px = w - panel_w - 15
        py = 100
        
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (px, py), (px + panel_w, py + panel_h), (20, 20, 40), -1)
        cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, "Corrections:",
                    (px + 10, py + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 2)
        
        for i, corr in enumerate(corrections):
            cy = py + 50 + i * 55
            severity = corr["severity"]
            
            if severity >= 2.0:
                dot_color = (0, 0, 255)
            elif severity >= 1.0:
                dot_color = (0, 165, 255)
            else:
                dot_color = (0, 255, 255)
            
            cv2.circle(frame, (px + 20, cy + 5), 6, dot_color, -1)
            cv2.putText(frame, corr["hint"],
                        (px + 35, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{corr['current']:.0f}° → {corr['target']:.0f}°",
                        (px + 35, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Draw correction indicator on skeleton
            kp_indices = corr["keypoint_indices"]
            vertex_idx = kp_indices[1]
            vx, vy = int(keypoints[vertex_idx][0]), int(keypoints[vertex_idx][1])
            if vx > 0 and vy > 0:
                cv2.circle(frame, (vx, vy), 18, dot_color, 3)
                cv2.circle(frame, (vx, vy), 22, dot_color, 1)
    else:
        # All good indicator
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (w - 280, 100), (w - 15, 155), (0, 80, 0), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "✓ Perfect Form!",
                    (w - 265, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # ── Draw angle values on joints ──
    angles = result.get("angles", {})
    for angle_name, (a_idx, b_idx, c_idx) in ANGLE_DEFINITIONS.items():
        if angle_name in angles and angles[angle_name] is not None:
            vx, vy = int(keypoints[b_idx][0]), int(keypoints[b_idx][1])
            if vx > 0 and vy > 0:
                angle_val = angles[angle_name]
                cv2.putText(frame, f"{angle_val:.0f}°",
                            (vx + 10, vy - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, (200, 200, 200), 1)
    
    return frame


def process_frame(frame):
    """Process a single frame: detect pose, classify, correct."""
    # Run YOLO
    results = yolo_model(frame, verbose=False)
    annotated = results[0].plot()
    
    result_data = None
    
    try:
        if (results[0].keypoints is not None and 
            len(results[0].keypoints.xy) > 0 and
            len(results[0].keypoints.xy[0]) > 0):
            
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            conf = results[0].keypoints.conf[0].cpu().numpy()
            
            if corrector is not None:
                result_data = corrector.classify_pose(keypoints)
                
                if result_data is not None:
                    annotated = draw_correction_overlay(annotated, result_data, keypoints)
            else:
                # Fallback: just show angles without classification
                for angle_name, (a_idx, b_idx, c_idx) in ANGLE_DEFINITIONS.items():
                    a, b, c = keypoints[a_idx], keypoints[b_idx], keypoints[c_idx]
                    if np.all(a != 0) and np.all(b != 0) and np.all(c != 0):
                        angle = calculate_angle(a, b, c)
                        vx, vy = int(b[0]), int(b[1])
                        cv2.putText(annotated, f"{angle:.0f}°",
                                    (vx + 10, vy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, (0, 255, 255), 1)
    except Exception as e:
        pass
    
    return annotated, result_data


def generate_frames():
    """Generate frames for MJPEG streaming."""
    cam = get_camera()
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        processed, _ = process_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG stream endpoint."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/process_frame', methods=['POST'])
def api_process_frame():
    """
    Process a single frame sent from the browser.
    Accepts base64-encoded JPEG image.
    Returns annotated image + correction data as JSON.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    
    # Decode base64 image
    img_data = data['image']
    if ',' in img_data:
        img_data = img_data.split(',')[1]
    
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400
    
    frame = cv2.flip(frame, 1)
    processed, result_data = process_frame(frame)
    
    # Encode result image
    ret, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
    result_img = base64.b64encode(buffer).decode('utf-8')
    
    response = {
        "image": f"data:image/jpeg;base64,{result_img}",
    }
    
    if result_data:
        response["pose"] = result_data["pose"]
        response["score"] = result_data["score"]
        response["confidence"] = result_data["confidence"]
        response["corrections"] = result_data["corrections"]
        response["angles"] = {k: v for k, v in result_data["angles"].items() if v is not None}
    
    return jsonify(response)


@app.route('/api/status')
def api_status():
    """Return system status."""
    return jsonify({
        "yolo_model": "loaded",
        "corrector": "loaded" if corrector else "not_trained",
        "pose_classes": corrector.pose_classes if corrector else [],
        "camera": "available" if camera and camera.isOpened() else "not_started"
    })


@app.route('/api/poses')
def api_poses():
    """Return available pose classes and their reference angles."""
    if corrector is None:
        return jsonify({"error": "Model not trained yet"}), 404
    
    return jsonify({
        "classes": corrector.pose_classes,
        "references": corrector.references
    })


# ────────────────────────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  🧘 Yoga Pose Corrector")
    print("=" * 60)
    print(f"  Model: {'Loaded' if corrector else 'Not trained — run train_model.py'}")
    if corrector:
        print(f"  Poses: {', '.join(corrector.pose_classes)}")
    print(f"  Open: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)