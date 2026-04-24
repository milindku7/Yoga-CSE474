import os
import sys
import cv2
import json
import base64
import numpy as np
from io import BytesIO
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

from pose_corrector import PoseCorrector, extract_features, calculate_angle, ANGLE_DEFINITIONS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading YOLO pose model...")
yolo_model = YOLO(os.path.join(BASE_DIR, "yolo11n-pose.pt"))
print("YOLO model loaded.")

corrector = None
try:
    corrector = PoseCorrector(BASE_DIR)
    print(f"Pose corrector loaded. Classes: {corrector.pose_classes}")
except FileNotFoundError as e:
    print(f"Warning: {e}")
    print("Running in detection-only mode (no pose classification).")



camera = None
target_pose = None  
last_result = None 


def get_pose_reference_image_filename(pose_name):
    pose_dir = os.path.join(BASE_DIR, "dataset", pose_name)
    if not os.path.isdir(pose_dir):
        return None

    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    files = [
        name for name in sorted(os.listdir(pose_dir))
        if os.path.isfile(os.path.join(pose_dir, name)) and name.lower().endswith(valid_exts)
    ]
    return files[0] if files else None

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera


def draw_correction_overlay(frame, result, keypoints):
    corrections = result["corrections"]
    for corr in corrections:
        severity = corr["severity"]
        if severity >= 2.0:
            dot_color = (0, 0, 255)
        elif severity >= 1.0:
            dot_color = (0, 165, 255)
        else:
            dot_color = (0, 255, 255)

        kp_indices = corr["keypoint_indices"]
        vertex_idx = kp_indices[1]
        vx, vy = int(keypoints[vertex_idx][0]), int(keypoints[vertex_idx][1])
        if vx > 0 and vy > 0:
            cv2.circle(frame, (vx, vy), 18, dot_color, 3)
            cv2.circle(frame, (vx, vy), 22, dot_color, 1)

    return frame


def process_frame(frame):
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
                result_data = corrector.classify_pose(keypoints, target_pose=target_pose)
                
                if result_data is not None:
                    annotated = draw_correction_overlay(annotated, result_data, keypoints)
            else:
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
    global last_result
    cam = get_camera()

    while True:
        success, frame = cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        processed, result_data = process_frame(frame)
        if result_data is not None:
            last_result = result_data

        ret, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/process_frame', methods=['POST'])
def api_process_frame():

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    
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
    return jsonify({
        "yolo_model": "loaded",
        "corrector": "loaded" if corrector else "not_trained",
        "pose_classes": corrector.pose_classes if corrector else [],
        "camera": "available" if camera and camera.isOpened() else "not_started"
    })


@app.route('/api/poses')
def api_poses():
    if corrector is None:
        return jsonify({"error": "Model not trained yet"}), 404

    pose_image_urls = {}
    for pose in corrector.pose_classes:
        if get_pose_reference_image_filename(pose):
            pose_image_urls[pose] = f"/api/pose_reference_image/{pose}"
    
    return jsonify({
        "classes": corrector.pose_classes,
        "references": corrector.references,
        "pose_image_urls": pose_image_urls
    })


@app.route('/api/set_target_pose', methods=['POST'])
def api_set_target_pose():
    global target_pose
    data = request.get_json()
    if not data or 'pose' not in data:
        return jsonify({"error": "No pose specified"}), 400
    
    pose = data['pose']
    if pose == '' or pose == 'auto':
        target_pose = None
        return jsonify({"target_pose": None, "message": "Auto-detect mode"})
    
    if corrector and pose in corrector.pose_classes:
        target_pose = pose
        return jsonify({"target_pose": pose, "message": f"Target set to {pose}"})
    else:
        return jsonify({"error": f"Unknown pose: {pose}"}), 400


@app.route('/api/current_result')
def api_current_result():
    if last_result is None:
        return jsonify({})
    return jsonify({
        "pose": last_result["pose"],
        "score": last_result["score"],
        "confidence": last_result["confidence"],
        "corrections": last_result["corrections"],
        "angles": {k: v for k, v in last_result["angles"].items() if v is not None},
    })


@app.route('/api/pose_reference_image/<pose_name>')
def api_pose_reference_image(pose_name):
    if corrector is not None and pose_name not in corrector.pose_classes:
        return jsonify({"error": f"Unknown pose: {pose_name}"}), 404

    filename = get_pose_reference_image_filename(pose_name)
    if not filename:
        return jsonify({"error": f"No reference image for pose: {pose_name}"}), 404

    pose_dir = os.path.join(BASE_DIR, "dataset", pose_name)
    return send_from_directory(pose_dir, filename)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  🧘 Yoga Pose Corrector")
    print("=" * 60)
    print(f"  Model: {'Loaded' if corrector else 'Not trained — run train_model.py'}")
    if corrector:
        print(f"  Poses: {', '.join(corrector.pose_classes)}")
    print(f"  Open: http://localhost:5001")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)