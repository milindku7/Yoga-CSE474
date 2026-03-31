import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load the YOLO11 Pose model
# Using the 'nano' (n) model for maximum real-time speed. 
# You can upgrade to 's', 'm', or 'l' for better accuracy if your GPU can handle it.
model = YOLO('yolo11n-pose.pt')

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.
    a, b, c are tuples or lists of (x, y) coordinates. b is the vertex.
    """
    a = np.array(a) # First point
    b = np.array(b) # Mid point (Vertex)
    c = np.array(c) # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def draw_alignment_grid(frame):
    """Draws a subtle grid to help with overall posture alignment."""
    h, w, _ = frame.shape
    # Draw vertical and horizontal center lines
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 255), 1)

# 2. Start Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # 3. Run YOLO11 inference
    results = model(frame, verbose=False)

    # Draw the base skeleton
    annotated_frame = results[0].plot()
    draw_alignment_grid(annotated_frame)

    # 4. Extract Keypoints and Evaluate Form
    try:
        # Check if any person is detected
        if results[0].keypoints is not None and len(results[0].keypoints.xy[0]) > 0:
            # Get the keypoints for the first detected person
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            
            # Confidence scores for keypoints (optional, useful for filtering bad data)
            # conf = results[0].keypoints.conf[0].cpu().numpy()

            # Example: Evaluate the Right Arm (Shoulder[6], Elbow[8], Wrist[10])
            r_shoulder = keypoints[6]
            r_elbow = keypoints[8]
            r_wrist = keypoints[10]

            # Check if all points are detected (YOLO outputs [0,0] if not confident/visible)
            if all(pt[0] != 0 and pt[1] != 0 for pt in [r_shoulder, r_elbow, r_wrist]):
                
                # Calculate the angle
                arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                
                # Visual Feedback Logic: Let's assume the user is trying to keep their arm straight (180 degrees)
                # like in a Warrior II pose.
                if 165 <= arm_angle <= 180:
                    feedback_color = (0, 255, 0) # Green for good form
                    feedback_text = "Good Arm Form!"
                else:
                    feedback_color = (0, 0, 255) # Red for correction
                    feedback_text = "Straighten your right arm"
                    
                    # Draw a guiding correction line from shoulder to where the wrist SHOULD be
                    # (This is simplified, but demonstrates the concept)
                    cv2.line(annotated_frame, 
                             (int(r_shoulder[0]), int(r_shoulder[1])), 
                             (int(r_wrist[0]), int(r_shoulder[1])), # Forcing a horizontal line
                             (0, 255, 255), 3) # Yellow guiding line
                
                # Display the angle at the elbow
                cv2.putText(annotated_frame, str(int(arm_angle)), 
                            (int(r_elbow[0]) + 15, int(r_elbow[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)
                
                # Display the instruction on the screen
                cv2.putText(annotated_frame, feedback_text, (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2, cv2.LINE_AA)

    except Exception as e:
        # Handle cases where keypoints might be out of bounds or missing
        pass

    # 5. Show the feed
    cv2.imshow('Yoga Pose Coach', annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()