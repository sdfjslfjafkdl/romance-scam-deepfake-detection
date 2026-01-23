import cv2
import mediapipe as mp
import numpy as np
import time

from detector import DeepfakeDetector, DetectorConfig
from preprocess import FaceCropConfig, crop_face_from_bbox

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.22  

det_cfg = DetectorConfig(
    model_id="prithivMLmods/Deep-Fake-Detector-v2-Model",
    device="auto",       
    use_fp16=True        
)
deepfake_detector = DeepfakeDetector(det_cfg)
crop_cfg = FaceCropConfig(margin=0.5, min_size=80)

def calculate_ear(landmarks, indices, w, h):
    coords = np.array([[landmarks[idx].x * w, landmarks[idx].y * h] for idx in indices])
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h_dist = np.linalg.norm(coords[0] - coords[3])
    return (v1 + v2) / (2.0 * h_dist)

def get_mediapipe_bbox(landmarks, w, h):
    x_min, y_min = w, h
    x_max, y_max = 0, 0

    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        if x < x_min: x_min = x
        if x > x_max: x_max = x
        if y < y_min: y_min = y
        if y > y_max: y_max = y
    
    return (x_min, y_min, x_max, y_max)

def draw_dashboard(frame, fake_prob, blink_count, fps):
    cv2.rectangle(frame, (10, 10), (350, 160), (0, 0, 0), -1)
    
    color = (0, 255, 0) if fake_prob < 0.5 else (0, 0, 255) 
    status = "REAL" if fake_prob < 0.5 else "FAKE WARNING"
    
    cv2.putText(frame, f"AI Analysis: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.rectangle(frame, (20, 50), (320, 70), (50, 50, 50), -1)
    fill_width = int(300 * fake_prob)
    cv2.rectangle(frame, (20, 50), (20 + fill_width, 70), color, -1)
    cv2.putText(frame, f"{fake_prob*100:.1f}%", (20 + fill_width + 5, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.putText(frame, f"Blinks: {blink_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

def main():
    cap = cv2.VideoCapture(0)
    
    blink_count = 0
    is_blinking = False
    
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(frame_rgb)
        
        fake_prob = 0.0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
                right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    if not is_blinking:
                        blink_count += 1
                        is_blinking = True
                    cv2.putText(frame, "BLINK!", (370, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    is_blinking = False

                bbox = get_mediapipe_bbox(face_landmarks.landmark, w, h)
                
                face_crop = crop_face_from_bbox(frame, bbox, crop_cfg)
                
                fake_prob, conf, label = deepfake_detector.predict_face(face_crop, is_bgr=True, apply_ema=True)
                
                color = (0, 0, 255) if fake_prob > 0.5 else (0, 255, 0)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        draw_dashboard(frame, fake_prob, blink_count, fps)

        cv2.imshow('DeepTrust Hybrid System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()