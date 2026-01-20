# deepfake/demo_webcam.py
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from detector import DeepfakeDetector, DetectorConfig
from preprocess import FaceCropConfig, crop_face_from_bbox


def draw_gauge(frame, prob: float, x: int = 20, y: int = 60, w: int = 250, h: int = 18):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
    fill = int(w * max(0.0, min(1.0, prob)))
    cv2.rectangle(frame, (x, y), (x + fill, y + h), (0, 0, 255), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)


def pick_largest_face_xyxy(faces_xywh) -> Optional[Tuple[int, int, int, int]]:
    """
    faces_xywh: list of (x, y, w, h)
    return: (x1, y1, x2, y2) for largest face
    """
    if faces_xywh is None or len(faces_xywh) == 0:
        return None
    x, y, w, h = max(faces_xywh, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(x + w), int(y + h)


def main():
    # 1) Deepfake detector (HF 모델) 준비 (그대로)
    det_cfg = DetectorConfig(
        model_id="prithivMLmods/Deep-Fake-Detector-v2-Model",
        device="auto",
        ema_alpha=0.15,
        unknown_hold_frames=15,
        use_fp16=True,
    )
    deepfake_detector = DeepfakeDetector(det_cfg)

    crop_cfg = FaceCropConfig(margin=0.35, min_size=80)

    # 2) OpenCV Haar face detector 준비 (MediaPipe 대체)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")

    # 3) Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found. Try changing VideoCapture index (0 -> 1).")

    prev_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Haar는 grayscale에서 동작
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출 (가벼운 데모용 세팅)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        bbox_xyxy: Optional[Tuple[int, int, int, int]] = pick_largest_face_xyxy(faces)
        face_crop = None
        if bbox_xyxy is not None:
            face_crop = crop_face_from_bbox(frame, bbox_xyxy, crop_cfg)

        fake_prob, conf, label = deepfake_detector.predict_face(face_crop, is_bgr=True, apply_ema=True)

        if fake_prob >= 0.75:
            status = "HIGH RISK"
        elif fake_prob >= 0.55:
            status = "CAUTION"
        else:
            status = "SAFE"

        if bbox_xyxy is not None:
            x1, y1, x2, y2 = bbox_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"Deepfake prob (EMA): {fake_prob:.3f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        cv2.putText(frame, f"Status: {status} | Conf: {conf:.3f} | Top1: {label}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
        draw_gauge(frame, fake_prob, x=20, y=60, w=250, h=18)

        now = time.time()
        dt = now - prev_t
        prev_t = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

        cv2.imshow("Deepfake Webcam Demo (OpenCV Face Detect)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord("q")]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()