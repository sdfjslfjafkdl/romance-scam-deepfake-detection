# deepfake/demo_webcam.py
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from detector import DeepfakeDetector, DetectorConfig
from preprocess import FaceCropConfig, crop_face_from_bbox


def mp_bbox_to_xyxy(
    rel_bbox, frame_w: int, frame_h: int
) -> Tuple[int, int, int, int]:
    x1 = int(rel_bbox.xmin * frame_w)
    y1 = int(rel_bbox.ymin * frame_h)
    x2 = int((rel_bbox.xmin + rel_bbox.width) * frame_w)
    y2 = int((rel_bbox.ymin + rel_bbox.height) * frame_h)
    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(0, min(frame_w - 1, x2))
    y2 = max(0, min(frame_h - 1, y2))
    return x1, y1, x2, y2


def draw_gauge(frame, prob: float, x: int = 20, y: int = 60, w: int = 250, h: int = 18):
    # 게이지 배경
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
    # 게이지 채움
    fill = int(w * max(0.0, min(1.0, prob)))
    cv2.rectangle(frame, (x, y), (x + fill, y + h), (0, 0, 255), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)


def main():
    # 1) detector 준비
    det_cfg = DetectorConfig(
        model_id="prithivMLmods/Deep-Fake-Detector-v2-Model",
        device="auto",
        ema_alpha=0.15,
        unknown_hold_frames=15,
        use_fp16=True,
    )
    detector = DeepfakeDetector(det_cfg)

    crop_cfg = FaceCropConfig(margin=0.35, min_size=80)

    # 2) mediapipe face detection
    mp_face = mp.solutions.face_detection

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found. Try changing VideoCapture index (0 -> 1).")

    prev_t = time.time()
    fps = 0.0

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_h, frame_w = frame.shape[:2]

            # MediaPipe는 RGB 입력 권장
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_det.process(rgb)

            face_crop = None
            bbox_xyxy: Optional[Tuple[int, int, int, int]] = None

            if res.detections:
                # 가장 confidence 높은 얼굴 1개만 사용 (데모 단순화)
                det0 = max(res.detections, key=lambda d: d.score[0])
                bbox_xyxy = mp_bbox_to_xyxy(det0.location_data.relative_bounding_box, frame_w, frame_h)
                face_crop = crop_face_from_bbox(frame, bbox_xyxy, crop_cfg)

            fake_prob, conf, label = detector.predict_face(face_crop, is_bgr=True, apply_ema=True)

            # 상태 텍스트
            if fake_prob >= 0.75:
                status = "HIGH RISK"
            elif fake_prob >= 0.55:
                status = "CAUTION"
            else:
                status = "SAFE"

            # bbox 표시
            if bbox_xyxy is not None:
                x1, y1, x2, y2 = bbox_xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 오버레이
            cv2.putText(frame, f"Deepfake prob (EMA): {fake_prob:.3f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
            cv2.putText(frame, f"Status: {status} | Conf: {conf:.3f} | Top1: {label}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
            draw_gauge(frame, fake_prob, x=20, y=60, w=250, h=18)

            # FPS
            now = time.time()
            dt = now - prev_t
            prev_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

            cv2.imshow("Deepfake Webcam Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord("q")]:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()