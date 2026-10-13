from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

model_path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
if not model_path.is_file():
    raise FileNotFoundError(
        f"MediaPipe hand model not found: {model_path}. "
        "Download hand_landmarker.task as described in README.md."
    )

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _annotate_frame(frame, result):
    annotated = frame.copy()
    if not result or not result.hand_landmarks:
        cv2.putText(
            annotated,
            "No hand",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        return annotated, "No hand"

    for hand_landmarks in result.hand_landmarks:
        height, width = annotated.shape[:2]
        points = [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in hand_landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(annotated, points[start], points[end], (0, 255, 0), 2)
        for point in points:
            cv2.circle(annotated, point, 4, (0, 0, 255), -1)

    cv2.putText(
        annotated,
        "Hand detected",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    return annotated, "Hand detected"


def process_frame(img_array):
    if img_array is None or getattr(img_array, "size", 0) == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8), "No frame"

    frame_bgr = np.asarray(img_array)
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        return np.zeros((480, 640, 3), dtype=np.uint8), "Invalid frame"

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = hand_landmarker.detect_for_video(image, time.monotonic_ns() // 1_000_000)
    annotated_frame, status = _annotate_frame(frame_bgr, result)
    return annotated_frame, status
