from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
CLASSIFIER_PATH = MODELS_DIR / "asl_classifier.onnx"
LABELS_PATH = MODELS_DIR / "labels.txt"
CONFIDENCE_THRESHOLD = 0.60

if not HAND_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"MediaPipe hand model not found: {HAND_MODEL_PATH}. "
        "Download hand_landmarker.task as described in README.md."
    )

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
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


def _load_labels(path):
    if not path.is_file():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        parts = text.split(" ", 1)
        labels.append(parts[1].strip() if len(parts) == 2 else parts[0])
    return labels


class OnnxClassifier:
    def __init__(self, model_path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        self.height = shape[1] if isinstance(shape[1], int) and shape[1] > 0 else 224
        self.width = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 224
        self.labels = _load_labels(LABELS_PATH)

    def predict(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        scores = self.session.run(
            None,
            {self.input_name: np.expand_dims(normalized, axis=0)},
        )[0][0]
        index = int(np.argmax(scores))
        label = self.labels[index] if index < len(self.labels) else f"Class {index}"
        return label, float(scores[index])


def _load_classifier():
    if not CLASSIFIER_PATH.is_file():
        print(f"ONNX classifier not found. Place asl_classifier.onnx in {MODELS_DIR}.")
        return None
    print(f"Loaded ONNX classifier: {CLASSIFIER_PATH.name}")
    return OnnxClassifier(CLASSIFIER_PATH)


classifier = _load_classifier()


def _draw_hand(annotated, result):
    if not result or not result.hand_landmarks:
        return
    height, width = annotated.shape[:2]
    for hand_landmarks in result.hand_landmarks:
        points = [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in hand_landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(annotated, points[start], points[end], (0, 255, 0), 2)
        for point in points:
            cv2.circle(annotated, point, 4, (0, 0, 255), -1)


def _draw_prediction(annotated, letter, confidence):
    if confidence < CONFIDENCE_THRESHOLD:
        text = f"...  {confidence * 100:.0f}%"
        color = (0, 255, 255)
    else:
        text = f"{letter}  {confidence * 100:.0f}%"
        color = (0, 255, 0)

    cv2.putText(annotated, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    bar_x, bar_y, bar_w, bar_h = 20, 70, 220, 20
    filled = int(bar_w * max(0.0, min(confidence, 1.0)))
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), color, -1)
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)


def process_frame(img_array):
    frame_bgr = np.array(img_array, copy=True)
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = hand_landmarker.detect_for_video(image, time.monotonic_ns() // 1_000_000)

    annotated = frame_bgr.copy()
    _draw_hand(annotated, result)
    if classifier is not None:
        letter, confidence = classifier.predict(frame_bgr)
        _draw_prediction(annotated, letter, confidence)
    elif result and result.hand_landmarks:
        cv2.putText(annotated, "Hand detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(annotated, "No hand", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return annotated
