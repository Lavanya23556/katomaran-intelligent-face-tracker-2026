"""Tests for face detector and ArcFace recognizer."""
import os
import cv2
import numpy as np
import pytest

from src.models.detector import calculate_face_quality, FaceDetector
from src.models.recognizer import FaceRecognizer


def test_face_quality_sharp_vs_flat(sample_face_crop):
    # Sharp crop with edges
    sharp_quality = calculate_face_quality(sample_face_crop)
    assert sharp_quality > 50.0

    # Flat blurred crop
    flat_crop = np.full((112, 112, 3), 128, dtype=np.uint8)
    flat_quality = calculate_face_quality(flat_crop)
    assert flat_quality == 0.0


def test_recognizer_math_and_preprocessing():
    rec = FaceRecognizer.__new__(FaceRecognizer)
    # Cosine similarity tests
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert abs(rec.compute_similarity(v1, v2) - 1.0) < 1e-5
    assert abs(rec.compute_similarity(v1, v3) - 0.0) < 1e-5

    # Crop preprocessing tensor test
    crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    tensor = rec.preprocess_crop(crop)
    assert tensor.shape == (1, 3, 112, 112)
    assert tensor.dtype == np.float32
    assert -1.1 <= tensor.min() <= 1.1
    assert -1.1 <= tensor.max() <= 1.1


def test_real_model_loading_and_inference():
    """Verify real YOLO face detector and InsightFace ArcFace models if weights exist."""
    det_path = "weights/yolov8n-face.pt"
    rec_path = "weights/w600k_mbf.onnx"

    if not os.path.exists(det_path) or not os.path.exists(rec_path):
        pytest.skip("Model weights not present; skipping real inference test.")

    detector = FaceDetector(model_path=det_path, confidence_threshold=0.3)
    recognizer = FaceRecognizer(model_path=rec_path)

    # Test dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple face-like shape
    cv2.circle(frame, (320, 240), 60, (200, 200, 200), -1)
    cv2.circle(frame, (300, 220), 8, (20, 20, 20), -1)
    cv2.circle(frame, (340, 220), 8, (20, 20, 20), -1)

    dets = detector.detect_faces(frame)
    assert isinstance(dets, list)

    # Test recognizer embedding generation on crop
    crop = frame[180:300, 260:380]
    emb = recognizer.extract_embedding(crop)
    if emb is not None:
        assert emb.shape == (512,)
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-4
