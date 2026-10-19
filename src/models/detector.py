"""
YOLO-based Face Detector using verified face weights.
Handles multi-face detection, quality estimation, and frame validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.logger import logger


def calculate_face_quality(crop: np.ndarray) -> float:
    """
    Compute blur/sharpness score using Laplacian variance.
    Higher values indicate sharper image.
    """
    if crop is None or crop.size == 0:
        return 0.0
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(var)


class FaceDetector:
    def __init__(
        self,
        model_path: str = "weights/yolov8n-face.pt",
        confidence_threshold: float = 0.50,
        min_face_size: int = 40,
        min_face_quality: float = 35.0,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.min_face_quality = min_face_quality
        self.device = device
        self._model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO face detector with validation."""
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Face detector model file not found at '{self.model_path}'. "
                "Ensure verified YOLO face weights are downloaded."
            )

        try:
            self._model = YOLO(str(path))
            # Verify class names: should have 'face'
            class_names = self._model.names
            if isinstance(class_names, dict):
                names_lower = [str(v).lower() for v in class_names.values()]
            else:
                names_lower = [str(v).lower() for v in class_names]

            if not any("face" in n for n in names_lower):
                logger.warning(
                    f"Model classes at {self.model_path} are {class_names}. "
                    "Expected a face-specific detector (class 0: 'face')."
                )
            else:
                logger.info(f"Loaded YOLO face detector from {self.model_path} with classes: {class_names}")

        except Exception as e:
            logger.error(f"Failed to load YOLO model from {self.model_path}: {e}")
            raise RuntimeError(f"Failed to initialize face detector: {e}") from e

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a video frame.
        Returns:
            List of dicts: [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'crop': np.ndarray,
                    'quality': float,
                    'is_valid_quality': bool
                }, ...
            ]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return []

        # Run inference with fast resolution scaling (384px) for high CPU FPS
        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device if self.device != "auto" else None,
            verbose=False,
            imgsz=384
        )

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf < self.confidence_threshold:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x1 = max(0, min(int(xyxy[0]), w - 1))
            y1 = max(0, min(int(xyxy[1]), h - 1))
            x2 = max(0, min(int(xyxy[2]), w))
            y2 = max(0, min(int(xyxy[3]), h))

            bw = x2 - x1
            bh = y2 - y1

            if bw < self.min_face_size or bh < self.min_face_size:
                # Reject transient tiny detections below minimum size
                continue

            # Extract face crop
            crop = frame[y1:y2, x1:x2].copy()
            quality = calculate_face_quality(crop)
            is_valid_quality = quality >= self.min_face_quality

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "crop": crop,
                "quality": quality,
                "is_valid_quality": is_valid_quality,
            })

        return detections
