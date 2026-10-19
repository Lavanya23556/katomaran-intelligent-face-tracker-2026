"""
Multi-face tracker with IoU matching and motion estimation.
Maintains consistent track IDs across frames and handles transient occlusions.
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import numpy as np


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute Intersection-over-Union between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union_area = areaA + areaB - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class TrackedFace:
    def __init__(self, track_id: int, bbox: List[float], confidence: float, crop: Optional[np.ndarray] = None, quality: float = 0.0):
        self.track_id: int = track_id
        self.bbox: List[float] = [float(v) for v in bbox]
        self.confidence: float = float(confidence)
        self.visitor_id: Optional[str] = None
        self.visit_id: Optional[str] = None
        self.is_new: bool = False
        self.state: str = "active"  # "active", "lost", "removed"
        self.age: int = 1
        self.hits: int = 1
        self.time_since_update: int = 0
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.best_face_crop: Optional[np.ndarray] = crop
        self.best_quality: float = float(quality)
        self.last_seen_time: float = 0.0

    def predict(self) -> List[float]:
        """Predict current bounding box based on velocity."""
        vx, vy = self.velocity
        x1, y1, x2, y2 = self.bbox
        return [x1 + vx, y1 + vy, x2 + vx, y2 + vy]

    def update(self, bbox: List[float], confidence: float, crop: Optional[np.ndarray] = None, quality: float = 0.0, current_time: float = 0.0) -> None:
        """Update track with newly matched detection."""
        new_bbox = [float(v) for v in bbox]
        # Calculate velocity based on center shift
        cx_old = (self.bbox[0] + self.bbox[2]) / 2.0
        cy_old = (self.bbox[1] + self.bbox[3]) / 2.0
        cx_new = (new_bbox[0] + new_bbox[2]) / 2.0
        cy_new = (new_bbox[1] + new_bbox[3]) / 2.0

        # Exponential moving average for velocity
        alpha = 0.4
        self.velocity = (
            alpha * (cx_new - cx_old) + (1.0 - alpha) * self.velocity[0],
            alpha * (cy_new - cy_old) + (1.0 - alpha) * self.velocity[1]
        )

        self.bbox = new_bbox
        self.confidence = float(confidence)
        self.age += 1
        self.hits += 1
        self.time_since_update = 0
        self.state = "active"
        self.last_seen_time = current_time

        if quality > self.best_quality and crop is not None:
            self.best_quality = quality
            self.best_face_crop = crop

    def mark_missed(self) -> None:
        """Increment time since update when detection is missed."""
        self.time_since_update += 1
        self.age += 1
        if self.time_since_update > 1:
            self.state = "lost"


class FaceTracker:
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.30):
        self.max_age: int = max_age
        self.iou_threshold: float = iou_threshold
        self._next_id: int = 1
        self.tracks: List[TrackedFace] = []

    def update(
        self,
        detections: List[Dict],
        current_time: float = 0.0
    ) -> List[TrackedFace]:
        """
        Associate detections with existing tracks using IoU matching.
        detections: list of dicts with keys:
            - 'bbox': [x1, y1, x2, y2]
            - 'confidence': float
            - 'crop': np.ndarray (optional)
            - 'quality': float (optional)
        Returns:
            List of active TrackedFace instances
        """
        # Predict locations for existing tracks
        predicted_boxes = [trk.predict() for trk in self.tracks]

        num_det = len(detections)
        num_trk = len(self.tracks)

        matched_trk: set = set()
        matched_det: set = set()

        if num_trk > 0 and num_det > 0:
            # Build IoU cost matrix
            iou_matrix = np.zeros((num_det, num_trk), dtype=np.float32)
            for d_idx, det in enumerate(detections):
                for t_idx, pred_box in enumerate(predicted_boxes):
                    iou_matrix[d_idx, t_idx] = compute_iou(det["bbox"], pred_box)

            # Greedy matching sorted by highest IoU
            indices = np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape)
            for d_idx, t_idx in zip(indices[0], indices[1]):
                if d_idx in matched_det or t_idx in matched_trk:
                    continue
                if iou_matrix[d_idx, t_idx] >= self.iou_threshold:
                    matched_det.add(d_idx)
                    matched_trk.add(t_idx)
                    det = detections[d_idx]
                    self.tracks[t_idx].update(
                        bbox=det["bbox"],
                        confidence=det["confidence"],
                        crop=det.get("crop"),
                        quality=det.get("quality", 0.0),
                        current_time=current_time,
                    )

        # Update unmatched tracks
        for t_idx, trk in enumerate(self.tracks):
            if t_idx not in matched_trk:
                trk.mark_missed()

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det:
                new_trk = TrackedFace(
                    track_id=self._next_id,
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    crop=det.get("crop"),
                    quality=det.get("quality", 0.0)
                )
                new_trk.last_seen_time = current_time
                self._next_id += 1
                self.tracks.append(new_trk)

        # Filter out tracks that exceeded max_age
        active_tracks: List[TrackedFace] = []
        remaining_tracks: List[TrackedFace] = []
        for trk in self.tracks:
            if trk.time_since_update > self.max_age:
                trk.state = "removed"
            else:
                remaining_tracks.append(trk)
                if trk.time_since_update == 0:
                    active_tracks.append(trk)

        self.tracks = remaining_tracks
        return active_tracks

    def get_all_tracks(self) -> List[TrackedFace]:
        """Return all tracks currently in memory (active or lost)."""
        return self.tracks

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self._next_id = 1
