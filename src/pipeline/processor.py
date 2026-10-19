"""
Frame processing pipeline for VisiTrace.
Coordinates face detection, ArcFace embedding, tracking, and visit logging.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.config import SystemConfig
from src.database.repository import DatabaseRepository
from src.models.detector import FaceDetector
from src.models.recognizer import FaceRecognizer
from src.tracking.tracker import FaceTracker, TrackedFace
from src.tracking.visit_manager import VisitLifecycleManager
from src.utils.image_store import ImageStore
from src.utils.visualizer import FrameVisualizer
from src.utils.logger import logger


class FrameProcessor:
    def __init__(
        self,
        config: SystemConfig,
        repository: DatabaseRepository,
        image_store: ImageStore,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        source_id: str = "camera_default",
    ):
        self.config = config
        self.repo = repository
        self.image_store = image_store
        self.detector = detector
        self.recognizer = recognizer
        self.source_id = source_id

        self.tracker = FaceTracker(
            max_age=config.tracker_max_age,
            iou_threshold=config.tracker_iou_threshold,
        )
        self.visit_manager = VisitLifecycleManager(
            repository=repository,
            image_store=image_store,
            exit_grace_seconds=config.exit_grace_seconds,
            source_id=source_id,
        )

        self.frame_count: int = 0
        self.last_detections: List[Dict[str, Any]] = []
        self._fps_time = time.time()
        self._fps_count: int = 0
        self.current_fps: float = 0.0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a single video frame:
        1. Run face detector every `frame_skip` frames.
        2. Update face tracker with detections.
        3. Extract embeddings and match/register new visitor identities.
        4. Manage visit entry and exit grace cycles.
        5. Render visual overlays.
        Returns:
            (annotated_frame, active_tracks_info, processing_stats)
        """
        now = timestamp if timestamp is not None else time.time()
        self.frame_count += 1
        self._fps_count += 1

        # Calculate FPS periodically
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self.current_fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_time = now

        # Frame skipping logic
        should_detect = (self.frame_count % self.config.frame_skip == 0) or (self.frame_count == 1)

        if should_detect:
            try:
                self.last_detections = self.detector.detect_faces(frame)
            except Exception as e:
                logger.error(f"Detector error on frame {self.frame_count}: {e}")
                self.last_detections = []

        # Update tracker
        active_tracks = self.tracker.update(self.last_detections, current_time=now)

        # Process recognition on tracks needing identity
        for trk in active_tracks:
            # Check if this track already has an identity
            if not trk.visitor_id:
                # Use best face crop available
                crop_to_use = trk.best_face_crop
                if crop_to_use is not None and trk.best_quality >= self.config.min_face_quality:
                    emb = self.recognizer.extract_embedding(crop_to_use)
                    if emb is not None:
                        vid, is_new, sim = self.repo.match_or_register_visitor(
                            embedding=emb,
                            similarity_threshold=self.config.recognition_similarity_threshold,
                            quality_score=trk.best_quality,
                            metadata={"track_id": trk.track_id}
                        )
                        self.visit_manager.process_recognized_track(
                            track=trk,
                            face_crop=crop_to_use,
                            visitor_id=vid,
                            is_new_visitor=is_new,
                            similarity=sim,
                            confidence=trk.confidence,
                            current_time=now,
                        )
            else:
                # Track already identified; keep active visit alive
                self.visit_manager.process_recognized_track(
                    track=trk,
                    face_crop=trk.best_face_crop if trk.best_face_crop is not None else frame,
                    visitor_id=trk.visitor_id,
                    is_new_visitor=trk.is_new,
                    similarity=1.0,
                    confidence=trk.confidence,
                    current_time=now,
                )

        # Update unseen tracks and apply exit grace duration
        self.visit_manager.update_unseen_tracks(active_tracks, current_time=now)

        # Prepare track metadata for visualization & API
        tracks_info = []
        for trk in active_tracks:
            tracks_info.append({
                "track_id": trk.track_id,
                "bbox": trk.bbox,
                "confidence": trk.confidence,
                "visitor_id": trk.visitor_id or "Identifying...",
                "is_new": trk.is_new,
                "quality": trk.best_quality,
            })

        # Get system statistics for HUD
        stats = self.repo.get_stats()
        annotated_frame = FrameVisualizer.draw_detections(
            frame=frame,
            tracks=tracks_info,
            fps=self.current_fps,
            source_label=self.source_id,
            unique_count=stats["unique_visitors"],
            active_count=len(self.visit_manager.active_sessions),
        )

        return annotated_frame, tracks_info, {
            "fps": self.current_fps,
            "frame_count": self.frame_count,
            "active_tracks": len(active_tracks),
            "active_visits": len(self.visit_manager.active_sessions),
            "unique_visitors": stats["unique_visitors"],
        }

    def close(self) -> None:
        """Close visits upon completion."""
        self.visit_manager.close_all_active_visits(reason="stream_closed")
