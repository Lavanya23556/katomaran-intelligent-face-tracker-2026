"""Tracking and visit management package."""
from src.tracking.tracker import FaceTracker, TrackedFace, compute_iou
from src.tracking.visit_manager import VisitLifecycleManager, ActiveVisitorSession

__all__ = [
    "FaceTracker",
    "TrackedFace",
    "compute_iou",
    "VisitLifecycleManager",
    "ActiveVisitorSession",
]
