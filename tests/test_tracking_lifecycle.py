"""Tests for multi-face tracking and visit lifecycle states."""
import numpy as np
import pytest

from src.tracking.tracker import FaceTracker, TrackedFace, compute_iou
from src.tracking.visit_manager import VisitLifecycleManager


def test_compute_iou():
    box1 = [10.0, 10.0, 50.0, 50.0]
    box2 = [10.0, 10.0, 50.0, 50.0]
    assert compute_iou(box1, box2) == 1.0

    box3 = [60.0, 60.0, 100.0, 100.0]
    assert compute_iou(box1, box3) == 0.0

    # Half overlap
    box4 = [10.0, 10.0, 30.0, 50.0]  # area 20*40=800; union: 1600+800-800 = 1600; iou=0.5
    assert abs(compute_iou(box1, box4) - 0.5) < 1e-4


def test_tracker_lifecycle():
    tracker = FaceTracker(max_age=3, iou_threshold=0.3)

    # Frame 1: Detection appears
    det1 = [{"bbox": [50, 50, 100, 100], "confidence": 0.9}]
    tracks1 = tracker.update(det1, current_time=1.0)
    assert len(tracks1) == 1
    trk_id = tracks1[0].track_id

    # Frame 2: Slightly shifted detection -> should maintain same track ID
    det2 = [{"bbox": [52, 53, 102, 103], "confidence": 0.92}]
    tracks2 = tracker.update(det2, current_time=2.0)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == trk_id

    # Frame 3, 4, 5: Missed detections
    tracks3 = tracker.update([], current_time=3.0)
    assert len(tracks3) == 0  # not active
    tracker.update([], current_time=4.0)
    tracker.update([], current_time=5.0)

    # Frame 6: Should be removed after max_age=3
    tracker.update([], current_time=6.0)
    assert len(tracker.get_all_tracks()) == 0


def test_visit_lifecycle_entry_and_grace_exit(test_repo, test_image_store, sample_face_crop):
    manager = VisitLifecycleManager(
        repository=test_repo,
        image_store=test_image_store,
        exit_grace_seconds=2.0,
        source_id="test_cam"
    )

    trk = TrackedFace(track_id=1, bbox=[50, 50, 100, 100], confidence=0.9, crop=sample_face_crop, quality=80.0)

    # 1. First recognized detection at t=0.0 -> Entry event triggered
    manager.process_recognized_track(
        track=trk,
        face_crop=sample_face_crop,
        visitor_id="VIS-0001",
        is_new_visitor=True,
        similarity=1.0,
        confidence=0.9,
        current_time=0.0
    )

    stats = test_repo.get_stats()
    assert stats["total_entries"] == 1
    assert stats["active_visits"] == 1
    assert "VIS-0001" in manager.active_sessions
    visit_id = manager.active_sessions["VIS-0001"].visit_id

    # 2. Track continues visible at t=1.0 -> No duplicate entry
    manager.update_unseen_tracks([trk], current_time=1.0)
    stats2 = test_repo.get_stats()
    assert stats2["total_entries"] == 1

    # 3. Track disappears at t=2.0 -> Grace period starts
    exits = manager.update_unseen_tracks([], current_time=2.0)
    assert len(exits) == 0
    assert manager.active_sessions["VIS-0001"].pending_exit is True

    # 4. Track reappears at t=2.8 (within 2.0s grace) -> Grace restored, no exit
    manager.process_recognized_track(
        track=trk,
        face_crop=sample_face_crop,
        visitor_id="VIS-0001",
        is_new_visitor=False,
        similarity=0.95,
        confidence=0.9,
        current_time=2.8
    )
    assert manager.active_sessions["VIS-0001"].pending_exit is False
    assert test_repo.get_stats()["total_exits"] == 0

    # 5. Track disappears again at t=3.0
    manager.update_unseen_tracks([], current_time=3.0)

    # 6. Time advances past grace period to t=5.5 (> 3.0 + 2.0s) -> Exit triggered!
    finalized = manager.update_unseen_tracks([], current_time=5.5)
    assert "VIS-0001" in finalized

    stats_final = test_repo.get_stats()
    assert stats_final["total_entries"] == 1
    assert stats_final["total_exits"] == 1
    assert stats_final["active_visits"] == 0
