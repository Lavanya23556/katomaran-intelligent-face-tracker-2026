"""
Critical regression tests covering all 15 mandatory scenarios
and 1,000-iteration stress testing of idempotency and identity deduplication.
"""
import os
import shutil
import tempfile
import numpy as np
import pytest

from src.config import SystemConfig
from src.database.schema import init_database
from src.database.repository import DatabaseRepository
from src.tracking.tracker import FaceTracker, TrackedFace
from src.tracking.visit_manager import VisitLifecycleManager
from src.utils.image_store import ImageStore


# Scenario 1: A new visitor registers once.
def test_scenario_01_new_visitor_registers_once(test_repo, sample_embedding):
    vid, is_new, _ = test_repo.match_or_register_visitor(sample_embedding)
    assert is_new is True
    assert test_repo.get_stats()["unique_visitors"] == 1


# Scenario 2: Ten consecutive detections of the same identity do not produce ten visitor registrations.
def test_scenario_02_consecutive_detections_no_duplicate_registrations(test_repo, sample_embedding):
    vid_initial, _, _ = test_repo.match_or_register_visitor(sample_embedding)

    for i in range(10):
        # Slightly jitter embedding (within similarity tolerance)
        jitter = sample_embedding + np.random.normal(0, 0.01, size=sample_embedding.shape)
        jitter /= np.linalg.norm(jitter)
        vid, is_new, sim = test_repo.match_or_register_visitor(jitter, similarity_threshold=0.60)
        assert is_new is False
        assert vid == vid_initial

    assert test_repo.get_stats()["unique_visitors"] == 1


# Scenario 3: The same visitor returns in a later session without increasing the unique count.
def test_scenario_03_visitor_returns_in_later_session(test_repo, sample_embedding):
    vid, _, _ = test_repo.match_or_register_visitor(sample_embedding)
    assert test_repo.get_stats()["unique_visitors"] == 1

    # Session 1: Entry & Exit
    vst1, _ = test_repo.start_visit(vid, "source1", "entry1.jpg")
    test_repo.finalize_visit(vst1, vid, "source1", "exit1.jpg")

    assert test_repo.get_stats()["unique_visitors"] == 1
    assert test_repo.get_stats()["total_entries"] == 1
    assert test_repo.get_stats()["total_exits"] == 1

    # Session 2: Same visitor returns later
    vid_recog, is_new, _ = test_repo.match_or_register_visitor(sample_embedding)
    assert is_new is False
    assert vid_recog == vid

    vst2, _ = test_repo.start_visit(vid_recog, "source1", "entry2.jpg")
    test_repo.finalize_visit(vst2, vid_recog, "source1", "exit2.jpg")

    # Unique count MUST still be 1; visits & events incremented
    assert test_repo.get_stats()["unique_visitors"] == 1
    assert test_repo.get_stats()["total_entries"] == 2
    assert test_repo.get_stats()["total_exits"] == 2


# Scenario 4: One completed visit has exactly one entry and one exit.
def test_scenario_04_completed_visit_exactly_one_entry_one_exit(test_repo, sample_embedding):
    vid, _, _ = test_repo.match_or_register_visitor(sample_embedding)
    visit_id, entry_event_id = test_repo.start_visit(vid, "source1", "entry.jpg")
    exit_event_id = test_repo.finalize_visit(visit_id, vid, "source1", "exit.jpg")

    events = test_repo.get_recent_events(limit=100)
    visit_events = [e for e in events if e["visit_id"] == visit_id]

    assert len(visit_events) == 2
    types = [e["event_type"] for e in visit_events]
    assert types.count("entry") == 1
    assert types.count("exit") == 1


# Scenario 5: Duplicate exit callbacks cannot produce duplicate exit events.
def test_scenario_05_duplicate_exit_callbacks_idempotency(test_repo, sample_embedding):
    vid, _, _ = test_repo.match_or_register_visitor(sample_embedding)
    visit_id, _ = test_repo.start_visit(vid, "source1", "entry.jpg")

    res1 = test_repo.finalize_visit(visit_id, vid, "source1", "exit1.jpg")
    res2 = test_repo.finalize_visit(visit_id, vid, "source1", "exit2.jpg")
    res3 = test_repo.finalize_visit(visit_id, vid, "source1", "exit3.jpg")

    assert res1 is not None
    assert res2 is None
    assert res3 is None
    assert test_repo.get_stats()["total_exits"] == 1


# Scenario 6: A brief missed detection does not prematurely close a visit.
def test_scenario_06_brief_missed_detection_preserves_visit(test_repo, test_image_store, sample_face_crop):
    manager = VisitLifecycleManager(test_repo, test_image_store, exit_grace_seconds=3.0, source_id="cam")
    trk = TrackedFace(track_id=1, bbox=[10, 10, 60, 60], confidence=0.9, crop=sample_face_crop, quality=60.0)

    # Visitor enters at t=0.0
    manager.process_recognized_track(trk, sample_face_crop, "VIS-001", True, 1.0, 0.9, current_time=0.0)
    assert len(manager.active_sessions) == 1

    # Brief missed detection for 1.5 seconds (less than 3.0s grace)
    manager.update_unseen_tracks([], current_time=1.5)
    assert len(manager.active_sessions) == 1
    assert test_repo.get_stats()["total_exits"] == 0

    # Detection resumes at t=2.0
    manager.process_recognized_track(trk, sample_face_crop, "VIS-001", False, 1.0, 0.9, current_time=2.0)
    assert manager.active_sessions["VIS-001"].pending_exit is False
    assert test_repo.get_stats()["total_exits"] == 0


# Scenario 7: A genuinely absent visitor is marked as exited after configured grace period.
def test_scenario_07_absent_visitor_marked_exited_after_grace(test_repo, test_image_store, sample_face_crop):
    manager = VisitLifecycleManager(test_repo, test_image_store, exit_grace_seconds=2.0, source_id="cam")
    trk = TrackedFace(track_id=1, bbox=[10, 10, 60, 60], confidence=0.9, crop=sample_face_crop, quality=60.0)

    manager.process_recognized_track(trk, sample_face_crop, "VIS-001", True, 1.0, 0.9, current_time=0.0)
    # Absent starts at t=1.0
    manager.update_unseen_tracks([], current_time=1.0)
    # Time reaches t=3.5 (> 1.0 + 2.0 grace)
    exited = manager.update_unseen_tracks([], current_time=3.5)

    assert "VIS-001" in exited
    assert len(manager.active_sessions) == 0
    assert test_repo.get_stats()["total_exits"] == 1


# Scenario 8: Two distinct faces can be tracked without sharing the same identity accidentally.
def test_scenario_08_two_distinct_faces_tracked_independently(test_repo, test_image_store, sample_face_crop):
    manager = VisitLifecycleManager(test_repo, test_image_store, exit_grace_seconds=2.0, source_id="cam")
    trk1 = TrackedFace(track_id=1, bbox=[10, 10, 60, 60], confidence=0.9, crop=sample_face_crop, quality=60.0)
    trk2 = TrackedFace(track_id=2, bbox=[200, 200, 250, 250], confidence=0.88, crop=sample_face_crop, quality=55.0)

    manager.process_recognized_track(trk1, sample_face_crop, "VIS-001", True, 1.0, 0.9, current_time=0.0)
    manager.process_recognized_track(trk2, sample_face_crop, "VIS-002", True, 1.0, 0.88, current_time=0.0)

    assert len(manager.active_sessions) == 2
    assert manager.active_sessions["VIS-001"].primary_track_id == 1
    assert manager.active_sessions["VIS-002"].primary_track_id == 2
    assert test_repo.get_stats()["active_visits"] == 2


# Scenario 9: A database restart preserves visitors, visits and event history.
def test_scenario_09_database_restart_preserves_records(test_config, sample_embedding):
    repoA = DatabaseRepository(test_config.database_path)
    vid, _, _ = repoA.match_or_register_visitor(sample_embedding)
    vst, evt_in = repoA.start_visit(vid, "source1", "entry.jpg")
    evt_out = repoA.finalize_visit(vst, vid, "source1", "exit.jpg")

    # Restart simulated by creating fresh repo instance
    repoB = DatabaseRepository(test_config.database_path)
    stats = repoB.get_stats()
    assert stats["unique_visitors"] == 1
    assert stats["total_entries"] == 1
    assert stats["total_exits"] == 1
    assert len(repoB.get_recent_events()) == 2


# Scenario 10: Failed camera connection produces controlled error.
def test_scenario_10_failed_camera_connection_controlled_error(test_config, test_repo, test_image_store):
    from src.pipeline.stream_manager import StreamManager
    sm = StreamManager(test_config, test_repo, test_image_store, detector=None, recognizer=None)

    # 1. Non-existent file source returns clear, actionable error
    success1, msg1 = sm.start_stream(source_type="file", source_path="data/nonexistent_camera_feed.mp4")
    assert success1 is False
    assert "not found" in msg1.lower()
    assert sm.is_running is False

    # 2. Invalid RTSP scheme returns validation error
    success2, msg2 = sm.start_stream(source_type="rtsp", source_path="http://not-an-rtsp-stream")
    assert success2 is False
    assert "invalid rtsp" in msg2.lower()
    assert sm.is_running is False


# Scenario 11: Processing stops cleanly.
def test_scenario_11_processing_stops_cleanly(test_config, test_repo, test_image_store):
    from src.pipeline.stream_manager import StreamManager
    sm = StreamManager(test_config, test_repo, test_image_store, detector=None, recognizer=None)

    # Start and stop cleanly
    sm.start_stream(source_type="file", source_path="data/sample_videos/video_sample1.mp4")
    import time
    time.sleep(0.2)
    success, msg = sm.stop_stream()
    assert success is True
    assert sm.is_running is False


# Scenario 12: Invalid configuration produces actionable error.
def test_scenario_12_invalid_configuration_actionable_error():
    with pytest.raises(ValueError, match="rtsp_url must be provided"):
        SystemConfig(video_source_type="rtsp", rtsp_url=None)


# Scenario 13: Dashboard API returns real database results.
def test_scenario_13_api_returns_real_database_results(test_config, test_repo, sample_embedding):
    from fastapi.testclient import TestClient
    from src.api.app import create_app

    # Pre-populate database
    vid, _, _ = test_repo.match_or_register_visitor(sample_embedding)
    vst, _ = test_repo.start_visit(vid, "source1", "entry.jpg")

    app = create_app(test_config)
    with TestClient(app) as client:
        res = client.get("/api/stats")
        assert res.status_code == 200
        data = res.json()
        assert data["unique_visitors"] == 1
        assert data["active_visits"] == 1
        assert data["total_entries"] == 1


# Scenario 14: Frontend controls invoke expected behaviour.
def test_scenario_14_frontend_controls_invoke_expected_behaviour(test_config):
    from fastapi.testclient import TestClient
    from src.api.app import create_app

    app = create_app(test_config)
    with TestClient(app) as client:
        # Check start stream control
        res_start = client.post("/api/streams/start", json={
            "source_type": "file",
            "source_path": "data/sample_videos/video_sample1.mp4",
            "loop": False
        })
        assert res_start.status_code in (200, 400)

        # Check stop stream control
        res_stop = client.post("/api/streams/stop")
        assert res_stop.status_code == 200
        assert res_stop.json()["success"] is True


# Scenario 15: Required logs and image directories created correctly.
def test_scenario_15_required_directories_created(temp_dir):
    cfg = SystemConfig(
        database_path=os.path.join(temp_dir, "db", "app.db"),
        events_log_path=os.path.join(temp_dir, "logs_test", "events.log"),
        entries_dir=os.path.join(temp_dir, "img", "entries"),
        exits_dir=os.path.join(temp_dir, "img", "exits"),
    )
    cfg.ensure_directories()

    assert os.path.exists(os.path.dirname(cfg.database_path))
    assert os.path.exists(os.path.dirname(cfg.events_log_path))
    assert os.path.exists(cfg.entries_dir)
    assert os.path.exists(cfg.exits_dir)


# STRESS TEST: 1,000 iterations of identity deduplication and idempotency
def test_1000_iterations_identity_deduplication_stress(temp_dir):
    """
    Stress test verifying that 1,000 detections of the same visitor
    across rapid cycles strictly preserve 1 unique visitor count and idempotency.
    """
    db_path = os.path.join(temp_dir, "stress_test.db")
    init_database(db_path)
    repo = DatabaseRepository(db_path)

    # Base embedding
    np.random.seed(123)
    base_emb = np.random.randn(512).astype(np.float32)
    base_emb /= np.linalg.norm(base_emb)

    primary_id, is_new, _ = repo.match_or_register_visitor(base_emb, similarity_threshold=0.60)
    assert is_new is True

    # 1,000 iterations with noise
    for i in range(1000):
        noisy_emb = base_emb + np.random.normal(0, 0.02, size=512).astype(np.float32)
        noisy_emb /= np.linalg.norm(noisy_emb)

        matched_id, is_new_det, sim = repo.match_or_register_visitor(noisy_emb, similarity_threshold=0.60)
        assert is_new_det is False
        assert matched_id == primary_id
        assert sim >= 0.85

    # Verification: Exactly 1 unique visitor after 1,000 iterations
    stats = repo.get_stats()
    assert stats["unique_visitors"] == 1
