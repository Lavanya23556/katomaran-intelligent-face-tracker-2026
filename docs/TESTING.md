# Testing & Verification Report — VisiTrace

## Test Strategy Overview

The testing suite rigorously verifies end-to-end functionality across configuration loading, model inference, tracking state transitions, database ACID properties, API contracts, and critical regression scenarios.

All tests are deterministic, automated via `pytest`, and executed against local SQLite test databases and mock/real inputs.

## Test Summary

```
============================== test session starts ==============================
Platform: Windows (Python 3.12.10, pytest-9.1.1)
Total Tests: 40
Passed: 40
Failed: 0
Skipped: 0
Duration: 9.33s
=================================================================================
```

## Detailed Test Coverage

### 1. Unit Tests (`tests/test_config.py`, `tests/test_database.py`, `tests/test_detector_recognizer.py`, `tests/test_tracking_lifecycle.py`)
- **Config**: Validates required default keys, frame skip bounds, and RTSP password masking.
- **Database**:
  - `PRAGMA journal_mode = WAL;` and foreign keys enabled.
  - New visitor auto-registration.
  - Cosine similarity matching of existing visitors.
  - Single entry and single exit per visit session.
  - Unique constraint `uq_visit_event` idempotency.
  - Database persistence across simulated application restarts.
- **Models**:
  - Laplacian variance sharpness calculation on sharp vs flat images.
  - ArcFace preprocessing tensor shape `(1, 3, 112, 112)` and normalized float range.
  - Cosine similarity mathematical identity ($v_1 \cdot v_1 = 1.0$, $v_1 \cdot v_{\perp} = 0.0$).
  - Real model loading and inference smoke test with `weights/yolov8n-face.pt` and `weights/w600k_mbf.onnx`.
- **Tracking Lifecycle**:
  - IoU cost matrix calculation (identical box = 1.0, disjoint = 0.0).
  - Track ID persistence across moved bounding boxes.
  - Track termination after `max_age` missed frames.
  - Visit entry creation upon track recognition.
  - Exit grace period handling and session recovery on reappearance.

### 2. Critical Regression Scenarios (`tests/test_regressions.py`)
Each of the 15 problem statement scenarios is tested:

1. **Scenario 1**: New visitor registers once (`test_scenario_01_new_visitor_registers_once`).
2. **Scenario 2**: Ten consecutive detections of the same identity do not produce 10 visitor registrations (`test_scenario_02_consecutive_detections_no_duplicate_registrations`).
3. **Scenario 3**: The same visitor returns in a later session without increasing the unique count (`test_scenario_03_visitor_returns_in_later_session`).
4. **Scenario 4**: One completed visit has exactly one entry and one exit (`test_scenario_04_completed_visit_exactly_one_entry_one_exit`).
5. **Scenario 5**: Duplicate exit callbacks cannot produce duplicate exit events (`test_scenario_05_duplicate_exit_callbacks_idempotency`).
6. **Scenario 6**: A brief missed detection does not prematurely close a visit (`test_scenario_06_brief_missed_detection_preserves_visit`).
7. **Scenario 7**: A genuinely absent visitor is marked as exited after configured grace period (`test_scenario_07_absent_visitor_marked_exited_after_grace`).
8. **Scenario 8**: Two distinct faces can be tracked without sharing the same identity accidentally (`test_scenario_08_two_distinct_faces_tracked_independently`).
9. **Scenario 9**: Database restart preserves visitors, visits, and event history (`test_scenario_09_database_restart_preserves_records`).
10. **Scenario 10**: Failed camera connection produces controlled error (`test_scenario_10_failed_camera_connection_controlled_error`).
11. **Scenario 11**: Processing stops cleanly and releases resources (`test_scenario_11_processing_stops_cleanly`).
12. **Scenario 12**: Invalid configuration produces actionable error (`test_scenario_12_invalid_configuration_actionable_error`).
13. **Scenario 13**: Dashboard API returns real database results (`test_scenario_13_api_returns_real_database_results`).
14. **Scenario 14**: Frontend controls invoke expected behaviour (`test_scenario_14_frontend_controls_invoke_expected_behaviour`).
15. **Scenario 15**: Required logs and image directories are created correctly (`test_scenario_15_required_directories_created`).

### 3. High-Iteration Stress Testing
- **1,000 Iterations Identity Deduplication Stress Test** (`test_1000_iterations_identity_deduplication_stress`):
  - 1,000 consecutive noisy embeddings of the same person matched against the database repository.
  - Successfully verified: Unique visitor count strictly remained **1**, with zero duplicate identity records created.

### 4. Real Video Processing Verification
- Executed on `data/sample_videos/video_sample1.mp4`:
  - Processed 120 frames at 14.6 FPS on CPU.
  - Successfully registered 16 unique visitors.
  - Logged 16 entry events and 16 exit events.
  - Saved 32 date-organized face crops into `logs/entries/2026-10-19/` and `logs/exits/2026-10-19/`.
  - Recorded detailed timeline in `events.log`.
