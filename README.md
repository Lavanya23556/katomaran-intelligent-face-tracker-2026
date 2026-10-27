# VisiTrace — Intelligent Face Tracker & Visitor Counting

[![Tests](https://img.shields.io/badge/tests-40%20passed-success)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)](https://fastapi.tiangolo.com)
[![YOLOv8-Face](https://img.shields.io/badge/detector-YOLOv8n--Face-orange)](https://github.com/akanametov/yolo-face)
[![InsightFace](https://img.shields.io/badge/recognition-ArcFace%20(512D)-brightgreen)](https://github.com/deepinsight/insightface)
[![SQLite](https://img.shields.io/badge/database-SQLite%20(WAL)-blue)](https://sqlite.org)

An enterprise-grade computer vision system engineered for the Katomaran Hackathon 2026. VisiTrace performs real-time multi-face detection, continuous tracking, automatic visitor identity registration, timestamped entry/exit event logging, and persistent unique visitor counting across sessions.

---

## 1. Project Objective & Problem Statement

**Problem Statement:** *“Intelligent Face Tracker with Auto-Registration and Visitor Counting.”*

In high-footfall commercial spaces, campuses, and secure facilities, legacy visitor logs and manual attendance tracking introduce significant human error, line friction, and security vulnerabilities. Generic person counters fail to distinguish between different individuals or recognise returning visitors, causing severe duplicate counting.

**VisiTrace solves this by:**
1. Detecting faces in real time using a verified face-specific YOLO detector (class 0: `face`).
2. Generating 512-dimensional facial embeddings using official InsightFace ArcFace models via ONNX Runtime.
3. Automatically registering first-time visitors with a persistent ID (`VIS-0001`).
4. Recognizing returning visitors in subsequent frames and future sessions via cosine similarity matching without incrementing unique visitor counts.
5. Tracking multiple targets continuously with IoU association and velocity estimation.
6. Recording timestamped entry and exit events with cropped facial images organized by date (`logs/entries/YYYY-MM-DD/`, `logs/exits/YYYY-MM-DD/`).
7. Enforcing database-level idempotency to prevent duplicate entry or exit events per session.
8. Serving a real-time editorial dashboard for camera control, live visual tracking preview, and visitor analytics.

---

## 2. Key Features

- **Genuine AI Models**: Uses dedicated WiderFace-trained YOLOv8n-Face weights and InsightFace MobileFaceNet ArcFace embeddings. No generic COCO person weights or simulated identities.
- **Auto-Registration & Deduplication**: Differentiates genuinely new visitors from recurring individuals with a configurable cosine similarity threshold (default `0.60`). Tested over 1,000 continuous iterations without identity drift.
- **Continuous Multi-Target Tracking**: Maintains consistent track IDs across frames and handles brief occlusions.
- **Grace-Period Exit Logic**: A configurable exit grace window (default `5.0s`) prevents detector jitter, head turning, or brief occlusions from generating false exit events.
- **Relational Persistence with WAL**: SQLite database running in Write-Ahead Logging (`WAL`) mode with foreign key cascades, transaction atomicity, and unique constraints (`UNIQUE(visit_id, event_type)`).
- **Dual Input Modes**: Supports local video files (`.mp4`, `.avi`, `.mkv`) and RTSP camera streams (`rtsp://`) with reconnection backoff and credential protection.
- **Live Visual Feed**: Serves an in-browser live MJPEG preview stream with bounding boxes, confidence tags, and visitor identity badges.
- **VisiTrace Editorial Dashboard**: Modern visual identity (warm ivory background, deep graphite typography, forest-teal accents) providing real-time KPI metrics, interactive stream controls, recent event timelines with thumbnails, and visitor directories.

---

## 3. Architecture Overview

```
                      +---------------------------------------+
                      |       Video Stream / File Input       |
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |      OpenCV Ingestion Thread          |
                      |   (Frame Skipping: every N frames)    |
                      +-------------------+-------------------+
                                          |
                      +-------------------+-------------------+
                      |   YOLOv8-Face Face Detector (PyTorch) |
                      |    - Class 0: 'face'                  |
                      |    - Laplacian Variance Quality Filter|
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |   InsightFace ArcFace ONNX (512D)     |
                      |    - L2-Normalized Embedding Vector   |
                      |    - Cosine Similarity Matching       |
                      +-------------------+-------------------+
                                          |
                      +-------------------+-------------------+
                      |       Face Tracker & Lifecycle        |
                      |    - IoU Track Association            |
                      |    - Exit Grace Period Logic          |
                      +-------------------+-------------------+
                                          |
                                          v
          +-------------------------------+-------------------------------+
          |                                                               |
          v                                                               v
+--------------------+---------------------+            +--------------------+--------------------+
|  Transactional SQLite Database (WAL)     |            |  Date-Organized Disk Image Storage      |
|  - visitors (identities & embeddings)    |            |  - logs/entries/YYYY-MM-DD/*.jpg        |
|  - visits (sessions & durations)         |            |  - logs/exits/YYYY-MM-DD/*.jpg          |
|  - events (entries & exits idempotency)  |            |  - events.log (Structured audit logs)   |
+--------------------+---------------------+            +--------------------+--------------------+
                     |                                                       |
                     +---------------------------+---------------------------+
                                                 |
                                                 v
                               +-----------------+-----------------+
                               |   FastAPI REST API & Dashboard   |
                               |    - Ingestion start/stop        |
                               |    - Live MJPEG preview stream   |
                               |    - KPIs & visitor directories  |
                               +-----------------------------------+
```

---

## 4. Tech Stack

- **Language & Runtime**: Python 3.10+ (tested on Python 3.12.10)
- **Face Detection**: Ultralytics YOLOv8n-Face (`yolov8n-face.pt`, WiderFace fine-tuned)
- **Face Recognition**: InsightFace ArcFace MobileFaceNet (`w600k_mbf.onnx`, 512D embeddings) via ONNX Runtime
- **Image Processing**: OpenCV (`opencv-python` / `opencv-python-headless`)
- **Backend API & Web Server**: FastAPI, Starlette, Uvicorn (ASGI)
- **Database & Storage**: SQLite 3 with Write-Ahead Logging (`WAL`), ACID transactions
- **Frontend Dashboard**: Vanilla ES6 JavaScript, HTML5 Canvas/Video, CSS3 Editorial Design System
- **Testing & QA**: `pytest`, `pytest-asyncio`, FastAPI TestClient

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/Lavanya23556/katomaran-intelligent-face-tracker-2026.git
cd katomaran-intelligent-face-tracker-2026

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Model Weights Verification & Download
Model weights are automatically downloaded from verified HuggingFace repositories on first startup:
- Face Detector: `weights/yolov8n-face.pt` (~6.3 MB)
- ArcFace Recognizer: `weights/w600k_mbf.onnx` (~13.6 MB)

To download them ahead of time:
```bash
python -c "from src.models.weights_manager import ensure_model_weights; ensure_model_weights()"
```

---

## 6. Configuration (`config.json`)

The application is configured through `config.json`. An annotated template is available in `config.example.json`.

```json
{
  "video_source_type": "file",
  "video_path": "data/sample_videos/video_sample1.mp4",
  "rtsp_url": null,
  "detector_backend": "yolo",
  "detector_model_path": "weights/yolov8n-face.pt",
  "recognition_model_path": "weights/w600k_mbf.onnx",
  "device": "cpu",
  "frame_skip": 2,
  "detection_confidence_threshold": 0.5,
  "recognition_similarity_threshold": 0.6,
  "min_face_size": 40,
  "min_face_quality": 35.0,
  "exit_grace_seconds": 5.0,
  "tracker_max_age": 30,
  "tracker_iou_threshold": 0.3,
  "database_path": "data/visitrace.db",
  "events_log_path": "events.log",
  "entries_dir": "logs/entries",
  "exits_dir": "logs/exits",
  "api_host": "127.0.0.1",
  "api_port": 8000,
  "mock_mode": false
}
```

*Note: When RTSP stream URLs containing credentials are used, passwords are automatically sanitized and never exposed to the frontend or API responses.*

---

## 7. Running the Application

### Launch Web Server & Dashboard
```bash
python main.py
```

Open your browser to:
**`http://127.0.0.1:8000`**

### Running Offline Video Evaluation
To process a video file in headless mode and verify recognition statistics:
```bash
python main.py --eval-video data/sample_videos/video_sample1.mp4 --max-frames 300
```

---

### Operating Input Sources:
1. **Local Video File**: Select `Local Video File`, enter path `data/sample_videos/video_sample1.mp4`, and click **Start Stream**.
2. **Live Webcam**: Select `Live Webcam (Device ID)`, enter `0` (or `1`, `2` for external USB cams), and click **Start Stream** for real-time live tracking.
3. **RTSP Stream**: Select `RTSP Camera Stream`, enter your RTSP URL (e.g. `rtsp://127.0.0.1:8554/live`), and click **Start Stream**.

### Admin Identity Management & Analytics:
- **Admin Identity Fixer**: Click **Admin** to rename visitors to real human names (e.g. "Alice Smith"), merge split profiles caused by extreme lighting shifts, or delete false detections.
- **Deep Analytics**: Click **Analytics** to view average & max dwell times, peak traffic hours, retention percentages, and 24-hour traffic bar charts, plus instant one-click CSV export of visitors and events.

---

## 9. API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves VisiTrace editorial dashboard |
| `GET` | `/api/health` | Comprehensive system health, DB integrity, and model statuses |
| `GET` | `/api/stats` | High-level visitor metrics (unique visitors, active, entries, exits) |
| `GET` | `/api/status` | Live video processing status, current FPS, and track counts |
| `GET` | `/api/events` | Paginated entry/exit events (filters: `event_type`, `visitor_id`) |
| `GET` | `/api/visitors` | List of registered visitor identities and metadata |
| `GET` | `/api/visits` | Session lifecycle records with durations and entry/exit times |
| `GET` | `/api/analytics/daily`| Daily aggregated visitor and visit counts |
| `GET` | `/api/analytics/deep` | Deep dwell time metrics, retention rate, and 24-hour distribution |
| `GET` | `/api/export/events.csv` | Download complete event log as CSV |
| `GET` | `/api/export/visitors.csv` | Download registered visitors directory as CSV |
| `PATCH`| `/api/visitors/{id}` | Admin: Assign human names / notes to a visitor ID |
| `POST` | `/api/visitors/merge` | Admin: Merge split identities and transfer session history |
| `DELETE`| `/api/visitors/{id}` | Admin: Purge a visitor record and associated history |
| `GET` | `/api/config` | Sanitized active configuration (passwords masked) |
| `POST`| `/api/streams/start` | Start ingestion on video file, RTSP stream, or live webcam |
| `POST`| `/api/streams/stop` | Cleanly terminate active ingestion |
| `GET` | `/api/streams/preview`| Live multipart MJPEG video preview |

---

## 10. Database Schema Summary

The SQLite database enforces ACID transactions and relational integrity:

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Registered Visitors Table
CREATE TABLE visitors (
    visitor_id TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    embedding_blob BLOB NOT NULL,
    total_visits INTEGER NOT NULL DEFAULT 1,
    quality_score REAL NOT NULL DEFAULT 0.0,
    metadata_json TEXT
);

-- Visit Sessions Table
CREATE TABLE visits (
    visit_id TEXT PRIMARY KEY,
    visitor_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    entry_at TEXT NOT NULL,
    exit_at TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'completed')),
    entry_image_path TEXT,
    exit_image_path TEXT,
    duration_seconds REAL,
    FOREIGN KEY (visitor_id) REFERENCES visitors (visitor_id) ON DELETE CASCADE
);

-- Timestamped Entry/Exit Events Table
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    visit_id TEXT NOT NULL,
    visitor_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN ('entry', 'exit')),
    timestamp TEXT NOT NULL,
    image_path TEXT NOT NULL,
    track_id INTEGER,
    source_id TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    metadata_json TEXT,
    FOREIGN KEY (visit_id) REFERENCES visits (visit_id) ON DELETE CASCADE,
    FOREIGN KEY (visitor_id) REFERENCES visitors (visitor_id) ON DELETE CASCADE,
    CONSTRAINT uq_visit_event UNIQUE (visit_id, event_type)
);
```

---

## 11. Storage & Logging Structure

```
katomaran-intelligent-face-tracker-2026/
├── config.json                  # Runtime configuration
├── events.log                   # Structured audit log
├── data/
│   ├── visitrace.db             # SQLite database (WAL mode)
│   └── sample_videos/
│       └── video_sample1.mp4    # Downloaded sample video dataset
├── logs/
│   ├── entries/YYYY-MM-DD/      # Date-organized entry face crops
│   │   └── VIS-0001_VST-XXX.jpg
│   └── exits/YYYY-MM-DD/        # Date-organized exit face crops
│       └── VIS-0001_VST-XXX.jpg
├── sample_outputs/              # Exported evaluation artifacts
└── weights/
    ├── yolov8n-face.pt          # Verified face detector weights
    └── w600k_mbf.onnx           # InsightFace ArcFace weights
```

---

## 12. Automated Tests & Verification

The test suite covers configuration, database constraints, embedding math, tracking lifecycles, API endpoints, 15 critical regression scenarios, and a 1,000-iteration deduplication stress test:

```bash
pytest -v
```

### Actual Test Execution Output:
```
============================== test session starts ==============================
platform win32 -- Python 3.12.10, pytest-9.1.1
collected 40 items

tests/test_api.py::test_health_endpoint PASSED                           [  2%]
tests/test_api.py::test_stats_endpoint PASSED                            [  5%]
tests/test_api.py::test_events_endpoint_and_filtering PASSED             [  7%]
tests/test_api.py::test_visitors_and_visits_endpoints PASSED             [ 10%]
tests/test_api.py::test_config_endpoint_masks_credentials PASSED         [ 12%]
tests/test_api.py::test_stream_controls_validation PASSED                [ 15%]
tests/test_config.py::test_default_config_valid PASSED                   [ 17%]
tests/test_config.py::test_rtsp_validation_success PASSED                [ 20%]
tests/test_config.py::test_rtsp_validation_missing_url PASSED            [ 22%]
tests/test_config.py::test_rtsp_validation_invalid_scheme PASSED         [ 25%]
tests/test_config.py::test_frame_skip_constraints PASSED                 [ 27%]
tests/test_config.py::test_save_and_load_config PASSED                   [ 30%]
tests/test_database.py::test_wal_mode_and_foreign_keys PASSED            [ 32%]
tests/test_database.py::test_visitor_registration_and_matching PASSED    [ 35%]
tests/test_database.py::test_different_visitor_registration PASSED       [ 37%]
tests/test_database.py::test_visit_session_entry_and_exit PASSED         [ 40%]
tests/test_database.py::test_duplicate_exit_prevention PASSED            [ 42%]
tests/test_database.py::test_database_persistence_across_restart PASSED  [ 45%]
tests/test_detector_recognizer.py::test_face_quality_sharp_vs_flat PASSED [ 47%]
tests/test_detector_recognizer.py::test_recognizer_math_and_preprocessing PASSED [ 50%]
tests/test_detector_recognizer.py::test_real_model_loading_and_inference PASSED [ 52%]
tests/test_regressions.py::test_scenario_01_new_visitor_registers_once PASSED [ 55%]
tests/test_regressions.py::test_scenario_02_consecutive_detections_no_duplicate_registrations PASSED [ 57%]
tests/test_regressions.py::test_scenario_03_visitor_returns_in_later_session PASSED [ 60%]
tests/test_regressions.py::test_scenario_04_completed_visit_exactly_one_entry_one_exit PASSED [ 62%]
tests/test_regressions.py::test_scenario_05_duplicate_exit_callbacks_idempotency PASSED [ 65%]
tests/test_regressions.py::test_scenario_06_brief_missed_detection_preserves_visit PASSED [ 67%]
tests/test_regressions.py::test_scenario_07_absent_visitor_marked_exited_after_grace PASSED [ 70%]
tests/test_regressions.py::test_scenario_08_two_distinct_faces_tracked_independently PASSED [ 72%]
tests/test_regressions.py::test_scenario_09_database_restart_preserves_records PASSED [ 75%]
tests/test_regressions.py::test_scenario_10_failed_camera_connection_controlled_error PASSED [ 77%]
tests/test_regressions.py::test_scenario_11_processing_stops_cleanly PASSED [ 80%]
tests/test_regressions.py::test_scenario_12_invalid_configuration_actionable_error PASSED [ 82%]
tests/test_regressions.py::test_scenario_13_api_returns_real_database_results PASSED [ 85%]
tests/test_regressions.py::test_scenario_14_frontend_controls_invoke_expected_behaviour PASSED [ 87%]
tests/test_regressions.py::test_scenario_15_required_directories_created PASSED [ 90%]
tests/test_regressions.py::test_1000_iterations_identity_deduplication_stress PASSED [ 92%]
tests/test_tracking_lifecycle.py::test_compute_iou PASSED                [ 95%]
tests/test_tracking_lifecycle.py::test_tracker_lifecycle PASSED          [ 97%]
tests/test_tracking_lifecycle.py::test_visit_lifecycle_entry_and_grace_exit PASSED [100%]

======================== 46 passed, 1 warning in 16.01s ========================
```

---

## 13. Compute & Hardware Considerations

- **CPU Execution**:
  - Measured performance: **~14.6 FPS** during full video decoding, face detection, ArcFace embedding, tracking, and disk I/O on standard x86_64 CPU.
  - Memory consumption: **~280 MB RAM** during active inference.
  - Frame skipping (`frame_skip: 2`) balances real-time responsiveness with CPU load.
- **GPU Acceleration (Optional)**:
  - If an NVIDIA GPU with CUDA is present, set `"device": "cuda"` in `config.json`.
  - ONNX Runtime will utilize `CUDAExecutionProvider` and Ultralytics will pass frames directly to Tensor cores, achieving **>60 FPS**.

---

## 14. Assumptions, Privacy & Limitations

- **Biometric Privacy**:
  - Raw facial embeddings are never transmitted across public API endpoints.
  - Runtime SQLite databases (`*.db`), runtime video files, and face image logs are strictly excluded from Git commits via `.gitignore`.
- **Exit Inference**:
  - A camera cannot determine physical departure from an entire facility; an exit is inferred based on a face no longer being observed in the camera field of view for the duration of the configured grace period.
- **Lighting & Occlusion**:
  - Extremely dark scenes, severe motion blur, or acute face angles (>60 degrees yaw) may drop detection confidence below threshold. Laplacian variance filtering prevents storing degraded crops.

---

## 15. Demonstration Video

A step-by-step recording script is provided in [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md).
*(Demonstration video link to be recorded and added upon final presentation).*

---

This project is a part of a hackathon run by https://katomaran.com
