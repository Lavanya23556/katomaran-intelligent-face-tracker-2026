# VisiTrace Demonstration Script

This script provides a step-by-step walkthrough for evaluating VisiTrace during hackathon reviews, technical interviews, and final-year CSE project viva.

---

## Prerequisites

1. Ensure Python 3.10+ is installed with packages from `requirements.txt`.
2. Model weights:
   - `weights/yolov8n-face.pt` (6.3 MB)
   - `weights/w600k_mbf.onnx` (13.6 MB)
   *(Automatically verified and downloaded upon startup).*
3. Sample video located at `data/sample_videos/video_sample1.mp4`.

---

## Step 1: Launch the Application

Run the server via terminal:

```bash
python main.py
```

Expected terminal output:
```
[INFO] [visitrace] Starting VisiTrace server at http://127.0.0.1:8000
[INFO] [visitrace] Loaded YOLO face detector from weights/yolov8n-face.pt with classes: {0: 'face'}
[INFO] [visitrace] Loaded InsightFace ArcFace model from weights/w600k_mbf.onnx
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Open your browser to:
**`http://127.0.0.1:8000`**

---

## Step 2: Examine the Dashboard Interface

1. **Brand & Visuals**: Observe the editorial aesthetic (warm ivory background, deep graphite typography, forest-teal accents).
2. **System Status**: Top status indicator shows `Standby (Idle)` with a gray dot.
3. **KPI Counters**: Top metrics row displays:
   - `Unique Visitors`: Persisted database count.
   - `Active In-Scene`: Currently tracked people.
   - `Total Entries`: Lifetime entry events.
   - `Total Exits`: Lifetime exit events.
   - `Today's Visitors`: Unique count for the current day.
4. **Live Visual Tracking Canvas**: Displays an idle placeholder indicating system readiness.

---

## Step 3: Start Real Video Stream Ingestion

1. In the **Input Mode** dropdown, ensure **Local Video File** is selected.
2. In the **Source Path** field, ensure `data/sample_videos/video_sample1.mp4` is specified (or click **Use Katomaran Sample**).
3. Click **Start Stream**.

### Observable Real-Time Behaviour:
- Status indicator transitions to green: `Active (file)`.
- Live Viewport starts streaming live video with:
  - Bounding boxes around every detected face.
  - Visitor IDs (`VIS-0001`, `VIS-0002`, etc.).
  - Tracker IDs (`TRK-1`, `TRK-3`).
  - Detection confidence scores (`0.92`, `0.85`).
  - Top HUD overlay with live FPS and active track count.
- **Entry & Exit Event Stream (Right Column)**:
  - Updates in real-time as visitors enter the camera field of view.
  - Displays cropped face thumbnails, timestamps, and visit badges (`Entry` in teal, `Exit` in amber).
- **KPI Counters**:
  - `Unique Visitors` increments as genuinely new identities appear.
  - `Active In-Scene` dynamically reflects the number of tracked people.

---

## Step 4: Verify Identity Deduplication & Return Visits

1. Observe that as people move across the camera frame, their assigned `VIS-XXXX` identity remains constant.
2. Watch the log or terminal: when a person briefly turns away or is occluded, a `[GRACE RESTORE]` event occurs rather than generating an immediate spurious exit.
3. When people leave the frame for longer than the grace period (5.0s), an `[EXIT]` event is logged and an exit thumbnail is saved.
4. If a previously registered person reappears, they retain their original `VIS-XXXX` ID. A new visit session (`VST-XXXX`) is started, while the `Unique Visitors` KPI count does **not** increase!

---

## Step 5: Stop the Stream and Inspect Data

1. Click **Stop Stream**.
2. Notice the live viewport enters standby and background processing threads terminate cleanly.
3. Open **Visitors Directory** (top navigation):
   - Review table of registered visitors, first-seen timestamps, last-seen timestamps, and total visit frequencies.
4. Open **Diagnostics**:
   - Inspect SQLite database integrity (`status: "healthy"`), CPU/Memory utilization, and loaded model paths.
5. Open **Config**:
   - Inspect sanitized runtime parameters (demonstrating credential masking for RTSP).

---

## Step 6: Verify Database Persistence

1. Stop the application in the terminal (`Ctrl+C`).
2. Restart the application:
   ```bash
   python main.py
   ```
3. Refresh `http://127.0.0.1:8000`:
   - All previously registered unique visitor identities, visit counts, and event records remain intact!

---

## Recording Note

If recording a demonstration video for hackathon submission:
- Capture screen at 1080p displaying the dashboard alongside the terminal.
- Demonstrate starting the stream, observing live bounding boxes and auto-registration, stopping the stream, and inspecting database records.
- *(Note: Actual demonstration video link to be recorded and added by the submitter).*
