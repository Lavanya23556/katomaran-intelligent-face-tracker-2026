"""
Stream Manager for VisiTrace.
Handles asynchronous video capture (file & RTSP), reconnection policies,
thread synchronization, and MJPEG preview frame broadcasting.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, Any

import cv2
import numpy as np

from src.config import SystemConfig
from src.database.repository import DatabaseRepository
from src.models.detector import FaceDetector
from src.models.recognizer import FaceRecognizer
from src.pipeline.processor import FrameProcessor
from src.utils.image_store import ImageStore
from src.utils.logger import logger


class StreamManager:
    def __init__(
        self,
        config: SystemConfig,
        repository: DatabaseRepository,
        image_store: ImageStore,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
    ):
        self.config = config
        self.repo = repository
        self.image_store = image_store
        self.detector = detector
        self.recognizer = recognizer

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running: bool = False

        self._processor: Optional[FrameProcessor] = None
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame_time: float = 0.0
        self._last_error: Optional[str] = None
        self._current_source: str = ""
        self._current_source_type: str = ""
        self._fps: float = 0.0
        self._total_frames_processed: int = 0

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_status(self) -> Dict[str, Any]:
        """Return real-time stream status and diagnostics."""
        with self._lock:
            return {
                "is_running": self._is_running,
                "source": self._current_source,
                "source_type": self._current_source_type,
                "fps": round(self._fps, 1),
                "frames_processed": self._total_frames_processed,
                "last_error": self._last_error,
                "active_tracks": len(self._processor.tracker.tracks) if self._processor else 0,
                "active_visits": len(self._processor.visit_manager.active_sessions) if self._processor else 0,
            }

    def start_stream(
        self,
        source_type: Optional[str] = None,
        source_path: Optional[str] = None,
        loop: bool = True
    ) -> Tuple[bool, str]:
        """
        Start stream processing in background thread.
        source_type: 'file' or 'rtsp'
        source_path: video file path or RTSP URL
        """
        with self._lock:
            if self._is_running:
                return False, "Stream processing is already running. Please stop current stream first."

            stype = source_type or self.config.video_source_type
            spath = source_path or (self.config.video_path if stype == "file" else self.config.rtsp_url)

            if not spath:
                return False, f"Missing source path/URL for source type '{stype}'."

            # Validate local file exists
            if stype == "file":
                p = Path(spath)
                if not p.exists():
                    msg = f"Video file not found at '{spath}'. Please provide a valid local video file."
                    self._last_error = msg
                    return False, msg

            # Validate RTSP format
            if stype == "rtsp":
                if not (spath.startswith("rtsp://") or spath.startswith("rtsps://")):
                    msg = f"Invalid RTSP URL format: '{spath}'."
                    self._last_error = msg
                    return False, msg

            self._stop_event.clear()
            self._current_source = spath
            self._current_source_type = stype
            self._last_error = None
            self._total_frames_processed = 0

            # Initialize new processor
            source_id = "rtsp_cam" if stype == "rtsp" else Path(spath).name
            self._processor = FrameProcessor(
                config=self.config,
                repository=self.repo,
                image_store=self.image_store,
                detector=self.detector,
                recognizer=self.recognizer,
                source_id=source_id,
            )

            self._thread = threading.Thread(
                target=self._run_capture_loop,
                args=(stype, spath, loop),
                daemon=True,
                name="VisiTraceStreamWorker"
            )
            self._is_running = True
            self._thread.start()

            logger.info(f"Started video stream processing on {stype}: {source_id}")
            return True, f"Stream processing started on {stype}: {source_id}"

    def stop_stream(self) -> Tuple[bool, str]:
        """Stop running stream cleanly and release video capture resources."""
        with self._lock:
            if not self._is_running:
                return True, "Stream is not currently running."

            self._stop_event.set()
            self._is_running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        with self._lock:
            if self._processor:
                self._processor.close()
                self._processor = None
            self._latest_jpeg = None
            self._fps = 0.0

        logger.info("Stream processing stopped and resources released.")
        return True, "Stream processing successfully stopped."

    def _run_capture_loop(self, source_type: str, source_path: str, loop: bool) -> None:
        """Background thread capture loop with reconnection support."""
        max_reconnects = 5 if source_type == "rtsp" else 1
        reconnect_count = 0

        while not self._stop_event.is_set():
            logger.info(f"Opening video capture from: {source_path}")
            cap = cv2.VideoCapture(source_path)

            if not cap.isOpened():
                err = f"Failed to connect or open video source: {source_path}"
                logger.error(err)
                self._last_error = err
                reconnect_count += 1
                if reconnect_count >= max_reconnects:
                    break
                time.sleep(2.0)
                continue

            reconnect_count = 0
            logger.info(f"Video capture opened successfully: {source_path}")

            while not self._stop_event.is_set():
                ret, frame = cap.read()

                if not ret or frame is None:
                    if source_type == "file":
                        if loop:
                            # Loop local video for continuous demonstration
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            logger.info("Reached end of video file.")
                            break
                    else:
                        # RTSP stream dropped: trigger reconnect
                        logger.warning("RTSP stream frame drop, attempting reconnect...")
                        break

                try:
                    annotated, tracks_info, stats = self._processor.process_frame(frame)
                    self._total_frames_processed += 1
                    self._fps = stats["fps"]

                    # Fast preview encoding: downscale if large for sub-2ms JPEG encoding
                    prev_frame = annotated
                    if annotated.shape[1] > 854:
                        scale = 854.0 / annotated.shape[1]
                        prev_frame = cv2.resize(annotated, (854, int(annotated.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)

                    ret_enc, jpeg = cv2.imencode(
                        ".jpg",
                        prev_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 75]
                    )
                    if ret_enc:
                        with self._lock:
                            self._latest_jpeg = jpeg.tobytes()
                            self._latest_frame_time = time.time()

                except Exception as e:
                    logger.error(f"Error processing frame: {e}", exc_info=True)
                    self._last_error = str(e)

                # Regulate playback speed smoothly
                if source_type == "file":
                    time.sleep(0.005)

            cap.release()

            if source_type == "file" and not loop:
                break

            if self._stop_event.is_set():
                break

            time.sleep(1.0)

        with self._lock:
            self._is_running = False
            if self._processor:
                self._processor.close()

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Fetch the most recent annotated frame encoded as JPEG bytes."""
        with self._lock:
            return self._latest_jpeg

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """Yield multipart MJPEG frame stream for HTTP browser streaming."""
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        blank_img = None

        while True:
            jpeg = self.get_latest_jpeg()
            if jpeg is None:
                # Generate a refined idle banner if no live frame is available
                if blank_img is None:
                    placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder,
                        "VisiTrace Standby - Stream Inactive",
                        (110, 175),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (180, 180, 180),
                        2,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        placeholder,
                        "Select local video or RTSP and click Start Stream",
                        (100, 215),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (120, 120, 120),
                        1,
                        cv2.LINE_AA
                    )
                    _, blank_img = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frame_bytes = blank_img.tobytes()
            else:
                frame_bytes = jpeg

            yield boundary + frame_bytes + b"\r\n"
            time.sleep(0.06)  # ~16 FPS preview
