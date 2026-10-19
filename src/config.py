"""
Configuration manager for VisiTrace face tracker.
Loads, validates, and serializes system settings from config.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class SystemConfig(BaseModel):
    # Video input source
    video_source_type: Literal["file", "rtsp"] = Field(
        default="file",
        description="Source type: 'file' for video file or 'rtsp' for camera stream"
    )
    video_path: str = Field(
        default="data/sample_videos/video_sample1.mp4",
        description="Path to local video file"
    )
    rtsp_url: Optional[str] = Field(
        default=None,
        description="RTSP stream URL (e.g. rtsp://user:pass@192.168.1.100:554/stream)"
    )

    # Models & Backend
    detector_backend: Literal["auto", "yolo", "onnx"] = Field(
        default="auto",
        description="Face detector backend: 'yolo' for Ultralytics YOLO face model, 'onnx' for ONNXRuntime, or 'auto'"
    )
    detector_model_path: str = Field(
        default="weights/yolov8n-face.pt",
        description="Path to YOLO face-specific weights (.pt or .onnx)"
    )
    recognition_model_path: str = Field(
        default="weights/w600k_mbf.onnx",
        description="Path to InsightFace/ArcFace embedding model (.onnx)"
    )
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="Inference device: 'auto', 'cpu', or 'cuda'"
    )

    # Detection & Recognition Parameters
    frame_skip: int = Field(
        default=2,
        ge=1,
        le=30,
        description="Number of frames skipped between full face detection cycles"
    )
    detection_confidence_threshold: float = Field(
        default=0.50,
        ge=0.1,
        le=1.0,
        description="Minimum confidence score for face detection"
    )
    recognition_similarity_threshold: float = Field(
        default=0.60,
        ge=0.2,
        le=0.95,
        description="Cosine similarity threshold to match existing registered visitor"
    )
    min_face_size: int = Field(
        default=40,
        ge=20,
        description="Minimum width and height in pixels for a valid face crop"
    )
    min_face_quality: float = Field(
        default=35.0,
        ge=5.0,
        description="Minimum Laplacian variance to reject blurry face crops"
    )

    # Tracking & Visit Lifecycle
    exit_grace_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Grace period in seconds before finalizing an exit event after track disappearance"
    )
    tracker_max_age: int = Field(
        default=30,
        ge=5,
        le=150,
        description="Maximum frames to maintain lost track before deletion"
    )
    tracker_iou_threshold: float = Field(
        default=0.30,
        ge=0.1,
        le=0.8,
        description="Intersection-over-Union matching threshold for tracker"
    )

    # Persistence & Logging Paths
    database_path: str = Field(
        default="data/visitrace.db",
        description="SQLite database path"
    )
    events_log_path: str = Field(
        default="events.log",
        description="Structured text event log file path"
    )
    entries_dir: str = Field(
        default="logs/entries",
        description="Directory for entry face crop images"
    )
    exits_dir: str = Field(
        default="logs/exits",
        description="Directory for exit face crop images"
    )

    # API & Serving
    api_host: str = Field(
        default="127.0.0.1",
        description="FastAPI host binding"
    )
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="FastAPI port"
    )

    # Simulation / Demo mode flag
    mock_mode: bool = Field(
        default=False,
        description="Explicit demo mode with synthetic data if hardware/models unavailable"
    )

    @field_validator("video_path")
    @classmethod
    def validate_video_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("video_path cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_source(self) -> SystemConfig:
        if self.video_source_type == "rtsp":
            if not self.rtsp_url or not self.rtsp_url.strip():
                raise ValueError("rtsp_url must be provided when video_source_type is 'rtsp'")
            if not (self.rtsp_url.startswith("rtsp://") or self.rtsp_url.startswith("rtsps://")):
                raise ValueError("rtsp_url must begin with rtsp:// or rtsps://")
        return self

    def ensure_directories(self) -> None:
        """Create required directories for database, logs, and images."""
        db_dir = Path(self.database_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        log_dir = Path(self.events_log_path).parent
        if str(log_dir) not in ("", "."):
            log_dir.mkdir(parents=True, exist_ok=True)

        Path(self.entries_dir).mkdir(parents=True, exist_ok=True)
        Path(self.exits_dir).mkdir(parents=True, exist_ok=True)
        Path("weights").mkdir(parents=True, exist_ok=True)

    def sanitized_dict(self) -> dict:
        """Return config dictionary with sensitive RTSP credentials masked."""
        data = self.model_dump()
        if data.get("rtsp_url"):
            url = data["rtsp_url"]
            # Mask user:pass in rtsp://user:pass@host:port/path
            import re
            masked = re.sub(r"://([^:]+):([^@]+)@", r"://\1:******@", url)
            data["rtsp_url"] = masked
        return data


def load_config(config_path: str = "config.json") -> SystemConfig:
    """
    Load configuration from JSON file with fallback to defaults.
    Ensures required directories exist upon loading.
    """
    path = Path(config_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
            config = SystemConfig.model_validate(content)
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}") from e
    else:
        config = SystemConfig()
        save_config(config, config_path)

    config.ensure_directories()
    return config


def save_config(config: SystemConfig, config_path: str = "config.json") -> None:
    """Save configuration to JSON file formatted nicely."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)
