"""
FastAPI application factory and lifecycle management for VisiTrace.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import SystemConfig, load_config
from src.database.schema import init_database
from src.database.repository import DatabaseRepository
from src.models.detector import FaceDetector
from src.models.recognizer import FaceRecognizer
from src.models.weights_manager import ensure_model_weights
from src.pipeline.stream_manager import StreamManager
from src.utils.image_store import ImageStore
from src.utils.logger import logger, setup_logger
from src.api.routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown procedures."""
    logger.info("Initializing VisiTrace Application...")
    config: SystemConfig = getattr(app.state, "config", None) or load_config()
    setup_logger(config.events_log_path)
    config.ensure_directories()
    app.state.config = config

    # Initialize Database
    init_database(config.database_path)
    repo = DatabaseRepository(config.database_path)
    app.state.repo = repo
    logger.info(f"Database initialized at {config.database_path}")

    # Initialize Image Store
    image_store = ImageStore(config.entries_dir, config.exits_dir)
    app.state.image_store = image_store

    # Verify and load AI models
    detector: Optional[FaceDetector] = None
    recognizer: Optional[FaceRecognizer] = None

    if not config.mock_mode:
        ok, msg = ensure_model_weights(
            config.detector_model_path,
            config.recognition_model_path
        )
        if ok:
            try:
                detector = FaceDetector(
                    model_path=config.detector_model_path,
                    confidence_threshold=config.detection_confidence_threshold,
                    min_face_size=config.min_face_size,
                    min_face_quality=config.min_face_quality,
                    device=config.device,
                )
                recognizer = FaceRecognizer(
                    model_path=config.recognition_model_path,
                    similarity_threshold=config.recognition_similarity_threshold,
                    device=config.device,
                )
                logger.info("AI Models (YOLO face detector & InsightFace recognizer) loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load AI models: {e}. Dashboard will operate in degraded mode.")
        else:
            logger.warning(f"Weights verification notice: {msg}")

    app.state.detector = detector
    app.state.recognizer = recognizer

    # Initialize Stream Manager
    stream_manager = StreamManager(
        config=config,
        repository=repo,
        image_store=image_store,
        detector=detector,
        recognizer=recognizer,
    )
    app.state.stream_manager = stream_manager

    # Auto-start stream if configured and not mock_mode
    if getattr(config, "auto_start_stream", False) and not config.mock_mode:
        source_type = config.video_source_type
        if source_type == "webcam":
            source_path = str(config.webcam_device_id)
        elif source_type == "rtsp":
            source_path = config.rtsp_url or ""
        else:
            source_path = config.video_path

        logger.info(f"Auto-initiating live stream capture on {source_type}: {source_path}")
        stream_manager.start_stream(source_type=source_type, source_path=source_path, loop=True)

    yield

    # Clean shutdown
    logger.info("Shutting down VisiTrace Application...")
    if stream_manager.is_running:
        stream_manager.stop_stream()
    logger.info("VisiTrace Application shutdown complete.")


def create_app(config: Optional[SystemConfig] = None) -> FastAPI:
    """Factory creating and configuring the FastAPI app."""
    app = FastAPI(
        title="VisiTrace — Visitor Intelligence System",
        description="Intelligent Face Tracker with Auto-Registration and Visitor Counting",
        version="1.0.0",
        lifespan=lifespan,
    )

    if config:
        app.state.config = config

    # Enable CORS for local dev and embedded dashboards
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API endpoints
    app.include_router(api_router)

    # Mount static files and face crop logs
    static_dir = Path("static").resolve()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    logs_dir = Path("logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/logs", StaticFiles(directory=str(logs_dir)), name="logs")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return HTMLResponse("<h2>VisiTrace API Running. Dashboard index.html is being prepared...</h2>")

    return app


app = create_app()
