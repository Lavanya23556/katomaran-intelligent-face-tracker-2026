"""
VisiTrace — Visitor Intelligence System
Main entry point for starting the web service or running offline video evaluation.
"""
from __future__ import annotations

import argparse
import sys
import uvicorn

from src.config import load_config
from src.database.schema import init_database
from src.database.repository import DatabaseRepository
from src.models.detector import FaceDetector
from src.models.recognizer import FaceRecognizer
from src.models.weights_manager import ensure_model_weights
from src.pipeline.stream_manager import StreamManager
from src.utils.image_store import ImageStore
from src.utils.logger import logger, setup_logger


def run_server(config_path: str = "config.json", host: str = None, port: int = None):
    """Run FastAPI application with Uvicorn."""
    config = load_config(config_path)
    bind_host = host or config.api_host
    bind_port = port or config.api_port

    logger.info(f"Starting VisiTrace server at http://{bind_host}:{bind_port}")
    uvicorn.run("src.api.app:app", host=bind_host, port=bind_port, reload=False)


def run_evaluation(video_path: str, config_path: str = "config.json", max_frames: int = 500):
    """
    Run offline evaluation on a video file to generate test outputs and verify pipeline.
    """
    import cv2
    import time

    config = load_config(config_path)
    setup_logger(config.events_log_path)
    init_database(config.database_path)

    ok, msg = ensure_model_weights(config.detector_model_path, config.recognition_model_path)
    if not ok:
        logger.error(f"Cannot run evaluation: {msg}")
        sys.exit(1)

    repo = DatabaseRepository(config.database_path)
    image_store = ImageStore(config.entries_dir, config.exits_dir)
    detector = FaceDetector(
        model_path=config.detector_model_path,
        confidence_threshold=config.detection_confidence_threshold,
        min_face_size=config.min_face_size,
        min_face_quality=config.min_face_quality,
        device=config.device
    )
    recognizer = FaceRecognizer(
        model_path=config.recognition_model_path,
        similarity_threshold=config.recognition_similarity_threshold,
        device=config.device
    )

    from src.pipeline.processor import FrameProcessor
    processor = FrameProcessor(
        config=config,
        repository=repo,
        image_store=image_store,
        detector=detector,
        recognizer=recognizer,
        source_id="eval_video",
    )

    logger.info(f"Evaluating video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        sys.exit(1)

    frames = 0
    t0 = time.time()
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames += 1
        annotated, tracks, stats = processor.process_frame(frame)
        if frames % 50 == 0:
            logger.info(f"Processed {frames} frames | Active Tracks: {stats['active_tracks']} | Unique Visitors: {stats['unique_visitors']}")

    cap.release()
    processor.close()
    elapsed = time.time() - t0
    final_stats = repo.get_stats()
    logger.info(
        f"Evaluation completed in {elapsed:.2f}s ({frames/elapsed:.1f} FPS) | "
        f"Frames: {frames} | Unique Visitors: {final_stats['unique_visitors']} | "
        f"Entries: {final_stats['total_entries']} | Exits: {final_stats['total_exits']}"
    )


def main():
    parser = argparse.ArgumentParser(description="VisiTrace — Intelligent Face Tracker & Visitor Counting")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--host", default=None, help="Server host binding")
    parser.add_argument("--port", type=int, default=None, help="Server port binding")
    parser.add_argument("--eval-video", default=None, help="Run offline video evaluation on specified file path")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames for offline evaluation")

    args = parser.parse_args()

    if args.eval_video:
        run_evaluation(args.eval_video, args.config, args.max_frames)
    else:
        run_server(args.config, args.host, args.port)


if __name__ == "__main__":
    main()
