"""
Model weights manager for VisiTrace.
Ensures verified face-detection and ArcFace recognition weights exist,
with automatic download and explicit error reporting.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

from src.utils.logger import logger

VERIFIED_MODELS = {
    "detector": {
        "path": "weights/yolov8n-face.pt",
        "url": "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt",
        "description": "YOLOv8n Face Detector fine-tuned on WiderFace (Class 0: face)",
        "min_size": 2_000_000,
    },
    "recognizer": {
        "path": "weights/w600k_mbf.onnx",
        "url": "https://huggingface.co/deepghs/insightface/resolve/4e1f33d3fe0e50a0945f3a53ab94ae8977ae7ddb/buffalo_s/w600k_mbf.onnx",
        "description": "InsightFace MobileFaceNet ArcFace embedding extractor (512D)",
        "min_size": 10_000_000,
    }
}


def ensure_model_weights(
    detector_path: str = "weights/yolov8n-face.pt",
    recognizer_path: str = "weights/w600k_mbf.onnx"
) -> Tuple[bool, str]:
    """
    Check if required weights exist and are valid.
    Downloads them if missing, or returns an explicit actionable error.
    """
    targets = [
        ("detector", detector_path, VERIFIED_MODELS["detector"]),
        ("recognizer", recognizer_path, VERIFIED_MODELS["recognizer"]),
    ]

    for name, req_path, spec in targets:
        p = Path(req_path)
        if not p.exists() or p.stat().st_size < spec["min_size"]:
            logger.info(f"Model weight {req_path} missing or incomplete. Attempting download from {spec['url']}...")
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                opener = urllib.request.build_opener()
                opener.addheaders = [("User-Agent", "Mozilla/5.0 (VisiTrace/1.0)")]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(spec["url"], str(p))
                if p.stat().st_size < spec["min_size"]:
                    msg = (
                        f"Downloaded {name} weights file {req_path} is smaller than expected "
                        f"({p.stat().st_size} < {spec['min_size']} bytes). Please verify network or download manually."
                    )
                    logger.error(msg)
                    return False, msg
                logger.info(f"Successfully downloaded {name} weights ({p.stat().st_size} bytes).")
            except Exception as e:
                msg = (
                    f"Failed to obtain {name} model weights at '{req_path}'.\n"
                    f"Target URL: {spec['url']}\n"
                    f"Error: {e}\n"
                    f"ACTION REQUIRED: Please place verified {spec['description']} weights into '{req_path}'."
                )
                logger.error(msg)
                return False, msg

    return True, "All model weights verified."
