"""Tests for configuration loading, validation, and sanitization."""
import json
import os
import pytest
from src.config import SystemConfig, load_config, save_config


def test_default_config_valid():
    cfg = SystemConfig()
    assert cfg.video_source_type == "file"
    assert cfg.frame_skip >= 1
    assert 0.0 < cfg.detection_confidence_threshold <= 1.0
    assert 0.0 < cfg.recognition_similarity_threshold <= 1.0


def test_rtsp_validation_success():
    cfg = SystemConfig(
        video_source_type="rtsp",
        rtsp_url="rtsp://admin:secret123@192.168.1.50:554/live"
    )
    assert cfg.video_source_type == "rtsp"
    sanitized = cfg.sanitized_dict()
    assert "secret123" not in sanitized["rtsp_url"]
    assert "******" in sanitized["rtsp_url"]


def test_rtsp_validation_missing_url():
    with pytest.raises(ValueError, match="rtsp_url must be provided"):
        SystemConfig(video_source_type="rtsp", rtsp_url=None)


def test_rtsp_validation_invalid_scheme():
    with pytest.raises(ValueError, match="rtsp_url must begin with"):
        SystemConfig(video_source_type="rtsp", rtsp_url="http://invalid.url")


def test_frame_skip_constraints():
    with pytest.raises(ValueError):
        SystemConfig(frame_skip=0)


def test_save_and_load_config(temp_dir):
    cfg_file = os.path.join(temp_dir, "test_config.json")
    cfg = SystemConfig(frame_skip=4, detection_confidence_threshold=0.65)
    save_config(cfg, cfg_file)

    loaded = load_config(cfg_file)
    assert loaded.frame_skip == 4
    assert loaded.detection_confidence_threshold == 0.65
