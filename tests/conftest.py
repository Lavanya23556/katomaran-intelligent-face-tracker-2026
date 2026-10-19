"""Test suite configuration and shared fixtures for VisiTrace."""
import os
import shutil
import tempfile
import numpy as np
import pytest

from src.config import SystemConfig
from src.database.schema import init_database
from src.database.repository import DatabaseRepository
from src.utils.image_store import ImageStore


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts and cleanup afterward."""
    tmp = tempfile.mkdtemp(prefix="visitrace_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir):
    """Provide a test SystemConfig mapped to the temporary directory."""
    db_path = os.path.join(temp_dir, "test_visitrace.db")
    entries_dir = os.path.join(temp_dir, "logs", "entries")
    exits_dir = os.path.join(temp_dir, "logs", "exits")
    log_path = os.path.join(temp_dir, "test_events.log")

    cfg = SystemConfig(
        database_path=db_path,
        entries_dir=entries_dir,
        exits_dir=exits_dir,
        events_log_path=log_path,
        video_path="data/sample_videos/video_sample1.mp4",
        frame_skip=2,
        detection_confidence_threshold=0.5,
        recognition_similarity_threshold=0.6,
        exit_grace_seconds=1.0,  # Fast for testing
        tracker_max_age=5,
        mock_mode=True,
    )
    cfg.ensure_directories()
    return cfg


@pytest.fixture
def test_repo(test_config):
    """Provide an initialized DatabaseRepository connected to a test database."""
    init_database(test_config.database_path)
    return DatabaseRepository(test_config.database_path)


@pytest.fixture
def test_image_store(test_config):
    """Provide an initialized ImageStore for test images."""
    return ImageStore(test_config.entries_dir, test_config.exits_dir)


@pytest.fixture
def sample_embedding():
    """Return a reproducible 512D normalized embedding vector."""
    np.random.seed(42)
    v = np.random.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def sample_face_crop():
    """Return a synthetic 112x112 BGR face crop image."""
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    # Draw some patterns to ensure high Laplacian variance
    img[20:90, 20:90] = 180
    img[30:50, 30:50] = 50
    img[30:50, 60:80] = 50
    img[70:80, 40:70] = 30
    return img
