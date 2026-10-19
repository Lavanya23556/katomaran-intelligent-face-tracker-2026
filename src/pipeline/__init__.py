"""Pipeline package for VisiTrace."""
from src.pipeline.processor import FrameProcessor
from src.pipeline.stream_manager import StreamManager

__all__ = ["FrameProcessor", "StreamManager"]
