"""
ArcFace / InsightFace face recognition model wrapper.
Extracts 512-dimensional facial embeddings using ONNX Runtime and computes cosine similarity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from src.utils.logger import logger


class FaceRecognizer:
    def __init__(
        self,
        model_path: str = "weights/w600k_mbf.onnx",
        similarity_threshold: float = 0.60,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.similarity_threshold = similarity_threshold
        self.device = device
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: str = "input.1"
        self._output_name: str = "516"
        self._load_model()

    def _load_model(self) -> None:
        """Initialize ONNXRuntime session for ArcFace model."""
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Face recognition model not found at '{self.model_path}'. "
                "Ensure InsightFace ArcFace weights are downloaded."
            )

        try:
            providers = ["CPUExecutionProvider"]
            if self.device in ("cuda", "auto"):
                available = ort.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    providers.insert(0, "CUDAExecutionProvider")

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            import os
            sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 2)

            self._session = ort.InferenceSession(str(path), sess_options=sess_opts, providers=providers)
            inputs = self._session.get_inputs()
            self._input_name = inputs[0].name
            outputs = self._session.get_outputs()
            self._output_name = outputs[0].name

            logger.info(
                f"Loaded InsightFace ArcFace model from {self.model_path} "
                f"(Input: {self._input_name}, Output: {self._output_name}, Provider: {self._session.get_providers()})"
            )
        except Exception as e:
            logger.error(f"Failed to load ArcFace model from {self.model_path}: {e}")
            raise RuntimeError(f"Failed to initialize face recognizer: {e}") from e

    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess face crop into standard ArcFace input tensor:
        - Resize to 112x112
        - Convert BGR to RGB
        - Transpose HWC -> CHW
        - Normalize: (x - 127.5) / 128.0
        """
        # Resize to standard InsightFace 112x112
        resized = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32)
        tensor = (tensor - 127.5) / 128.0
        # Transpose from (112, 112, 3) to (3, 112, 112)
        tensor = np.transpose(tensor, (2, 0, 1))
        # Add batch dim -> (1, 3, 112, 112)
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    def extract_embedding(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate L2-normalized 512-dimensional embedding from face crop.
        Returns None if crop is invalid or extraction fails.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        try:
            input_tensor = self.preprocess_crop(crop)
            outputs = self._session.run([self._output_name], {self._input_name: input_tensor})
            raw_emb = outputs[0][0].astype(np.float32)

            # L2 normalisation
            norm = np.linalg.norm(raw_emb)
            if norm < 1e-6:
                return None
            norm_emb = raw_emb / norm
            return norm_emb
        except Exception as e:
            logger.error(f"Failed to extract face embedding: {e}")
            return None

    @staticmethod
    def compute_similarity(embA: np.ndarray, embB: np.ndarray) -> float:
        """Calculate cosine similarity between two normalized embeddings."""
        if embA is None or embB is None:
            return 0.0
        dot = float(np.dot(embA.flatten(), embB.flatten()))
        # Clamp to [-1.0, 1.0] range
        return max(-1.0, min(1.0, dot))
