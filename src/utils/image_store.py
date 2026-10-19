"""
Image persistence service for VisiTrace.
Organizes face crop images by date into logs/entries/YYYY-MM-DD/ and logs/exits/YYYY-MM-DD/.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class ImageStore:
    def __init__(self, entries_base_dir: str = "logs/entries", exits_base_dir: str = "logs/exits"):
        self.entries_base_dir = Path(entries_base_dir).resolve()
        self.exits_base_dir = Path(exits_base_dir).resolve()
        self.entries_base_dir.mkdir(parents=True, exist_ok=True)
        self.exits_base_dir.mkdir(parents=True, exist_ok=True)

    def save_face_crop(
        self,
        face_img: np.ndarray,
        event_type: str,  # 'entry' or 'exit'
        visitor_id: str,
        visit_id: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        Save a face crop image organized by date.
        Returns the relative path formatted for web/API access, or None if save failed.
        """
        if face_img is None or face_img.size == 0:
            return None

        dt = timestamp or datetime.now(timezone.utc)
        date_folder = dt.strftime("%Y-%m-%d")
        time_tag = dt.strftime("%H%M%S_%f")[:10]

        target_base = self.entries_base_dir if event_type == "entry" else self.exits_base_dir
        dest_dir = target_base / date_folder
        dest_dir.mkdir(parents=True, exist_ok=True)

        clean_visitor = visitor_id.replace("/", "_").replace("\\", "_")
        clean_visit = visit_id.replace("/", "_").replace("\\", "_")
        filename = f"{clean_visitor}_{clean_visit}_{time_tag}.jpg"
        abs_path = dest_dir / filename

        try:
            # Ensure 3-channel BGR for writing
            if len(face_img.shape) == 2:
                img_to_write = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
            elif face_img.shape[2] == 4:
                img_to_write = cv2.cvtColor(face_img, cv2.COLOR_BGRA2BGR)
            else:
                img_to_write = face_img

            success = cv2.imwrite(str(abs_path), img_to_write, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if success:
                # Store relative forward-slash path for portability across OS & web URLs
                event_folder = "entries" if event_type == "entry" else "exits"
                rel_path = f"logs/{event_folder}/{date_folder}/{filename}"
                return rel_path
            return None
        except Exception as e:
            from src.utils.logger import logger
            logger.error(f"Failed to save face crop to {abs_path}: {e}")
            return None
