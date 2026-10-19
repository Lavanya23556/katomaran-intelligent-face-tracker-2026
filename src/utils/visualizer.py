"""
Visualization utility to draw bounding boxes, visitor labels, and tracking info on video frames.
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import cv2
import numpy as np


class FrameVisualizer:
    # Editorial brand colors (BGR):
    # Forest Teal: (80, 120, 40) -> BGR: (40, 120, 80)
    # Warm Amber: (30, 150, 230) -> BGR: (30, 150, 230)
    COLOR_TEAL = (80, 140, 40)
    COLOR_AMBER = (30, 160, 230)
    COLOR_DARK = (30, 30, 30)
    COLOR_WHITE = (245, 245, 245)

    @classmethod
    def draw_detections(
        cls,
        frame: np.ndarray,
        tracks: List[dict],
        fps: float = 0.0,
        source_label: str = "Live Feed",
        unique_count: int = 0,
        active_count: int = 0,
    ) -> np.ndarray:
        """
        Draw detected face boxes, visitor IDs, tracker IDs, confidence, and dashboard stats overlay.
        tracks: list of dicts with keys:
            - bbox: [x1, y1, x2, y2]
            - track_id: int
            - visitor_id: str
            - confidence: float
            - is_new: bool
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for trk in tracks:
            bbox = trk.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Clip coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            visitor_id = trk.get("visitor_id", "Unknown")
            track_id = trk.get("track_id", 0)
            conf = trk.get("confidence", 0.0)
            is_new = trk.get("is_new", False)

            box_color = cls.COLOR_AMBER if is_new else cls.COLOR_TEAL

            # Draw sleek bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # Corner accents
            corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            if corner_len > 3:
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), box_color, 3)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), box_color, 3)
                cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), box_color, 3)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), box_color, 3)
                cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), box_color, 3)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), box_color, 3)
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), box_color, 3)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), box_color, 3)

            # Label text
            label = f"{visitor_id} [TRK-{track_id}] {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Background header pill
            lbl_y1 = max(0, y1 - txt_h - 8)
            lbl_y2 = y1
            lbl_x2 = min(w - 1, x1 + txt_w + 10)
            cv2.rectangle(annotated, (x1, lbl_y1), (lbl_x2, lbl_y2), box_color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - 4),
                font,
                font_scale,
                cls.COLOR_WHITE,
                thickness,
                cv2.LINE_AA
            )

        # Header HUD banner
        hud_bg = np.zeros((36, w, 3), dtype=np.uint8)
        cv2.rectangle(hud_bg, (0, 0), (w, 36), cls.COLOR_DARK, -1)
        # Alpha blend HUD
        annotated[0:36, 0:w] = cv2.addWeighted(annotated[0:36, 0:w], 0.35, hud_bg, 0.65, 0)

        # HUD Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        hud_text = f"VisiTrace Live | {source_label} | FPS: {fps:.1f} | Active: {active_count} | Unique Visitors: {unique_count}"
        cv2.putText(annotated, hud_text, (12, 23), font, 0.48, cls.COLOR_WHITE, 1, cv2.LINE_AA)

        return annotated
