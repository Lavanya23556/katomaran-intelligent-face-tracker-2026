"""
Visit Manager for VisiTrace.
Orchestrates visit lifecycle, entry/exit detection, grace periods, and identity mapping.
Guarantees at most one entry and one exit event per visit session.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.database.repository import DatabaseRepository
from src.utils.image_store import ImageStore
from src.utils.logger import logger
from src.tracking.tracker import TrackedFace


class ActiveVisitorSession:
    """Represents a visitor currently present in the scene."""
    def __init__(self, visitor_id: str, visit_id: str, track_id: int, source_id: str):
        self.visitor_id: str = visitor_id
        self.visit_id: str = visit_id
        self.primary_track_id: int = track_id
        self.source_id: str = source_id
        self.last_seen_time: float = time.time()
        self.last_face_crop: Optional[np.ndarray] = None
        self.last_confidence: float = 0.0
        self.pending_exit: bool = False
        self.disappeared_at: Optional[float] = None


class VisitLifecycleManager:
    def __init__(
        self,
        repository: DatabaseRepository,
        image_store: ImageStore,
        exit_grace_seconds: float = 5.0,
        source_id: str = "camera_default",
    ):
        self.repo = repository
        self.image_store = image_store
        self.exit_grace_seconds = float(exit_grace_seconds)
        self.source_id = source_id

        # Mapping of visitor_id -> ActiveVisitorSession
        self.active_sessions: Dict[str, ActiveVisitorSession] = {}
        # Mapping of track_id -> visitor_id (to avoid redundant recognitions on same track)
        self.track_to_visitor: Dict[int, str] = {}

    def process_recognized_track(
        self,
        track: TrackedFace,
        face_crop: np.ndarray,
        visitor_id: str,
        is_new_visitor: bool,
        similarity: float,
        confidence: float,
        current_time: Optional[float] = None,
    ) -> None:
        """
        Handle a recognized face track.
        If visitor does not have an active session, creates a new visit and fires an entry event.
        If visitor was pending exit (during grace period), cancels exit and restores session.
        """
        now = current_time if current_time is not None else time.time()
        track.visitor_id = visitor_id
        track.is_new = is_new_visitor
        self.track_to_visitor[track.track_id] = visitor_id

        session = self.active_sessions.get(visitor_id)

        if session is None:
            # New visit session!
            # 1. Save entry image crop
            img_path = self.image_store.save_face_crop(
                face_img=face_crop,
                event_type="entry",
                visitor_id=visitor_id,
                visit_id="pending"
            ) or ""

            # 2. Start visit transactionally in database
            meta = {
                "similarity": round(float(similarity), 4),
                "is_first_registration": is_new_visitor,
                "track_id": track.track_id,
            }
            visit_id, event_id = self.repo.start_visit(
                visitor_id=visitor_id,
                source_id=self.source_id,
                entry_image_path=img_path,
                track_id=track.track_id,
                confidence=confidence,
                metadata=meta,
            )

            # 3. Create active session
            session = ActiveVisitorSession(
                visitor_id=visitor_id,
                visit_id=visit_id,
                track_id=track.track_id,
                source_id=self.source_id,
            )
            session.last_seen_time = now
            session.last_face_crop = face_crop
            session.last_confidence = confidence
            self.active_sessions[visitor_id] = session

            track.visit_id = visit_id

            logger.info(
                f"[ENTRY] Visitor={visitor_id} Visit={visit_id} Event={event_id} "
                f"Track={track.track_id} NewRegistration={is_new_visitor} Sim={similarity:.2f}"
            )

        else:
            # Reconnected or continuing visit
            track.visit_id = session.visit_id
            session.last_seen_time = now
            session.last_face_crop = face_crop
            session.last_confidence = confidence

            if session.pending_exit:
                # Cancel pending exit within grace period
                session.pending_exit = False
                session.disappeared_at = None
                logger.info(f"[GRACE RESTORE] Visitor {visitor_id} reappeared. Exit cancelled.")

    def update_unseen_tracks(
        self,
        current_tracks: List[TrackedFace],
        current_time: Optional[float] = None
    ) -> List[str]:
        """
        Evaluate currently visible tracks versus active sessions.
        If a visitor is no longer observed by any track, start their grace period.
        If their grace period has expired, finalize the visit and fire an exit event.
        Returns:
            List of visitor_ids that finalized an exit event during this call.
        """
        now = current_time if current_time is not None else time.time()
        finalized_exits: List[str] = []

        # Find which visitor IDs are currently visible on active tracks
        visible_visitors = set()
        for trk in current_tracks:
            if trk.visitor_id:
                visible_visitors.add(trk.visitor_id)

        # Check each active session
        visitors_to_remove = []

        for visitor_id, session in self.active_sessions.items():
            if visitor_id in visible_visitors:
                # Still visible: refresh state
                session.last_seen_time = now
                session.pending_exit = False
                session.disappeared_at = None
            else:
                # Not currently observed
                if not session.pending_exit:
                    session.pending_exit = True
                    session.disappeared_at = now
                else:
                    # Already in grace period: check expiration
                    disappeared_duration = now - (session.disappeared_at or now)
                    if disappeared_duration >= self.exit_grace_seconds:
                        # Grace period expired! Finalize visit
                        self._trigger_exit(session)
                        finalized_exits.append(visitor_id)
                        visitors_to_remove.append(visitor_id)

        for vid in visitors_to_remove:
            del self.active_sessions[vid]

        # Cleanup track mapping for removed tracks
        active_trk_ids = {trk.track_id for trk in current_tracks}
        stale_trks = [tid for tid in self.track_to_visitor if tid not in active_trk_ids]
        for tid in stale_trks:
            del self.track_to_visitor[tid]

        return finalized_exits

    def _trigger_exit(self, session: ActiveVisitorSession) -> None:
        """Trigger single exit event and write to database transactionally."""
        img_path = ""
        if session.last_face_crop is not None:
            img_path = self.image_store.save_face_crop(
                face_img=session.last_face_crop,
                event_type="exit",
                visitor_id=session.visitor_id,
                visit_id=session.visit_id,
            ) or ""

        event_id = self.repo.finalize_visit(
            visit_id=session.visit_id,
            visitor_id=session.visitor_id,
            source_id=session.source_id,
            exit_image_path=img_path,
            track_id=session.primary_track_id,
            confidence=session.last_confidence,
            metadata={"grace_duration_applied": self.exit_grace_seconds}
        )

        logger.info(
            f"[EXIT] Visitor={session.visitor_id} Visit={session.visit_id} "
            f"Event={event_id} (Inferred after {self.exit_grace_seconds}s absence)"
        )

    def close_all_active_visits(self, reason: str = "shutdown") -> None:
        """
        Close all active visit sessions upon stream termination or app shutdown.
        Guarantees no orphaned visits remain in 'active' state.
        """
        for visitor_id, session in list(self.active_sessions.items()):
            self._trigger_exit(session)
        self.active_sessions.clear()
        self.track_to_visitor.clear()
        logger.info(f"Closed all active visits (Reason: {reason})")
