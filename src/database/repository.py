"""
Database repository providing transactional queries and persistence for VisiTrace.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class DatabaseRepository:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).resolve())
        # Ensure database tables exist
        from src.database.schema import init_database
        init_database(self.db_path)
        # Cache for fast embedding matching: dict of {visitor_id: np.ndarray}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._reload_embedding_cache()

    def get_connection(self) -> sqlite3.Connection:
        """Get SQLite connection configured with WAL and row factory."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    def _reload_embedding_cache(self) -> None:
        """Reload all registered visitor embeddings into memory cache."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT visitor_id, embedding_blob FROM visitors")
                rows = cursor.fetchall()
                cache = {}
                for row in rows:
                    visitor_id = row["visitor_id"]
                    blob = row["embedding_blob"]
                    emb = np.frombuffer(blob, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        cache[visitor_id] = emb / norm
                    else:
                        cache[visitor_id] = emb
                self._embedding_cache = cache
        except sqlite3.OperationalError:
            # Table might not exist yet if init_database hasn't run
            self._embedding_cache = {}

    def match_or_register_visitor(
        self,
        embedding: np.ndarray,
        similarity_threshold: float = 0.60,
        quality_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool, float]:
        """
        Compare face embedding against registered visitors.
        Returns:
            (visitor_id, is_new, best_similarity)
        If similarity >= threshold: matches existing visitor (is_new=False).
        Otherwise: registers new visitor in DB and cache (is_new=True).
        """
        # Ensure embedding is 1D float32 normalized
        emb = np.asarray(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        else:
            return "VIS-UNKNOWN", False, 0.0

        best_sim = -1.0
        best_id: Optional[str] = None

        if self._embedding_cache:
            ids = list(self._embedding_cache.keys())
            matrix = np.stack([self._embedding_cache[i] for i in ids])
            sims = np.dot(matrix, emb)
            max_idx = int(np.argmax(sims))
            best_sim = float(sims[max_idx])
            if best_sim >= similarity_threshold:
                best_id = ids[max_idx]

        now = utc_now_iso()

        if best_id is not None:
            # Existing visitor recognized
            with self.get_connection() as conn:
                conn.execute(
                    "UPDATE visitors SET last_seen_at = ? WHERE visitor_id = ?",
                    (now, best_id)
                )
                conn.commit()
            return best_id, False, best_sim

        # Genuinely new visitor: generate new sequential or unique ID
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM visitors")
            count = cursor.fetchone()[0] + 1
            new_id = f"VIS-{count:04d}"

            # If by chance ID exists, fallback to short uuid
            cursor.execute("SELECT 1 FROM visitors WHERE visitor_id = ?", (new_id,))
            if cursor.fetchone():
                new_id = f"VIS-{uuid.uuid4().hex[:6].upper()}"

            meta_json = json.dumps(metadata or {})
            conn.execute(
                """
                INSERT INTO visitors (visitor_id, first_seen_at, last_seen_at, embedding_blob, total_visits, quality_score, metadata_json)
                VALUES (?, ?, ?, ?, 1, ?, ?)
                """,
                (new_id, now, now, emb.tobytes(), float(quality_score), meta_json)
            )
            conn.commit()

        # Update cache
        self._embedding_cache[new_id] = emb
        return new_id, True, 1.0

    def start_visit(
        self,
        visitor_id: str,
        source_id: str,
        entry_image_path: str,
        track_id: Optional[int] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """
        Create a new visit session and log its single entry event in one transaction.
        Returns:
            (visit_id, event_id)
        """
        visit_id = f"VST-{uuid.uuid4().hex[:8].upper()}"
        event_id = f"EVT-{uuid.uuid4().hex[:8].upper()}"
        now = utc_now_iso()
        meta_json = json.dumps(metadata or {})

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM visitors WHERE visitor_id = ?", (visitor_id,))
            if not cursor.fetchone():
                conn.execute(
                    """
                    INSERT INTO visitors (visitor_id, first_seen_at, last_seen_at, embedding_blob, total_visits, quality_score, metadata_json)
                    VALUES (?, ?, ?, ?, 0, 0.0, ?)
                    """,
                    (visitor_id, now, now, np.zeros(512, dtype=np.float32).tobytes(), meta_json)
                )

            # 1. Update total_visits on visitor
            conn.execute(
                "UPDATE visitors SET total_visits = total_visits + 1, last_seen_at = ? WHERE visitor_id = ?",
                (now, visitor_id)
            )

            # 2. Insert into visits
            conn.execute(
                """
                INSERT INTO visits (visit_id, visitor_id, source_id, entry_at, status, entry_image_path)
                VALUES (?, ?, ?, ?, 'active', ?)
                """,
                (visit_id, visitor_id, source_id, now, entry_image_path)
            )

            # 3. Insert entry event (guaranteed at most 1 entry via DB constraint)
            conn.execute(
                """
                INSERT INTO events (event_id, visit_id, visitor_id, event_type, timestamp, image_path, track_id, source_id, confidence, metadata_json)
                VALUES (?, ?, ?, 'entry', ?, ?, ?, ?, ?, ?)
                """,
                (event_id, visit_id, visitor_id, now, entry_image_path, track_id, source_id, confidence, meta_json)
            )
            conn.commit()

        return visit_id, event_id

    def finalize_visit(
        self,
        visit_id: str,
        visitor_id: str,
        source_id: str,
        exit_image_path: str,
        track_id: Optional[int] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Finalize an active visit session and log its single exit event in one transaction.
        Enforces one exit per visit at database level.
        Returns:
            event_id if created, or None if visit was already completed / exit exists.
        """
        now = utc_now_iso()
        event_id = f"EVT-{uuid.uuid4().hex[:8].upper()}"
        meta_json = json.dumps(metadata or {})

        with self.get_connection() as conn:
            # Check if visit is active
            cursor = conn.cursor()
            cursor.execute(
                "SELECT entry_at, status FROM visits WHERE visit_id = ?",
                (visit_id,)
            )
            row = cursor.fetchone()
            if not row or row["status"] != "active":
                # Already finalized or invalid visit
                return None

            entry_at_str = row["entry_at"]
            try:
                entry_dt = datetime.fromisoformat(entry_at_str)
                exit_dt = datetime.fromisoformat(now)
                duration = max(0.0, (exit_dt - entry_dt).total_seconds())
            except Exception:
                duration = 0.0

            try:
                # 1. Update visit status
                conn.execute(
                    """
                    UPDATE visits
                    SET exit_at = ?, status = 'completed', exit_image_path = ?, duration_seconds = ?
                    WHERE visit_id = ? AND status = 'active'
                    """,
                    (now, exit_image_path, duration, visit_id)
                )

                # 2. Insert exit event
                conn.execute(
                    """
                    INSERT INTO events (event_id, visit_id, visitor_id, event_type, timestamp, image_path, track_id, source_id, confidence, metadata_json)
                    VALUES (?, ?, ?, 'exit', ?, ?, ?, ?, ?, ?)
                    """,
                    (event_id, visit_id, visitor_id, now, exit_image_path, track_id, source_id, confidence, meta_json)
                )
                conn.commit()
                return event_id
            except sqlite3.IntegrityError:
                # Duplicate exit prevention triggered by database constraint
                conn.rollback()
                return None

    # --- Analytics & Reporting Queries ---

    def get_stats(self) -> Dict[str, Any]:
        """Return system-wide statistics for the dashboard."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM visitors")
            unique_visitors = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM visits WHERE status = 'active'")
            active_visits = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM events WHERE event_type = 'entry'")
            total_entries = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM events WHERE event_type = 'exit'")
            total_exits = cursor.fetchone()[0]

            # Daily unique count (visits started today UTC)
            today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            cursor.execute(
                "SELECT COUNT(DISTINCT visitor_id) FROM visits WHERE entry_at LIKE ?",
                (f"{today_prefix}%",)
            )
            today_unique_visitors = cursor.fetchone()[0]

        return {
            "unique_visitors": unique_visitors,
            "active_visits": active_visits,
            "total_entries": total_entries,
            "total_exits": total_exits,
            "today_unique_visitors": today_unique_visitors,
        }

    def get_recent_events(
        self,
        limit: int = 50,
        offset: int = 0,
        event_type: Optional[str] = None,
        visitor_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch recent events with optional filtering and pagination."""
        query = "SELECT * FROM events WHERE 1=1"
        params: List[Any] = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if visitor_id:
            query += " AND visitor_id = ?"
            params.append(visitor_id)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_all_visitors(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Fetch registered visitors list without exposing raw embedding blobs."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT visitor_id, first_seen_at, last_seen_at, total_visits, quality_score, metadata_json
                FROM visitors
                ORDER BY first_seen_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset)
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if item.get("metadata_json"):
                    try:
                        item["metadata"] = json.loads(item["metadata_json"])
                    except Exception:
                        item["metadata"] = {}
                else:
                    item["metadata"] = {}
                results.append(item)
            return results

    def get_visits(
        self,
        limit: int = 100,
        offset: int = 0,
        visitor_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch visit sessions with duration and entry/exit details."""
        query = "SELECT * FROM visits WHERE 1=1"
        params: List[Any] = []

        if visitor_id:
            query += " AND visitor_id = ?"
            params.append(visitor_id)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY entry_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_daily_visitor_summary(self, days: int = 7) -> List[Dict[str, Any]]:
        """Return aggregated daily visitor and visit counts for chart rendering."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    SUBSTR(entry_at, 1, 10) as visit_date,
                    COUNT(DISTINCT visitor_id) as unique_visitors,
                    COUNT(*) as total_visits
                FROM visits
                GROUP BY visit_date
                ORDER BY visit_date DESC
                LIMIT ?
                """,
                (days,)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in reversed(rows)]

    def check_health(self) -> Dict[str, Any]:
        """Check database connectivity and integrity."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check;")
                status = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM visitors;")
                v_count = cursor.fetchone()[0]
                return {
                    "status": "healthy" if status == "ok" else "corrupt",
                    "integrity": status,
                    "registered_visitors": v_count,
                    "db_path": self.db_path,
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "db_path": self.db_path,
            }
