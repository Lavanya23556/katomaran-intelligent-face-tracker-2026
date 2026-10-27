"""
Database schema definition and initialization for VisiTrace.
Uses SQLite with WAL mode, foreign keys, and transactional constraints.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

INIT_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;
PRAGMA mmap_size = 268435456;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS visitors (
    visitor_id TEXT PRIMARY KEY,
    name TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    embedding_blob BLOB NOT NULL,
    total_visits INTEGER NOT NULL DEFAULT 1,
    quality_score REAL NOT NULL DEFAULT 0.0,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS visits (
    visit_id TEXT PRIMARY KEY,
    visitor_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    entry_at TEXT NOT NULL,
    exit_at TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'completed')),
    entry_image_path TEXT,
    exit_image_path TEXT,
    duration_seconds REAL,
    FOREIGN KEY (visitor_id) REFERENCES visitors (visitor_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    visit_id TEXT NOT NULL,
    visitor_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN ('entry', 'exit')),
    timestamp TEXT NOT NULL,
    image_path TEXT NOT NULL,
    track_id INTEGER,
    source_id TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    metadata_json TEXT,
    FOREIGN KEY (visit_id) REFERENCES visits (visit_id) ON DELETE CASCADE,
    FOREIGN KEY (visitor_id) REFERENCES visitors (visitor_id) ON DELETE CASCADE,
    CONSTRAINT uq_visit_event UNIQUE (visit_id, event_type)
);

CREATE INDEX IF NOT EXISTS idx_visits_visitor_id ON visits(visitor_id);
CREATE INDEX IF NOT EXISTS idx_visits_status ON visits(status);
CREATE INDEX IF NOT EXISTS idx_visits_entry_at ON visits(entry_at);
CREATE INDEX IF NOT EXISTS idx_events_visitor_id ON events(visitor_id);
CREATE INDEX IF NOT EXISTS idx_events_visit_id ON events(visit_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
"""


def init_database(db_path: str) -> None:
    """Initialize SQLite database with schema, WAL mode, and indices."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.executescript(INIT_SCHEMA_SQL)

        # Ensure 'name' column exists in visitors table
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(visitors);")
        existing_cols = [row[1] for row in cursor.fetchall()]
        if "name" not in existing_cols:
            conn.execute("ALTER TABLE visitors ADD COLUMN name TEXT;")

        # Record schema version
        cursor.execute("SELECT MAX(version) FROM schema_version")
        row = cursor.fetchone()
        current_version = row[0] if row and row[0] is not None else 0

        if current_version < SCHEMA_VERSION:
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,)
            )
        conn.commit()
