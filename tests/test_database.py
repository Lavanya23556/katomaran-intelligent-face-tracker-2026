"""Tests for database transactions, integrity, and queries."""
import numpy as np
import pytest
from src.database.repository import DatabaseRepository


def test_wal_mode_and_foreign_keys(test_repo):
    with test_repo.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"

        cursor.execute("PRAGMA foreign_keys;")
        fk = cursor.fetchone()[0]
        assert fk == 1


def test_visitor_registration_and_matching(test_repo, sample_embedding):
    # 1. Register new visitor
    vid, is_new, sim = test_repo.match_or_register_visitor(
        embedding=sample_embedding,
        similarity_threshold=0.60,
        quality_score=85.0
    )
    assert is_new is True
    assert vid.startswith("VIS-")
    assert sim == 1.0

    stats = test_repo.get_stats()
    assert stats["unique_visitors"] == 1

    # 2. Present same embedding -> should recognize existing visitor
    vid2, is_new2, sim2 = test_repo.match_or_register_visitor(
        embedding=sample_embedding,
        similarity_threshold=0.60
    )
    assert is_new2 is False
    assert vid2 == vid
    assert sim2 >= 0.999

    # Unique count must remain 1
    assert test_repo.get_stats()["unique_visitors"] == 1


def test_different_visitor_registration(test_repo, sample_embedding):
    vid1, is_new1, _ = test_repo.match_or_register_visitor(sample_embedding, similarity_threshold=0.6)
    assert is_new1 is True

    # Generate orthogonal embedding
    np.random.seed(99)
    other_emb = np.random.randn(512).astype(np.float32)
    other_emb /= np.linalg.norm(other_emb)

    vid2, is_new2, sim = test_repo.match_or_register_visitor(other_emb, similarity_threshold=0.6)
    assert is_new2 is True
    assert vid2 != vid1
    assert test_repo.get_stats()["unique_visitors"] == 2


def test_visit_session_entry_and_exit(test_repo, sample_embedding):
    vid, _, _ = test_repo.match_or_register_visitor(sample_embedding)

    # Start visit
    visit_id, event_id = test_repo.start_visit(
        visitor_id=vid,
        source_id="test_cam",
        entry_image_path="logs/entries/2026-10-19/test_entry.jpg",
        track_id=1,
        confidence=0.92
    )
    assert visit_id.startswith("VST-")
    assert event_id.startswith("EVT-")

    stats = test_repo.get_stats()
    assert stats["active_visits"] == 1
    assert stats["total_entries"] == 1
    assert stats["total_exits"] == 0

    # Finalize visit
    exit_event_id = test_repo.finalize_visit(
        visit_id=visit_id,
        visitor_id=vid,
        source_id="test_cam",
        exit_image_path="logs/exits/2026-10-19/test_exit.jpg",
        track_id=1,
        confidence=0.88
    )
    assert exit_event_id is not None
    assert exit_event_id.startswith("EVT-")

    stats = test_repo.get_stats()
    assert stats["active_visits"] == 0
    assert stats["total_entries"] == 1
    assert stats["total_exits"] == 1


def test_duplicate_exit_prevention(test_repo, sample_embedding):
    vid, _, _ = test_repo.match_or_register_visitor(sample_embedding)
    visit_id, _ = test_repo.start_visit(vid, "cam1", "img_entry.jpg")

    # First exit succeeds
    ev1 = test_repo.finalize_visit(visit_id, vid, "cam1", "img_exit1.jpg")
    assert ev1 is not None

    # Repeated exit attempt must return None and NOT create a second event
    ev2 = test_repo.finalize_visit(visit_id, vid, "cam1", "img_exit2.jpg")
    assert ev2 is None

    stats = test_repo.get_stats()
    assert stats["total_exits"] == 1


def test_database_persistence_across_restart(test_config, sample_embedding):
    repo1 = DatabaseRepository(test_config.database_path)
    vid, is_new, _ = repo1.match_or_register_visitor(sample_embedding)
    visit_id, _ = repo1.start_visit(vid, "cam1", "entry.jpg")
    repo1.finalize_visit(visit_id, vid, "cam1", "exit.jpg")

    # Instantiate brand new repository instance pointing to the same file
    repo2 = DatabaseRepository(test_config.database_path)
    stats = repo2.get_stats()
    assert stats["unique_visitors"] == 1
    assert stats["total_entries"] == 1
    assert stats["total_exits"] == 1

    # Match again in repo2: should recognize same visitor
    vid_matched, is_new2, sim = repo2.match_or_register_visitor(sample_embedding)
    assert is_new2 is False
    assert vid_matched == vid
