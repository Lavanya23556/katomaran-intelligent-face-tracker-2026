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


def test_update_visitor_name(test_repo, sample_embedding):
    vid, _, _ = test_repo.match_or_register_visitor(sample_embedding)
    updated = test_repo.update_visitor(vid, name="Alice Smith")
    assert updated is True

    visitors = test_repo.get_visitors()
    v = next(item for item in visitors if item["visitor_id"] == vid)
    assert v["name"] == "Alice Smith"


def test_merge_visitors(test_repo, sample_embedding):
    vid1, _, _ = test_repo.match_or_register_visitor(sample_embedding)
    np.random.seed(99)
    other_emb = np.random.randn(512).astype(np.float32)
    other_emb /= np.linalg.norm(other_emb)
    vid2, _, _ = test_repo.match_or_register_visitor(other_emb)

    # Start and finish visits for both
    vst1, _ = test_repo.start_visit(vid1, "cam1", "entry1.jpg")
    test_repo.finalize_visit(vst1, vid1, "cam1", "exit1.jpg")

    vst2, _ = test_repo.start_visit(vid2, "cam1", "entry2.jpg")
    test_repo.finalize_visit(vst2, vid2, "cam1", "exit2.jpg")

    assert test_repo.get_stats()["unique_visitors"] == 2

    # Merge vid2 into vid1
    merged = test_repo.merge_visitors(source_visitor_id=vid2, target_visitor_id=vid1)
    assert merged is True
    assert test_repo.get_stats()["unique_visitors"] == 1

    # vid1 should now have total_visits == 2
    v1 = next(item for item in test_repo.get_all_visitors() if item["visitor_id"] == vid1)
    assert v1["total_visits"] == 2


def test_delete_visitor(test_repo, sample_embedding):
    vid, _, _ = test_repo.match_or_register_visitor(sample_embedding)
    vst, _ = test_repo.start_visit(vid, "cam1", "entry.jpg")
    test_repo.finalize_visit(vst, vid, "cam1", "exit.jpg")

    deleted = test_repo.delete_visitor(vid)
    assert deleted is True
    assert test_repo.get_stats()["unique_visitors"] == 0
    assert len(test_repo.get_all_visitors()) == 0


def test_deep_analytics_and_csv_export(test_repo, sample_embedding):
    vid, _, _ = test_repo.match_or_register_visitor(sample_embedding)
    vst, _ = test_repo.start_visit(vid, "cam1", "entry.jpg")
    test_repo.finalize_visit(vst, vid, "cam1", "exit.jpg")

    analytics = test_repo.get_deep_analytics()
    assert "hourly_entries" in analytics
    assert "dwell_time" in analytics
    assert analytics["visitor_breakdown"]["total_unique"] == 1

    events_csv = test_repo.export_events_csv()
    assert "event_id,visit_id,visitor_id" in events_csv
    assert vid in events_csv

    visitors_csv = test_repo.export_visitors_csv()
    assert "visitor_id,name" in visitors_csv
    assert "total_visits" in visitors_csv
    assert vid in visitors_csv



