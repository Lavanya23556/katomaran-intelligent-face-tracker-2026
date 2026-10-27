"""API endpoint verification tests using FastAPI TestClient."""
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture
def client(test_config, test_repo, test_image_store):
    app = create_app(test_config)
    # Use TestClient with lifespan
    with TestClient(app) as tc:
        yield tc


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("healthy", "degraded")
    assert "database" in data
    assert "models" in data
    assert "system" in data


def test_stats_endpoint(client):
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.json()
    assert "unique_visitors" in data
    assert "active_visits" in data
    assert "total_entries" in data
    assert "total_exits" in data


def test_events_endpoint_and_filtering(client):
    res = client.get("/api/events")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    res_filtered = client.get("/api/events?event_type=entry")
    assert res_filtered.status_code == 200
    assert isinstance(res_filtered.json(), list)


def test_visitors_and_visits_endpoints(client):
    res_vis = client.get("/api/visitors")
    assert res_vis.status_code == 200
    assert isinstance(res_vis.json(), list)

    res_vst = client.get("/api/visits")
    assert res_vst.status_code == 200
    assert isinstance(res_vst.json(), list)


def test_config_endpoint_masks_credentials(client):
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "database_path" in data
    # Ensure sensitive credentials would be masked
    if data.get("rtsp_url"):
        assert "password" not in data["rtsp_url"].lower()


def test_stream_controls_validation(client):
    # Missing source file
    res = client.post("/api/streams/start", json={
        "source_type": "file",
        "source_path": "non_existent_path_xyz.mp4"
    })
    assert res.status_code == 400
    assert "not found" in res.json()["detail"].lower()

    # Invalid RTSP URL
    res2 = client.post("/api/streams/start", json={
        "source_type": "rtsp",
        "source_path": "invalid_url_no_rtsp_prefix"
    })
    assert res2.status_code == 400

    # Stop stream
    res_stop = client.post("/api/streams/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["success"] is True


def test_deep_analytics_and_export_endpoints(client):
    res_analytics = client.get("/api/analytics/deep")
    assert res_analytics.status_code == 200
    data = res_analytics.json()
    assert "hourly_entries" in data
    assert "dwell_time" in data
    assert "visitor_breakdown" in data

    res_events_csv = client.get("/api/export/events.csv")
    assert res_events_csv.status_code == 200
    assert "text/csv" in res_events_csv.headers["content-type"]
    assert "event_id,visit_id,visitor_id" in res_events_csv.text

    res_visitors_csv = client.get("/api/export/visitors.csv")
    assert res_visitors_csv.status_code == 200
    assert "text/csv" in res_visitors_csv.headers["content-type"]
    assert "visitor_id,name" in res_visitors_csv.text
    assert "total_visits" in res_visitors_csv.text


def test_admin_visitor_management_endpoints(client, test_repo, sample_embedding):
    # Setup test visitors
    vid1, _, _ = test_repo.match_or_register_visitor(sample_embedding)
    import numpy as np
    np.random.seed(99)
    emb2 = np.random.randn(512).astype(np.float32)
    emb2 /= np.linalg.norm(emb2)
    vid2, _, _ = test_repo.match_or_register_visitor(emb2)

    # 1. PATCH rename
    res_patch = client.patch(f"/api/visitors/{vid1}", json={"name": "Jane Doe"})
    assert res_patch.status_code == 200
    assert res_patch.json()["success"] is True

    # Verify name updated in visitors list
    res_list = client.get("/api/visitors")
    v = next(item for item in res_list.json() if item["visitor_id"] == vid1)
    assert v["name"] == "Jane Doe"

    # 2. POST merge
    res_merge = client.post("/api/visitors/merge", json={
        "source_visitor_id": vid2,
        "target_visitor_id": vid1
    })
    assert res_merge.status_code == 200
    assert res_merge.json()["success"] is True

    # 3. DELETE visitor
    res_del = client.delete(f"/api/visitors/{vid1}")
    assert res_del.status_code == 200
    assert res_del.json()["success"] is True


