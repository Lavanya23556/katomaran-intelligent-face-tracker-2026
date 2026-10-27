"""
FastAPI route definitions for VisiTrace visitor intelligence system.
"""
from __future__ import annotations

import os
import platform
import psutil
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    DailySummaryItem,
    DeepAnalyticsResponse,
    EventItem,
    HealthResponse,
    MergeVisitorsRequest,
    StartStreamRequest,
    StatsResponse,
    StreamActionResponse,
    UpdateVisitorRequest,
    VisitItem,
    VisitorItem,
)
from src.database.repository import DatabaseRepository
from src.pipeline.stream_manager import StreamManager

router = APIRouter(prefix="/api", tags=["VisiTrace API"])


def get_repo(request: Request) -> DatabaseRepository:
    return request.app.state.repo


def get_stream_manager(request: Request) -> StreamManager:
    return request.app.state.stream_manager


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request, repo: DatabaseRepository = Depends(get_repo)):
    """Comprehensive system health and resource status."""
    db_health = repo.check_health()
    sm: StreamManager = request.app.state.stream_manager

    models_info = {
        "detector_loaded": request.app.state.detector is not None,
        "detector_model": request.app.state.config.detector_model_path,
        "recognizer_loaded": request.app.state.recognizer is not None,
        "recognizer_model": request.app.state.config.recognition_model_path,
    }

    sys_info = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
    }

    overall_status = "healthy" if db_health.get("status") == "healthy" else "degraded"

    return {
        "status": overall_status,
        "database": db_health,
        "models": models_info,
        "stream": sm.get_status(),
        "system": sys_info,
    }


@router.get("/stats", response_model=StatsResponse)
def get_stats(repo: DatabaseRepository = Depends(get_repo)):
    """Fetch high-level visitor intelligence KPIs."""
    return repo.get_stats()


@router.get("/status")
def get_stream_status(sm: StreamManager = Depends(get_stream_manager)):
    """Fetch live video processing stream status and FPS."""
    return sm.get_status()


@router.get("/events", response_model=List[EventItem])
def get_events(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    event_type: Optional[str] = Query(default=None),
    visitor_id: Optional[str] = Query(default=None),
    repo: DatabaseRepository = Depends(get_repo),
):
    """Fetch recent visitor events with filtering by type or visitor."""
    return repo.get_recent_events(
        limit=limit,
        offset=offset,
        event_type=event_type,
        visitor_id=visitor_id
    )


@router.get("/visitors", response_model=List[VisitorItem])
def get_visitors(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repo: DatabaseRepository = Depends(get_repo),
):
    """Fetch list of all registered unique visitors."""
    return repo.get_all_visitors(limit=limit, offset=offset)


@router.get("/visits", response_model=List[VisitItem])
def get_visits(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    visitor_id: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    repo: DatabaseRepository = Depends(get_repo),
):
    """Fetch visit session records."""
    return repo.get_visits(
        limit=limit,
        offset=offset,
        visitor_id=visitor_id,
        status=status
    )


@router.get("/analytics/daily", response_model=List[DailySummaryItem])
def get_daily_analytics(
    days: int = Query(default=7, ge=1, le=90),
    repo: DatabaseRepository = Depends(get_repo),
):
    """Fetch daily aggregated unique visitor counts."""
    return repo.get_daily_visitor_summary(days=days)


@router.get("/config")
def get_sanitized_config(request: Request):
    """Fetch active configuration with credentials safely masked."""
    return request.app.state.config.sanitized_dict()


@router.post("/streams/start", response_model=StreamActionResponse)
def start_stream(
    payload: StartStreamRequest,
    sm: StreamManager = Depends(get_stream_manager)
):
    """Start video stream ingestion from local file or RTSP stream."""
    success, msg = sm.start_stream(
        source_type=payload.source_type,
        source_path=payload.source_path,
        loop=payload.loop
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@router.post("/streams/stop", response_model=StreamActionResponse)
def stop_stream(sm: StreamManager = Depends(get_stream_manager)):
    """Stop active video stream ingestion cleanly."""
    success, msg = sm.stop_stream()
    return {"success": success, "message": msg}


@router.get("/streams/preview")
def stream_preview(sm: StreamManager = Depends(get_stream_manager)):
    """Stream live annotated video feed as MJPEG."""
    return StreamingResponse(
        sm.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.patch("/visitors/{visitor_id}", response_model=StreamActionResponse)
def update_visitor_identity(
    visitor_id: str,
    payload: UpdateVisitorRequest,
    repo: DatabaseRepository = Depends(get_repo),
):
    """Admin action: Update visitor name, notes, or metadata."""
    ok = repo.update_visitor(visitor_id=visitor_id, name=payload.name, metadata=payload.metadata)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Visitor '{visitor_id}' not found.")
    return {"success": True, "message": f"Visitor '{visitor_id}' identity updated."}


@router.post("/visitors/merge", response_model=StreamActionResponse)
def merge_visitor_identities(
    payload: MergeVisitorsRequest,
    repo: DatabaseRepository = Depends(get_repo),
):
    """Admin action: Merge two split identities of the same person."""
    ok = repo.merge_visitors(
        source_visitor_id=payload.source_visitor_id,
        target_visitor_id=payload.target_visitor_id
    )
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to merge visitors. Ensure both IDs exist and are distinct.")
    return {"success": True, "message": f"Successfully merged {payload.source_visitor_id} into {payload.target_visitor_id}."}


@router.delete("/visitors/{visitor_id}", response_model=StreamActionResponse)
def delete_visitor_identity(
    visitor_id: str,
    repo: DatabaseRepository = Depends(get_repo),
):
    """Admin action: Delete a false or unwanted visitor record."""
    ok = repo.delete_visitor(visitor_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Visitor '{visitor_id}' not found.")
    return {"success": True, "message": f"Visitor '{visitor_id}' deleted."}


@router.get("/analytics/deep", response_model=DeepAnalyticsResponse)
def get_deep_analytics(repo: DatabaseRepository = Depends(get_repo)):
    """Deep analytics: hourly distributions, dwell times, retention rates."""
    return repo.get_deep_analytics()


@router.get("/export/events.csv")
def export_events_csv(repo: DatabaseRepository = Depends(get_repo)):
    """Export all events as downloadable CSV."""
    csv_data = repo.export_events_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=visitrace_events.csv"}
    )


@router.get("/export/visitors.csv")
def export_visitors_csv(repo: DatabaseRepository = Depends(get_repo)):
    """Export all registered visitors as downloadable CSV."""
    csv_data = repo.export_visitors_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=visitrace_visitors.csv"}
    )
