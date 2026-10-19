"""
Pydantic schemas for VisiTrace API requests and responses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StartStreamRequest(BaseModel):
    source_type: Optional[str] = Field(default="file", description="'file' or 'rtsp'")
    source_path: Optional[str] = Field(default=None, description="Local file path or RTSP URL")
    loop: bool = Field(default=True, description="Loop local video continuously")


class StreamActionResponse(BaseModel):
    success: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    database: Dict[str, Any]
    models: Dict[str, Any]
    stream: Dict[str, Any]
    system: Dict[str, Any]


class StatsResponse(BaseModel):
    unique_visitors: int
    active_visits: int
    total_entries: int
    total_exits: int
    today_unique_visitors: int


class EventItem(BaseModel):
    event_id: str
    visit_id: str
    visitor_id: str
    event_type: str
    timestamp: str
    image_path: str
    track_id: Optional[int] = None
    source_id: str
    confidence: float
    metadata_json: Optional[str] = None


class VisitorItem(BaseModel):
    visitor_id: str
    first_seen_at: str
    last_seen_at: str
    total_visits: int
    quality_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VisitItem(BaseModel):
    visit_id: str
    visitor_id: str
    source_id: str
    entry_at: str
    exit_at: Optional[str] = None
    status: str
    entry_image_path: Optional[str] = None
    exit_image_path: Optional[str] = None
    duration_seconds: Optional[float] = None


class DailySummaryItem(BaseModel):
    visit_date: str
    unique_visitors: int
    total_visits: int
