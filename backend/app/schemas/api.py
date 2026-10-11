from typing import Any, Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    id: str
    title: str
    organization: str
    url: str
    page: int | None = None
    section: str | None = None
    region: str
    official: bool
    modality: Literal["text", "table", "image"] = "text"
    excerpt: str | None = None
    score: float | None = None


class Evidence(BaseModel):
    id: str
    title: str
    text: str
    modality: Literal["text", "table", "image"]
    score: float
    source_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    label: str
    detail: str
    status: Literal["complete", "fallback", "skipped"] = "complete"


class ResponseImage(BaseModel):
    url: str
    title: str
    description: str
    category: str
    source: str
    source_url: str
    region: str
    license: str | None = None
    date: str | None = None
    location: str | None = None
    record_id: str | None = None
    verified_record_match: bool = False
    author: str | None = None
    license_url: str | None = None
    date_kind: str | None = None
    width: int | None = None
    height: int | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=2000)
    session_id: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_-]{1,80}$")
    top_k: int = Field(default=5, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    intent: str
    citations: list[Citation]
    evidence: list[Evidence]
    trace: list[TraceStep]
    retrieval_engine: str
    model: str | None
    session_id: str
    images: list[ResponseImage] = Field(default_factory=list)
    query_understanding: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    results: list[Evidence]
