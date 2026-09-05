"""
SatQuery AI — Pydantic Schemas
================================
"""
from typing import Optional
from pydantic import BaseModel, Field


# ─── VQA ─────────────────────────────────────
class VQARequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image (PNG/JPEG)")
    question: str = Field(..., description="Question about the image")

class VQAResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_trace: str
    query_id: Optional[str] = None
    model: str = "satquery-ai-vqa"
    task: str = "vqa"


# ─── Caption ─────────────────────────────────
class CaptionRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image (PNG/JPEG)")

class CaptionResponse(BaseModel):
    caption: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_trace: str
    query_id: Optional[str] = None
    model: str = "satquery-ai-vqa"
    task: str = "caption"


# ─── Referring ───────────────────────────────
class ReferringRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image (PNG/JPEG)")
    expression: str = Field(..., description="Object description to locate")

class ReferringResponse(BaseModel):
    raw_output: str
    bbox: Optional[list[float]] = Field(None, description="[x1, y1, x2, y2] normalized 0-1")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_trace: str
    query_id: Optional[str] = None
    model: str = "satquery-ai-vqa"
    task: str = "referring"


# ─── Firebase direct submit ───────────────────
class FirebaseQueryRequest(BaseModel):
    image_url: str = Field(..., description="Firebase Storage download URL of the image")
    question: str  = Field(..., description="User's question")
    task: str      = Field("vqa", description="vqa | caption | refer")

class FirebaseQueryResponse(BaseModel):
    query_id: str
    status: str = "pending"
    message: str = "Query submitted. Listen to /results/{query_id} in Firebase."


# ─── Health ──────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_name: str
    gpu_name: Optional[str] = None
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    queries_processed: int = 0
    errors: int = 0
    firebase_listener: bool = False


# ─── Query status ────────────────────────────
class QueryStatusResponse(BaseModel):
    query_id: str
    status: str       # pending | processing | done | error
    question: Optional[str] = None
    task: Optional[str] = None
    timestamp: Optional[int] = None

class QueryResultResponse(BaseModel):
    query_id: str
    answer: str
    confidence: float
    task: str
    processed_at: Optional[int] = None


# ─── Stats ───────────────────────────────────
class StatsResponse(BaseModel):
    total_queries: int
    errors: int
    model_loaded: bool
    firebase_connected: bool
    uptime_seconds: float
