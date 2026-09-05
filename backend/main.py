"""
SatQuery AI — Complete FastAPI Backend
========================================
Endpoints:

  System
  ──────
  GET  /                      → Redirect to /docs
  GET  /api/health            → Health check (GPU, VRAM, model status)
  GET  /api/stats             → Query counts, uptime

  REST Inference (direct — no Firebase)
  ──────────────────────────────────────
  POST /api/vqa               → VQA with base64 image JSON
  POST /api/vqa/upload        → VQA with file upload
  POST /api/caption           → Caption with base64 image JSON
  POST /api/caption/upload    → Caption with file upload
  POST /api/refer             → Referring grounding with base64 JSON
  POST /api/refer/upload      → Referring grounding with file upload

  Firebase Integration
  ─────────────────────
  POST /api/firebase/submit   → Submit query via Firebase (writes to Realtime DB)
  GET  /api/firebase/query/{id}  → Get query status from Firebase
  GET  /api/firebase/result/{id} → Get result from Firebase

Usage:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    Swagger UI → http://localhost:8000/docs
"""

import base64
import io
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from PIL import Image

from backend.config import API_HOST, API_PORT, CORS_ORIGINS, MAX_IMAGE_SIZE_MB
from backend.schemas import (
    CaptionRequest, CaptionResponse,
    FirebaseQueryRequest, FirebaseQueryResponse,
    HealthResponse, QueryResultResponse, QueryStatusResponse,
    ReferringRequest, ReferringResponse,
    StatsResponse,
    VQARequest, VQAResponse,
)
from backend.model import model_manager
from backend.firebase_config import init_firebase, FIREBASE_CONFIG
from backend.firebase_listener import FirebaseListener

# ─────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────
_start_time = time.time()
_firebase_listener = FirebaseListener(model_manager)
_firebase_ok = False


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _firebase_ok

    print()
    print("=" * 60)
    print("  SatQuery AI Backend — Starting")
    print("=" * 60)

    # Load model
    print("\n[1/2] Loading Qwen2.5-VL-3B model...")
    try:
        model_manager.load()
        print("  Model ready!")
    except Exception as e:
        print(f"  Model load failed: {e}")
        print("  REST endpoints will return 503 until model is ready.")

    # Firebase
    print("\n[2/2] Connecting to Firebase...")
    try:
        init_firebase()
        _firebase_listener.start()
        _firebase_ok = True
        print("  Firebase Realtime DB listener active.")
    except Exception as e:
        print(f"  Firebase failed: {e}")
        print("  Add serviceAccountKey.json to backend/ to enable Firebase.")

    print("\n" + "=" * 60)
    print("  Server ready at http://localhost:8000")
    print("  Swagger UI  → http://localhost:8000/docs")
    print("=" * 60 + "\n")

    yield

    # Shutdown
    _firebase_listener.stop()
    print("SatQuery AI — Shut down.")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="SatQuery AI",
    description=(
        "Remote sensing image analysis API powered by fine-tuned Qwen2.5-VL-3B.\n\n"
        "**Tasks:** Visual QA · Image Captioning · Referring Expression Grounding\n\n"
        "**Firebase:** Automatically processes queries submitted via Firebase Realtime DB."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _check_model():
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Check server logs for loading status.",
        )

def _decode_b64_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 string.")
    if len(raw) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_IMAGE_SIZE_MB} MB.")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode image from base64.")

async def _read_upload(file: UploadFile) -> Image.Image:
    raw = await file.read()
    if len(raw) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_IMAGE_SIZE_MB} MB.")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode uploaded image.")


# ─────────────────────────────────────────────
# System endpoints
# ─────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check model status, GPU info, Firebase listener state."""
    status = model_manager.get_status()
    return HealthResponse(
        **status,
        firebase_listener=_firebase_ok,
    )


@app.get("/api/stats", response_model=StatsResponse, tags=["System"])
async def stats():
    """Query count, uptime, and connection status."""
    s = model_manager.stats
    return StatsResponse(
        total_queries=s["queries"],
        errors=s["errors"],
        model_loaded=model_manager.is_loaded,
        firebase_connected=_firebase_ok,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


# ─────────────────────────────────────────────
# VQA
# ─────────────────────────────────────────────
@app.post("/api/vqa", response_model=VQAResponse, tags=["Inference"])
async def vqa_json(req: VQARequest):
    """
    Visual Question Answering — send base64 image + question.

    Example question: *"How many buildings are visible?"*
    """
    _check_model()
    if not req.image_base64:
        raise HTTPException(400, "image_base64 is required")
    image = _decode_b64_image(req.image_base64)
    result = model_manager.answer_vqa(image, req.question)
    return VQAResponse(**result, task="vqa")


@app.post("/api/vqa/upload", response_model=VQAResponse, tags=["Inference"])
async def vqa_upload(
    file: UploadFile = File(..., description="Image file (PNG/JPEG)"),
    question: str = Form(..., description="Your question about the image"),
):
    """Visual Question Answering — upload image as multipart form."""
    _check_model()
    image = await _read_upload(file)
    result = model_manager.answer_vqa(image, question)
    return VQAResponse(**result, task="vqa")


# ─────────────────────────────────────────────
# Caption
# ─────────────────────────────────────────────
@app.post("/api/caption", response_model=CaptionResponse, tags=["Inference"])
async def caption_json(req: CaptionRequest):
    """Generate a detailed description of a satellite/aerial image (base64)."""
    _check_model()
    if not req.image_base64:
        raise HTTPException(400, "image_base64 is required")
    image = _decode_b64_image(req.image_base64)
    result = model_manager.generate_caption(image)
    return CaptionResponse(**result, task="caption")


@app.post("/api/caption/upload", response_model=CaptionResponse, tags=["Inference"])
async def caption_upload(
    file: UploadFile = File(..., description="Image file (PNG/JPEG)"),
):
    """Generate a detailed description — upload image as multipart form."""
    _check_model()
    image = await _read_upload(file)
    result = model_manager.generate_caption(image)
    return CaptionResponse(**result, task="caption")


# ─────────────────────────────────────────────
# Referring / Grounding
# ─────────────────────────────────────────────
@app.post("/api/refer", response_model=ReferringResponse, tags=["Inference"])
async def refer_json(req: ReferringRequest):
    """
    Referring Expression Grounding — locate an object described in text (base64 image).

    Returns bounding box `[x1, y1, x2, y2]` normalized to `[0, 1]`.
    Example expression: *"The large white building in the top-left"*
    """
    _check_model()
    if not req.image_base64:
        raise HTTPException(400, "image_base64 is required")
    image = _decode_b64_image(req.image_base64)
    result = model_manager.locate_object(image, req.expression)
    return ReferringResponse(**result, task="referring")


@app.post("/api/refer/upload", response_model=ReferringResponse, tags=["Inference"])
async def refer_upload(
    file: UploadFile = File(..., description="Image file (PNG/JPEG)"),
    expression: str = Form(..., description="Text description of object to locate"),
):
    """Referring Expression Grounding — upload image as multipart form."""
    _check_model()
    image = await _read_upload(file)
    result = model_manager.locate_object(image, expression)
    return ReferringResponse(**result, task="referring")


# ─────────────────────────────────────────────
# Firebase Integration
# ─────────────────────────────────────────────
@app.post("/api/firebase/submit", response_model=FirebaseQueryResponse, tags=["Firebase"])
async def firebase_submit(req: FirebaseQueryRequest):
    """
    Submit a query via Firebase Realtime DB.

    The backend listener will pick it up automatically and write the answer
    to `/results/{query_id}`. The frontend should listen there.

    Use this if you want the backend (not the frontend) to write to Firebase.
    """
    if not _firebase_ok:
        raise HTTPException(503, "Firebase not connected. Check serviceAccountKey.json.")

    from backend.firebase_config import get_db
    import time

    ref = get_db().child("queries")
    new_entry = ref.push({
        "image_url": req.image_url,
        "question":  req.question,
        "task":      req.task,
        "status":    "pending",
        "timestamp": int(time.time() * 1000),
        "source":    "rest_api",
    })
    query_id = new_entry.key
    return FirebaseQueryResponse(
        query_id=query_id,
        status="pending",
        message=f"Query submitted. Listen to /results/{query_id} in Firebase.",
    )


@app.get("/api/firebase/query/{query_id}", response_model=QueryStatusResponse, tags=["Firebase"])
async def firebase_query_status(query_id: str):
    """
    Get the current status of a Firebase query.

    Status values: `pending` → `processing` → `done` | `error`
    """
    if not _firebase_ok:
        raise HTTPException(503, "Firebase not connected.")

    from backend.firebase_config import get_db
    data = get_db().child("queries").child(query_id).get()
    if not data:
        raise HTTPException(404, f"Query '{query_id}' not found.")
    return QueryStatusResponse(
        query_id=query_id,
        status=data.get("status", "unknown"),
        question=data.get("question"),
        task=data.get("task"),
        timestamp=data.get("timestamp"),
    )


@app.get("/api/firebase/result/{query_id}", response_model=QueryResultResponse, tags=["Firebase"])
async def firebase_result(query_id: str):
    """
    Get the model's answer for a completed Firebase query.

    Returns 404 if the result is not ready yet (check query status first).
    """
    if not _firebase_ok:
        raise HTTPException(503, "Firebase not connected.")

    from backend.firebase_config import get_db
    data = get_db().child("results").child(query_id).get()
    if not data:
        raise HTTPException(404, f"Result for '{query_id}' not ready yet.")
    return QueryResultResponse(
        query_id=query_id,
        answer=data.get("answer", ""),
        confidence=data.get("confidence", 0.0),
        task=data.get("task", "vqa"),
        processed_at=data.get("processed_at"),
    )


# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
