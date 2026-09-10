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
import importlib
import io
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from PIL import Image

# ─ Ensure top-level packages ("AI Agents", "CNN") are importable ────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Alias "AI Agents" → AI_Agents so Python can import it
if "AI_Agents" not in sys.modules:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "AI_Agents",
        _ROOT / "AI Agents" / "__init__.py",
        submodule_search_locations=[str(_ROOT / "AI Agents")],
    )
    if _spec:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["AI_Agents"] = _mod
        # Register all sub-modules so relative imports work
        for _sub in ["router", "planner", "tools", "executor", "verifier", "merger", "agent"]:
            _sspec = importlib.util.spec_from_file_location(
                f"AI_Agents.{_sub}",
                _ROOT / "AI Agents" / f"{_sub}.py",
            )
            if _sspec:
                _smod = importlib.util.module_from_spec(_sspec)
                sys.modules[f"AI_Agents.{_sub}"] = _smod
        _spec.loader.exec_module(_mod)

from backend.config import API_HOST, API_PORT, CORS_ORIGINS, MAX_IMAGE_SIZE_MB
from backend.schemas import (
    CaptionRequest, CaptionResponse,
    FirebaseQueryRequest, FirebaseQueryResponse,
    HealthResponse, QueryResultResponse, QueryStatusResponse,
    ReferringRequest, ReferringResponse,
    StatsResponse,
    VQARequest, VQAResponse,
    # CNN
    CNNDetectRequest, CNNDetectResponse,
    CNNSegmentRequest, CNNSegmentResponse,
    CNNFullRequest, CNNFullResponse,
    # Agent
    AgentAnalyzeRequest, AgentAnalyzeResponse, AgentStatusResponse,
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
    print("  Swagger UI  -> http://localhost:8000/docs")
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
# CNN Endpoints
# ─────────────────────────────────────────────
@app.post("/api/cnn/detect", response_model=CNNDetectResponse, tags=["CNN"])
async def cnn_detect_endpoint(req: CNNDetectRequest):
    """
    **YOLOv8 Object Detection** — instant counting + square bounding boxes.

    Returns detected objects with normalized bboxes [x1,y1,x2,y2] and
    an annotated image (base64 PNG) with square boxes drawn around each object.
    """
    if not req.image_base64:
        raise HTTPException(400, "image_base64 is required")
    image = _decode_b64_image(req.image_base64)
    from CNN.yolo_detector import yolo_detector
    det = yolo_detector.detect(image, conf_threshold=req.conf_threshold)
    if det.error:
        raise HTTPException(500, f"YOLO detection failed: {det.error}")
    # Filter by target classes if requested
    detections = det.detections
    counts = det.object_counts
    if req.target_classes:
        tc = [c.lower() for c in req.target_classes]
        detections = [d for d in detections if d.label.lower() in tc]
        counts = {k: v for k, v in counts.items() if k.lower() in tc}
    return CNNDetectResponse(
        object_counts=counts,
        total_detections=len(detections),
        bboxes=[d.bbox_norm for d in detections],
        bbox_labels=[d.label for d in detections],
        bbox_confidences=[d.confidence for d in detections],
        annotated_image_b64=det.annotated_image_b64,
        confidence=det.confidence,
        duration_ms=det.duration_ms,
    )


@app.post("/api/cnn/detect/upload", response_model=CNNDetectResponse, tags=["CNN"])
async def cnn_detect_upload(
    file: UploadFile = File(..., description="Image file (PNG/JPEG)"),
    conf_threshold: float = Form(0.25),
    target_classes: Optional[str] = Form(None, description="Comma-separated class names"),
):
    """YOLOv8 Object Detection — multipart file upload."""
    image = await _read_upload(file)
    from CNN.yolo_detector import yolo_detector
    det = yolo_detector.detect(image, conf_threshold=conf_threshold)
    if det.error:
        raise HTTPException(500, f"YOLO detection failed: {det.error}")
    tc = [c.strip().lower() for c in target_classes.split(",")] if target_classes else []
    detections = [d for d in det.detections if not tc or d.label.lower() in tc]
    counts = {k: v for k, v in det.object_counts.items() if not tc or k.lower() in tc}
    return CNNDetectResponse(
        object_counts=counts,
        total_detections=len(detections),
        bboxes=[d.bbox_norm for d in detections],
        bbox_labels=[d.label for d in detections],
        bbox_confidences=[d.confidence for d in detections],
        annotated_image_b64=det.annotated_image_b64,
        confidence=det.confidence,
        duration_ms=det.duration_ms,
    )


@app.post("/api/cnn/segment", response_model=CNNSegmentResponse, tags=["CNN"])
async def cnn_segment_endpoint(req: CNNSegmentRequest):
    """
    **ResNet/FCN Land Cover Segmentation** — pixel-level surface density.

    Returns exact percentages for:
    - Vegetation / Forest Canopy
    - Water Bodies / Rivers
    - Urban / Built-up Structures
    - Agricultural Fields
    """
    if not req.image_base64:
        raise HTTPException(400, "image_base64 is required")
    image = _decode_b64_image(req.image_base64)
    from CNN.segmentation import land_cover_segmenter
    seg = land_cover_segmenter.segment(image)
    if seg.error:
        raise HTTPException(500, f"Segmentation failed: {seg.error}")
    from CNN.segmentation import CLASS_DISPLAY_NAMES
    return CNNSegmentResponse(
        land_cover_percentages=seg.percentages,
        land_cover_display={CLASS_DISPLAY_NAMES.get(k, k): v for k, v in seg.percentages.items()},
        dominant_land_cover=seg.dominant_class,
        mask_image_b64=seg.mask_image_b64,
        confidence=seg.confidence,
        duration_ms=seg.duration_ms,
    )


@app.post("/api/cnn/segment/upload", response_model=CNNSegmentResponse, tags=["CNN"])
async def cnn_segment_upload(file: UploadFile = File(..., description="Image file (PNG/JPEG)")):
    """ResNet/FCN Land Cover Segmentation — multipart file upload."""
    image = await _read_upload(file)
    from CNN.segmentation import land_cover_segmenter, CLASS_DISPLAY_NAMES
    seg = land_cover_segmenter.segment(image)
    if seg.error:
        raise HTTPException(500, f"Segmentation failed: {seg.error}")
    return CNNSegmentResponse(
        land_cover_percentages=seg.percentages,
        land_cover_display={CLASS_DISPLAY_NAMES.get(k, k): v for k, v in seg.percentages.items()},
        dominant_land_cover=seg.dominant_class,
        mask_image_b64=seg.mask_image_b64,
        confidence=seg.confidence,
        duration_ms=seg.duration_ms,
    )


@app.post("/api/cnn/full", response_model=CNNFullResponse, tags=["CNN"])
async def cnn_full_endpoint(req: CNNFullRequest):
    """
    **Full CNN Pipeline** — detection + segmentation in parallel.

    Runs YOLOv8 detection and ResNet/FCN segmentation concurrently.
    """
    if not req.image_base64:
        raise HTTPException(400, "image_base64 is required")
    image = _decode_b64_image(req.image_base64)
    from CNN.cnn_pipeline import cnn_pipeline
    result = await cnn_pipeline.run(image, tasks=req.tasks, conf_threshold=req.conf_threshold)
    return CNNFullResponse(**result.to_dict())


@app.post("/api/cnn/full/upload", response_model=CNNFullResponse, tags=["CNN"])
async def cnn_full_upload(
    file: UploadFile = File(...),
    tasks: str = Form("detect,segment"),
    conf_threshold: float = Form(0.25),
):
    """Full CNN Pipeline — multipart file upload."""
    image = await _read_upload(file)
    from CNN.cnn_pipeline import cnn_pipeline
    task_list = [t.strip() for t in tasks.split(",")]
    result = await cnn_pipeline.run(image, tasks=task_list, conf_threshold=conf_threshold)
    return CNNFullResponse(**result.to_dict())


# ─────────────────────────────────────────────
# AI Agent Endpoints
# ─────────────────────────────────────────────
@app.get("/api/agent/status", response_model=AgentStatusResponse, tags=["AI Agent"])
async def agent_status():
    """
    Check which AI agent components are ready.

    - `cnn_detect_ready`: YOLOv8 model is loaded
    - `cnn_segment_ready`: Segmentation model is loaded
    - `vlm_ready`: Qwen2.5-VL model is loaded
    """
    vlm_ready = model_manager.is_loaded
    # CNN models lazy-load on first call; we check if ultralytics is importable
    try:
        import ultralytics  # noqa: F401
        cnn_detect_ready = True
    except ImportError:
        cnn_detect_ready = False
    try:
        import torchvision  # noqa: F401
        cnn_segment_ready = True
    except ImportError:
        cnn_segment_ready = False

    all_ready = vlm_ready and cnn_detect_ready and cnn_segment_ready
    return AgentStatusResponse(
        agent_ready=all_ready,
        cnn_detect_ready=cnn_detect_ready,
        cnn_segment_ready=cnn_segment_ready,
        vlm_ready=vlm_ready,
        message=(
            "All components ready." if all_ready
            else "Some components not yet loaded (see individual flags)."
        ),
    )


@app.post("/api/agent/analyze", response_model=AgentAnalyzeResponse, tags=["AI Agent"])
async def agent_analyze(req: AgentAnalyzeRequest):
    """
    **Full AI Agent Pipeline** — intelligent multi-tool satellite image analysis.

    The agent:
    1. **Routes** the query (fast CNN vs heavy VLM vs Hybrid)
    2. **Plans** execution steps from complex compound questions
    3. **Executes** independent steps in parallel (asyncio.gather)
    4. **Verifies** low-confidence results by triggering re-analysis
    5. **Merges** numeric stats, bboxes, land-cover %, and NL into one report

    Returns a StructuredReport with all analysis results.
    """
    if not req.image_base64:
        raise HTTPException(400, "image_base64 is required")
    image = _decode_b64_image(req.image_base64)

    from AI_Agents.agent import sat_agent
    report = await sat_agent.run(image, req.question, req.conf_threshold)
    return AgentAnalyzeResponse(**report.to_dict())


@app.post("/api/agent/analyze/upload", response_model=AgentAnalyzeResponse, tags=["AI Agent"])
async def agent_analyze_upload(
    file: UploadFile = File(..., description="Satellite/aerial image (PNG/JPEG)"),
    question: str = Form(..., description="Your question about the image"),
    conf_threshold: float = Form(0.25, description="YOLO detection confidence threshold"),
):
    """
    **Full AI Agent Pipeline** — multipart file upload version.

    Same as `/api/agent/analyze` but accepts a file upload instead of base64.
    """
    image = await _read_upload(file)
    from AI_Agents.agent import sat_agent
    report = await sat_agent.run(image, question, conf_threshold)
    return AgentAnalyzeResponse(**report.to_dict())


# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
