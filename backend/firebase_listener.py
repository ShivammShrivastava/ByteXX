"""
Firebase Realtime Database Listener -- Extended
================================================
Listens for new queries written by the frontend,
dispatches to the correct pipeline (VQA / Agent / CNN),
and writes structured results back to Firebase RTDB.

Realtime DB structure:
  /queries/{queryId}/
    uid:           str
    image_base64:  str   (base64 JPEG, no data: prefix)
    question:      str
    task:          str   ("vqa" | "caption" | "refer" | "agent" | "cnn")
    status:        str   ("pending" | "processing" | "done" | "error")
    timestamp:     int   (Unix ms)
    source:        str   ("frontend_web")

  /results/{queryId}/         <- VQA / caption / refer answers
    answer:        str
    confidence:    float
    bboxes:        list   (optional)
    bbox_object:   str    (optional)
    task:          str
    processed_at:  int

  /agent_results/{queryId}/   <- Full AI Agent structured reports
    summary:            str
    object_counts:      dict
    total_detections:   int
    bboxes:             list
    land_cover:         dict
    land_cover_display: dict
    dominant_land_cover: str
    vlm_answer:         str | None
    confidence:         float
    tools_used:         list
    routing_mode:       str
    duration_ms:        float
    errors:             list
    processed_at:       int

  /cnn_results/{queryId}/     <- CNN-only results
    object_counts:      dict
    total_detections:   int
    land_cover:         dict
    dominant_land_cover: str
    confidence:         float
    tasks_run:          list
    duration_ms:        float
    processed_at:       int
"""

import asyncio
import io
import threading
import time
import urllib.request
import base64

from PIL import Image

from backend.firebase_config import (
    FIREBASE_CONFIG,
    DB_QUERIES_PATH,
    DB_RESULTS_PATH,
    DB_AGENT_PATH,
    DB_CNN_PATH,
    init_firebase,
    get_db,
    is_initialized,
)


class FirebaseListener:
    """
    Background thread that polls Firebase Realtime DB
    for pending queries and processes them via the correct pipeline.

    Supported task values:
      "vqa"     -> Qwen2.5-VL VQA  (writes to /results/)
      "caption" -> Qwen2.5-VL caption (writes to /results/)
      "refer"   -> Qwen2.5-VL referring expression (writes to /results/)
      "agent"   -> Full AI Agent pipeline (writes to /agent_results/)
      "cnn"     -> CNN-only pipeline (writes to /cnn_results/)
    """

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self._running = False
        self._thread = None
        self._poll_interval = 2        # seconds between polls
        self._loop = None              # asyncio event loop for agent pipeline
        self._auth_error_count = 0     # track consecutive JWT failures
        self._last_auth_error = ""     # avoid repeated identical prints

    def start(self):
        """Start the background listener thread."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="FirebaseListener"
        )
        self._thread.start()
        print("  [OK] Firebase listener started (polling every 2s)")

    def stop(self):
        """Stop the listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("  Firebase listener stopped")

    def _poll_loop(self):
        """Main polling loop -- creates a dedicated asyncio loop for async tools."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        while self._running:
            try:
                self._process_pending()
                # Reset backoff on success
                if self._auth_error_count > 0:
                    self._auth_error_count = 0
                    self._poll_interval = 2
                    print("  [OK] Firebase connection restored")
            except Exception as e:
                err_str = str(e)
                is_jwt = "JWT" in err_str or "invalid_grant" in err_str
                if is_jwt:
                    self._auth_error_count += 1
                    if err_str != self._last_auth_error:
                        # Only print when the error message changes
                        print(
                            f"  [WARN]  Firebase JWT auth error (count={self._auth_error_count}): "
                            f"Invalid JWT Signature -- run: w32tm /resync /force"
                        )
                        self._last_auth_error = err_str
                    # Exponential backoff: 2s -> 4s -> 8s ... max 60s
                    self._poll_interval = min(60, 2 ** min(self._auth_error_count, 6))
                else:
                    print(f"  Firebase listener error: {e}")
            time.sleep(self._poll_interval)
        self._loop.close()

    def _process_pending(self):
        """Find and process all pending queries."""
        if not is_initialized():
            return
        try:
            ref = get_db().child(DB_QUERIES_PATH)
            all_queries = ref.get()
            if not all_queries:
                return

            for query_id, query_data in all_queries.items():
                if not isinstance(query_data, dict):
                    continue
                if query_data.get("status") != "pending":
                    continue
                self._handle_query(query_id, query_data)

        except Exception as e:
            err_str = str(e)
            if "initialize_app()" in err_str:
                time.sleep(10)
            else:
                # Re-raise so _poll_loop can classify it (JWT vs other)
                raise


    def _handle_query(self, query_id: str, query_data: dict):
        """Dispatch to the correct pipeline based on task type."""
        import threading as _th
        task = query_data.get("task", "vqa")
        question = str(query_data.get('question', ''))[:50]
        print(f"  [IN]  Query {query_id} | task={task} | '{question}'")
        t_dispatch = time.time()

        # Mark as processing
        get_db().child(DB_QUERIES_PATH).child(query_id).update({"status": "processing"})

        try:
            image = self._load_image(query_data)
            print(f"  [OK]  Image loaded | {image.size[0]}x{image.size[1]} | {time.time()-t_dispatch:.1f}s")
        except Exception as e:
            self._write_error(query_id, task, f"Image load failed: {e}")
            return

        # Heartbeat: keep writing to RTDB so frontend knows backend is alive
        _stop_hb = _th.Event()
        def _heartbeat():
            n = 0
            while not _stop_hb.wait(30):
                n += 1
                elapsed = int(time.time() - t_dispatch)
                try:
                    get_db().child(DB_QUERIES_PATH).child(query_id).update(
                        {"heartbeat": elapsed, "status": "processing"}
                    )
                    print(f"  [HB]  {query_id} still running ({elapsed}s elapsed)")
                except Exception:
                    pass
        _hb_thread = _th.Thread(target=_heartbeat, daemon=True)
        _hb_thread.start()

        try:
            if task == "agent":
                self._handle_agent(query_id, query_data, image, task)
            elif task == "cnn":
                self._handle_cnn(query_id, query_data, image, task)
            else:
                self._handle_vlm(query_id, query_data, image, task)
            print(f"  [OK]  Query {query_id} done in {time.time()-t_dispatch:.1f}s")
        except Exception as e:
            print(f"  [ERR] Query {query_id} failed after {time.time()-t_dispatch:.1f}s: {e}")
            self._write_error(query_id, task, str(e))
        finally:
            _stop_hb.set()

    # ─── VLM pipeline ─────────────────────────────────────────────────────────

    def _handle_vlm(self, query_id, query_data, image, task):
        """Handle VQA / caption / refer tasks via Qwen2.5-VL."""
        question = query_data.get("question", "")

        if not self.model_manager.is_loaded:
            # Model not loaded — return an informative stub response
            print(f"  [WARN] Model not loaded yet for {query_id}; returning stub.")
            payload = {
                "answer":       "Model is still loading. Please retry in 60 seconds.",
                "confidence":   0.0,
                "task":         task,
                "processed_at": int(time.time() * 1000),
            }
            get_db().child(DB_RESULTS_PATH).child(query_id).set(payload)
            get_db().child(DB_QUERIES_PATH).child(query_id).update({"status": "done"})
            return

        bboxes = []
        bbox_object = None

        if task == "caption":
            result = self.model_manager.generate_caption(image)
            answer = result["caption"]
            confidence = result["confidence"]

        elif task == "refer":
            expr = query_data.get("expression", question)
            result = self.model_manager.locate_object(image, expr)
            answer = str(result["bbox"]) if result["bbox"] else result["raw_output"]
            confidence = result["confidence"]

        else:  # vqa (default)
            obj_label = self._detect_object_keyword(question)
            is_count  = self._is_counting_query(question)

            # Always run VQA first (works on both CPU and GPU)
            vqa_result = self.model_manager.answer_vqa(image, question)
            answer     = vqa_result["answer"]
            confidence = vqa_result["confidence"]

            # Try bounding-box detection (works on both CPU and GPU,
            # but skipped if no matching object keyword in query)
            if is_count and obj_label:
                try:
                    detect_result = self.model_manager.detect_objects(image, obj_label)
                    if detect_result.get("bboxes"):
                        bboxes = detect_result["bboxes"]
                        bbox_object = obj_label
                except Exception as det_err:
                    print(f"  Detection skipped: {det_err}")

        payload = {
            "answer":       answer,
            "confidence":   round(confidence, 3),
            "task":         task,
            "processed_at": int(time.time() * 1000),
        }
        if bboxes:
            payload["bboxes"] = bboxes
            payload["bbox_object"] = bbox_object

        get_db().child(DB_RESULTS_PATH).child(query_id).set(payload)
        get_db().child(DB_QUERIES_PATH).child(query_id).update({"status": "done"})
        print(f"  [OK] VLM done {query_id}: '{answer[:60]}'")

    # ─── AI Agent pipeline ────────────────────────────────────────────────────

    def _handle_agent(self, query_id, query_data, image, task):
        """Handle full AI Agent pipeline -- writes to /agent_results/."""
        question = query_data.get("question", "")

        # AI_Agents is registered by main.py's startup block; fall back to
        # direct sys.modules lookup so the listener thread also finds it.
        import sys
        if "AI_Agents.agent" in sys.modules:
            agent_mod = sys.modules["AI_Agents.agent"]
        else:
            # Fallback: register manually (same logic as main.py)
            import importlib.util as _ilu
            from pathlib import Path
            _ai_dir = Path(__file__).resolve().parent.parent / "AI Agents"
            if "AI_Agents" not in sys.modules:
                _pkg_spec = _ilu.spec_from_file_location(
                    "AI_Agents", _ai_dir / "__init__.py",
                    submodule_search_locations=[str(_ai_dir)],
                )
                _pkg_mod = _ilu.module_from_spec(_pkg_spec)
                sys.modules["AI_Agents"] = _pkg_mod
                for _sub in ["router", "planner", "tools", "executor", "verifier", "merger", "agent"]:
                    _ss = _ilu.spec_from_file_location(f"AI_Agents.{_sub}", _ai_dir / f"{_sub}.py")
                    if _ss:
                        _sm = _ilu.module_from_spec(_ss)
                        sys.modules[f"AI_Agents.{_sub}"] = _sm
                        _ss.loader.exec_module(_sm)
                _pkg_spec.loader.exec_module(_pkg_mod)
            agent_mod = sys.modules["AI_Agents.agent"]

        sat_agent = agent_mod.sat_agent

        print(f"  [Agent] Starting pipeline for {query_id}...")
        try:
            # Run the async agent pipeline in our dedicated event loop
            report = self._loop.run_until_complete(
                sat_agent.run(image, question)
            )
            rd = report.to_dict()
        except Exception as agent_err:
            print(f"  [Agent] Pipeline error: {agent_err}")
            # Write a graceful partial result so frontend gets something back
            error_payload = {
                "summary":          f"Agent pipeline encountered an error: {agent_err}",
                "vlm_answer":       None,
                "object_counts":    {},
                "total_detections": 0,
                "bboxes":           [],
                "land_cover":       {},
                "confidence":       0.0,
                "tools_used":       [],
                "routing_mode":     "error",
                "errors":           [str(agent_err)],
                "duration_ms":      0.0,
                "question":         question,
                "processed_at":     int(time.time() * 1000),
            }
            get_db().child(DB_AGENT_PATH).child(query_id).set(error_payload)
            get_db().child(DB_QUERIES_PATH).child(query_id).update(
                {"status": "done", "error": str(agent_err)}
            )
            return

        # Strip large base64 images before writing to RTDB (keep counts/text/percentages)
        payload = {
            "summary":             rd.get("summary", ""),
            "vlm_answer":          rd.get("vlm_answer"),
            "caption":             rd.get("caption"),
            "object_counts":       rd.get("object_counts", {}),
            "total_detections":    rd.get("total_detections", 0),
            "bboxes":              rd.get("bboxes", []),
            "bbox_labels":         rd.get("bbox_labels", []),
            "bbox_confidences":    rd.get("bbox_confidences", []),
            "land_cover":          rd.get("land_cover", {}),
            "land_cover_display":  rd.get("land_cover_display", {}),
            "dominant_land_cover": rd.get("dominant_land_cover", ""),
            "confidence":          rd.get("confidence", 0.0),
            "tools_used":          rd.get("tools_used", []),
            "routing_mode":        rd.get("routing_mode", ""),
            "reasoning_trace":     rd.get("reasoning_trace", ""),
            "errors":              rd.get("errors", []),
            "duration_ms":         rd.get("duration_ms", 0.0),
            "question":            question,
            "processed_at":        int(time.time() * 1000),
        }

        get_db().child(DB_AGENT_PATH).child(query_id).set(payload)
        get_db().child(DB_QUERIES_PATH).child(query_id).update({"status": "done"})
        print(
            f"  [OK] Agent done {query_id}: "
            f"conf={rd.get('confidence',0):.2f} | "
            f"tools={rd.get('tools_used',[])} | "
            f"{rd.get('duration_ms',0):.0f}ms"
        )

    # ─── CNN pipeline ─────────────────────────────────────────────────────────

    def _handle_cnn(self, query_id, query_data, image, task):
        """Handle CNN-only pipeline (detect + segment) -- writes to /cnn_results/."""
        tasks_param = query_data.get("cnn_tasks", ["detect", "segment"])
        conf_threshold = float(query_data.get("conf_threshold", 0.25))

        from CNN.cnn_pipeline import cnn_pipeline

        result = self._loop.run_until_complete(
            cnn_pipeline.run(image, tasks=tasks_param, conf_threshold=conf_threshold)
        )
        rd = result.to_dict()

        payload = {
            "object_counts":       rd.get("object_counts", {}),
            "total_detections":    rd.get("total_detections", 0),
            "bboxes":              rd.get("bboxes", []),
            "bbox_labels":         rd.get("bbox_labels", []),
            "bbox_confidences":    rd.get("bbox_confidences", []),
            "land_cover_percentages": rd.get("land_cover_percentages", {}),
            "land_cover_display":  rd.get("land_cover_display", {}),
            "dominant_land_cover": rd.get("dominant_land_cover", ""),
            "tasks_run":           rd.get("tasks_run", []),
            "overall_confidence":  rd.get("overall_confidence", 0.0),
            "total_duration_ms":   rd.get("total_duration_ms", 0.0),
            "errors":              rd.get("errors", []),
            "processed_at":        int(time.time() * 1000),
        }

        get_db().child(DB_CNN_PATH).child(query_id).set(payload)
        get_db().child(DB_QUERIES_PATH).child(query_id).update({"status": "done"})
        print(
            f"  [OK] CNN done {query_id}: "
            f"detections={rd.get('total_detections',0)} | "
            f"{rd.get('total_duration_ms',0):.0f}ms"
        )

    # ─── Error writer ──────────────────────────────────────────────────────────

    def _write_error(self, query_id: str, task: str, error_msg: str):
        """Write error to both /queries/ and the appropriate results path."""
        print(f"  [ERROR] Error {query_id}: {error_msg}")

        result_path = (
            DB_AGENT_PATH if task == "agent"
            else DB_CNN_PATH if task == "cnn"
            else DB_RESULTS_PATH
        )

        error_payload = {
            "answer":       f"Error: {error_msg}",
            "confidence":   0.0,
            "task":         task,
            "processed_at": int(time.time() * 1000),
            "error":        error_msg,
        }

        get_db().child(DB_QUERIES_PATH).child(query_id).update(
            {"status": "error", "error": error_msg}
        )
        get_db().child(result_path).child(query_id).set(error_payload)

    # ─── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_image(query_data: dict) -> Image.Image:
        """Load PIL image from base64 or URL in the query payload."""
        image_b64 = query_data.get("image_base64", "")
        image_url = query_data.get("image_url", "")

        if image_b64:
            raw = base64.b64decode(image_b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        elif image_url:
            req = urllib.request.Request(
                image_url, headers={"User-Agent": "SatQueryAI/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        else:
            raise ValueError("Query has neither image_base64 nor image_url")

    @staticmethod
    def _is_counting_query(question: str) -> bool:
        q = question.lower()
        return any(kw in q for kw in [
            "how many", "count", "number of", "total", "detect", "find all",
            "locate all", "identify all", "show all",
        ])

    @staticmethod
    def _detect_object_keyword(question: str) -> str | None:
        q = question.lower()
        OBJECT_MAP = [
            (["airplane", "aircraft", "plane", "planes", "airplanes", "aeroplane", "aeroplanes", "jet", "helicopter"], "airplane"),
            (["car", "cars", "vehicle", "vehicles", "automobile"], "car"),
            (["ship", "ships", "vessel", "vessels", "boat", "boats"], "ship"),
            (["building", "buildings", "structure", "structures", "house"], "building"),
            (["truck", "trucks"], "truck"),
            (["tank", "tanks", "storage tank"], "storage tank"),
            (["bridge", "bridges"], "bridge"),
            (["person", "people", "pedestrian", "human"], "person"),
        ]
        for keywords, label in OBJECT_MAP:
            if any(kw in q for kw in keywords):
                return label
        return None
