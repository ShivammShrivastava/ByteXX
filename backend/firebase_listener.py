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

    # Maximum age (seconds) for a pending query before it is considered stale.
    # Queries older than this were likely submitted before the backend started
    # and would just block fresh requests.
    _STALE_AFTER_S = 1800  # 30 minutes — relaxed so minor restarts don't expire valid queries

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self._running = False
        self._thread = None
        self._poll_interval = 2        # seconds between polls
        self._auth_error_count = 0     # track consecutive JWT failures
        self._last_auth_error = ""     # avoid repeated identical prints
        self._in_flight = set()        # query IDs currently being processed
        # CPU can only run one model at a time — enforce serial inference.
        # Set to Semaphore(2) if you have GPU and want mild concurrency.
        self._inference_sem = threading.Semaphore(1)

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
        """Find pending queries, drop stale ones, dispatch fresh ones newest-first.

        Also recovers orphaned queries that are stuck in 'processing' with no
        recent heartbeat (> 4 min without an update) — these are left over from
        backend restarts and would block the queue forever otherwise.
        """
        if not is_initialized():
            return
        try:
            ref = get_db().child(DB_QUERIES_PATH)
            all_queries = ref.get()
            if not all_queries:
                return

            now_ms = int(time.time() * 1000)
            stale_cutoff_ms = now_ms - self._STALE_AFTER_S * 1000
            # Queries stuck in 'processing' without a heartbeat for > 4 min are orphaned
            orphan_cutoff_ms = now_ms - 240_000  # 4 minutes

            # ── Pass 1: recover orphaned 'processing' queries ──────────────────
            for query_id, query_data in all_queries.items():
                if not isinstance(query_data, dict):
                    continue
                if query_data.get("status") != "processing":
                    continue
                if query_id in self._in_flight:
                    # We own it — heartbeat keeps it alive, not orphaned
                    continue

                ts = query_data.get("timestamp", now_ms)
                last_hb_s = query_data.get("heartbeat")  # elapsed seconds, written every 30s

                # Estimate last activity time
                if last_hb_s is not None:
                    # heartbeat is elapsed seconds since dispatch; reconstruct last-active ms
                    last_active_ms = ts + int(last_hb_s) * 1000
                else:
                    last_active_ms = ts

                if last_active_ms < orphan_cutoff_ms:
                    print(
                        f"  [RECOVER] Orphaned query {query_id} (last_active={(now_ms - last_active_ms)//1000}s ago)"
                        " — marking error"
                    )
                    try:
                        get_db().child(DB_QUERIES_PATH).child(query_id).update({
                            "status": "error",
                            "error": "Query was abandoned mid-processing (backend restarted). Please resubmit.",
                        })
                    except Exception:
                        pass

            # ── Pass 2: collect genuine pending entries ────────────────────────
            pending = []
            for query_id, query_data in all_queries.items():
                if not isinstance(query_data, dict):
                    continue
                if query_data.get("status") != "pending":
                    continue
                if query_id in self._in_flight:
                    continue

                ts = query_data.get("timestamp", now_ms)
                if ts < stale_cutoff_ms:
                    # Too old — mark as error so it doesn't block the queue
                    print(f"  [SKIP] Stale query {query_id} (age={(now_ms-ts)//1000}s) — marking error")
                    try:
                        get_db().child(DB_QUERIES_PATH).child(query_id).update(
                            {"status": "error", "error": "Query expired while backend was busy"}
                        )
                    except Exception:
                        pass
                    continue

                pending.append((ts, query_id, query_data))

            # Process newest queries first so fresh requests don't wait behind old ones
            pending.sort(key=lambda x: x[0], reverse=True)

            for ts, query_id, query_data in pending:
                # Mark as processing immediately to prevent re-picking on next poll
                self._in_flight.add(query_id)
                get_db().child(DB_QUERIES_PATH).child(query_id).update({"status": "processing"})

                # Dispatch to its own thread — the semaphore inside will serialise
                # actual model inference so only 1 runs at a time on CPU
                worker = threading.Thread(
                    target=self._run_query_thread,
                    args=(query_id, query_data),
                    daemon=True,
                    name=f"QW-{query_id[:8]}",
                )
                worker.start()

        except Exception as e:
            err_str = str(e)
            if "initialize_app()" in err_str:
                time.sleep(10)
            else:
                raise

    def _run_query_thread(self, query_id: str, query_data: dict):
        """Run one query: wait for the inference semaphore, then process it."""
        # Each thread gets its own event loop for async pipelines
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # ── Serialise model inference: only 1 query runs at a time on CPU ──
            print(f"  [WAIT] {query_id[:12]} waiting for inference slot...")
            with self._inference_sem:
                print(f"  [RUN]  {query_id[:12]} acquired inference slot")
                # Pass the loop so agent/CNN pipelines can use it
                self._handle_query(query_id, query_data, loop)
        finally:
            self._in_flight.discard(query_id)
            loop.close()


    def _handle_query(self, query_id: str, query_data: dict, loop: asyncio.AbstractEventLoop):
        """Dispatch to the correct pipeline based on task type."""
        import threading as _th
        task = query_data.get("task", "vqa")
        question = str(query_data.get('question', ''))[:50]
        print(f"  [IN]  Query {query_id} | task={task} | '{question}'")
        t_dispatch = time.time()

        try:
            image = self._load_image(query_data)
            print(f"  [OK]  Image loaded | {image.size[0]}x{image.size[1]} | {time.time()-t_dispatch:.1f}s")
        except Exception as e:
            self._write_error(query_id, task, f"Image load failed: {e}")
            return

        # Heartbeat: keep writing to RTDB so frontend knows backend is alive
        _stop_hb = _th.Event()
        def _heartbeat():
            while not _stop_hb.wait(30):
                elapsed = int(time.time() - t_dispatch)
                try:
                    get_db().child(DB_QUERIES_PATH).child(query_id).update(
                        {"heartbeat": elapsed, "status": "processing"}
                    )
                    print(f"  [HB]  {query_id[:12]} still running ({elapsed}s elapsed)")
                except Exception:
                    pass
        _hb_thread = _th.Thread(target=_heartbeat, daemon=True)
        _hb_thread.start()

        try:
            if task == "agent":
                self._handle_agent(query_id, query_data, image, task, loop)
            elif task == "cnn":
                self._handle_cnn(query_id, query_data, image, task, loop)
            else:
                self._handle_vlm(query_id, query_data, image, task)
            print(f"  [OK]  Query {query_id[:12]} done in {time.time()-t_dispatch:.1f}s")
        except Exception as e:
            print(f"  [ERR] Query {query_id[:12]} failed after {time.time()-t_dispatch:.1f}s: {e}")
            self._write_error(query_id, task, str(e))
        finally:
            _stop_hb.set()


    # ─── VLM pipeline ─────────────────────────────────────────────────────────

    def _handle_vlm(self, query_id, query_data, image, task):
        """Handle VQA / caption / refer tasks.

        CPU mode (no GPU):
          ALL queries use YOLO-based analysis (~0.5 s per query).
          - Object/count queries  → YOLO counts + bboxes
          - Scene/description     → structured scene summary from YOLO detections
          - Caption               → YOLO scene summary
          - No VLM calls on CPU (VLM takes 90+ s per query on CPU)

        GPU mode:
          Full VLM (Qwen2.5-VL) inference with bounding-box detection on top.
        """
        question = query_data.get("question", "")

        if not self.model_manager.is_loaded:
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
        has_gpu = (self.model_manager._device is not None and
                   "cuda" in str(self.model_manager._device))

        # Classify the query
        obj_label = self._detect_object_keyword(question)
        is_count  = self._is_counting_query(question)
        is_object_query = obj_label is not None

        # ════════════════════════════════════════════════════════════
        # SMART ROUTING: decide CPU vs GPU per query
        #   Fast path (YOLO/CPU):  counting + object detection queries
        #   Deep path (VLM/GPU):   captions, scene description, reasoning
        # ════════════════════════════════════════════════════════════
        use_fast_path = (is_count or is_object_query) and task == "vqa"

        if not has_gpu:
            # No GPU at all — everything goes through YOLO
            use_fast_path = True

        if use_fast_path:
            # ── FAST PATH (CPU / YOLO) ─────────────────────────────
            route = "YOLO/CPU"
            answer, confidence, bboxes, bbox_object = self._yolo_answer(
                image, question, task
            )
        else:
            # ── DEEP PATH (GPU / VLM) ──────────────────────────────
            if not has_gpu:
                # Fallback: no GPU, YOLO already handled above
                route = "YOLO/CPU-fallback"
                answer, confidence, bboxes, bbox_object = self._yolo_answer(
                    image, question, task
                )
            elif task == "caption":
                route = "VLM/GPU"
                answer, confidence = self._run_vlm_with_timeout(
                    lambda: self.model_manager.generate_caption(image),
                    result_key="caption",
                )
            elif task == "refer":
                route = "VLM/GPU"
                expr = query_data.get("expression", question)
                answer, confidence = self._run_vlm_with_timeout(
                    lambda: self.model_manager.locate_object(image, expr),
                    result_key="raw_output",
                )
            else:
                route = "VLM/GPU"
                answer, confidence = self._run_vlm_with_timeout(
                    lambda: self.model_manager.answer_vqa(image, question),
                    result_key="answer",
                )
                # Also overlay bounding-box detection for object queries
                if is_object_query and obj_label:
                    try:
                        detect_result = self.model_manager.detect_objects(image, obj_label)
                        if detect_result.get("bboxes"):
                            bboxes = detect_result["bboxes"]
                            bbox_object = obj_label
                    except Exception as det_err:
                        print(f"  Detection skipped: {det_err}")

        payload = {
            "answer":       answer,
            "confidence":   round(float(confidence), 3),
            "task":         task,
            "processed_at": int(time.time() * 1000),
        }
        if bboxes:
            payload["bboxes"] = bboxes
            payload["bbox_object"] = bbox_object

        get_db().child(DB_RESULTS_PATH).child(query_id).set(payload)
        get_db().child(DB_QUERIES_PATH).child(query_id).update({"status": "done"})
        print(f"  [OK] {route} done {query_id}: '{answer[:60]}'")

    # ─── VLM timeout helper ────────────────────────────────────────────────────

    def _run_vlm_with_timeout(
        self, fn, result_key: str = "answer", timeout_s: int = 90
    ) -> tuple[str, float]:
        """Run a VLM callable in a thread with a hard timeout.

        Args:
            fn:         Zero-arg callable that returns a dict with at least
                        result_key and 'confidence' keys.
            result_key: Dict key to extract the text answer from.
            timeout_s:  Hard wall-clock timeout in seconds (default 90s).

        Returns:
            (answer_str, confidence_float)
        """
        import threading as _thr
        result_holder: list = []
        exc_holder:    list = []

        def _worker():
            try:
                result_holder.append(fn())
            except Exception as e:
                exc_holder.append(e)

        t = _thr.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout_s)

        if t.is_alive():
            # Thread is still running — VLM timed out
            print(f"  [TIMEOUT] VLM inference exceeded {timeout_s}s — returning error")
            return (
                f"The AI model did not respond within {timeout_s} seconds on CPU. "
                f"This image/query combination is too complex for CPU inference. "
                f"Please try a simpler query (e.g. 'describe this image') or restart "
                f"the backend with GPU support.",
                0.0,
            )

        if exc_holder:
            err = str(exc_holder[0])
            print(f"  [ERR] VLM inference raised: {err}")
            return f"Model error: {err}", 0.0

        if not result_holder:
            return "Model returned no result.", 0.0

        res = result_holder[0]
        answer     = str(res.get(result_key, res.get("answer", "No answer returned.")))
        confidence = float(res.get("confidence", 0.75))
        return answer, confidence

    # ─── YOLO-based answer generator (CPU mode) ────────────────────────────────

    def _yolo_answer(
        self, image, question: str, task: str
    ) -> tuple[str, float, list, str | None]:
        """Generate an answer using only YOLO — no VLM. Used on CPU.

        Handles all query types:
          - Object/count queries  → YOLO count + bboxes
          - Scene/description     → structured summary of all detected objects
          - Caption               → same as scene summary

        Returns:
            (answer, confidence, bboxes, bbox_object)
        """
        from CNN.yolo_detector import yolo_detector

        q = question.lower().strip()
        obj_label = self._detect_object_keyword(question)

        try:
            yolo_result = yolo_detector.detect(image)
        except Exception as e:
            print(f"  [YOLO] detect() failed: {e}")
            return (
                f"Object detection failed on CPU: {e}. "
                f"Please try again or use Agent mode.",
                0.0, [], None
            )

        counts   = yolo_result.object_counts   # e.g. {"Vehicle": 3, "Person": 1}
        total    = yolo_result.total_count
        conf_avg = yolo_result.confidence or 0.5
        dets     = yolo_result.detections

        # ── YOLO label mapping (object keyword → COCO/satellite labels) ──────────
        # YOLOv8n on COCO-80: class 4 = "airplane" but satellite images are
        # typically overhead, so COCO airplane class rarely fires. We still try.
        YOLO_LABEL_MAP: dict[str, list[str]] = {
            "airplane":     ["airplane", "aircraft", "Airplane", "Aircraft"],
            "car":          ["vehicle", "car", "Vehicle", "Car"],
            "ship":         ["vessel", "ship", "boat", "Vessel", "Ship", "Boat"],
            "building":     ["building", "Building"],
            "truck":        ["truck", "Truck"],
            "storage tank": ["storage tank", "Storage Tank"],
            "bridge":       ["bridge", "Bridge"],
            "person":       ["person", "Person"],
        }

        bboxes: list = []
        bbox_object: str | None = None

        # ── Object / count query ───────────────────────────────────────────────
        if obj_label:
            yolo_labels = YOLO_LABEL_MAP.get(obj_label, [obj_label])
            matching = [
                d for d in dets
                if any(yl.lower() in d.label.lower() for yl in yolo_labels)
            ]
            count = len(matching)
            bboxes = [d.bbox_norm for d in matching]
            bbox_object = obj_label

            if count > 0:
                answer = (
                    f"There {'is' if count == 1 else 'are'} {count} "
                    f"{obj_label}{'s' if count != 1 else ''} detected in this image."
                )
                confidence = conf_avg if conf_avg > 0 else 0.7
            else:
                # Object not found — report what WAS found
                if counts:
                    found_str = ", ".join(
                        f"{v} {k.lower()}{'s' if v != 1 else ''}"
                        for k, v in counts.items()
                    )
                    answer = (
                        f"No {obj_label}s were detected. "
                        f"The image contains: {found_str}. "
                        f"Note: aircraft, buildings, and storage tanks are outside "
                        f"the base detector's scope — use Agent mode for those."
                    )
                else:
                    answer = (
                        f"No {obj_label}s detected. The scene appears to contain no "
                        f"recognisable objects at the current detection threshold."
                    )
                confidence = 0.5

            print(f"  [YOLO] '{obj_label}': found {count} | conf={confidence:.2f}")
            return answer, confidence, bboxes, bbox_object

        # ── Scene / description / caption query (no specific object) ─────────
        # Build a structured description from everything YOLO found
        if total == 0:
            answer = (
                "No objects were detected in this image at the current confidence "
                "threshold (0.25). The scene may be empty, contain very small objects, "
                "or consist mostly of natural features (water, vegetation, terrain) "
                "which are not in the detector's object class list. "
                "For detailed scene description, GPU inference is required."
            )
            confidence = 0.3
        else:
            # Build count summary
            summary_parts = [
                f"{v} {k.lower()}{'s' if v != 1 else ''}"
                for k, v in counts.items()
            ]
            summary = ", ".join(summary_parts)

            # Build a scene context hint from the question
            q_hints: dict[str, str] = {
                "water":        "No water bodies were identified (water is not an YOLO object class).",
                "vegetation":   "Vegetation/greenery is not classified by the object detector.",
                "building":     "No structures were detected by the base object detector.",
                "road":         "Road network is not classified by the base object detector.",
                "color":        "Colour analysis requires GPU vision model inference.",
                "colour":       "Colour analysis requires GPU vision model inference.",
            }
            extra = ""
            for kw, hint in q_hints.items():
                if kw in q:
                    extra = f" {hint}"
                    break

            answer = (
                f"Scene analysis (CPU / object detector mode):\n\n"
                f"**Detected objects:** {summary} ({total} total, "
                f"avg. confidence {conf_avg:.0%}).\n\n"
                f"The detector identified {len(counts)} distinct object class(es).{extra}\n\n"
                f"*Note: Detailed answers to '{question[:80]}' require GPU-accelerated "
                f"VLM inference. Use Agent mode for richer analysis.*"
            )
            confidence = conf_avg if conf_avg > 0 else 0.6

        print(f"  [YOLO-scene] total={total} | conf={confidence:.2f}")
        return answer, confidence, [], None

    # ─── AI Agent pipeline ────────────────────────────────────────────────────

    def _handle_agent(self, query_id, query_data, image, task, loop: asyncio.AbstractEventLoop):
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

        print(f"  [Agent] Starting pipeline for {query_id[:12]}...")
        try:
            # Hard 4-minute timeout — guarantees Firebase always gets a result.
            # On CPU (verifier reruns disabled) YOLO-only queries finish in < 60s.
            # Hybrid queries with one VLM call typically finish in 2-3 min on CPU.
            report = loop.run_until_complete(
                asyncio.wait_for(sat_agent.run(image, question), timeout=240)
            )
            rd = report.to_dict()
        except asyncio.TimeoutError:
            print(f"  [Agent] Pipeline TIMED OUT after 4 min for {query_id[:12]}")
            error_payload = {
                "summary":          "Analysis timed out after 4 minutes on CPU. "
                                    "Try a simpler query (e.g. detection-only) or "
                                    "enable GPU inference for longer analyses.",
                "vlm_answer":       None,
                "object_counts":    {},
                "total_detections": 0,
                "bboxes":           [],
                "land_cover":       {},
                "confidence":       0.0,
                "tools_used":       [],
                "routing_mode":     "timeout",
                "errors":           ["Pipeline timed out after 240 seconds on CPU."],
                "duration_ms":      240000.0,
                "question":         question,
                "processed_at":     int(time.time() * 1000),
            }
            get_db().child(DB_AGENT_PATH).child(query_id).set(error_payload)
            get_db().child(DB_QUERIES_PATH).child(query_id).update({"status": "done"})
            return
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

    def _handle_cnn(self, query_id, query_data, image, task, loop: asyncio.AbstractEventLoop):
        """Handle CNN-only pipeline (detect + segment) -- writes to /cnn_results/."""
        tasks_param = query_data.get("cnn_tasks", ["detect", "segment"])
        conf_threshold = float(query_data.get("conf_threshold", 0.25))

        from CNN.cnn_pipeline import cnn_pipeline

        result = loop.run_until_complete(
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
