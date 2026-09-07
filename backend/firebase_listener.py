"""
Firebase Realtime Database Listener
=====================================
Listens for new VQA queries written by the frontend,
runs inference, and writes answers back to Firebase.

Realtime DB structure:
  /queries/{queryId}/
    image_url:  str   (Firebase Storage download URL)
    question:   str
    status:     "pending" | "processing" | "done" | "error"
    timestamp:  int   (Unix ms)

  /results/{queryId}/
    answer:     str
    confidence: float
    task:       str
    processed_at: int
"""

import io
import threading
import time
import urllib.request

from PIL import Image

from backend.firebase_config import (
    FIREBASE_CONFIG,
    DB_QUERIES_PATH,
    DB_RESULTS_PATH,
    init_firebase,
    get_db,
)


class FirebaseListener:
    """
    Background thread that polls Firebase Realtime DB
    for pending queries and processes them via the model.
    """

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self._running = False
        self._thread = None
        self._poll_interval = 2  # seconds between polls

    def start(self):
        """Start the background listener thread."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True
        )
        self._thread.start()
        print("  Firebase listener started (polling every 2s)")

    def stop(self):
        """Stop the listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("  Firebase listener stopped")

    def _poll_loop(self):
        """Main polling loop — checks for pending queries."""
        while self._running:
            try:
                self._process_pending()
            except Exception as e:
                print(f"  Firebase listener error: {e}")
            time.sleep(self._poll_interval)

    def _process_pending(self):
        """Find and process all pending queries."""
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
            print(f"  DB read error: {e}")

    def _handle_query(self, query_id: str, query_data: dict):
        """Process a single query."""
        question  = query_data.get("question", "")
        image_b64 = query_data.get("image_base64", "")
        image_url = query_data.get("image_url", "")
        task      = query_data.get("task", "vqa")

        print(f"  Processing query {query_id}: '{question[:50]}'")

        # Mark as processing
        get_db().child(DB_QUERIES_PATH).child(query_id).update(
            {"status": "processing"}
        )

        try:
            # Load image — prefer base64 (free RTDB), fall back to URL (legacy)
            if image_b64:
                image = self._decode_base64_image(image_b64)
            elif image_url:
                image = self._download_image(image_url)
            else:
                raise ValueError("Query has neither image_base64 nor image_url")

            # Run inference
            if not self.model_manager.is_loaded:
                raise RuntimeError("Model not loaded yet")

            bboxes      = []
            bbox_object = None

            if task == "caption":
                result     = self.model_manager.generate_caption(image)
                answer     = result["caption"]
                confidence = result["confidence"]

            elif task == "refer":
                expr       = query_data.get("expression", question)
                result     = self.model_manager.locate_object(image, expr)
                answer     = str(result["bbox"]) if result["bbox"] else result["raw_output"]
                confidence = result["confidence"]

            else:  # default: vqa
                # ── Check if this is an object-counting/detection question ──
                obj_label = self._detect_object_keyword(question)
                is_count  = self._is_counting_query(question)

                if is_count and obj_label:
                    # Step 1: VQA for the CORRECT text answer to the user's question
                    vqa_result = self.model_manager.answer_vqa(image, question)
                    answer     = vqa_result["answer"]
                    confidence = vqa_result["confidence"]

                    # Step 2: Separate detection pass for real bounding boxes
                    try:
                        detect_result = self.model_manager.detect_objects(image, obj_label)
                        bboxes        = detect_result["bboxes"]
                        bbox_object   = obj_label
                        print(f"  Detection found {len(bboxes)} {obj_label}(s)")
                    except Exception as det_err:
                        print(f"  Detection failed ({det_err}) — visual boxes skipped")
                        bboxes      = []
                        bbox_object = None
                else:
                    result     = self.model_manager.answer_vqa(image, question)
                    answer     = result["answer"]
                    confidence = result["confidence"]

            # Write result to /results/{queryId}
            result_payload = {
                "answer":       answer,
                "confidence":   round(confidence, 3),
                "task":         task,
                "processed_at": int(time.time() * 1000),
            }
            if bboxes:
                result_payload["bboxes"]      = bboxes
                result_payload["bbox_object"] = bbox_object

            get_db().child(DB_RESULTS_PATH).child(query_id).set(result_payload)

            # Mark query as done
            get_db().child(DB_QUERIES_PATH).child(query_id).update(
                {"status": "done"}
            )
            print(f"  Done {query_id}: '{answer[:60]}' | bboxes: {len(bboxes)}")

        except Exception as e:
            error_msg = str(e)
            print(f"  Error processing {query_id}: {error_msg}")
            get_db().child(DB_QUERIES_PATH).child(query_id).update(
                {"status": "error", "error": error_msg}
            )
            get_db().child(DB_RESULTS_PATH).child(query_id).set({
                "answer":       f"Error: {error_msg}",
                "confidence":   0.0,
                "task":         task,
                "processed_at": int(time.time() * 1000),
            })

    @staticmethod
    def _is_counting_query(question: str) -> bool:
        """Returns True if the question is asking to count or detect objects."""
        q = question.lower()
        return any(kw in q for kw in [
            "how many", "count", "number of", "total", "detect", "find all",
            "locate all", "identify all", "show all",
        ])

    @staticmethod
    def _detect_object_keyword(question: str) -> str | None:
        """
        Returns the primary object label if the question mentions a known
        detectable object type, else None.
        """
        q = question.lower()
        OBJECT_MAP = [
            (["airplane", "aircraft", "plane", "planes", "airplanes", "jet", "helicopter"], "airplane"),
            (["car", "cars", "vehicle", "vehicles", "automobile"],                          "car"),
            (["ship", "ships", "vessel", "vessels", "boat", "boats"],                       "ship"),
            (["building", "buildings", "structure", "structures", "house"],                 "building"),
            (["truck", "trucks"],                                                            "truck"),
            (["tank", "tanks", "storage tank"],                                             "storage tank"),
            (["bridge", "bridges"],                                                         "bridge"),
            (["person", "people", "pedestrian", "human"],                                   "person"),
        ]
        for keywords, label in OBJECT_MAP:
            if any(kw in q for kw in keywords):
                return label
        return None

    @staticmethod
    def _decode_base64_image(b64: str) -> Image.Image:
        """Decode a plain base64 string (no data: prefix) into a PIL Image."""
        import base64
        try:
            raw = base64.b64decode(b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}")

    @staticmethod
    def _download_image(url: str) -> Image.Image:
        """Download image from URL (Firebase Storage download URL)."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SatQueryAI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            img_bytes = response.read()
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

