"""
SatQuery AI — Model Manager (Python 3.13 Compatible)
=====================================================
Uses standard transformers + PEFT + BitsAndBytes.
No Unsloth dependency — works on Python 3.13/Windows.

Load order:
  1. Try local LoRA adapter (fastest — no download)
  2. Try HuggingFace repo adapter
  3. Fall back to base model (no fine-tuning)

Stub mode (SATQUERY_MODEL_MODE=stub):
  Returns placeholder responses without loading any model.
  Use this to test the full Firebase pipeline before Qwen is available.
"""

import io
import math
import os
import re
import threading
from typing import Optional

import torch
from PIL import Image


class SatQueryModel:
    """Thread-safe singleton model manager."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model     = None
        self._processor = None
        self._device    = None
        self._initialized = True
        self._stats = {"queries": 0, "errors": 0}
        self._stub_mode = False

    # ─────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return (self._model is not None and self._processor is not None) or self._stub_mode

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    # ─────────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────────
    def load(self):
        """Load base model + LoRA adapter with 4-bit quantization.
        
        If SATQUERY_MODEL_MODE=stub, skips loading and enables stub responses.
        """
        from backend.config import MODEL_MODE

        if MODEL_MODE == "stub":
            self._stub_mode = True
            print("  ⚙  Stub mode enabled — Qwen will NOT be loaded.")
            print("     Set SATQUERY_MODEL_MODE=real to load the actual model.")
            return

        if self._model is not None:
            return

        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        from backend.config import (
            BASE_MODEL_NAME,
            LOCAL_ADAPTER_PATH,
            HF_LORA_REPO,
            LOAD_IN_4BIT,
        )

        model_path, is_local = self._resolve_model_path(BASE_MODEL_NAME)
        print(f"  Base model : {BASE_MODEL_NAME}")
        if is_local:
            print(f"  Local snapshot : {model_path}")
        print(f"  Quantize   : {'4-bit NF4' if LOAD_IN_4BIT else 'FP16'}")

        # Quantization config
        quant_config = None
        if LOAD_IN_4BIT and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        # Load base model
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            local_files_only=is_local,
        )

        # Try loading LoRA adapter
        adapter_path = self._find_adapter(LOCAL_ADAPTER_PATH, HF_LORA_REPO)
        if adapter_path:
            try:
                from peft import PeftModel
                print(f"  Adapter    : {adapter_path}")
                model = PeftModel.from_pretrained(base_model, adapter_path)
                self._model = model.merge_and_unload()
                print(f"  LoRA merged into base model")
            except Exception as e:
                print(f"  Adapter load failed ({e}) — using base model")
                self._model = base_model
        else:
            print(f"  No adapter found — using base model")
            self._model = base_model

        self._model.eval()
        try:
            self._processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=is_local
            )
        except Exception:
            self._processor = AutoProcessor.from_pretrained(
                BASE_MODEL_NAME, trust_remote_code=True
            )
        self._device = next(self._model.parameters()).device

        vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"  Device     : {self._device}")
        print(f"  VRAM used  : {vram:.1f} GB")

    @staticmethod
    def _resolve_model_path(base_model_name: str) -> tuple[str, bool]:
        """Resolve base model path. If a cached snapshot exists in HF_HOME,
        returns (snapshot_path, True). Otherwise returns (base_model_name, False).
        Using snapshot directory directly bypasses Windows / OneDrive PermissionError
        on refs/main and prevents unnecessary HF Hub network requests.
        """
        if os.path.isdir(base_model_name):
            return base_model_name, True

        hf_home = os.environ.get("HF_HOME") or os.path.expanduser(r"~\.cache\huggingface")
        hub_dir = os.path.join(hf_home, "hub")
        clean_name = f"models--{base_model_name.replace('/', '--')}"
        snapshots_dir = os.path.join(hub_dir, clean_name, "snapshots")
        if os.path.isdir(snapshots_dir):
            try:
                snapshots = [
                    os.path.join(snapshots_dir, s)
                    for s in os.listdir(snapshots_dir)
                    if os.path.isdir(os.path.join(snapshots_dir, s))
                ]
                if snapshots:
                    snapshots.sort(key=os.path.getmtime, reverse=True)
                    return snapshots[0], True
            except Exception:
                pass
        return base_model_name, False

    @staticmethod
    def _find_adapter(local_path: str, hf_repo: str) -> Optional[str]:
        """Return the best available adapter path."""
        if local_path and os.path.isdir(local_path):
            # Check it has adapter weights
            if any(f.endswith(".bin") or f.endswith(".safetensors")
                   for f in os.listdir(local_path)):
                return local_path
        # Fall back to HF repo (requires internet)
        try:
            from huggingface_hub import repo_exists
            if repo_exists(hf_repo):
                return hf_repo
        except Exception:
            pass
        return None

    # ─────────────────────────────────────────────
    # Public inference methods
    # ─────────────────────────────────────────────
    def answer_vqa(self, image: Image.Image, question: str) -> dict:
        if self._stub_mode:
            return self._stub_response("vqa", image, question)

        on_gpu = self._device is not None and "cuda" in str(self._device)

        if on_gpu:
            # ── GPU: full research-quality prompt, 512 tokens (≈ 8-30 s) ──
            prompt = (
                f"You are SatQuery AI, an expert remote sensing analyst. "
                f"A user has asked the following question about this satellite/aerial image:\n"
                f"QUESTION: {question}\n\n"
                f"Provide a thorough, research-quality analysis. Your response MUST include:\n"
                f"1. A direct, precise answer to the question\n"
                f"2. Detailed observations about what you see in the image related to the question\n"
                f"3. Spatial context: describe locations (top-left, center, etc.) and relationships\n"
                f"4. Any relevant characteristics: size, shape, color, density, arrangement\n"
                f"5. Confidence assessment and any limitations or caveats\n"
                f"6. Remote sensing insights relevant to the scene\n\n"
                f"Write in clear, professional language as if writing a geospatial intelligence report. "
                f"Be specific and detailed — minimum 150 words."
            )
            max_tok = 512
        else:
            # ── CPU: concise prompt, 128 tokens (≈ 30-90 s) ──
            prompt = (
                f"You are a satellite image analyst. "
                f"Answer this question about the image: {question}\n"
                f"Give a clear, direct answer in 2-4 sentences. "
                f"Include the count/number if asked, describe what you see, and note the location."
            )
            max_tok = 48   # cap on CPU — most answers need < 20 tokens; cuts gen time ~halved

        answer, confidence = self._generate(image, prompt, "vqa", max_new_tokens=max_tok)
        self._stats["queries"] += 1
        return {
            "answer":          answer,
            "confidence":      confidence,
            "reasoning_trace": f"{'Research' if on_gpu else 'CPU'}-mode VQA on {image.size[0]}x{image.size[1]} image. Q: '{question}'",
        }

    def detect_objects(
        self, image: Image.Image, object_label: str, count_hint: int = 0
    ) -> dict:
        """
        Ask Qwen2.5-VL to locate all instances of `object_label` in the image.
        Returns normalized [0-1] bounding boxes alongside the answer.
        """
        if self._stub_mode:
            return {"answer": "[STUB] Detection skipped.", "confidence": 0.0,
                    "bboxes": [], "object": object_label}

        # Step 1 — Ask Qwen for bounding boxes
        on_gpu = self._device is not None and "cuda" in str(self._device)
        bbox_max_tokens = 96 if on_gpu else 48   # fewer tokens on CPU = faster

        prompt = (
            f"Task: Detect ALL {object_label}s in this satellite/aerial image. "
            f"Scan the ENTIRE image carefully — do NOT stop after the first one. "
            f"For EVERY {object_label} you see (even partially visible ones), "
            f"output one line per object in this EXACT format:\n"
            f"BBOX: [x1, y1, x2, y2]\n"
            f"Coordinates are 0-1000 (top-left=0,0 bottom-right=1000,1000). "
            f"Example for 3 objects:\n"
            f"BBOX: [120, 80, 280, 200]\n"
            f"BBOX: [450, 300, 620, 430]\n"
            f"BBOX: [700, 150, 850, 280]\n"
            f"Now detect all {object_label}s and list every BBOX:"
        )
        raw_output, confidence = self._generate(image, prompt, "refer", max_new_tokens=bbox_max_tokens)
        bboxes = self._parse_plain_bboxes(raw_output)

        # Also try Qwen native grounding format if plain text parsing failed
        if not bboxes:
            bboxes = self._parse_grounding_boxes(raw_output)
        if not bboxes:
            bboxes = self._parse_legacy_boxes(raw_output)

        # Build answer sentence
        count = len(bboxes)
        if count > 0:
            answer = (
                f"There {'is' if count == 1 else 'are'} {count} "
                f"{object_label}{'s' if count != 1 else ''} visible in the image."
            )
        else:
            # Fallback: plain VQA count question (no boxes)
            count_q = (
                f"How many {object_label}s are visible in this satellite image? "
                f"Answer with a number only."
            )
            answer, confidence = self._generate(image, count_q, "vqa")
            bboxes = []

        self._stats["queries"] += 1
        return {
            "answer":     answer,
            "confidence": confidence,
            "bboxes":     bboxes,
            "object":     object_label,
        }

    def generate_caption(self, image: Image.Image) -> dict:
        if self._stub_mode:
            return self._stub_response("caption", image, "")
        prompt = "Describe the contents of this remote sensing image in detail."
        caption, confidence = self._generate(image, prompt, "caption")
        self._stats["queries"] += 1
        return {
            "caption":         caption,
            "confidence":      confidence,
            "reasoning_trace": f"Caption for {image.size[0]}x{image.size[1]} image.",
        }

    def locate_object(self, image: Image.Image, expression: str) -> dict:
        if self._stub_mode:
            return self._stub_response("refer", image, expression)
        prompt = f"Give me the location of {expression}"
        raw_output, confidence = self._generate(image, prompt, "refer")
        bbox = self._parse_bbox(raw_output)
        self._stats["queries"] += 1
        return {
            "raw_output":      raw_output,
            "bbox":            bbox,
            "confidence":      confidence,
            "reasoning_trace": f"Referring '{expression}' in {image.size[0]}x{image.size[1]} image.",
        }

    # ─────────────────────────────────────────────
    # Stub responses (no model loaded)
    # ─────────────────────────────────────────────
    def _stub_response(self, task: str, image: Image.Image, text: str) -> dict:
        """Return a clearly-labeled placeholder when model is not loaded."""
        self._stats["queries"] += 1
        size_str = f"{image.size[0]}x{image.size[1]}"
        stub_answers = {
            "vqa":     f"[STUB MODE] Received your question: \"{text}\". "
                       f"The Qwen2.5-VL model is not yet loaded on this machine. "
                       f"Image size detected: {size_str}. "
                       f"Please start the backend with SATQUERY_MODEL_MODE=real once Qwen is available.",
            "caption": f"[STUB MODE] This is a {size_str} satellite/aerial image. "
                       f"Detailed caption will be generated once the Qwen VLM is connected.",
            "refer":   f"[STUB MODE] Referring expression received: \"{text}\". "
                       f"Bounding box will be returned once the Qwen model is running.",
        }
        answer = stub_answers.get(task, "[STUB MODE] Model not loaded.")
        result = {
            "answer":         answer,
            "confidence":     0.0,
            "reasoning_trace": f"Stub mode — no inference performed.",
        }
        if task == "caption":
            result["caption"] = answer
        if task == "refer":
            result["raw_output"] = answer
            result["bbox"] = None
        return result

    # ─────────────────────────────────────────────
    # Core generation
    # ─────────────────────────────────────────────
    def _generate(
        self, image: Image.Image, text: str, task: str, max_new_tokens: Optional[int] = None
    ) -> tuple[str, float]:
        from backend.config import (
            RS_SYSTEM_PROMPT,
            MAX_NEW_TOKENS_VQA,
            MAX_NEW_TOKENS_CAPTION,
            MAX_NEW_TOKENS_REFER,
            TEMPERATURE,
            DO_SAMPLE,
        )

        max_tokens = max_new_tokens or {
            "vqa":     MAX_NEW_TOKENS_VQA,
            "caption": MAX_NEW_TOKENS_CAPTION,
            "refer":   MAX_NEW_TOKENS_REFER,
        }.get(task, MAX_NEW_TOKENS_VQA)

        messages = [
            {"role": "system", "content": RS_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ]},
        ]

        prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        on_cpu = self._device is None or "cpu" in str(self._device)

        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                do_sample=DO_SAMPLE,
                # On CPU, output_scores=True forces a per-token softmax over the full
                # vocabulary (32k–150k entries) for every generated token — this alone
                # can take several minutes and blocks the response from being written
                # back to Firebase.  Skip score collection on CPU and use a fixed
                # placeholder confidence instead; GPU keeps the full calculation.
                return_dict_in_generate=not on_cpu,
                output_scores=not on_cpu,
                repetition_penalty=1.05,   # prevent looping in long research answers
            )
            # temperature is only valid when sampling; greedy decoding ignores it
            if DO_SAMPLE:
                gen_kwargs["temperature"] = TEMPERATURE

            outputs = self._model.generate(**inputs, **gen_kwargs)

        if on_cpu:
            # CPU: outputs is a plain tensor (no GenerateDecoderOnlyOutput wrapper)
            generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
            confidence = 0.75  # fixed placeholder — avoids vocab-wide softmax on CPU
        else:
            generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:]
            confidence = self._compute_confidence(outputs.scores, generated_ids[0])

        answer = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return answer, confidence

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────
    @staticmethod
    def _compute_confidence(scores: tuple, generated_ids: torch.Tensor) -> float:
        if not scores:
            return 0.5
        token_probs = []
        for i, score in enumerate(scores):
            if i >= len(generated_ids):
                break
            probs = torch.softmax(score[0], dim=-1)
            token_probs.append(probs[generated_ids[i].item()].item())
        if not token_probs:
            return 0.5
        mean_log = sum(math.log(max(p, 1e-10)) for p in token_probs) / len(token_probs)
        return round(max(0.0, min(1.0, math.exp(mean_log))), 3)

    @staticmethod
    def _parse_plain_bboxes(text: str) -> list:
        """
        Parse simple plain-text format: BBOX: [x1, y1, x2, y2]
        Coordinates in 0-1000 scale, returned normalized to 0-1.
        """
        pattern = r'BBOX:\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
        matches = re.findall(pattern, text, re.IGNORECASE)
        boxes = []
        for x1, y1, x2, y2 in matches:
            boxes.append([
                float(x1) / 1000.0,
                float(y1) / 1000.0,
                float(x2) / 1000.0,
                float(y2) / 1000.0,
            ])
        return boxes

    @staticmethod
    def _parse_grounding_boxes(text: str) -> list:
        """
        Parse Qwen2.5-VL native grounding format:
          <|box_start|>(x1,y1),(x2,y2)<|box_end|>
        Coordinates are in 0-1000 scale; returned normalized to 0-1.
        """
        pattern = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
        matches = re.findall(pattern, text)
        boxes = []
        for x1, y1, x2, y2 in matches:
            boxes.append([
                int(x1) / 1000.0,
                int(y1) / 1000.0,
                int(x2) / 1000.0,
                int(y2) / 1000.0,
            ])
        return boxes

    @staticmethod
    def _parse_legacy_boxes(text: str) -> list:
        """
        Parse older Qwen angle-bracket format: <x1><y1><x2><y2>
        Coordinates in 0-100 scale.
        """
        nums = re.findall(r'<(\d+)>', text)
        boxes = []
        for i in range(0, len(nums) - 3, 4):
            boxes.append([
                int(nums[i])   / 100.0,
                int(nums[i+1]) / 100.0,
                int(nums[i+2]) / 100.0,
                int(nums[i+3]) / 100.0,
            ])
        return boxes

    @staticmethod
    def _parse_bbox(text: str) -> Optional[list[float]]:
        nums = re.findall(r'<(\d+)>', text)
        if len(nums) != 4:
            return None
        return [max(0.0, min(1.0, int(n) / 100.0)) for n in nums]

    def get_status(self) -> dict:
        info = {
            "model_loaded": self.is_loaded,
            "model_name":   "Qwen/Qwen2.5-VL-3B-Instruct" + (" (stub)" if self._stub_mode else ""),
            "gpu_name":     None,
            "vram_used_gb": None,
            "vram_total_gb": None,
            "queries_processed": self._stats["queries"],
            "errors":            self._stats["errors"],
            "stub_mode":         self._stub_mode,
        }
        if torch.cuda.is_available() and not self._stub_mode:
            info["gpu_name"]      = torch.cuda.get_device_name(0)
            info["vram_used_gb"]  = round(torch.cuda.memory_allocated() / 1024**3, 2)
            info["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )
        return info


# Module-level singleton
model_manager = SatQueryModel()
