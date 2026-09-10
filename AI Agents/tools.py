"""
AI Agents/tools.py — Tool Registry
=====================================
Defines all callable tools with a standard interface.

Each tool:
  - Accepts a PIL image + optional arguments
  - Returns a ToolResult with: result dict, confidence, duration_ms, tool_name
  - Is async-native (uses asyncio executor for CPU-bound CNN calls)

Available tools:
  cnn_detect(image, target_classes, conf_threshold)  → YOLOv8 detection
  cnn_segment(image)                                 → Land-cover segmentation
  vlm_vqa(image, question)                           → Qwen2.5-VL VQA
  vlm_caption(image)                                 → Qwen2.5-VL captioning
  vlm_detect(image, object_label)                    → Qwen2.5-VL detection + bboxes
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ─── Ensure parent directory is on path for both module styles ────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class ToolResult:
    """
    Standard result container returned by every tool.
    """
    tool_name: str
    result: dict[str, Any]
    confidence: float              # 0.0 – 1.0
    duration_ms: float
    success: bool = True
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool_name":   self.tool_name,
            "result":      self.result,
            "confidence":  self.confidence,
            "duration_ms": self.duration_ms,
            "success":     self.success,
            "error":       self.error,
            "metadata":    self.metadata,
        }


# ─── Tool Implementations ────────────────────────────────────────────────────

async def cnn_detect(
    image: Image.Image,
    target_classes: list[str] | None = None,
    conf_threshold: float = 0.25,
) -> ToolResult:
    """
    YOLOv8 object detection tool.

    Detects all objects in the image and returns counts + bboxes.
    Optionally filters to specific target classes.
    """
    t0 = time.perf_counter()
    try:
        from CNN.yolo_detector import yolo_detector

        loop = asyncio.get_event_loop()
        det_result = await loop.run_in_executor(
            None, yolo_detector.detect, image, conf_threshold
        )

        if det_result.error:
            return ToolResult(
                tool_name="cnn_detect",
                result={},
                confidence=0.0,
                duration_ms=(time.perf_counter() - t0) * 1000,
                success=False,
                error=det_result.error,
            )

        # Filter to target classes if specified
        detections = det_result.detections
        counts     = det_result.object_counts
        if target_classes:
            tc_lower = [c.lower() for c in target_classes]
            detections = [d for d in detections
                          if d.label.lower() in tc_lower or
                          any(tc in d.label.lower() for tc in tc_lower)]
            counts = {k: v for k, v in counts.items()
                      if k.lower() in tc_lower or
                      any(tc in k.lower() for tc in tc_lower)}

        result = {
            "object_counts":       counts,
            "total_detections":    len(detections),
            "bboxes":              [d.bbox_norm for d in detections],
            "bbox_labels":         [d.label for d in detections],
            "bbox_confidences":    [d.confidence for d in detections],
            "annotated_image_b64": det_result.annotated_image_b64,
            "model":               "yolov8n",
        }
        return ToolResult(
            tool_name="cnn_detect",
            result=result,
            confidence=det_result.confidence,
            duration_ms=det_result.duration_ms,
            metadata={"target_classes": target_classes},
        )

    except Exception as e:
        logger.exception("cnn_detect tool failed")
        return ToolResult(
            tool_name="cnn_detect",
            result={},
            confidence=0.0,
            duration_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            error=str(e),
        )


async def cnn_segment(image: Image.Image) -> ToolResult:
    """
    ResNet/FCN land-cover segmentation tool.

    Returns pixel-level percentages for Vegetation, Water, Urban, Agricultural.
    """
    t0 = time.perf_counter()
    try:
        from CNN.segmentation import land_cover_segmenter

        loop = asyncio.get_event_loop()
        seg_result = await loop.run_in_executor(
            None, land_cover_segmenter.segment, image
        )

        if seg_result.error:
            return ToolResult(
                tool_name="cnn_segment",
                result={},
                confidence=0.0,
                duration_ms=(time.perf_counter() - t0) * 1000,
                success=False,
                error=seg_result.error,
            )

        from CNN.segmentation import CLASS_DISPLAY_NAMES
        result = {
            "land_cover_percentages": seg_result.percentages,
            "land_cover_display": {
                CLASS_DISPLAY_NAMES.get(k, k): v
                for k, v in seg_result.percentages.items()
            },
            "dominant_class":     seg_result.dominant_class,
            "mask_image_b64":     seg_result.mask_image_b64,
            "model":              "resnet50-fcn-landcover",
        }
        return ToolResult(
            tool_name="cnn_segment",
            result=result,
            confidence=seg_result.confidence,
            duration_ms=seg_result.duration_ms,
        )

    except Exception as e:
        logger.exception("cnn_segment tool failed")
        return ToolResult(
            tool_name="cnn_segment",
            result={},
            confidence=0.0,
            duration_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            error=str(e),
        )


async def vlm_vqa(image: Image.Image, question: str) -> ToolResult:
    """Qwen2.5-VL Visual Question Answering tool."""
    t0 = time.perf_counter()
    try:
        from backend.model import model_manager

        if not model_manager.is_loaded:
            return ToolResult(
                tool_name="vlm_vqa",
                result={"answer": "[VLM not loaded — stub response]", "confidence": 0.0},
                confidence=0.0,
                duration_ms=(time.perf_counter() - t0) * 1000,
                success=True,
                error=None,
                metadata={"stub": True},
            )

        loop = asyncio.get_event_loop()
        vqa_result = await loop.run_in_executor(
            None, model_manager.answer_vqa, image, question
        )

        return ToolResult(
            tool_name="vlm_vqa",
            result={
                "answer":          vqa_result["answer"],
                "reasoning_trace": vqa_result.get("reasoning_trace", ""),
                "model":           "qwen2.5-vl-3b",
            },
            confidence=vqa_result.get("confidence", 0.5),
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    except Exception as e:
        logger.exception("vlm_vqa tool failed")
        return ToolResult(
            tool_name="vlm_vqa",
            result={},
            confidence=0.0,
            duration_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            error=str(e),
        )


async def vlm_caption(image: Image.Image) -> ToolResult:
    """Qwen2.5-VL image captioning tool."""
    t0 = time.perf_counter()
    try:
        from backend.model import model_manager

        if not model_manager.is_loaded:
            return ToolResult(
                tool_name="vlm_caption",
                result={"caption": "[VLM not loaded — stub caption]"},
                confidence=0.0,
                duration_ms=(time.perf_counter() - t0) * 1000,
                metadata={"stub": True},
            )

        loop = asyncio.get_event_loop()
        cap_result = await loop.run_in_executor(
            None, model_manager.generate_caption, image
        )

        return ToolResult(
            tool_name="vlm_caption",
            result={
                "caption": cap_result["caption"],
                "model":   "qwen2.5-vl-3b",
            },
            confidence=cap_result.get("confidence", 0.5),
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    except Exception as e:
        logger.exception("vlm_caption tool failed")
        return ToolResult(
            tool_name="vlm_caption",
            result={},
            confidence=0.0,
            duration_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            error=str(e),
        )


async def vlm_detect(image: Image.Image, object_label: str) -> ToolResult:
    """Qwen2.5-VL object detection + bounding boxes tool."""
    t0 = time.perf_counter()
    try:
        from backend.model import model_manager

        if not model_manager.is_loaded:
            return ToolResult(
                tool_name="vlm_detect",
                result={"answer": "[VLM not loaded]", "bboxes": []},
                confidence=0.0,
                duration_ms=(time.perf_counter() - t0) * 1000,
                metadata={"stub": True},
            )

        loop = asyncio.get_event_loop()
        det_result = await loop.run_in_executor(
            None, model_manager.detect_objects, image, object_label
        )

        return ToolResult(
            tool_name="vlm_detect",
            result={
                "answer":       det_result["answer"],
                "bboxes":       det_result.get("bboxes", []),
                "object":       det_result.get("object", object_label),
                "model":        "qwen2.5-vl-3b",
            },
            confidence=det_result.get("confidence", 0.5),
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    except Exception as e:
        logger.exception("vlm_detect tool failed")
        return ToolResult(
            tool_name="vlm_detect",
            result={},
            confidence=0.0,
            duration_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            error=str(e),
        )


# ─── Tool Registry ────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Any] = {
    "cnn_detect":  cnn_detect,
    "cnn_segment": cnn_segment,
    "vlm_vqa":     vlm_vqa,
    "vlm_caption": vlm_caption,
    "vlm_detect":  vlm_detect,
}


async def call_tool(
    tool_name: str,
    image: Image.Image,
    args: dict,
) -> ToolResult:
    """
    Call a tool by name with the given image and args.

    Args:
        tool_name: Name from TOOL_REGISTRY.
        image:     PIL image to analyse.
        args:      Keyword arguments for the tool.

    Returns:
        ToolResult from the called tool.
    """
    if tool_name not in TOOL_REGISTRY:
        return ToolResult(
            tool_name=tool_name,
            result={},
            confidence=0.0,
            duration_ms=0.0,
            success=False,
            error=f"Unknown tool: '{tool_name}'. Available: {list(TOOL_REGISTRY)}",
        )

    fn = TOOL_REGISTRY[tool_name]
    return await fn(image, **args)
