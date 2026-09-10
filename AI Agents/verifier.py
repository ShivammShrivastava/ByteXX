"""
AI Agents/verifier.py — Confidence Verifier
=============================================
Checks each tool result's confidence score and triggers
additional analysis for low-confidence outputs.

Strategy:
  - Low-confidence YOLO detection  → re-run with Qwen2.5-VL detect
  - Low-confidence VLM answer      → re-run with a refined prompt
  - Both still low                 → flag as uncertain in final report

Threshold: LOW_CONFIDENCE_THRESHOLD = 0.45
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from AI_Agents.executor import StepExecutionRecord
from AI_Agents.tools import ToolResult

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = 0.45
MAX_REVERIFICATION_ROUNDS = 1    # prevent infinite loops


@dataclass
class VerificationRecord:
    """Tracks what the verifier did for a specific step result."""
    original_step_index: int
    original_tool: str
    original_confidence: float
    triggered: bool                             # Was re-analysis triggered?
    rerun_tool: Optional[str] = None
    rerun_result: Optional[ToolResult] = None
    final_confidence: float = 0.0
    verification_note: str = ""

    def to_dict(self) -> dict:
        return {
            "step_index":          self.original_step_index,
            "original_tool":       self.original_tool,
            "original_confidence": round(self.original_confidence, 3),
            "triggered":           self.triggered,
            "rerun_tool":          self.rerun_tool,
            "final_confidence":    round(self.final_confidence, 3),
            "note":                self.verification_note,
        }


class Verifier:
    """
    Post-execution confidence verifier.

    For each step result, checks if the confidence is below threshold
    and optionally triggers a complementary tool to improve accuracy.
    """

    async def verify(
        self,
        records: list[StepExecutionRecord],
        image: Image.Image,
        original_question: str,
    ) -> tuple[list[StepExecutionRecord], list[VerificationRecord]]:
        """
        Verify all step records and trigger re-analysis where needed.

        Args:
            records:           Execution records from the executor.
            image:             Original PIL image.
            original_question: The user's original question.

        Returns:
            (updated_records, verification_records)
        """
        verification_logs: list[VerificationRecord] = []

        for record in records:
            if not record.result or not record.result.success:
                verification_logs.append(VerificationRecord(
                    original_step_index=record.step.index,
                    original_tool=record.step.tool,
                    original_confidence=0.0,
                    triggered=False,
                    final_confidence=0.0,
                    verification_note="Step failed — skipping verification.",
                ))
                continue

            conf = record.result.confidence
            vlog = VerificationRecord(
                original_step_index=record.step.index,
                original_tool=record.step.tool,
                original_confidence=conf,
                triggered=False,
                final_confidence=conf,
            )

            if conf >= LOW_CONFIDENCE_THRESHOLD:
                vlog.verification_note = f"Confidence {conf:.1%} ≥ threshold — accepted."
                verification_logs.append(vlog)
                continue

            # ── Low confidence — trigger additional analysis ──────────────────
            logger.info(
                f"  [verifier] Low confidence {conf:.3f} for {record.step.tool} "
                f"(step {record.step.index}) — triggering re-analysis."
            )
            vlog.triggered = True

            rerun_result = await self._rerun(
                record.step.tool, image, original_question, record.step.args
            )

            if rerun_result:
                vlog.rerun_tool       = rerun_result.tool_name
                vlog.rerun_result     = rerun_result
                new_conf              = rerun_result.confidence

                if new_conf > conf:
                    # Re-run improved confidence — use new result
                    record.result        = rerun_result
                    vlog.final_confidence = new_conf
                    vlog.verification_note = (
                        f"Re-analysis with {rerun_result.tool_name} improved "
                        f"confidence: {conf:.1%} → {new_conf:.1%}."
                    )
                    logger.info(
                        f"  [verifier] Improved: {conf:.3f} → {new_conf:.3f} "
                        f"via {rerun_result.tool_name}"
                    )
                else:
                    # Re-run didn't help — keep original but flag as uncertain
                    vlog.final_confidence = max(conf, new_conf)
                    vlog.verification_note = (
                        f"Re-analysis with {rerun_result.tool_name} did not improve "
                        f"confidence ({new_conf:.1%}). Marking result as uncertain."
                    )
                    # Attach uncertainty flag to metadata
                    record.result.metadata["uncertain"] = True
                    record.result.metadata["verification_note"] = vlog.verification_note
            else:
                vlog.verification_note = "Re-analysis could not be triggered — no fallback available."

            verification_logs.append(vlog)

        return records, verification_logs

    @staticmethod
    async def _rerun(
        tool_name: str,
        image: Image.Image,
        question: str,
        original_args: dict,
    ) -> Optional[ToolResult]:
        """
        Choose and run the complementary fallback tool.

        Fallback map:
          cnn_detect  → vlm_detect (Qwen grounding)
          cnn_segment → vlm_vqa (ask VLM about land cover)
          vlm_vqa     → vlm_vqa with refined prompt
          vlm_detect  → cnn_detect (YOLO)
          vlm_caption → vlm_caption (retry once)
        """
        from AI_Agents.tools import call_tool

        fallbacks = {
            "cnn_detect":  ("vlm_detect",  {"object_label": _primary_from_args(original_args)}),
            "cnn_segment": ("vlm_vqa",     {"question": _land_cover_fallback_q(question)}),
            "vlm_vqa":     ("vlm_vqa",     {"question": _refine_question(question)}),
            "vlm_detect":  ("cnn_detect",  {"conf_threshold": 0.15}),  # lower threshold
            "vlm_caption": ("vlm_caption", {}),
        }

        if tool_name not in fallbacks:
            return None

        fallback_tool, fallback_args = fallbacks[tool_name]
        try:
            return await call_tool(fallback_tool, image, fallback_args)
        except Exception as e:
            logger.warning(f"Fallback {fallback_tool} also failed: {e}")
            return None


def _primary_from_args(args: dict) -> str:
    classes = args.get("target_classes") or []
    return classes[0] if classes else "object"


def _land_cover_fallback_q(question: str) -> str:
    return (
        "Estimate the approximate percentage coverage of: "
        "vegetation/forest, water bodies, urban/built-up areas, "
        f"and agricultural fields in this image. Original question: {question}"
    )


def _refine_question(question: str) -> str:
    return (
        f"Please carefully re-examine the image and answer this question with high detail: "
        f"{question}\n\nBe very specific about what you observe visually."
    )


# Module-level singleton
verifier = Verifier()
