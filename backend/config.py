"""
SatQuery AI — Backend Configuration
=====================================
"""
import os

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
HF_LORA_REPO   = "shivamw/satquery-ai-vqa-lora"   # HF repo after training
BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# If you saved the adapter locally (from Jupyter Cell 8):
LOCAL_ADAPTER_PATH = os.environ.get(
    "SATQUERY_ADAPTER_PATH",
    r"C:\Users\IPS\Desktop\Satellite\satquery-ai-vqa-lora"
)

# 4-bit quantization — keeps model within 4 GB VRAM on RTX 3050
LOAD_IN_4BIT = True

# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────
MAX_NEW_TOKENS_VQA     = 64
MAX_NEW_TOKENS_CAPTION = 256
MAX_NEW_TOKENS_REFER   = 32
TEMPERATURE            = 0.1
DO_SAMPLE              = False

# ─────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────
RS_SYSTEM_PROMPT = (
    "You are SatQuery AI, a remote sensing image analysis expert specializing in "
    "satellite and aerial imagery interpretation. You can identify land cover types, "
    "objects, spatial relationships, and scene characteristics in remote sensing images. "
    "Provide accurate, concise answers based on visual evidence in the image."
)

# ─────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────
API_HOST       = "0.0.0.0"
API_PORT       = 8000
CORS_ORIGINS   = ["*"]
MAX_IMAGE_SIZE_MB = 10
