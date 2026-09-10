"""
SatQuery AI — Backend Configuration
=====================================
All values can be overridden via environment variables.
Copy backend/.env.example → backend/.env and fill in your values.
"""
import os

# ─────────────────────────────────────────────
# HuggingFace Cache — point to local desktop folder
# ─────────────────────────────────────────────
# This tells transformers to use the locally downloaded model
# instead of downloading from the internet.
_HF_HOME = os.environ.get("HF_HOME", r"C:\Users\Suryansh\OneDrive\Desktop\huggingface")
os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_HF_HOME, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_HF_HOME, "hub"))

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
# The base Qwen2.5-VL-3B model — uses local cache set above
BASE_MODEL_NAME = os.environ.get(
    "SATQUERY_BASE_MODEL",
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

# HuggingFace repo for LoRA adapter (used if LOCAL_ADAPTER_PATH is missing)
HF_LORA_REPO = os.environ.get(
    "SATQUERY_HF_LORA_REPO",
    "shivamw/satquery-ai-vqa-lora"
)

# Local path to a saved LoRA adapter directory (fastest — no download)
LOCAL_ADAPTER_PATH = os.environ.get(
    "SATQUERY_ADAPTER_PATH",
    r"C:\Users\Suryansh\OneDrive\Desktop\satquery-ai-vqa-lora"
)

# Enable 4-bit NF4 quantization (needs bitsandbytes + CUDA)
LOAD_IN_4BIT = os.environ.get("SATQUERY_LOAD_4BIT", "true").lower() == "true"

# Model mode: "real" loads Qwen2.5-VL-3B + LoRA. "stub" skips it (for pipeline testing).
MODEL_MODE = os.environ.get("SATQUERY_MODEL_MODE", "real")   # "real" | "stub"

# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────
MAX_NEW_TOKENS_VQA     = int(os.environ.get("SATQUERY_MAX_TOKENS_VQA",     "128"))
MAX_NEW_TOKENS_CAPTION = int(os.environ.get("SATQUERY_MAX_TOKENS_CAPTION", "256"))
MAX_NEW_TOKENS_REFER   = int(os.environ.get("SATQUERY_MAX_TOKENS_REFER",   "512"))
TEMPERATURE            = float(os.environ.get("SATQUERY_TEMPERATURE",      "0.1"))
DO_SAMPLE              = os.environ.get("SATQUERY_DO_SAMPLE", "false").lower() == "true"

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
API_HOST          = os.environ.get("SATQUERY_HOST", "0.0.0.0")
API_PORT          = int(os.environ.get("SATQUERY_PORT", "8000"))
CORS_ORIGINS      = os.environ.get("SATQUERY_CORS_ORIGINS", "*").split(",")
MAX_IMAGE_SIZE_MB = int(os.environ.get("SATQUERY_MAX_IMAGE_MB", "10"))
