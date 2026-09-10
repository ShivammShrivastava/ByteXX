@echo off
title SatQuery AI Backend
cd /d "C:\Users\Suryansh\ByteXX"
echo ============================================
echo   SatQuery AI Backend - Auto Starting...
echo ============================================
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

:: ── Point to your local model and adapter ─────────────────────────────────────
set HF_HOME=C:\Users\Suryansh\OneDrive\Desktop\huggingface
set TRANSFORMERS_CACHE=%HF_HOME%\hub
set HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub
set SATQUERY_ADAPTER_PATH=C:\Users\Suryansh\OneDrive\Desktop\satquery-ai-vqa-lora
set SATQUERY_MODEL_MODE=real
:: ──────────────────────────────────────────────────────────────────────────────

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
pause
