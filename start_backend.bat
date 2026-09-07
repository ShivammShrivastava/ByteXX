@echo off
title SatQuery AI Backend
cd /d "C:\Users\IPS\Desktop\Byte\ByteXX"
echo ============================================
echo   SatQuery AI Backend - Auto Starting...
echo ============================================
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat
set SATQUERY_MODE=real
uvicorn backend.main:app --host 0.0.0.0 --port 8000
pause
