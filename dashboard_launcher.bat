@echo off
setlocal
title BorsaBot Terminal -- Canli Dashboard
cd /d "%~dp0"

echo.
echo =========================================================
echo    BORSABOT TERMINAL -- Canli Dashboard
echo =========================================================
echo.

call venv\Scripts\activate.bat
python scripts\dashboard.py --broker mt5 --symbols EURUSD XAUUSD --model-dir models/

echo.
echo  Dashboard durdu.
pause
