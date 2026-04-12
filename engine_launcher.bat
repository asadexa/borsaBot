@echo off
setlocal
title BorsaBot Engine -- Kural Secimi
cd /d "%~dp0"

echo.
echo =========================================================
echo    BORSABOT ENGINE -- Kural Secimi ve Trade Motoru
echo =========================================================
echo.

call venv\Scripts\activate.bat
python scripts\main.py --broker mt5 --symbols EURUSD XAUUSD

echo.
echo  Engine durdu.
pause
