@echo off
setlocal

echo =======================================================
echo          BorsaBot Startup Script (Windows)
echo =======================================================
echo.

:: 1. Docker Desktop'un calistigindan emin olalim
echo [1/3] Checking Docker daemon...
docker ps >nul 2>&1
if "%ERRORLEVEL%" neq "0" (
    echo [HATA] Docker Desktop calismiyor veya yonetici izni yok!
    echo Lutfen once Docker Desktop uygulamasini baslatin.
    pause
    exit /b 1
)
echo Docker is running.
echo.

:: 2. Veritabani ve Grafana Web Arayuzunu (Arka planda) baslatalim
echo [2/3] Starting Background Services (TimescaleDB, Redis, Grafana)...
docker compose up -d timescaledb redis prometheus grafana
echo Background services started.
echo.

:: 3. Web UI bilgileri
echo =======================================================
echo [WEB UI] Grafana Live Dashboard hazir!
echo Link: http://localhost:3000
echo Kullanici  : admin
echo Sifre      : borsabot
echo.
echo NOT: Asagidaki bot kapansa bile web arayuzu arkaplanda
echo Docker uzerinden calismaya devam edecektir. 
echo Hizmetleri durdurmak icin baska bir komut satirinda:
echo   "docker compose down" 
echo yazabilirsiniz.
echo =======================================================
echo.

:: 4. BorsaBot Python yazilimini MT5 modunda baslat
echo [3/3] Starting BorsaBot Live Trader (MT5)...
echo.
python scripts/main.py --broker mt5 --symbols EURUSD XAUUSD
echo.

echo Bot durduruldu! Baska tusa basarak cikabilirsiniz.
pause
