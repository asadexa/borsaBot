@echo off
setlocal EnableDelayedExpansion

:: =========================================================
::  BORSABOT -- Tam Baslama Scripti v3.1
::  Tum kontroller, kurulum, engine ve dashboard otomasyonu
:: =========================================================

cd /d "%~dp0"
set "ROOT=%~dp0"
set "VENV=%ROOT%venv"
set "PY=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"
set "MT5_EXE=C:\Program Files\MetaTrader 5\terminal64.exe"

title BorsaBot -- Baslatiliyor
echo.
echo =========================================================
echo    BORSABOT -- Kapsamli Baslama Scripti v3.1
echo =========================================================
echo.

:: =========================================================
:: [1/6]  Python kontrolu
:: =========================================================
echo [1/6] Python kontrol ediliyor...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [HATA] Python bulunamadi!
    echo         https://www.python.org/downloads/ adresinden
    echo         Python 3.11+ indirin. Kurulumda "Add Python to PATH"
    echo         secenegini isaretleyin.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] Python %PYVER% bulundu.
echo.

:: =========================================================
:: [2/6]  Sanal ortam (venv) kontrolu + kurulum
:: =========================================================
echo [2/6] Sanal ortam (venv) kontrol ediliyor...
if not exist "%VENV%\Scripts\activate.bat" (
    echo  [..] venv yok -- olusturuluyor...
    python -m venv "%VENV%"
    if !ERRORLEVEL! neq 0 (
        echo  [HATA] venv olusturulamadi!
        pause
        exit /b 1
    )
    echo  [OK] venv olusturuldu.
) else (
    echo  [OK] venv mevcut.
)
echo.

:: =========================================================
:: [3/6]  Paket kontrolu + kurulum (tam venv yolu kullan)
:: =========================================================
echo [3/6] Python paketleri kontrol ediliyor...

"%PIP%" install --upgrade pip --quiet >nul 2>&1

"%PY%" -m py_compile scripts\main.py >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [!!] Syntax hatasi scripts\main.py icinde!
    pause
    exit /b 1
)

:: pandas yoksa tum paketi kur
"%PY%" -c "import pandas" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [..] Temel paketler eksik, pip install -e . calistiriliyor...
    echo       (ilk kurulumda 2-5 dakika surebilir)
    echo.
    "%PIP%" install -e "%ROOT%." --quiet
    if !ERRORLEVEL! neq 0 (
        echo  [HATA] Paket kurulumu basarisiz! Lutfen elle deneyin:
        echo         %PIP% install -e .
        pause
        exit /b 1
    )
    echo  [OK] Tum bagimliliklar yuklendi.
) else (
    echo  [OK] Temel paketler yuklu.
)

:: rich yoksa ayri yukle
"%PY%" -c "import rich" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [..] rich yukleniyor (dashboard icin gerekli)...
    "%PIP%" install rich --quiet
    echo  [OK] rich yuklendi.
)

:: MetaTrader5 python paketi yoksa yukle
"%PY%" -c "import MetaTrader5" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [..] MetaTrader5 python paketi yukleniyor...
    "%PIP%" install MetaTrader5 --quiet
    echo  [OK] MetaTrader5 yuklendi.
)

echo  [OK] Tum Python paketleri hazir.
echo.

:: =========================================================
:: [4/6]  MetaTrader 5 uygulamasi kontrolu
:: =========================================================
echo [4/6] MetaTrader 5 uygulamasi kontrol ediliyor...
tasklist /FI "IMAGENAME eq terminal64.exe" 2>nul | find /I "terminal64.exe" >nul
if %ERRORLEVEL% equ 0 (
    echo  [OK] MetaTrader 5 zaten calisuyor.
) else (
    echo  [..] MT5 acik degil -- baslatilmaya calisiliyor...
    if exist "%MT5_EXE%" (
        start "" "%MT5_EXE%"
        echo  [OK] MT5 baslatildi. Giris yapilmasini bekliyoruz (15sn)...
        timeout /t 15 /nobreak >nul
        echo  [!!] ONEMLI: MT5 icinde "Algorithmic Trading" aktif olmali!
        echo       Araçlar >> Seçenekler >> Uzman Danismanlar >> Algoritmik islemlere izin ver
    ) else (
        echo  [!!] MT5 bulunamadi: %MT5_EXE%
        echo       Lutfen MetaTrader 5 uygulamasini el ile acin.
        echo       Algoritmik trading acik olmali!
        echo.
        echo  MT5 acik degilse engine baglanti hatasi alirsiniz.
        echo  Devam etmek icin bir tusa basin...
        pause >nul
    )
)
echo.

:: =========================================================
:: [5/6]  Docker Desktop kontrolu + Servisler
:: =========================================================
echo [5/6] Docker servisleri kontrol ediliyor...

docker ps >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo  [OK] Docker calisuyor.

    :: timescaledb konteyneri calisiyor mu?
    for /f %%i in ('docker ps --filter "name=borsabot_timescaledb" --filter "status=running" -q 2^>nul') do set TSDB_ID=%%i

    if not defined TSDB_ID (
        echo  [..] Docker servisleri baslatiliyor...
        docker compose up -d timescaledb redis prometheus grafana >nul 2>&1
        if !ERRORLEVEL! neq 0 (
            echo  [!!] docker compose up basarisiz oldu, devam ediliyor...
        ) else (
            echo  [..] Servislerin hazir olmasini bekliyoruz (10sn)...
            timeout /t 10 /nobreak >nul
            echo  [OK] Docker servisleri baslatildi.
        )
    ) else (
        echo  [OK] Docker servisleri zaten calisuyor.
    )

    echo.
    echo  Servis durumu:
    docker compose ps --format "  {{.Name}} -- {{.Status}}" 2>nul
    echo.
    echo  Grafana  : http://localhost:3000 (admin / borsabot)
    echo  Kibana   : http://localhost:5601
    echo  Prometheus: http://localhost:9090
) else (
    echo  [!!] Docker calismiyor veya Docker Desktop kapali.
    echo       Redis ve TimescaleDB olmadan bot calisabilir fakat
    echo       veri kaydedilmez ve Grafana goruntulenemez.
    echo.
    echo  Docker Desktop'i baslatip tekrar deneyebilirsiniz.
    echo  Simdilik devam ediliyor...
)
echo.

:: =========================================================
:: [6/6]  Borsa Engine + Terminal pencereleri
:: =========================================================
echo [6/6] Engine ve Terminal baslatiliyor...
echo.

:: Engine -- ayri bat dosyasiyla aciyor (guvenilir yontem)
echo  [>>] Borsa Engine penceresi aciliyor (kural secimi orada yapilacak)...
start "BorsaBot Engine" "%ROOT%engine_launcher.bat"

:: Dashboard MT5 baglantisini yakalasin diye 8 saniye bekle
echo  [..] Dashboard icin 8 saniye bekleniyor...
timeout /t 8 /nobreak >nul

:: Dashboard -- ayri bat dosyasiyla aciyor
echo  [>>] Borsa Terminal (dashboard) penceresi aciliyor...
start "BorsaBot Terminal" "%ROOT%dashboard_launcher.bat"

echo.
echo =========================================================
echo  BorsaBot tamamen ayaga kaldirildi!
echo ---------------------------------------------------------
echo  [Pencere 1] BorsaBot Engine    --> Kural secimi burada
echo  [Pencere 2] BorsaBot Terminal  --> Canli dashboard
echo ---------------------------------------------------------
echo  Grafana   : http://localhost:3000 (admin / borsabot)
echo  Kibana    : http://localhost:5601
echo  Durdurmak : docker compose down  (bu klasorde)
echo =========================================================
echo.
echo Bu pencereyi kapatabilirsiniz.
echo Engine ve Terminal bagimsiz calismaya devam eder.
echo.
pause
