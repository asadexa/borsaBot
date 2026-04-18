"""
BorsaBot Build Script
python launcher/build.py [--clean] [--test]

Adimlar:
  1. PyInstaller kurulumunu kontrol et
  2. Spec dosyasiyla build et
  3. dist/BorsaBot/ klasorunu dogrula
"""
from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Windows konsolunda UTF-8 zorla
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
DIST = ROOT / "dist" / "BorsaBot"
SPEC = ROOT / "BorsaBot.spec"

# ANSI renkleri
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def log(msg: str, color: str = RESET) -> None:
    print(f"{color}{msg}{RESET}", flush=True)


def step(n: int, total: int, msg: str) -> None:
    print(f"\n{CYAN}{BOLD}[{n}/{total}]{RESET} {msg}", flush=True)


def run(cmd: list, cwd: Path | None = None, **kwargs) -> subprocess.CompletedProcess:
    log(f"  $ {' '.join(str(c) for c in cmd)}", YELLOW)
    return subprocess.run(cmd, cwd=str(cwd or ROOT), **kwargs)


def check_pyinstaller() -> bool:
    result = run(
        [sys.executable, "-m", "PyInstaller", "--version"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        log(f"  PyInstaller {result.stdout.strip()} mevcut", GREEN)
        return True
    return False


def install_pyinstaller() -> None:
    log("  PyInstaller kuruluyor...", YELLOW)
    result = run([sys.executable, "-m", "pip", "install", "pyinstaller", "--quiet"])
    if result.returncode != 0:
        log("HATA: PyInstaller kurulamadi!", RED)
        sys.exit(1)
    log("  PyInstaller kuruldu.", GREEN)


def ensure_hooks_dir() -> None:
    hooks = ROOT / "launcher" / "hooks"
    hooks.mkdir(exist_ok=True)
    mt5_hook = hooks / "hook-MetaTrader5.py"
    if not mt5_hook.exists():
        mt5_hook.write_text(
            "from PyInstaller.utils.hooks import collect_all\n"
            "datas, binaries, hiddenimports = collect_all('MetaTrader5')\n",
            encoding="utf-8",
        )
    ib_hook = hooks / "hook-ib_insync.py"
    if not ib_hook.exists():
        ib_hook.write_text(
            "from PyInstaller.utils.hooks import collect_all\n"
            "datas, binaries, hiddenimports = collect_all('ib_insync')\n",
            encoding="utf-8",
        )


def clean_dist() -> None:
    if DIST.exists():
        log(f"  Temizleniyor: {DIST}", YELLOW)
        shutil.rmtree(DIST)
    build_dir = ROOT / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    log("  Temizlik tamamlandi.", GREEN)


def _post_copy_files() -> None:
    """Build sonrasi runtime dosyalari dist/ icine kopyala."""
    copies = [
        (ROOT / "models",            DIST / "models"),
        (ROOT / "scripts",           DIST / "scripts"),
        (ROOT / "docker-compose.yml", DIST / "docker-compose.yml"),
        (ROOT / ".env.example",      DIST / ".env.example"),
    ]
    env_file = ROOT / ".env"
    if env_file.exists():
        copies.append((env_file, DIST / ".env"))

    for src, dst in copies:
        if not src.exists():
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def build() -> None:
    log("\n" + "=" * 60)
    log("  BorsaBot Production EXE Build", BOLD)
    log("=" * 60 + "\n")

    steps = 5
    t0 = time.time()

    # 1. PyInstaller
    step(1, steps, "PyInstaller kontrol ediliyor...")
    if not check_pyinstaller():
        install_pyinstaller()

    # 2. Hooks
    step(2, steps, "Hook dizini hazirlaniyor...")
    ensure_hooks_dir()
    log("  Hooks hazir.", GREEN)

    # 3. Build
    step(3, steps, "PyInstaller build baslatiliyor...")
    result = run([
        sys.executable, "-m", "PyInstaller",
        str(SPEC),
        "--noconfirm",
        "--log-level", "WARN",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
    ])

    if result.returncode != 0:
        log("\n[HATA] Build BASARISIZ!", RED)
        sys.exit(1)

    # 4. Dogrula
    step(4, steps, "Build dogrulaniyor...")
    exe_path = DIST / "BorsaBot.exe"
    if not exe_path.exists():
        log(f"[HATA] EXE bulunamadi: {exe_path}", RED)
        sys.exit(1)

    size_mb = exe_path.stat().st_size / 1_048_576
    log(f"  BorsaBot.exe -- {size_mb:.1f} MB", GREEN)
    models_in_dist = DIST / "models"
    if not models_in_dist.exists():
        log("  [!!] models/ dist icinde yok -- kopyalaniyor...", YELLOW)
        _post_copy_files()
    pkl_count = len(list((DIST / "models").glob("*.pkl"))) if (DIST / "models").exists() else 0
    log(f"  models/ -- {pkl_count} .pkl dosyasi", GREEN)

    # 5. README
    step(5, steps, "Kullanim kilavuzu olusturuluyor...")
    _write_readme(DIST)
    log("  KULLANIM_KILAVUZU.txt olusturuldu.", GREEN)

    elapsed = time.time() - t0
    log("\n" + "=" * 60, CYAN)
    log(f"  [OK] Build tamamlandi! ({elapsed:.0f} saniye)", GREEN + BOLD)
    log(f"  EXE konumu: {DIST}", CYAN)
    log(f"  Dagitim icin tum {DIST.name}/ klasorunu kopyalayin.", CYAN)
    log("=" * 60 + "\n", CYAN)


def _write_readme(dist_dir: Path) -> None:
    readme = dist_dir / "KULLANIM_KILAVUZU.txt"
    readme.write_text(
        """\
+==================================================================+
|               BorsaBot -- Kullanim Kilavuzu                      |
+==================================================================+

KURULUM
-------
1. Bu klasorun (BorsaBot/) tamamini hedef bilgisayara kopyalayin.
   Ornek: C:\\BorsaBot\\

2. .env dosyasini duzenleyin:
   - MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER degerlerini girin
   - BINANCE_API_KEY / BINANCE_API_SECRET (Binance kullaniyorsaniz)
   - DEFAULT_SYMBOLS, NAV_USD, LOG_LEVEL

3. MetaTrader 5 terminalini acin ve:
   - Araclar > Secenekler > Uzman Danismanlar
   - "Algoritmik islemlere izin ver" kutusunu isaretleyin

4. Docker Desktop'i baslatin (TimescaleDB & Grafana icin).

5. BorsaBot.exe'yi calistirin.

GEREKSINIMLER
-------------
- Windows 10/11 (64-bit)
- MetaTrader 5 (broker hesabi)
- Docker Desktop (opsiyonel -- veri kaydi icin)
- Internet baglantisi

EXE YAPILANDIRMASI
------------------
BorsaBot/
|-- BorsaBot.exe          <- Ana baslatic (buradan baslayin)
|-- scripts/              <- Engine & Dashboard scriptleri
|   |-- main.py           <- Trading engine
|   +-- dashboard.py      <- Terminal dashboard
|-- models/               <- Egitimli ML modelleri (.pkl)
|-- .env                  <- Konfigurasyon (duzenleyin)
|-- docker-compose.yml    <- Docker servisleri
+-- KULLANIM_KILAVUZU.txt <- Bu dosya

BASLATMA
--------
- BorsaBot.exe cift-tiklayarak acin
- Broker secin: mt5 (veya binance/mock)
- Semboller: EURUSD XAUUSD (bosluk ile ayirin)
- Paper Mode: Gercek islem yapmadan test
- Preset: Ana (uretim) / Test (agresif) / Ozel (interaktif)
- "Engine Baslat" veya "Ikisini Birden Baslat" butonuna tiklayin

IZLEME
------
- Grafana Dashboard: http://localhost:3000
  Kullanici: admin / Sifre: borsabot
- logs/live_trades_datacollection.csv -- canli sinyal logu

SORUN GIDERME
-------------
- MT5 baglanamıyor -> terminal64.exe calisiyor mu? (Ctrl+Alt+Del)
- Docker kapali -> Docker Desktop'i baslatin
- Model bulunamadi -> models/ klasorunde .pkl dosyalari var mi?
- .env bulunamadi -> .env.example'dan kopyalayip doldurun

DESTEK
------
Loglar: BorsaBot.exe klasorudeki logs/ dizini
""",
        encoding="utf-8",
    )


def test_import() -> bool:
    """Temel importlari test et (EXE build oncesi)."""
    log("\nImport testi baslatiliyor...", CYAN)
    ok = True
    packages = [
        "tkinter", "pandas", "numpy", "sklearn",
        "xgboost", "lightgbm", "hmmlearn", "rich",
        "pydantic_settings", "structlog", "zmq",
        "redis", "prometheus_client", "transitions",
    ]
    for pkg in packages:
        try:
            __import__(pkg)
            log(f"  [OK] {pkg}", GREEN)
        except ImportError as e:
            log(f"  [!!] {pkg}: {e}", RED)
            ok = False
    if ok:
        log("\n[OK] Tum importlar basarili -- build hazir!", GREEN + BOLD)
    else:
        log("\n[HATA] Bazi paketler eksik -- pip install -e . calistirin", RED)
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BorsaBot EXE Build")
    parser.add_argument("--clean", action="store_true", help="dist/ ve build/ temizle")
    parser.add_argument("--test",  action="store_true", help="Sadece import testi yap")
    args = parser.parse_args()

    if args.test:
        sys.exit(0 if test_import() else 1)

    if args.clean:
        log("\nTemizlik modu aktif...", YELLOW)
        clean_dist()

    build()
