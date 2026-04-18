# -*- mode: python ; coding: utf-8 -*-
"""
BorsaBot PyInstaller Spec
EXE çıktısı: dist/BorsaBot/BorsaBot.exe
"""

import sys
from pathlib import Path

ROOT = Path(SPECPATH)  # pyproject.toml ile aynı klasör

# ─────────────────────────────────────────────────────────────────────────────
# Gizli importları elle bil
# ─────────────────────────────────────────────────────────────────────────────
hidden_imports = [
    # Core async
    "asyncio",
    "asyncio.windows_events",
    "asyncio.windows_utils",

    # Standard libs sık kullanılan
    "tkinter",
    "tkinter.ttk",
    "tkinter.messagebox",
    "tkinter.font",
    "threading",
    "subprocess",
    "pathlib",
    "signal",
    "logging",
    "logging.handlers",

    # Borsabot package
    "borsabot",
    "borsabot.config",
    "borsabot.core",
    "borsabot.core.bus",
    "borsabot.core.events",
    "borsabot.core.logging",
    "borsabot.core.tick_handler",
    "borsabot.core.trader",
    "borsabot.brokers",
    "borsabot.brokers.base",
    "borsabot.brokers.binance_adapter",
    "borsabot.brokers.ib_adapter",
    "borsabot.brokers.mt5_adapter",
    "borsabot.features",
    "borsabot.features.builder",
    "borsabot.market_data",
    "borsabot.market_data.order_book",
    "borsabot.market_data.feed_monitor",
    "borsabot.market_data.tick_normalizer",
    "borsabot.models",
    "borsabot.models.regime",
    "borsabot.models.colab_adapter",
    "borsabot.models.meta_model",
    "borsabot.risk",
    "borsabot.risk.engine",
    "borsabot.execution",
    "borsabot.execution.engine",
    "borsabot.execution.order_fsm",
    "borsabot.storage",
    "borsabot.storage.redis_cache",
    "borsabot.storage.timescale",
    "borsabot.storage.parquet_lake",
    "borsabot.monitoring",
    "borsabot.monitoring.metrics",
    "borsabot.monitoring.health",

    # ML / Data
    "pandas",
    "pandas._libs.tslibs.np_datetime",
    "pandas._libs.tslibs.nattype",
    "pandas._libs.tslibs.timestamps",
    "pandas._libs.hashtable",
    "pandas._libs.lib",
    "pandas._libs.missing",
    "pandas._libs.ops",
    "numpy",
    "numpy.core._methods",
    "numpy.lib.format",
    "sklearn",
    "sklearn.utils._cython_blas",
    "sklearn.neighbors._typedefs",
    "sklearn.neighbors._quad_tree",
    "sklearn.tree._utils",
    "xgboost",
    "lightgbm",
    "hmmlearn",
    "hmmlearn.hmm",
    "imbalanced_learn",
    "statsmodels",
    "statsmodels.tsa",

    # Messaging
    "zmq",
    "zmq.backend.cython",
    "zmq.backend.cffi",
    "redis",
    "redis.asyncio",

    # Pydantic
    "pydantic",
    "pydantic_settings",
    "pydantic.v1",

    # Other
    "structlog",
    "prometheus_client",
    "dotenv",
    "rich",
    "rich.console",
    "rich.table",
    "rich.live",
    "rich.panel",
    "rich.layout",
    "rich.text",
    "rich.box",
    "rich.columns",
    "transitions",
    "sortedcontainers",
    "pyarrow",
    "msgpack",
    "websockets",
    "websockets.legacy",
    "MetaTrader5",
    "binance",
    "binance.client",
    "asyncpg",
    "pyzmq",
]

# ─────────────────────────────────────────────────────────────────────────────
# Collect all
# ─────────────────────────────────────────────────────────────────────────────
collect_all_packages = [
    "borsabot",
    "sklearn",
    "xgboost",
    "lightgbm",
    "hmmlearn",
    "statsmodels",
    "pandas",
    "numpy",
    "pyarrow",
    "zmq",
    "rich",
    "pydantic",
    "pydantic_settings",
    "structlog",
    "prometheus_client",
    "transitions",
    "sortedcontainers",
    "msgpack",
    "websockets",
    "binance",
    "redis",
    "asyncpg",
]

all_datas = []
all_binaries = []
all_hiddenimports = list(set(hidden_imports))

for pkg in collect_all_packages:
    try:
        from PyInstaller.utils.hooks import collect_all
        d, b, h = collect_all(pkg)
        all_datas      += d
        all_binaries   += b
        all_hiddenimports += h
    except Exception as e:
        print(f"[WARN] collect_all failed for {pkg}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Extra datas (dosya kopyalama)
# ─────────────────────────────────────────────────────────────────────────────
extra_datas = [
    # scripts
    (str(ROOT / "scripts"), "scripts"),
    # modeller
    (str(ROOT / "models"), "models"),
    # .env.example — kullanıcı ilk çalıştırmada bunu düzenler
    (str(ROOT / ".env.example"), "."),
    # docker-compose.yml — docker servisleri için
    (str(ROOT / "docker-compose.yml"), "."),
]

# .env varsa onu da dahil et (opsiyonel — hassas veriler içinn warning ver)
env_file = ROOT / ".env"
if env_file.exists():
    extra_datas.append((str(env_file), "."))

all_datas += extra_datas

# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────
a = Analysis(
    [str(ROOT / "launcher" / "borsabot_launcher.py")],
    pathex=[str(ROOT)],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=list(set(all_hiddenimports)),
    hookspath=[str(ROOT / "launcher" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "PIL",
        "cv2",
        "tensorflow",
        "torch",
        "PySide6",
        "PyQt5",
        "PyQt6",
        "wx",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # onedir modu
    name="BorsaBot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # GUI: siyah konsol penceresi açılmasın
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "launcher" / "icon.ico") if (ROOT / "launcher" / "icon.ico").exists() else None,
    version=str(ROOT / "launcher" / "version_info.txt") if (ROOT / "launcher" / "version_info.txt").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="BorsaBot",
)
