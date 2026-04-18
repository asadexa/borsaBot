"""
BorsaBot Launcher — Production EXE Entry Point
Tüm bağımlılıkları kontrol ederek Engine ve Dashboard'u başlatır.
"""
from __future__ import annotations

import os
import sys
import subprocess
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, font
from pathlib import Path

# ── PyInstaller: çalışma dizinini exe'nin bulunduğu klasöre ayarla ────────────
if getattr(sys, "frozen", False):
    # Exe ile çalışıyoruz
    BASE_DIR = Path(sys.executable).parent
else:
    # Doğrudan Python ile çalışıyoruz
    BASE_DIR = Path(__file__).parent.parent

# .env dosyasını yükle (en başta)
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            v = v.split("#")[0].strip()
            os.environ.setdefault(k.strip(), v)

MODEL_DIR  = BASE_DIR / "models"
SCRIPTS_DIR = BASE_DIR / "scripts"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_detached(cmd: list[str], title: str, cwd: Path) -> subprocess.Popen:
    """Ayrı bir konsol penceresinde komut çalıştır (Windows)."""
    startupinfo = subprocess.STARTUPINFO()
    creationflags = subprocess.CREATE_NEW_CONSOLE
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        creationflags=creationflags,
        startupinfo=startupinfo,
    )


def _check_docker() -> bool:
    try:
        result = subprocess.run(
            ["docker", "ps"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _check_mt5_running() -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq terminal64.exe"],
            capture_output=True, text=True, timeout=5
        )
        return "terminal64.exe" in result.stdout
    except Exception:
        return False


def _docker_services_up(cwd: Path) -> bool:
    """docker compose up -d servislerini başlat."""
    try:
        result = subprocess.run(
            ["docker", "compose", "up", "-d",
             "timescaledb", "redis", "prometheus", "grafana"],
            cwd=str(cwd), capture_output=True, timeout=60
        )
        return result.returncode == 0
    except Exception:
        return False


def _get_python_exe() -> str:
    """EXE içinde gömülü Python kullanılır; doğrudan scriptler için sistem Python."""
    if getattr(sys, "frozen", False):
        return sys.executable
    # Geliştirme modunda venv veya sistem Python
    venv_py = BASE_DIR / "venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


# ─────────────────────────────────────────────────────────────────────────────
# GUI Launcher
# ─────────────────────────────────────────────────────────────────────────────

class BorsaBotLauncher(tk.Tk):
    DARK_BG   = "#0d1117"
    PANEL_BG  = "#161b22"
    ACCENT    = "#58a6ff"
    SUCCESS   = "#3fb950"
    ERROR     = "#f85149"
    WARNING   = "#d29922"
    TEXT      = "#e6edf3"
    MUTED     = "#8b949e"
    BORDER    = "#30363d"

    def __init__(self):
        super().__init__()
        self.title("BorsaBot — Production Launcher")
        self.geometry("760x660")
        self.resizable(False, False)
        self.configure(bg=self.DARK_BG)
        self._set_icon()

        self._procs: list[subprocess.Popen] = []
        self._engine_running  = False
        self._dash_running    = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # İlk kontrol gecikmeli (UI tamamen yüklendikten sonra)
        self.after(300, self._initial_check)

    # ── Icon ─────────────────────────────────────────────────────────────────
    def _set_icon(self):
        icon_path = BASE_DIR / "launcher" / "icon.ico"
        if icon_path.exists():
            try:
                self.iconbitmap(str(icon_path))
            except Exception:
                pass

    # ── UI Builder ────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Fonts
        title_font  = font.Font(family="Segoe UI", size=18, weight="bold")
        header_font = font.Font(family="Segoe UI", size=10, weight="bold")
        body_font   = font.Font(family="Segoe UI", size=9)
        mono_font   = font.Font(family="Consolas",  size=8)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=self.PANEL_BG, pady=16)
        hdr.pack(fill="x")
        tk.Label(
            hdr, text="⚡ BorsaBot", fg=self.ACCENT, bg=self.PANEL_BG,
            font=title_font
        ).pack(side="left", padx=20)
        tk.Label(
            hdr, text="AI-Native Algorithmic Trading Platform",
            fg=self.MUTED, bg=self.PANEL_BG, font=body_font
        ).pack(side="left", padx=6)

        # Version tag
        tk.Label(
            hdr, text="v1.0  Production", fg=self.WARNING, bg=self.PANEL_BG,
            font=body_font
        ).pack(side="right", padx=20)

        # Separator
        tk.Frame(self, bg=self.BORDER, height=1).pack(fill="x")

        # ── Status Cards ──────────────────────────────────────────────────────
        cards_frame = tk.Frame(self, bg=self.DARK_BG, pady=12)
        cards_frame.pack(fill="x", padx=16)

        self._status_vars: dict[str, tk.StringVar] = {}
        self._status_labels: dict[str, tk.Label] = {}

        statuses = [
            ("docker",  "🐳 Docker"),
            ("mt5",     "📊 MetaTrader 5"),
            ("models",  "🧠 Modeller"),
            ("env",     "⚙️  .env"),
        ]

        for col, (key, label) in enumerate(statuses):
            card = tk.Frame(cards_frame, bg=self.PANEL_BG, bd=0,
                            highlightbackground=self.BORDER,
                            highlightthickness=1, padx=12, pady=8)
            card.grid(row=0, column=col, padx=6, sticky="nsew")
            cards_frame.columnconfigure(col, weight=1)

            tk.Label(card, text=label, fg=self.MUTED, bg=self.PANEL_BG,
                     font=header_font).pack(anchor="w")
            var = tk.StringVar(value="Kontrol ediliyor...")
            lbl = tk.Label(card, textvariable=var, fg=self.WARNING,
                           bg=self.PANEL_BG, font=body_font)
            lbl.pack(anchor="w", pady=(2, 0))
            self._status_vars[key]  = var
            self._status_labels[key] = lbl

        # ── Config Section ────────────────────────────────────────────────────
        cfg_frame = tk.LabelFrame(self, text=" Başlatma Konfigürasyonu ",
                                  fg=self.ACCENT, bg=self.DARK_BG,
                                  font=header_font, bd=1,
                                  highlightbackground=self.BORDER)
        cfg_frame.pack(fill="x", padx=16, pady=8)

        # Broker seçimi
        row1 = tk.Frame(cfg_frame, bg=self.DARK_BG)
        row1.pack(fill="x", padx=12, pady=6)
        tk.Label(row1, text="Broker:", fg=self.TEXT, bg=self.DARK_BG,
                 font=body_font, width=14, anchor="w").pack(side="left")
        self._broker_var = tk.StringVar(value="mt5")
        broker_combo = ttk.Combobox(
            row1, textvariable=self._broker_var,
            values=["mt5", "binance", "ib", "mock"],
            state="readonly", width=16
        )
        broker_combo.pack(side="left", padx=4)

        # Paper mode
        self._paper_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            row1, text="Paper Mode (Gerçek işlem yok)",
            variable=self._paper_var, fg=self.TEXT, bg=self.DARK_BG,
            selectcolor=self.PANEL_BG, activebackground=self.DARK_BG,
            activeforeground=self.TEXT, font=body_font
        ).pack(side="left", padx=16)

        # Semboller
        row2 = tk.Frame(cfg_frame, bg=self.DARK_BG)
        row2.pack(fill="x", padx=12, pady=4)
        tk.Label(row2, text="Semboller:", fg=self.TEXT, bg=self.DARK_BG,
                 font=body_font, width=14, anchor="w").pack(side="left")
        self._symbols_var = tk.StringVar(
            value=os.environ.get("DEFAULT_SYMBOLS", "EURUSD XAUUSD")
        )
        tk.Entry(row2, textvariable=self._symbols_var,
                 bg=self.PANEL_BG, fg=self.TEXT,
                 insertbackground=self.TEXT, width=30,
                 font=mono_font).pack(side="left", padx=4)

        # NAV
        row3 = tk.Frame(cfg_frame, bg=self.DARK_BG)
        row3.pack(fill="x", padx=12, pady=4)
        tk.Label(row3, text="NAV (USD):", fg=self.TEXT, bg=self.DARK_BG,
                 font=body_font, width=14, anchor="w").pack(side="left")
        self._nav_var = tk.StringVar(
            value=os.environ.get("NAV_USD", "100000")
        )
        tk.Entry(row3, textvariable=self._nav_var,
                 bg=self.PANEL_BG, fg=self.TEXT,
                 insertbackground=self.TEXT, width=12,
                 font=mono_font).pack(side="left", padx=4)

        # Docker servis başlatma
        self._docker_start_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            row3, text="Docker servislerini otomatik başlat",
            variable=self._docker_start_var,
            fg=self.TEXT, bg=self.DARK_BG,
            selectcolor=self.PANEL_BG,
            activebackground=self.DARK_BG,
            activeforeground=self.TEXT, font=body_font
        ).pack(side="left", padx=16)

        # ── Ruleset ───────────────────────────────────────────────────────────
        rule_frame = tk.LabelFrame(self, text=" Trade Ruleset ",
                                   fg=self.ACCENT, bg=self.DARK_BG,
                                   font=header_font, bd=1)
        rule_frame.pack(fill="x", padx=16, pady=4)

        row_r = tk.Frame(rule_frame, bg=self.DARK_BG)
        row_r.pack(fill="x", padx=12, pady=6)
        tk.Label(row_r, text="Preset:", fg=self.TEXT, bg=self.DARK_BG,
                 font=body_font, width=14, anchor="w").pack(side="left")
        self._preset_var = tk.StringVar(value="ana")
        preset_combo = ttk.Combobox(
            row_r, textvariable=self._preset_var,
            values=["ana", "test", "ozel"],
            state="readonly", width=10
        )
        preset_combo.pack(side="left", padx=4)
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)

        tk.Label(row_r, text="Ana = Üretim filtreli  |  Test = Agresif (kısıtsız)  |  Özel = Manuel",
                 fg=self.MUTED, bg=self.DARK_BG, font=body_font
                 ).pack(side="left", padx=10)

        # ── Action Buttons ────────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=self.DARK_BG, pady=10)
        btn_frame.pack(fill="x", padx=16)

        btn_style = dict(
            font=font.Font(family="Segoe UI", size=10, weight="bold"),
            relief="flat", cursor="hand2", padx=16, pady=8, bd=0
        )

        self._engine_btn = tk.Button(
            btn_frame, text="▶  Engine Başlat",
            bg=self.SUCCESS, fg="white",
            command=self._start_engine, **btn_style
        )
        self._engine_btn.pack(side="left", padx=4)

        self._dash_btn = tk.Button(
            btn_frame, text="📊  Dashboard Aç",
            bg=self.ACCENT, fg="white",
            command=self._start_dashboard, **btn_style
        )
        self._dash_btn.pack(side="left", padx=4)

        self._both_btn = tk.Button(
            btn_frame, text="⚡  İkisini Birden Başlat",
            bg="#6e40c9", fg="white",
            command=self._start_both, **btn_style
        )
        self._both_btn.pack(side="left", padx=4)

        tk.Button(
            btn_frame, text="🛑  Durdur",
            bg=self.ERROR, fg="white",
            command=self._stop_all, **btn_style
        ).pack(side="right", padx=4)

        tk.Button(
            btn_frame, text="🔄  Yenile",
            bg=self.PANEL_BG, fg=self.TEXT,
            command=self._initial_check, **btn_style
        ).pack(side="right", padx=4)

        # ── Log Frame ─────────────────────────────────────────────────────────
        log_frame = tk.LabelFrame(
            self, text=" Launcher Log ", fg=self.ACCENT,
            bg=self.DARK_BG, font=header_font, bd=1
        )
        log_frame.pack(fill="both", expand=True, padx=16, pady=8)

        self._log_text = tk.Text(
            log_frame, bg="#0a0e14", fg=self.TEXT,
            font=mono_font, height=10, state="disabled",
            wrap="word", bd=0, relief="flat"
        )
        scrollbar = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Tag renkler
        self._log_text.tag_configure("ok",   foreground=self.SUCCESS)
        self._log_text.tag_configure("err",  foreground=self.ERROR)
        self._log_text.tag_configure("warn", foreground=self.WARNING)
        self._log_text.tag_configure("info", foreground=self.TEXT)
        self._log_text.tag_configure("muted", foreground=self.MUTED)

        # ── Status Bar ────────────────────────────────────────────────────────
        self._statusbar_var = tk.StringVar(value="Hazır")
        tk.Label(
            self, textvariable=self._statusbar_var,
            fg=self.MUTED, bg=self.PANEL_BG, anchor="w", pady=4
        ).pack(fill="x", side="bottom")

    # ── Logging ───────────────────────────────────────────────────────────────
    def _log(self, msg: str, tag: str = "info"):
        ts = time.strftime("%H:%M:%S")
        self._log_text.configure(state="normal")
        self._log_text.insert("end", f"[{ts}] {msg}\n", tag)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _set_status(self, key: str, text: str, color: str):
        self._status_vars[key].set(text)
        self._status_labels[key].configure(fg=color)

    # ── Checks ────────────────────────────────────────────────────────────────
    def _initial_check(self):
        threading.Thread(target=self._run_checks, daemon=True).start()

    def _run_checks(self):
        self._log("Sistem kontrolleri başlatılıyor...", "muted")

        # Docker
        if _check_docker():
            self.after(0, lambda: self._set_status("docker", "✅ Çalışıyor", self.SUCCESS))
            self._log("Docker: Çalışıyor", "ok")
        else:
            self.after(0, lambda: self._set_status("docker", "❌ Kapalı", self.ERROR))
            self._log("Docker: Kapalı — Docker Desktop'ı başlatın", "warn")

        # MT5
        if _check_mt5_running():
            self.after(0, lambda: self._set_status("mt5", "✅ Çalışıyor", self.SUCCESS))
            self._log("MetaTrader 5: Çalışıyor", "ok")
        else:
            self.after(0, lambda: self._set_status("mt5", "⚠️  Kapalı", self.WARNING))
            self._log("MetaTrader 5: Kapalı — Engine başlatmadan önce açın", "warn")

        # Modeller
        pkls = list(MODEL_DIR.glob("*.pkl")) if MODEL_DIR.exists() else []
        if pkls:
            self.after(0, lambda: self._set_status("models", f"✅ {len(pkls)} dosya", self.SUCCESS))
            self._log(f"Modeller: {len(pkls)} .pkl dosyası bulundu", "ok")
        else:
            self.after(0, lambda: self._set_status("models", "❌ Bulunamadı", self.ERROR))
            self._log(f"Modeller: models/ klasörü boş veya yok ({MODEL_DIR})", "err")

        # .env
        if _env_path.exists():
            self.after(0, lambda: self._set_status("env", "✅ Yüklendi", self.SUCCESS))
            self._log(f".env: Yüklendi ({_env_path})", "ok")
        else:
            self.after(0, lambda: self._set_status("env", "❌ Bulunamadı", self.ERROR))
            self._log(f".env bulunamadı — {_env_path} oluşturun", "err")

        self.after(0, lambda: self._statusbar_var.set("Kontrol tamamlandı"))

    # ── Preset ───────────────────────────────────────────────────────────────
    def _on_preset_change(self, event=None):
        preset = self._preset_var.get()
        desc = {
            "ana":   "Üretim preset — yüksek güven filtreli, düşük risk",
            "test":  "Agresif preset — tüm filtreler kapalı, hızlı giriş",
            "ozel":  "Özel — engine başladığında interaktif menüden ayar",
        }
        self._log(f"Preset seçildi: {preset} — {desc.get(preset,'')}", "info")

    # ── Start Engine ─────────────────────────────────────────────────────────
    def _start_engine(self):
        if self._engine_running:
            self._log("Engine zaten çalışıyor!", "warn")
            return

        broker  = self._broker_var.get()
        symbols = self._symbols_var.get().strip().split()
        nav     = self._nav_var.get().strip()
        paper   = self._paper_var.get()
        preset  = self._preset_var.get()

        if not symbols:
            messagebox.showerror("Hata", "En az bir sembol girin!")
            return

        self._statusbar_var.set("Engine başlatılıyor...")
        self._log(f"Engine başlatılıyor — broker={broker} symbols={symbols} paper={paper}", "info")

        # Docker servislerini başlat (arka planda)
        if self._docker_start_var.get():
            self._log("Docker servisleri kontrol ediliyor/başlatılıyor...", "muted")
            threading.Thread(
                target=lambda: _docker_services_up(BASE_DIR), daemon=True
            ).start()

        threading.Thread(
            target=self._launch_engine,
            args=(broker, symbols, nav, paper, preset),
            daemon=True
        ).start()

    def _launch_engine(self, broker, symbols, nav, paper, preset):
        py_exe = _get_python_exe()
        script = SCRIPTS_DIR / "main.py"

        cmd = [py_exe, str(script),
               "--broker", broker,
               "--symbols", *symbols,
               "--model-dir", str(MODEL_DIR),
               "--nav", nav]
        if paper:
            cmd.append("--paper")

        # Preset otomatik seçimi için env var kullan
        env = os.environ.copy()
        env["BORSABOT_PRESET"] = preset

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(BASE_DIR), env=env,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            self._procs.append(proc)
            self._engine_running = True
            self.after(0, lambda: self._log(f"Engine başlatıldı (PID={proc.pid})", "ok"))
            self.after(0, lambda: self._statusbar_var.set(f"Engine çalışıyor (PID={proc.pid})"))
            self.after(0, lambda: self._engine_btn.configure(bg="#2d6a2d"))
        except Exception as e:
            self.after(0, lambda: self._log(f"Engine başlatma HATASI: {e}", "err"))
            self._engine_running = False

    # ── Start Dashboard ───────────────────────────────────────────────────────
    def _start_dashboard(self):
        if self._dash_running:
            self._log("Dashboard zaten çalışıyor!", "warn")
            return

        broker  = self._broker_var.get()
        symbols = self._symbols_var.get().strip().split()

        self._log(f"Dashboard açılıyor — broker={broker} symbols={symbols}", "info")
        threading.Thread(
            target=self._launch_dashboard,
            args=(broker, symbols),
            daemon=True
        ).start()

    def _launch_dashboard(self, broker, symbols):
        py_exe  = _get_python_exe()
        script  = SCRIPTS_DIR / "dashboard.py"

        cmd = [py_exe, str(script),
               "--broker", broker,
               "--symbols", *symbols,
               "--model-dir", str(MODEL_DIR)]

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(BASE_DIR),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            self._procs.append(proc)
            self._dash_running = True
            self.after(0, lambda: self._log(f"Dashboard açıldı (PID={proc.pid})", "ok"))
            self.after(0, lambda: self._dash_btn.configure(bg="#1a4a7a"))
        except Exception as e:
            self.after(0, lambda: self._log(f"Dashboard HATASI: {e}", "err"))
            self._dash_running = False

    # ── Start Both ────────────────────────────────────────────────────────────
    def _start_both(self):
        self._start_engine()
        # Dashboard'u engine 4 saniye sonra başlasın
        self.after(4000, self._start_dashboard)

    # ── Stop All ──────────────────────────────────────────────────────────────
    def _stop_all(self):
        if not self._procs:
            self._log("Çalışan proses yok.", "warn")
            return
        for proc in self._procs:
            try:
                proc.terminate()
                self._log(f"PID {proc.pid} durduruldu", "warn")
            except Exception:
                pass
        self._procs.clear()
        self._engine_running = False
        self._dash_running   = False
        self._engine_btn.configure(bg=self.SUCCESS)
        self._dash_btn.configure(bg=self.ACCENT)
        self._statusbar_var.set("Tüm prosesler durduruldu")

    # ── Close ─────────────────────────────────────────────────────────────────
    def _on_close(self):
        if self._procs:
            if messagebox.askyesno(
                "Çıkış",
                "Çalışan Engine/Dashboard prosesleri var.\n"
                "Bunları da durdurmak istiyor musunuz?"
            ):
                self._stop_all()
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = BorsaBotLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
