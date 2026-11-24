"""Isolated Xvfb display, static HTTP server, and plain Chrome launcher.

Strictly enforces the Engine-Isolation Firewall:
- Serves unmodified `text.html` over loopback HTTP.
- Launches plain Google Chrome (`--app=...`) with NO remote debugging port,
  NO browser automation protocol, and NO injected JavaScript.
- The AI interacts ONLY through X11 screenshots and OS pointer/keyboard events.
"""

from __future__ import annotations

import functools
import http.server
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def _find_free_port(preferred: int = 18080) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return int(sock.getsockname()[1])
            except OSError:
                continue
    raise RuntimeError("No free TCP port available for static game HTTP server")


class GameDisplaySession:
    """Manages an isolated Xvfb display + local HTTP server + plain Chrome window."""

    def __init__(
        self,
        game_html_path: str | Path = "text.html",
        display_num: int = 99,
        width: int = 1280,
        height: int = 720,
        color_depth: int = 24,
        http_port: int = 18080,
    ) -> None:
        self.game_html_path = Path(game_html_path).resolve()
        if not self.game_html_path.exists():
            raise FileNotFoundError(f"Game source file not found: {self.game_html_path}")
        self.display_num = display_num
        self.display_str = f":{display_num}"
        self.width = width
        self.height = height
        self.color_depth = color_depth
        self.http_port = http_port

        self._http_server: http.server.ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._xvfb_proc: subprocess.Popen[bytes] | None = None
        self._chrome_proc: subprocess.Popen[bytes] | None = None
        self._profile_dir: str | None = None
        self._running = False

    @property
    def game_url(self) -> str:
        return f"http://127.0.0.1:{self.http_port}/{self.game_html_path.name}"

    def start(self, startup_wait_sec: float = 3.0) -> None:
        if self._running:
            return
        self._start_http_server()
        self._start_xvfb()
        self._start_chrome()
        self._running = True
        time.sleep(startup_wait_sec)

    def _start_http_server(self) -> None:
        self.http_port = _find_free_port(self.http_port)
        handler = functools.partial(_QuietHandler, directory=str(self.game_html_path.parent))
        self._http_server = http.server.ThreadingHTTPServer(("127.0.0.1", self.http_port), handler)
        self._http_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()

    def _start_xvfb(self) -> None:
        chosen = self.display_num
        for candidate in range(self.display_num, self.display_num + 50):
            lock_file = Path(f"/tmp/.X{candidate}-lock")
            if not lock_file.exists():
                chosen = candidate
                break
            try:
                pid = int(lock_file.read_text().strip())
                os.kill(pid, 0)
                continue
            except (ProcessLookupError, ValueError):
                lock_file.unlink(missing_ok=True)
                chosen = candidate
                break
        else:
            raise RuntimeError(f"No free Xvfb display found in :{self.display_num}..:{self.display_num+49}")

        self.display_num = chosen
        self.display_str = f":{chosen}"

        cmd = [
            "Xvfb",
            self.display_str,
            "-screen",
            "0",
            f"{self.width}x{self.height}x{self.color_depth}",
            "-ac",
            "+extension",
            "XTEST",
            "-nolisten",
            "tcp",
        ]
        self._xvfb_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait until X socket exists
        sock_path = Path(f"/tmp/.X11-unix/X{self.display_num}")
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if sock_path.exists():
                return
            if self._xvfb_proc.poll() is not None:
                raise RuntimeError(f"Xvfb failed to start on {self.display_str}")
            time.sleep(0.05)
        raise TimeoutError(f"Timed out waiting for Xvfb socket {sock_path}")

    def _start_chrome(self) -> None:
        chrome_bin = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
        if not chrome_bin:
            raise FileNotFoundError("Google Chrome / Chromium binary not found on PATH")

        self._profile_dir = tempfile.mkdtemp(prefix=f"tio_chrome_d{self.display_num}_")
        env = os.environ.copy()
        env["DISPLAY"] = self.display_str

        # Plain app window: strictly NO debugging port, NO automation switches
        cmd = [
            chrome_bin,
            f"--app={self.game_url}",
            f"--user-data-dir={self._profile_dir}",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--use-gl=angle",
            "--use-angle=swiftshader-webgl",
            "--enable-unsafe-swiftshader",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-notifications",
            "--disable-translate",
            "--autoplay-policy=no-user-gesture-required",
            "--window-position=0,0",
            f"--window-size={self.width},{self.height}",
            "--kiosk",
        ]
        self._chrome_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        self._running = False
        for proc in (self._chrome_proc, self._xvfb_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._chrome_proc = None
        self._xvfb_proc = None

        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
        if self._profile_dir and Path(self._profile_dir).exists():
            shutil.rmtree(self._profile_dir, ignore_errors=True)
            self._profile_dir = None

    def __enter__(self) -> "GameDisplaySession":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
