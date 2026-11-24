"""External human-like X11 mouse and keyboard controller.

Operates exclusively on the visible OS window via X11 XTEST events (`python-xlib`).
Supports smooth Bezier cursor motion, clicks, hold/release, drags, scroll wheel zoom,
and legitimate Territorial.io keyboard controls.
"""

from __future__ import annotations

import math
import random
import time

from Xlib import X, XK, display as xdisplay
from Xlib.ext import xtest

from core.config import ControlConfig
from core.types import BBox

_KEYSYM_MAP: dict[str, str] = {
    "Space": "space",
    "Enter": "Return",
    "Escape": "Escape",
    "KeyW": "w",
    "KeyA": "a",
    "KeyS": "s",
    "KeyD": "d",
    "KeyB": "b",
    "KeyI": "i",
    "KeyM": "m",
    "KeyP": "p",
    "KeyH": "h",
    "ArrowUp": "Up",
    "ArrowDown": "Down",
    "ArrowLeft": "Left",
    "ArrowRight": "Right",
    "NumpadAdd": "KP_Add",
    "NumpadSubtract": "KP_Subtract",
}


class ExternalInputController:
    """Drives real X11 pointer and keyboard events on the visible game display."""

    def __init__(self, display_str: str = ":99", config: ControlConfig | None = None) -> None:
        self.display_str = display_str
        self.cfg = config or ControlConfig()
        self._dpy = xdisplay.Display(display_str)
        self._root = self._dpy.screen().root
        self._cur_x = 100
        self._cur_y = 100
        self._rng = random.Random(1337)

    def _screen_clamp(self, x: int, y: int) -> tuple[int, int]:
        geom = self._root.get_geometry()
        return max(1, min(int(geom.width) - 2, x)), max(1, min(int(geom.height) - 2, y))

    def norm_to_screen(self, x_norm: float, y_norm: float, bbox: BBox) -> tuple[int, int]:
        px = bbox.x + int(round(max(0.0, min(1.0, x_norm)) * max(1, bbox.w - 1)))
        py = bbox.y + int(round(max(0.0, min(1.0, y_norm)) * max(1, bbox.h - 1)))
        return self._screen_clamp(px, py)

    def move_to(self, x: int, y: int, duration_sec: float | None = None) -> None:
        tx, ty = self._screen_clamp(int(x), int(y))
        if not self.cfg.human_like_motion:
            xtest.fake_input(self._dpy, X.MotionNotify, x=tx, y=ty)
            self._dpy.sync()
            self._cur_x, self._cur_y = tx, ty
            return

        sx, sy = self._cur_x, self._cur_y
        dist = math.hypot(tx - sx, ty - sy)
        steps = max(2, min(self.cfg.move_steps, int(dist / 25) + 3))
        dur = self.cfg.move_duration_sec if duration_sec is None else duration_sec

        # Slight natural arc control point
        mx = 0.5 * (sx + tx) + self._rng.uniform(-0.08, 0.08) * min(dist, 220.0)
        my = 0.5 * (sy + ty) + self._rng.uniform(-0.08, 0.08) * min(dist, 220.0)

        for i in range(1, steps + 1):
            t = i / steps
            # Smoothstep ease-in-out
            u = t * t * (3.0 - 2.0 * t)
            inv = 1.0 - u
            px = int(round(inv * inv * sx + 2 * inv * u * mx + u * u * tx))
            py = int(round(inv * inv * sy + 2 * inv * u * my + u * u * ty))
            px, py = self._screen_clamp(px, py)
            xtest.fake_input(self._dpy, X.MotionNotify, x=px, y=py)
            self._dpy.flush()
            if dur > 0:
                time.sleep(dur / steps)
        self._dpy.sync()
        self._cur_x, self._cur_y = tx, ty

    def mouse_down(self, button: int = 1) -> None:
        xtest.fake_input(self._dpy, X.ButtonPress, detail=button)
        self._dpy.sync()

    def mouse_up(self, button: int = 1) -> None:
        xtest.fake_input(self._dpy, X.ButtonRelease, detail=button)
        self._dpy.sync()

    def click(self, x: int, y: int, button: int = 1, hold_sec: float = 0.025) -> None:
        self.move_to(x, y)
        self.mouse_down(button)
        time.sleep(max(0.01, hold_sec))
        self.mouse_up(button)

    def drag(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        button: int = 1,
        duration_sec: float = 0.14,
    ) -> None:
        self.move_to(x0, y0, duration_sec=0.03)
        self.mouse_down(button)
        time.sleep(0.02)
        self.move_to(x1, y1, duration_sec=duration_sec)
        time.sleep(0.02)
        self.mouse_up(button)

    def scroll(self, x: int, y: int, notches: int) -> None:
        """Scroll wheel at (x, y). Positive notches = zoom in (button 4), negative = zoom out (button 5)."""
        self.move_to(x, y, duration_sec=0.025)
        button = 4 if notches > 0 else 5
        for _ in range(abs(notches)):
            xtest.fake_input(self._dpy, X.ButtonPress, detail=button)
            xtest.fake_input(self._dpy, X.ButtonRelease, detail=button)
            self._dpy.sync()
            time.sleep(0.02)

    def press_key(self, key_name: str, hold_sec: float = 0.025) -> None:
        sym_name = _KEYSYM_MAP.get(key_name, key_name)
        keysym = XK.string_to_keysym(sym_name)
        if keysym == 0:
            raise ValueError(f"Unknown X11 keysym for key '{key_name}' -> '{sym_name}'")
        keycode = self._dpy.keysym_to_keycode(keysym)
        if keycode == 0:
            raise RuntimeError(f"Keycode unavailable for keysym '{sym_name}'")
        xtest.fake_input(self._dpy, X.KeyPress, detail=keycode)
        self._dpy.sync()
        time.sleep(max(0.01, hold_sec))
        xtest.fake_input(self._dpy, X.KeyRelease, detail=keycode)
        self._dpy.sync()

    def close(self) -> None:
        self._dpy.close()
