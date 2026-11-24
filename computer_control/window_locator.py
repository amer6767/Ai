"""Dynamic X11 game window locator. Never hardcodes fixed screen coordinates."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from Xlib import display as xdisplay

from core.types import BBox


@dataclass
class WindowInfo:
    window_id: int
    title: str
    bbox: BBox


class WindowLocator:
    """Finds the visible Territorial.io browser window geometry dynamically via X11."""

    def __init__(self, display_str: str | None = None, title_hint: str = "Territorial") -> None:
        self.display_str = display_str or os.environ.get("DISPLAY", ":99")
        self.title_hint = title_hint.lower()

    def get_screen_bbox(self) -> BBox:
        dpy = xdisplay.Display(self.display_str)
        try:
            root = dpy.screen().root
            geom = root.get_geometry()
            return BBox(x=0, y=0, w=int(geom.width), h=int(geom.height))
        finally:
            dpy.close()

    def locate_game_window(self) -> WindowInfo:
        """Return the largest visible matching browser window, falling back to full display."""
        screen_bbox = self.get_screen_bbox()
        best: WindowInfo | None = None
        best_area = 0

        dpy = xdisplay.Display(self.display_str)
        try:
            root = dpy.screen().root
            stack = [root]
            while stack:
                win = stack.pop()
                try:
                    children = win.query_tree().children
                except Exception:
                    children = []
                for child in children:
                    stack.append(child)
                    try:
                        attrs = child.get_attributes()
                        if attrs.map_state != 2:  # IsViewable
                            continue
                        wm_name = child.get_wm_name() or ""
                        wm_class = child.get_wm_class() or ("", "")
                        title = str(wm_name)
                        cls_str = " ".join(str(c) for c in wm_class).lower()
                        if self.title_hint in title.lower() or "chrome" in cls_str or "chromium" in cls_str:
                            geom = child.get_geometry()
                            coords = root.translate_coords(child, 0, 0)
                            w = int(geom.width)
                            h = int(geom.height)
                            if w >= 320 and h >= 240:
                                area = w * h
                                if area > best_area:
                                    x0 = max(0, int(coords.x))
                                    y0 = max(0, int(coords.y))
                                    w_clip = min(w, screen_bbox.w - x0)
                                    h_clip = min(h, screen_bbox.h - y0)
                                    best_area = area
                                    best = WindowInfo(
                                        window_id=int(child.id),
                                        title=title or cls_str,
                                        bbox=BBox(x=x0, y=y0, w=w_clip, h=h_clip),
                                    )
                    except Exception:
                        continue
        finally:
            dpy.close()

        if best is not None:
            return best
        return WindowInfo(window_id=0, title="root_display", bbox=screen_bbox)

    def focus_window(self, window_id: int) -> None:
        if window_id <= 0:
            return
        env = os.environ.copy()
        env["DISPLAY"] = self.display_str
        subprocess.run(
            ["xdotool", "windowactivate", "--sync", str(window_id)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
