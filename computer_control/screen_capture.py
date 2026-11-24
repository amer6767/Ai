"""High-speed external X11 screenshot capture with rolling frame ring buffer."""

from __future__ import annotations

import collections
import time
from typing import Deque

import mss
import numpy as np

from computer_control.window_locator import WindowLocator
from core.types import BBox, RawFrame


class ScreenCapturer:
    """Captures raw RGB screenshots directly from the external X11 display."""

    def __init__(
        self,
        display_str: str = ":99",
        ring_size: int = 8,
        recheck_window_every: int = 30,
    ) -> None:
        self.display_str = display_str
        self.locator = WindowLocator(display_str=display_str)
        self._sct = mss.mss(display=display_str)
        self._frame_id = 0
        self._recheck_every = max(1, recheck_window_every)
        self._window_bbox: BBox | None = None
        self._ring: Deque[RawFrame] = collections.deque(maxlen=ring_size)

    def refresh_window_bbox(self) -> BBox:
        info = self.locator.locate_game_window()
        self._window_bbox = info.bbox
        return self._window_bbox

    def capture(self) -> RawFrame:
        if self._window_bbox is None or (self._frame_id % self._recheck_every == 0):
            self.refresh_window_bbox()
        bbox = self._window_bbox
        assert bbox is not None

        monitor = {"left": bbox.x, "top": bbox.y, "width": bbox.w, "height": bbox.h}
        raw_bgra = np.asarray(self._sct.grab(monitor), dtype=np.uint8)
        # Convert BGRA -> RGB without extra copy overhead
        rgb = np.ascontiguousarray(raw_bgra[:, :, 2::-1])

        self._frame_id += 1
        frame = RawFrame(
            rgb=rgb,
            timestamp=time.monotonic(),
            frame_id=self._frame_id,
            window_bbox=bbox,
        )
        self._ring.append(frame)
        return frame

    @property
    def recent_frames(self) -> list[RawFrame]:
        return list(self._ring)

    def close(self) -> None:
        self._sct.close()
