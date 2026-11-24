"""Temporal frame differencing to detect active battles, border surges, and rapid changes."""

from __future__ import annotations

import cv2
import numpy as np


class TemporalChangeTracker:
    """Computes temporally smoothed combat flux heatmaps directly from frame pixel/label changes."""

    def __init__(self, grid_size: int = 128, decay: float = 0.65) -> None:
        self.grid_size = grid_size
        self.decay = decay
        self._prev_rgb_ds: np.ndarray | None = None
        self._prev_label_map: np.ndarray | None = None
        self._flux_map = np.zeros((grid_size, grid_size), dtype=np.float32)

    def reset(self) -> None:
        self._prev_rgb_ds = None
        self._prev_label_map = None
        self._flux_map.fill(0.0)

    def update(self, rgb: np.ndarray, label_map: np.ndarray, ui_mask: np.ndarray | None = None) -> np.ndarray:
        ds = self.grid_size
        rgb_ds = cv2.resize(rgb, (ds, ds), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._prev_rgb_ds is None or self._prev_label_map is None or self._prev_label_map.shape != label_map.shape:
            self._prev_rgb_ds = rgb_ds
            self._prev_label_map = label_map.copy()
            self._flux_map.fill(0.0)
            return self._flux_map.copy()

        # Global camera movement guard: if >45% of screen pixels changed simultaneously, camera zoomed/panned!
        pix_diff = np.abs(rgb_ds - self._prev_rgb_ds).mean(axis=2)
        changed_fraction = float((pix_diff > 18.0).mean())
        if changed_fraction > 0.45:
            self._flux_map *= 0.3
        else:
            label_changed = (label_map != self._prev_label_map) & (label_map >= 0) & (self._prev_label_map >= 0)
            instant_flux = np.clip(pix_diff / 80.0, 0.0, 1.0) * 0.5 + label_changed.astype(np.float32) * 0.8
            if ui_mask is not None:
                ui_ds = cv2.resize(ui_mask.astype(np.uint8), (ds, ds), interpolation=cv2.INTER_NEAREST) > 0
                instant_flux[ui_ds] = 0.0
            self._flux_map = np.clip(self.decay * self._flux_map + (1.0 - self.decay) * instant_flux, 0.0, 1.0)

        self._prev_rgb_ds = rgb_ds
        self._prev_label_map = label_map.copy()
        return self._flux_map.copy()
