"""Visual camera odometry estimating viewport pan (dx, dy) and zoom factor from consecutive frames."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraPose:
    cx: float = 0.5        # viewport center X in canonical world [0, 1]
    cy: float = 0.5        # viewport center Y in canonical world [0, 1]
    zoom: float = 1.0      # >= 1.0 means zoomed in
    last_dx: float = 0.0
    last_dy: float = 0.0
    confidence: float = 1.0


class VisualCameraOdometry:
    """Tracks camera pan/zoom across frames using phase correlation on terrain/coastline edges."""

    def __init__(self, work_size: int = 128) -> None:
        self.work_size = work_size
        self.pose = CameraPose()
        self._prev_edge: np.ndarray | None = None
        self._hann = cv2.createHanningWindow((work_size, work_size), cv2.CV_32F)

    def reset(self) -> None:
        self.pose = CameraPose()
        self._prev_edge = None

    def register_action_hint(self, dx_norm: float = 0.0, dy_norm: float = 0.0, zoom_mult: float = 1.0) -> None:
        """Incorporate known external pan/zoom motor command as a weak prior before visual alignment."""
        self.pose.zoom = float(np.clip(self.pose.zoom * zoom_mult, 0.5, 8.0))
        self.pose.cx = float(np.clip(self.pose.cx + dx_norm / max(0.5, self.pose.zoom), 0.05, 0.95))
        self.pose.cy = float(np.clip(self.pose.cy + dy_norm / max(0.5, self.pose.zoom), 0.05, 0.95))

    def update(self, water_mask: np.ndarray, label_map: np.ndarray) -> CameraPose:
        ds = self.work_size
        # Use stable coastline + terrain structure (water boundary + neutral contour) for shift estimation
        structure = (water_mask.astype(np.float32) * 2.0 + (label_map == -1).astype(np.float32))
        if structure.shape != (ds, ds):
            structure = cv2.resize(structure, (ds, ds), interpolation=cv2.INTER_AREA)
        edge = cv2.GaussianBlur(structure, (5, 5), 0)

        if self._prev_edge is None:
            self._prev_edge = edge
            return self.pose

        if float(edge.std()) < 1e-3 or float(self._prev_edge.std()) < 1e-3:
            self._prev_edge = edge
            self.pose.last_dx = 0.0
            self.pose.last_dy = 0.0
            self.pose.confidence = 0.25
            return self.pose

        (shift_x, shift_y), response = cv2.phaseCorrelate(self._prev_edge, edge, window=self._hann)
        # Positive image shift means camera moved in the opposite world direction
        if response > 0.15 and abs(shift_x) < ds * 0.35 and abs(shift_y) < ds * 0.35:
            dx_world = (-shift_x / ds) / max(0.5, self.pose.zoom)
            dy_world = (-shift_y / ds) / max(0.5, self.pose.zoom)
            self.pose.cx = float(np.clip(self.pose.cx + dx_world, 0.05, 0.95))
            self.pose.cy = float(np.clip(self.pose.cy + dy_world, 0.05, 0.95))
            self.pose.last_dx = float(dx_world)
            self.pose.last_dy = float(dy_world)
            self.pose.confidence = float(np.clip(response, 0.0, 1.0))
        else:
            self.pose.last_dx = 0.0
            self.pose.last_dy = 0.0
            self.pose.confidence = float(np.clip(response, 0.0, 1.0))

        self._prev_edge = edge
        return self.pose
