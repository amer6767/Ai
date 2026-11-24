"""Persistent allocentric world map with fog-of-vision staleness tracking."""

from __future__ import annotations

import cv2
import numpy as np

from core.types import PerceivedFrame
from map_analysis.camera_odometry import CameraPose, VisualCameraOdometry


class PersistentWorldMap:
    """Stitches observed viewport frames into a persistent allocentric world representation.

    Retains off-screen water, neutral, self, and enemy density memory across camera pan/zoom,
    and tracks `staleness` (0 = freshly seen, 1 = unseen/stale) so the AI knows when to scout.
    """

    def __init__(self, world_size: int = 256) -> None:
        self.world_size = world_size
        self.odometry = VisualCameraOdometry(work_size=128)
        self.water_prob = np.zeros((world_size, world_size), dtype=np.float32)
        self.self_prob = np.zeros((world_size, world_size), dtype=np.float32)
        self.neutral_prob = np.zeros((world_size, world_size), dtype=np.float32)
        self.enemy_density = np.zeros((world_size, world_size), dtype=np.float32)
        self.staleness = np.ones((world_size, world_size), dtype=np.float32)
        self._step = 0

    def reset(self) -> None:
        self.odometry.reset()
        self.water_prob.fill(0.0)
        self.self_prob.fill(0.0)
        self.neutral_prob.fill(0.0)
        self.enemy_density.fill(0.0)
        self.staleness.fill(1.0)
        self._step = 0

    def update(self, pf: PerceivedFrame) -> CameraPose:
        self._step += 1
        pose = self.odometry.update(pf.water_mask, pf.label_map)

        # Age entire world map slightly each step
        self.staleness = np.clip(self.staleness + 0.015, 0.0, 1.0)

        # Compute viewport rectangle inside [0, world_size]
        span = 1.0 / max(1.0, pose.zoom)
        x0 = max(0, int(round((pose.cx - 0.5 * span) * self.world_size)))
        y0 = max(0, int(round((pose.cy - 0.5 * span) * self.world_size)))
        x1 = min(self.world_size, max(x0 + 4, int(round((pose.cx + 0.5 * span) * self.world_size))))
        y1 = min(self.world_size, max(y0 + 4, int(round((pose.cy + 0.5 * span) * self.world_size))))
        tw, th = x1 - x0, y1 - y0

        # Build log-density enemy map in viewport space
        h_ds, w_ds = pf.label_map.shape
        enemy_map = np.zeros((h_ds, w_ds), dtype=np.float32)
        for e in pf.entities:
            if e.is_self:
                continue
            density = float(np.log1p(e.estimated_troops / max(1.0, e.area_px))) / 6.0
            enemy_map[pf.label_map == e.entity_id] = float(np.clip(density, 0.0, 1.0))

        w_patch = cv2.resize(pf.water_mask.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
        s_patch = cv2.resize(pf.self_mask.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
        n_patch = cv2.resize(pf.neutral_mask.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
        e_patch = cv2.resize(enemy_map, (tw, th), interpolation=cv2.INTER_AREA)

        alpha = 0.75
        self.water_prob[y0:y1, x0:x1] = (1 - alpha) * self.water_prob[y0:y1, x0:x1] + alpha * w_patch
        self.self_prob[y0:y1, x0:x1] = (1 - alpha) * self.self_prob[y0:y1, x0:x1] + alpha * s_patch
        self.neutral_prob[y0:y1, x0:x1] = (1 - alpha) * self.neutral_prob[y0:y1, x0:x1] + alpha * n_patch
        self.enemy_density[y0:y1, x0:x1] = (1 - alpha) * self.enemy_density[y0:y1, x0:x1] + alpha * e_patch
        self.staleness[y0:y1, x0:x1] = 0.0
        return pose

    def viewport_staleness(self, target_size: int = 128) -> np.ndarray:
        pose = self.odometry.pose
        span = 1.0 / max(1.0, pose.zoom)
        x0 = max(0, int(round((pose.cx - 0.5 * span) * self.world_size)))
        y0 = max(0, int(round((pose.cy - 0.5 * span) * self.world_size)))
        x1 = min(self.world_size, max(x0 + 4, int(round((pose.cx + 0.5 * span) * self.world_size))))
        y1 = min(self.world_size, max(y0 + 4, int(round((pose.cy + 0.5 * span) * self.world_size))))
        crop = self.staleness[y0:y1, x0:x1]
        return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)

    def stalest_direction(self) -> tuple[float, float, float]:
        """Return (dx, dy, max_staleness) pointing from current camera toward the stalest unseen region."""
        # Divide world into 4x4 blocks
        blk = self.world_size // 4
        best_s = 0.0
        best_cx, best_cy = 0.5, 0.5
        for by in range(4):
            for bx in range(4):
                s = float(self.staleness[by * blk : (by + 1) * blk, bx * blk : (bx + 1) * blk].mean())
                if s > best_s:
                    best_s = s
                    best_cx = (bx + 0.5) / 4.0
                    best_cy = (by + 0.5) / 4.0
        return best_cx - self.odometry.pose.cx, best_cy - self.odometry.pose.cy, best_s
