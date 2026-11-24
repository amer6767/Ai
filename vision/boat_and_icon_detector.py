"""Pure-pixel naval boat and crown/leader icon detector."""

from __future__ import annotations

import cv2
import numpy as np

from core.types import BoatDetection


class BoatAndIconDetector:
    """Detects moving naval ships traversing water masks and crown icons over leaders."""

    def __init__(self) -> None:
        self._prev_boats: list[tuple[float, float]] = []

    def detect_boats(
        self,
        rgb: np.ndarray,
        water_mask: np.ndarray,
        self_mask: np.ndarray,
    ) -> list[BoatDetection]:
        """Locate compact high-contrast boat sprites inside ocean/water regions."""
        h, w = rgb.shape[:2]
        # Boats appear as bright/saturated compact sprites sitting strictly surrounded by water
        water_u8 = water_mask.astype(np.uint8) * 255
        if water_u8.sum() < 50:
            self._prev_boats = []
            return []

        # Dilate water slightly then find non-water holes enclosed completely by water (small area 8..400 px)
        closed_water = cv2.morphologyEx(water_u8, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        holes = ((closed_water > 0) & (water_u8 == 0)).astype(np.uint8) * 255
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(holes, connectivity=8)

        # Precompute distance transform to own coastline
        self_u8 = (self_mask > 0).astype(np.uint8)
        if self_u8.any():
            inv_self = (1 - self_u8) * 255
            dist_to_self = cv2.distanceTransform(inv_self, cv2.DIST_L2, 3)
            diag = float(np.hypot(self_mask.shape[0], self_mask.shape[1])) + 1e-6
        else:
            dist_to_self = None
            diag = 1.0

        detections: list[BoatDetection] = []
        cur_pts: list[tuple[float, float]] = []
        for i in range(1, num_labels):
            bx, by, bw, bh, area = stats[i]
            if area < 6 or area > 420:
                continue
            aspect = bw / max(1, bh)
            if not (0.35 <= aspect <= 2.8):
                continue
            patch = rgb[by : by + bh, bx : bx + bw]
            # Boats have bright white hull/sail pixels or vivid player-color badges
            if patch.size == 0 or float(patch.max()) < 160:
                continue

            cx_n = float(centroids[i][0]) / max(1, w)
            cy_n = float(centroids[i][1]) / max(1, h)
            cur_pts.append((cx_n, cy_n))

            vx, vy = 0.0, 0.0
            for px, py in self._prev_boats:
                d = float(np.hypot(cx_n - px, cy_n - py))
                if d < 0.06:
                    vx, vy = cx_n - px, cy_n - py
                    break

            dist_self = 1.0
            if dist_to_self is not None:
                sy = min(self_mask.shape[0] - 1, max(0, int(cy_n * self_mask.shape[0])))
                sx = min(self_mask.shape[1] - 1, max(0, int(cx_n * self_mask.shape[1])))
                dist_self = float(dist_to_self[sy, sx]) / diag

            detections.append(
                BoatDetection(
                    x_norm=cx_n,
                    y_norm=cy_n,
                    vx_norm=vx,
                    vy_norm=vy,
                    is_hostile=dist_self > 0.02,
                    distance_to_self_coast=dist_self,
                )
            )
            if len(detections) >= 16:
                break

        self._prev_boats = cur_pts
        return detections

    def detect_crown_point(self, rgb: np.ndarray, ui_mask: np.ndarray | None = None) -> tuple[float, float] | None:
        """Detect golden crown icon rendered on the #1 ranked kingdom on the map."""
        h, w = rgb.shape[:2]
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        gold = (r > 205) & (g > 175) & (b < 95) & (r - b > 120)
        if ui_mask is not None:
            gold &= ~ui_mask
        ys, xs = gold.nonzero()
        if len(xs) < 8 or len(xs) > 1500:
            return None
        return float(xs.mean()) / max(1, w), float(ys.mean()) / max(1, h)
