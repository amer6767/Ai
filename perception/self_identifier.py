"""Robust visual identification of the AI's own territory across pan, zoom, and color themes.

Uses strictly visual cues:
1. Initial spawn lock: when entering IN_GAME, Territorial.io centers the camera directly
   on the player's own territory (around viewport center (0.5, 0.5)).
2. Persistent RGB/LAB color signature with EMA tracking as territory grows.
3. Spatial overlap continuity with previous self_mask across consecutive frames.
"""

from __future__ import annotations

import numpy as np

from vision.segmentation import SegmentationResult


class SelfTerritoryIdentifier:
    """Tracks which segmented territory polygon belongs to the AI itself."""

    def __init__(self) -> None:
        self._locked_rgb: np.ndarray | None = None
        self._prev_self_mask: np.ndarray | None = None
        self._prev_centroid: tuple[float, float] = (0.5, 0.5)

    def reset(self) -> None:
        self._locked_rgb = None
        self._prev_self_mask = None
        self._prev_centroid = (0.5, 0.5)

    def set_color_hint(self, rgb: tuple[int, int, int]) -> None:
        self._locked_rgb = np.array(rgb, dtype=np.float32)

    def identify(self, seg: SegmentationResult) -> tuple[int | None, np.ndarray]:
        h, w = seg.label_map.shape
        if not seg.component_areas:
            empty = np.zeros((h, w), dtype=np.uint8)
            return None, empty

        best_id: int | None = None
        best_score = -1e9

        # First active frame without locked color: Territorial.io spawns camera centered on self!
        center_id = int(seg.label_map[h // 2, w // 2])
        if self._locked_rgb is None and center_id > 0:
            best_id = center_id
            self._locked_rgb = np.array(seg.component_colors[center_id], dtype=np.float32)
        else:
            for cid, area in seg.component_areas.items():
                bx, by, bw, bh = seg.component_bboxes[cid]
                cx = (bx + bw * 0.5) / w
                cy = (by + bh * 0.5) / h
                col = np.array(seg.component_colors[cid], dtype=np.float32)
                mask_c = seg.label_map == cid

                score = 0.0
                if self._locked_rgb is not None:
                    color_dist = float(np.linalg.norm(col - self._locked_rgb))
                    score -= color_dist * 0.08
                else:
                    # Prefer near center when uninitialized
                    score -= float(np.hypot(cx - 0.5, cy - 0.5)) * 6.0

                if self._prev_self_mask is not None and self._prev_self_mask.shape == mask_c.shape:
                    inter = float((mask_c & (self._prev_self_mask > 0)).sum())
                    union = float((mask_c | (self._prev_self_mask > 0)).sum()) + 1e-6
                    iou = inter / union
                    score += iou * 12.0

                dist_prev = float(np.hypot(cx - self._prev_centroid[0], cy - self._prev_centroid[1]))
                score -= dist_prev * 2.5
                score += 0.15 * float(np.log1p(area))

                if score > best_score:
                    best_score = score
                    best_id = cid

        if best_id is None:
            return None, np.zeros((h, w), dtype=np.uint8)

        # Also merge any disconnected islands sharing exact same RGB signature as self (e.g., after boat landing!)
        self_col = np.array(seg.component_colors[best_id], dtype=np.float32)
        if self._locked_rgb is None:
            self._locked_rgb = self_col
        else:
            # Slow EMA update if close enough (prevents drifting onto enemy on sudden death)
            if float(np.linalg.norm(self_col - self._locked_rgb)) < 45.0:
                self._locked_rgb = 0.85 * self._locked_rgb + 0.15 * self_col

        self_mask = (seg.label_map == best_id).astype(np.uint8)
        for cid, c_rgb in seg.component_colors.items():
            if cid == best_id:
                continue
            if float(np.linalg.norm(np.array(c_rgb, dtype=np.float32) - self._locked_rgb)) < 18.0:
                self_mask |= (seg.label_map == cid).astype(np.uint8)

        ys, xs = self_mask.nonzero()
        if len(xs) > 0:
            self._prev_centroid = (float(xs.mean()) / w, float(ys.mean()) / h)
        self._prev_self_mask = self_mask
        return best_id, self_mask
