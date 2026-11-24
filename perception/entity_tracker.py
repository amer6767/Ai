"""Entity tracker associating OCR troop counts, border contacts, coastlines, and crown cues to territories."""

from __future__ import annotations

import cv2
import numpy as np

from core.types import BBox, TerritoryEntity
from vision.ocr import FastCanvasOCR
from vision.segmentation import SegmentationResult


class TerritoryEntityTracker:
    """Builds structured `TerritoryEntity` objects purely from segmented polygons and OCR crops."""

    def __init__(self, ocr: FastCanvasOCR | None = None, max_entities: int = 32) -> None:
        self.ocr = ocr or FastCanvasOCR()
        self.max_entities = max_entities
        self._prev_troops: dict[tuple[int, int, int], float] = {}
        self._prev_areas: dict[tuple[int, int, int], int] = {}

    def reset(self) -> None:
        self._prev_troops.clear()
        self._prev_areas.clear()

    @staticmethod
    def _color_key(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        return (rgb[0] // 16, rgb[1] // 16, rgb[2] // 16)

    def _read_territory_troops(self, full_rgb: np.ndarray, box: BBox, mean_rgb: tuple[int, int, int]) -> tuple[float | None, float]:
        """Read troop count printed inside a territory (handles both white-on-dark and black-on-light text!)."""
        if box.w < 18 or box.h < 14:
            return None, 0.0
        # Troop count sits in the lower-middle of the inscribed territory text block
        y0 = max(0, box.y + int(box.h * 0.42))
        y1 = min(full_rgb.shape[0], box.y + int(box.h * 0.85))
        x0 = max(0, box.x + int(box.w * 0.15))
        x1 = min(full_rgb.shape[1], box.x + int(box.w * 0.85))
        patch = full_rgb[y0:y1, x0:x1]
        if patch.size == 0:
            return None, 0.0

        luma = 0.299 * mean_rgb[0] + 0.587 * mean_rgb[1] + 0.114 * mean_rgb[2]
        if luma > 135.0:
            # Bright territory renders BLACK text -> invert so FastCanvasOCR sees bright white glyphs!
            patch = 255 - patch
        return self.ocr.read_troops(patch)

    def extract_entities(
        self,
        full_rgb: np.ndarray,
        seg: SegmentationResult,
        self_entity_id: int | None,
        self_mask: np.ndarray,
        hud_self_troops: int | None = None,
        crown_norm: tuple[float, float] | None = None,
        enable_map_ocr: bool = True,
    ) -> list[TerritoryEntity]:
        h_ds, w_ds = seg.label_map.shape

        # Dilate self_mask by 2 cells to find exact border contact pixels with every neighbor!
        self_dilated = cv2.dilate(self_mask, np.ones((5, 5), np.uint8)) > 0
        water_dilated = cv2.dilate(seg.water_mask, np.ones((3, 3), np.uint8)) > 0

        entities: list[TerritoryEntity] = []
        ordered_ids = sorted(
            seg.component_areas.keys(),
            key=lambda cid: (
                0 if cid == self_entity_id else 1,
                -int(((seg.label_map == cid) & self_dilated).sum()),
                -seg.component_areas[cid],
            ),
        )

        new_troops: dict[tuple[int, int, int], float] = {}
        new_areas: dict[tuple[int, int, int], int] = {}

        for cid in ordered_ids[: self.max_entities]:
            mask = seg.label_map == cid
            area = int(mask.sum())
            if area < 4:
                continue
            ys, xs = mask.nonzero()
            cx_n = float(xs.mean()) / w_ds
            cy_n = float(ys.mean()) / h_ds
            bx, by, bw, bh = seg.component_bboxes[cid]
            full_box = BBox(
                x=int(bx * seg.scale_x),
                y=int(by * seg.scale_y),
                w=max(1, int(bw * seg.scale_x)),
                h=max(1, int(bh * seg.scale_y)),
            )
            mean_rgb = seg.component_colors[cid]
            ck = self._color_key(mean_rgb)
            is_self = cid == self_entity_id

            contact_mask = mask & self_dilated if not is_self else np.zeros_like(mask)
            contact_px = int(contact_mask.sum())
            coastal_px = int((mask & water_dilated).sum())

            if contact_px > 0:
                c_ys, c_xs = contact_mask.nonzero()
                atk_pt = (float(c_xs.mean()) / w_ds, float(c_ys.mean()) / h_ds)
            else:
                atk_pt = (cx_n, cy_n)

            if is_self and hud_self_troops is not None and hud_self_troops >= 0:
                est_troops = float(hud_self_troops)
                conf = 0.95
            else:
                ocr_val, ocr_conf = (
                    self._read_territory_troops(full_rgb, full_box, mean_rgb)
                    if enable_map_ocr
                    else (None, 0.0)
                )
                if ocr_val is not None and ocr_conf > 0.42:
                    est_troops = float(ocr_val)
                    conf = ocr_conf
                elif ck in self._prev_troops:
                    est_troops = self._prev_troops[ck]
                    conf = 0.30
                else:
                    est_troops = float(max(20.0, area * 22.0))
                    conf = 0.15

            prev_t = self._prev_troops.get(ck, est_troops)
            prev_a = self._prev_areas.get(ck, area)
            dt = est_troops - prev_t
            da = float(area - prev_a)

            has_crown = False
            if crown_norm is not None:
                c_gx = int(crown_norm[0] * (w_ds - 1))
                c_gy = int(crown_norm[1] * (h_ds - 1))
                has_crown = bool(seg.label_map[c_gy, c_gx] == cid)

            entities.append(
                TerritoryEntity(
                    entity_id=cid,
                    is_self=is_self,
                    is_neutral=False,
                    mean_rgb=mean_rgb,
                    area_px=area,
                    centroid_norm=(cx_n, cy_n),
                    bbox=full_box,
                    estimated_troops=est_troops,
                    troop_confidence=conf,
                    border_contact_with_self_px=contact_px,
                    coastal_px=coastal_px,
                    delta_area_px=da,
                    delta_troops=dt,
                    is_under_attack=(dt < -max(30.0, 0.12 * prev_t) or da < -2.0),
                    has_crown=has_crown,
                    attack_point_norm=atk_pt,
                )
            )
            new_troops[ck] = est_troops
            new_areas[ck] = area

        self._prev_troops = new_troops
        self._prev_areas = new_areas
        return entities
