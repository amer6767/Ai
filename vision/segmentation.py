"""Hybrid classical + neural territory segmentation from raw RGB screenshots.

Separates water, mountains, neutral land, and individual occupied player territories
even when adjacent players share similar colors by exploiting Territorial.io's darkened
inner territory borders and optional `VisionUNetMobile` semantic predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from models.vision_unet import VisionUNetMobile
from vision.color_calibration import DynamicColorCalibrator, TerrainMasks


@dataclass
class SegmentationResult:
    label_map: np.ndarray        # [H_ds, W_ds] int32: -2=water, -1=mountain, 0=neutral, >0=territory component id
    water_mask: np.ndarray       # [H_ds, W_ds] uint8
    mountain_mask: np.ndarray    # [H_ds, W_ds] uint8
    neutral_mask: np.ndarray     # [H_ds, W_ds] uint8
    border_mask: np.ndarray      # [H_ds, W_ds] uint8
    component_colors: dict[int, tuple[int, int, int]]
    component_areas: dict[int, int]
    component_bboxes: dict[int, tuple[int, int, int, int]]
    scale_x: float
    scale_y: float


class TerritorySegmenter:
    """Segments Territorial.io screenshots into terrain + individual player territories."""

    def __init__(
        self,
        target_size: int = 128,
        unet_checkpoint: str | Path | None = "models/checkpoints/vision_unet.pt",
        device: str = "cpu",
    ) -> None:
        self.target_size = target_size
        self.calibrator = DynamicColorCalibrator()
        self.device = torch.device(device)
        self.unet = VisionUNetMobile().to(self.device).eval()
        self.neural_ready = False
        if unet_checkpoint and Path(unet_checkpoint).exists():
            try:
                ckpt = torch.load(unet_checkpoint, map_location=self.device, weights_only=False)
                self.unet.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
                self.neural_ready = True
            except Exception:
                self.neural_ready = False

    def _predict_neural_classes(self, rgb_ds: np.ndarray) -> np.ndarray | None:
        if not self.neural_ready:
            return None
        inp = torch.from_numpy(rgb_ds.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.unet(inp)
            return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    def segment(self, rgb: np.ndarray, ui_mask: np.ndarray | None = None) -> SegmentationResult:
        h_orig, w_orig = rgb.shape[:2]
        ds = self.target_size
        rgb_ds = cv2.resize(rgb, (ds, ds), interpolation=cv2.INTER_AREA)
        ui_ds = (
            cv2.resize(ui_mask.astype(np.uint8), (ds, ds), interpolation=cv2.INTER_NEAREST) > 0
            if ui_mask is not None
            else np.zeros((ds, ds), dtype=bool)
        )

        terrain: TerrainMasks = self.calibrator.classify_terrain(rgb_ds, ui_mask=ui_ds)
        neural_cls = self._predict_neural_classes(rgb_ds)

        if neural_cls is not None:
            # Fuse classical color-calibrated geometry with learned UNet border & terrain classes
            water = (terrain.water | (neural_cls == 1)) & (~ui_ds)
            mountain = (terrain.mountain | (neural_cls == 2)) & (~water) & (~ui_ds)
            border_ridge = (terrain.border_ridge | (neural_cls == 5)) & (~water) & (~ui_ds)
            player_land = (terrain.player_land | (neural_cls == 4)) & (~water) & (~mountain) & (~ui_ds)
            neutral = (~water) & (~mountain) & (~player_land) & (~ui_ds)
        else:
            water = terrain.water
            mountain = terrain.mountain
            border_ridge = terrain.border_ridge
            player_land = terrain.player_land
            neutral = terrain.neutral_land

        label_map = np.zeros((ds, ds), dtype=np.int32)
        label_map[water] = -2
        label_map[mountain] = -1
        label_map[ui_ds] = -3

        # Split player territories cleanly along darkened inner border ridges + LAB color jumps
        lab = cv2.cvtColor(rgb_ds, cv2.COLOR_RGB2LAB).astype(np.float32)
        dx = np.abs(np.diff(lab, axis=1, prepend=lab[:, :1, :])).sum(axis=2)
        dy = np.abs(np.diff(lab, axis=0, prepend=lab[:1, :, :])).sum(axis=2)
        color_edge = (dx > 28.0) | (dy > 28.0)

        interior_seeds = (player_land & (~color_edge)).astype(np.uint8)
        # Connected components over uniform color interiors
        num_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(interior_seeds, connectivity=4)

        # Keep meaningful territories and expand 1px across border ridge into their own border pixels
        comp_colors: dict[int, tuple[int, int, int]] = {}
        comp_areas: dict[int, int] = {}
        comp_bboxes: dict[int, tuple[int, int, int, int]] = {}

        # Sort by area descending and keep top 48 visible components
        candidates = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, num_cc) if stats[i, cv2.CC_STAT_AREA] >= 4]
        candidates.sort(reverse=True)
        next_id = 1
        for area, cc_idx in candidates[:48]:
            mask = cc_map == cc_idx
            mean_c = rgb_ds[mask].mean(axis=0)
            label_map[mask] = next_id
            comp_colors[next_id] = (int(mean_c[0]), int(mean_c[1]), int(mean_c[2]))
            comp_areas[next_id] = area
            x = int(stats[cc_idx, cv2.CC_STAT_LEFT])
            y = int(stats[cc_idx, cv2.CC_STAT_TOP])
            w = int(stats[cc_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[cc_idx, cv2.CC_STAT_HEIGHT])
            comp_bboxes[next_id] = (x, y, w, h)
            next_id += 1

        # Reclaim 1px border-ridge pixels belonging to player_land via nearest assigned neighbor
        unassigned_player = player_land & (label_map == 0)
        if unassigned_player.any() and next_id > 1:
            dilated = cv2.dilate(label_map.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
            fill_mask = unassigned_player & (dilated > 0)
            label_map[fill_mask] = dilated[fill_mask]
            # Refresh area counts
            for eid in list(comp_areas.keys()):
                comp_areas[eid] = int((label_map == eid).sum())

        return SegmentationResult(
            label_map=label_map,
            water_mask=water.astype(np.uint8),
            mountain_mask=mountain.astype(np.uint8),
            neutral_mask=(label_map == 0).astype(np.uint8),
            border_mask=border_ridge.astype(np.uint8),
            component_colors=comp_colors,
            component_areas=comp_areas,
            component_bboxes=comp_bboxes,
            scale_x=w_orig / float(ds),
            scale_y=h_orig / float(ds),
        )
