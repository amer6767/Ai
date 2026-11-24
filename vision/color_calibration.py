"""Dynamic color calibration & terrain classification for variable map themes and player palettes."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TerrainMasks:
    water: np.ndarray        # bool [H, W]
    mountain: np.ndarray     # bool [H, W]
    neutral_land: np.ndarray # bool [H, W]
    player_land: np.ndarray  # bool [H, W]
    border_ridge: np.ndarray # bool [H, W] darkened inner territory borders


class DynamicColorCalibrator:
    """Separates water, mountain, neutral terrain, and dynamic player territory fills
    without assuming a single fixed palette or screen resolution.
    """

    def __init__(self) -> None:
        self._water_hue_anchor: float | None = None
        self._self_color_rgb: np.ndarray | None = None

    def set_self_color_hint(self, rgb: tuple[int, int, int]) -> None:
        self._self_color_rgb = np.array(rgb, dtype=np.float32)

    @property
    def self_color_rgb(self) -> tuple[int, int, int] | None:
        if self._self_color_rgb is None:
            return None
        return (int(self._self_color_rgb[0]), int(self._self_color_rgb[1]), int(self._self_color_rgb[2]))

    def classify_terrain(
        self,
        rgb: np.ndarray,
        ui_mask: np.ndarray | None = None,
    ) -> TerrainMasks:
        """Classify every map pixel into water, mountain, neutral land, player territory, and borders."""
        h, w = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        valid = np.ones((h, w), dtype=bool)
        if ui_mask is not None:
            valid &= ~ui_mask

        # 1. Water: Territorial.io oceans/lakes are blue-teal with B clearly dominating R and moderate-high depth.
        # Dynamically calibrate background water from screen margins (outside UI).
        water_raw = (
            valid
            & (b - r >= 22)
            & (b >= 65)
            & (hue >= 90)
            & (hue <= 125)
            & (sat >= 60)
            & (val >= 35)
            & (val <= 205)
        )
        # Also detect coastal shallow blue glow surrounding islands
        water_shallow = valid & (b - r >= 18) & (g - r >= 12) & (hue >= 88) & (hue <= 118) & (sat >= 55)
        water = water_raw | water_shallow

        # 2. Mountains / dark rocky coastlines / high-elevation impassable terrain
        mountain = (
            valid
            & (~water)
            & (sat < 48)
            & ((val < 62) | ((val >= 62) & (val <= 115) & (np.abs(r - g) < 16) & (np.abs(g - b) < 16)))
        )

        # 3. Neutral unowned terrain: natural textured greens, olive-yellows, sandy beaches, white snow peaks
        # Territorial.io neutral terrain has characteristic organic gradient texture vs flat solid player fills!
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        local_tex = cv2.blur(lap, (5, 5))

        # Neutral biomes: natural terrain greens/tans/snow with subtle elevation shading
        neutral_green = (
            valid
            & (~water)
            & (~mountain)
            & (g - r >= 12)
            & (g - b >= 18)
            & (hue >= 36)
            & (hue <= 74)
            & (val >= 55)
        )
        neutral_snow_sand = (
            valid
            & (~water)
            & (~mountain)
            & (sat <= 55)
            & (val >= 145)
        )

        # Flat solid-fill detector: occupied territories have ultra-low interior color variance
        # while natural green biome terrain has smooth gradient variation or micro-texture.
        rgb_f = rgb.astype(np.float32)
        mean_box = cv2.blur(rgb_f, (5, 5))
        sq_box = cv2.blur(rgb_f * rgb_f, (5, 5))
        color_std = np.sqrt(np.maximum(0.0, sq_box - mean_box * mean_box)).mean(axis=2)

        # Darkened inner border ridge around player territories (2-4 px strong dark edge between flat regions)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(sobel_x, sobel_y)
        border_ridge = valid & (~water) & (grad_mag > 42.0)

        # Natural green/olive/tan biome band (Territorial.io unowned land)
        is_biome_green = (hue >= 34) & (hue <= 76) & (g >= r + 10) & (g >= b + 15)
        is_biome_neutral = is_biome_green | ((sat <= 52) & (val >= 135))

        # Player land = non-biome hues (reds, purples, oranges, yellows, cyans, pinks, dark olives)
        # OR ultra-uniform flat green fills clearly distinct from natural terrain gradients (e.g. near-zero std)
        ultra_flat_solid = (color_std < 1.35) & (local_tex < 2.5)
        player_land = (
            valid
            & (~water)
            & (~mountain)
            & (
                ((~is_biome_neutral) & (sat >= 42) & (val >= 38))
                | (is_biome_green & ultra_flat_solid & (sat >= 90) & (val >= 50) & (val <= 185))
            )
        )
        neutral_land = valid & (~water) & (~mountain) & (~player_land)

        return TerrainMasks(
            water=water,
            mountain=mountain,
            neutral_land=neutral_land,
            player_land=player_land,
            border_ridge=border_ridge,
        )
