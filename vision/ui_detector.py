"""Scale-invariant pure-pixel UI detector for Territorial.io.

Locates menus, Custom Scenario panels, Top-Center Balance Box, Top-Right Stats Panel,
Bottom-Center Troop Slider, Top-Left Leaderboard, and Spawn Banner dynamically using
white-border geometry and relative layout cues (zero hardcoded pixel coordinates).
"""

from __future__ import annotations

import cv2
import numpy as np

from core.types import BBox, ScreenState, UILayout
from vision.ocr import FastCanvasOCR


def _find_white_rectangles(
    rgb: np.ndarray,
    min_w_ratio: float = 0.05,
    min_h_ratio: float = 0.02,
    max_w_ratio: float = 0.60,
    max_h_ratio: float = 0.52,
) -> list[BBox]:
    """Detect crisp white-bordered UI rectangles present across all Territorial.io screens."""
    h_img, w_img = rgb.shape[:2]
    white = ((rgb[:, :, 0] > 180) & (rgb[:, :, 1] > 180) & (rgb[:, :, 2] > 180)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[BBox] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (w_img * min_w_ratio <= w <= w_img * max_w_ratio):
            continue
        if not (h_img * min_h_ratio <= h <= h_img * max_h_ratio):
            continue
        top_edge = float((white[y : min(h_img, y + 3), x : x + w] > 0).mean())
        bot_edge = float((white[max(0, y + h - 3) : y + h, x : x + w] > 0).mean())
        left_edge = float((white[y : y + h, x : min(w_img, x + 3)] > 0).mean())
        right_edge = float((white[y : y + h, max(0, x + w - 3) : x + w] > 0).mean())
        if (top_edge > 0.30 and bot_edge > 0.30) or (left_edge > 0.30 and right_edge > 0.30):
            boxes.append(BBox(x=int(x), y=int(y), w=int(w), h=int(h)))
    return boxes


class ScaleInvariantUIDetector:
    """Dynamically localizes and reads Territorial.io UI widgets at any screen resolution."""

    def __init__(self, ocr: FastCanvasOCR | None = None) -> None:
        self.ocr = ocr or FastCanvasOCR()

    def build_ui_mask(self, h: int, w: int, layout: UILayout) -> np.ndarray:
        """Return boolean mask [H, W] covering detected HUD boxes so map CV ignores UI overlays."""
        mask = np.zeros((h, w), dtype=bool)
        for box in (
            layout.balance_box,
            layout.stats_panel,
            layout.slider_box,
            layout.leaderboard_box,
        ):
            if box is not None:
                y0 = max(0, box.y - 3)
                y1 = min(h, box.y + box.h + 3)
                x0 = max(0, box.x - 3)
                x1 = min(w, box.x + box.w + 3)
                mask[y0:y1, x0:x1] = True
        mask[int(h * 0.91) :, : int(w * 0.055)] = True
        mask[int(h * 0.39) : int(h * 0.62), int(w * 0.935) :] = True
        mask[int(h * 0.77) :, int(w * 0.80) :] = True
        if layout.leaderboard_box is not None:
            lb = layout.leaderboard_box
            mask[lb.y + lb.h : min(h, lb.y + lb.h + int(h * 0.10)), : min(w, int(w * 0.13))] = True
        return mask

    def detect(self, rgb: np.ndarray) -> UILayout:
        h, w = rgb.shape[:2]
        rects = _find_white_rectangles(rgb)

        leaderboard_box: BBox | None = None
        stats_panel: BBox | None = None
        balance_box: BBox | None = None
        slider_box: BBox | None = None

        for b in rects:
            cx_n, cy_n = b.cx / w, b.cy / h
            w_n, h_n = b.w / w, b.h / h

            # Top-Left Leaderboard
            if b.x < w * 0.05 and b.y < h * 0.06 and 0.12 <= w_n <= 0.32 and 0.18 <= h_n <= 0.48:
                if leaderboard_box is None or (b.w * b.h > leaderboard_box.w * leaderboard_box.h):
                    leaderboard_box = b

            # Top-Right Stats Panel
            if b.x + b.w > w * 0.94 and b.y < h * 0.06 and 0.07 <= w_n <= 0.24 and 0.08 <= h_n <= 0.26:
                if stats_panel is None or (b.w * b.h > stats_panel.w * stats_panel.h):
                    stats_panel = b

            # Top-Center Troop Balance Box (aspect ~ 5:1 to 7:1, top margin)
            if 0.35 <= cx_n <= 0.68 and b.y < h * 0.08 and 0.09 <= w_n <= 0.30 and 0.022 <= h_n <= 0.09:
                aspect = b.w / max(1, b.h)
                if aspect >= 3.2:
                    balance_box = b

            # Bottom-Center Troop Slider (aspect ~ 8:1 to 15:1, bottom margin)
            if 0.35 <= cx_n <= 0.65 and b.y + b.h > h * 0.90 and 0.18 <= w_n <= 0.55 and 0.022 <= h_n <= 0.09:
                aspect = b.w / max(1, b.h)
                if aspect >= 5.5:
                    slider_box = b

        # 2. Check Custom Scenario / Submenu Footer Bar (horizontal white line around y ~ 0.89 * H)
        gray = rgb.mean(axis=2)
        row_white_ratio = (gray > 180).mean(axis=1)
        top_sep_rows = np.where((np.arange(h) > h * 0.08) & (np.arange(h) < h * 0.22) & (row_white_ratio > 0.70))[0]
        bot_sep_rows = np.where((np.arange(h) > h * 0.80) & (np.arange(h) < h * 0.95) & (row_white_ratio > 0.70))[0]

        has_lobby_header_footer = len(top_sep_rows) > 0 and len(bot_sep_rows) > 0
        start_match_btn: BBox | None = None
        back_btn: BBox | None = None
        if has_lobby_header_footer:
            fy = int(bot_sep_rows[0])
            fh = h - fy
            mid_band = gray[fy + 2 : h - 2, int(w * 0.44) : int(w * 0.56)]
            col_white = (mid_band > 180).mean(axis=0) if mid_band.size else np.zeros(1)
            div_cols = np.where(col_white > 0.40)[0]
            if len(div_cols) > 0:
                split_x = int(w * 0.44) + int(div_cols[0])
                back_btn = BBox(0, fy, split_x, fh)
                start_match_btn = BBox(split_x, fy, w - split_x, fh)
            else:
                back_btn = BBox(0, fy, w, fh)

        # 3. Check Main Menu 2x2 central buttons
        multiplayer_btn: BBox | None = None
        singleplayer_btn: BBox | None = None
        if not has_lobby_header_footer and leaderboard_box is None and stats_panel is None:
            for b in rects:
                cx_n, cy_n = b.cx / w, b.cy / h
                if 0.32 <= cx_n <= 0.50 and 0.42 <= cy_n <= 0.62 and b.w > w * 0.07:
                    patch = rgb[b.y + 4 : b.y + b.h - 4, b.x + 4 : b.x + b.w - 4]
                    if patch.size and patch[:, :, 1].mean() > patch[:, :, 0].mean() + 18:
                        multiplayer_btn = b
                elif 0.50 <= cx_n <= 0.68 and 0.42 <= cy_n <= 0.62 and b.w > w * 0.07:
                    patch = rgb[b.y + 4 : b.y + b.h - 4, b.x + 4 : b.x + b.w - 4]
                    if patch.size and patch[:, :, 2].mean() > patch[:, :, 0].mean() + 15:
                        singleplayer_btn = b

        # 4. Classify ScreenState cleanly from detected widgets
        screen_state = ScreenState.UNKNOWN
        if balance_box is not None and slider_box is not None:
            screen_state = ScreenState.IN_GAME
        elif leaderboard_box is not None and stats_panel is not None:
            # Distinguish SPAWN_SELECTION vs SPECTATING/END
            # Spawn phase shows "Choose your start position!" dark banner in bottom-right (around y=0.77..0.82)
            br_strip = rgb[int(h * 0.77) : int(h * 0.83), int(w * 0.81) : int(w * 0.99)]
            dark_ratio = float((br_strip.max(axis=2) < 40).mean()) if br_strip.size else 0.0
            white_text = float((br_strip.min(axis=2) > 200).mean()) if br_strip.size else 0.0
            # Also check if game clock is still 0:00 in stats panel
            if dark_ratio > 0.35 and white_text > 0.04:
                screen_state = ScreenState.SPAWN_SELECTION
            else:
                screen_state = ScreenState.SPECTATING
        elif has_lobby_header_footer:
            screen_state = ScreenState.SINGLEPLAYER_LOBBY if start_match_btn is not None else ScreenState.MODAL_OVERLAY
        elif multiplayer_btn is not None or singleplayer_btn is not None:
            screen_state = ScreenState.MAIN_MENU
        else:
            # Check victory / defeat / modal popup (dark centered box over map)
            center_patch = rgb[int(h * 0.30) : int(h * 0.70), int(w * 0.30) : int(w * 0.70)]
            if center_patch.size and float((center_patch.max(axis=2) < 45).mean()) > 0.5:
                screen_state = ScreenState.MODAL_OVERLAY

        layout = UILayout(
            screen_state=screen_state,
            balance_box=balance_box,
            stats_panel=stats_panel,
            slider_box=slider_box,
            leaderboard_box=leaderboard_box,
            singleplayer_btn=singleplayer_btn,
            multiplayer_btn=multiplayer_btn,
            start_match_btn=start_match_btn,
            back_btn=back_btn,
        )

        # 5. Parse fine-grained HUD indicators purely from pixels
        if balance_box is not None:
            self._parse_balance_box(rgb, balance_box, layout)
        if slider_box is not None:
            self._parse_slider_box(rgb, slider_box, layout)
        if stats_panel is not None:
            self._parse_stats_panel(rgb, stats_panel, layout)

        return layout

    def _parse_balance_box(self, rgb: np.ndarray, box: BBox, layout: UILayout) -> None:
        patch = rgb[box.y + 2 : box.y + box.h - 2, box.x + 2 : box.x + box.w - 2]
        if patch.size == 0:
            return
        # Green bg (G > R) = gaining troops; Red bg (R > G) = losing troops
        mean_r = float(patch[:, :, 0].mean())
        mean_g = float(patch[:, :, 1].mean())
        layout.balance_gaining = mean_g >= mean_r - 5.0

        # Red-interest text cue: light pink rgb(255, 160, 160) -> R > 215 and R - G > 45
        pink_text = (patch[:, :, 0] > 215) & (patch[:, :, 0].astype(np.int16) - patch[:, :, 1].astype(np.int16) > 45)
        if int(pink_text.sum()) >= 12:
            layout.red_interest_active = True

        # Left/Right white vertical cycle bars: measure white pixel fill ratio along left strip
        side_w = max(2, int(box.h * 0.16))
        left_bar = patch[:, :side_w]
        white_rows = (left_bar.min(axis=2) > 170).mean(axis=1)
        layout.cycle_progress = float(np.clip(1.0 - white_rows.mean(), 0.0, 1.0))

        # OCR central troop count (exclude left/right cycle bars & small +income superscript on right)
        inner = patch[:, side_w + 2 : max(side_w + 4, patch.shape[1] - side_w - 2)]
        # Split off '+income' text on right side if present (upper half right portion)
        troops, conf = self.ocr.read_troops(inner[:, : int(inner.shape[1] * 0.72)])
        if troops is not None and conf > 0.35:
            layout.hud_troops = int(round(troops))

    def _parse_slider_box(self, rgb: np.ndarray, box: BBox, layout: UILayout) -> None:
        cap_w = int(round(1.5 * box.h))
        x0 = box.x + cap_w
        x1 = box.x + box.w - cap_w
        layout.slider_track_x0 = x0
        layout.slider_track_x1 = max(x0 + 4, x1)

        track = rgb[box.y + 2 : box.y + box.h - 2, x0 + 2 : max(x0 + 3, x1 - 2)]
        if track.size == 0 or track.shape[1] < 4:
            return
        # Exact visual cue from canvas rendering: a crisp 1px full-height vertical white bar separates filled vs unfilled
        white_cols = (track.min(axis=2) > 180).mean(axis=0)
        # Exclude text strokes (which only occupy the middle vertical band, never > 82% of full column height)
        divider_cols = np.where(white_cols > 0.78)[0]
        if len(divider_cols) > 0:
            div_x = int(divider_cols[0])
            layout.estimated_slider_pct = float(np.clip((div_x + 2) / max(1, x1 - x0), 0.01, 1.0))
            return

        # Fallback: top+bottom margin strip luminance jump (filled colored bar is much brighter than dark 0.85 black unfilled)
        top_strip = track[: max(2, track.shape[0] // 5), :, :]
        bot_strip = track[max(1, (4 * track.shape[0]) // 5) :, :, :]
        strips = np.concatenate([top_strip, bot_strip], axis=0).astype(np.float32)
        col_r = strips[:, :, 0].mean(axis=0)
        col_g = strips[:, :, 1].mean(axis=0)
        filled_cols = (col_r + col_g) > float(np.median(col_r + col_g)) * 1.15
        if filled_cols.any():
            last_filled = int(np.where(filled_cols)[0][-1]) + 1
            layout.estimated_slider_pct = float(np.clip(last_filled / track.shape[1], 0.01, 1.0))
        else:
            layout.estimated_slider_pct = 0.50

    def _parse_stats_panel(self, rgb: np.ndarray, box: BBox, layout: UILayout) -> None:
        patch = rgb[box.y + 2 : box.y + box.h - 2, box.x + 2 : box.x + box.w - 2]
        if patch.size == 0:
            return
        ph, pw = patch.shape[:2]
        # Right half contains numeric values across 5 rows: Players, Percentage, Interest, Income, Time
        right_half = patch[:, int(pw * 0.48) :]
        row_h = max(6, ph // 5)

        # Row 0: Players alive
        p_val, p_conf = self.ocr.read_troops(right_half[0:row_h])
        if p_val is not None and p_conf > 0.35 and 1 <= p_val <= 600:
            layout.hud_players_alive = int(round(p_val))

        # Row 2: Interest % (and detect coral-red text rgb(255,120,100)!)
        r2 = right_half[2 * row_h : 3 * row_h]
        red_int = (r2[:, :, 0] > 195) & (r2[:, :, 0].astype(np.int16) - r2[:, :, 1].astype(np.int16) > 70)
        if int(red_int.sum()) >= 8:
            layout.red_interest_active = True
        int_pct, int_conf = self.ocr.read_percent(r2)
        if int_pct is not None and int_conf > 0.35 and int_pct <= 15.0:
            layout.hud_interest_pct = int_pct

        # Row 3: Income (= land pixels in standard Territorial.io!)
        inc_val, inc_conf = self.ocr.read_troops(right_half[3 * row_h : 4 * row_h])
        if inc_val is not None and inc_conf > 0.35:
            layout.hud_land = int(round(inc_val))

        # Row 4: Clock M:SS
        sec, clk_conf = self.ocr.read_clock(right_half[4 * row_h : ph])
        if sec is not None and clk_conf > 0.35:
            layout.hud_elapsed_sec = sec
