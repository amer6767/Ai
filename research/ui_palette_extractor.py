"""Offline UI color palette & layout grammar extractor for Territorial.io (text.html).

ZONE A RESEARCH TOOL ONLY. Never imported by runtime AI packages.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from research.string_table_decoder import (
    extract_string_table,
    load_source,
    substitute_string_table,
)


def extract_ui_palette(html_path: str | Path = "text.html") -> dict[str, Any]:
    raw = load_source(html_path)
    src = substitute_string_table(raw, extract_string_table(raw))

    # Extract named palette entries from c0() (bB)
    m = re.search(r"function c0\(\)\{(.*?)ea=null;\}", src, re.S)
    colors: dict[str, str] = {}
    if m:
        body = m.group(1)
        for cm in re.finditer(r"this\.([a-zA-Z0-9_]+)=ea\.(nn|np)\(([^)]+)\)", body):
            key, fn, args = cm.group(1), cm.group(2), cm.group(3)
            prefix = "rgb" if fn == "nn" else "rgba"
            colors[key] = f"{prefix}({args})"

    return {
        "named_palette_count": len(colors),
        "key_ui_colors": {
            "panel_dark_bg": colors.get("ns", "rgba(0,0,0,0.75)"),
            "panel_darker_bg": colors.get("nr", "rgba(0,0,0,0.85)"),
            "white_border_text": colors.get("nv", "rgb(255,255,255)"),
            "hud_progress_green": colors.get("oF", "rgba(0,255,0,0.6)"),
            "red_interest_hud_text": colors.get("or", "rgb(255,120,100)"),
            "red_interest_balance_text": colors.get("oS", "rgb(255,160,160)"),
            "cycle_bonus_yellow": colors.get("p2", "rgb(255,255,100)"),
            "balance_gain_bg": "rgba(0,100,0,0.8)",
            "balance_loss_bg": "rgba(150,0,0,0.8)",
            "menu_multiplayer_btn": "rgba(22,88,22,0.8)",
            "menu_singleplayer_btn": "rgba(22,88,88,0.8)",
        },
        "slider_color_gradient": {
            "0_to_33_pct": "green (0,130,0) -> yellow (130,130,0)",
            "33_to_66_pct": "yellow (130,130,0) -> red (130,0,0)",
            "66_to_100_pct": "red (130,0,0) -> magenta (130,0,130)",
        },
        "relative_ui_geometry": {
            "main_menu_play_bar_y_ratio": 0.54,
            "main_menu_multiplayer_width_share": 0.60,
            "main_menu_singleplayer_width_share": 0.40,
            "top_center_balance_box": "width ~ 0.24 * min(W,H) desktop, height ~ width/6, 2 vertical white 10-tick cycle bars on sides",
            "top_right_stats_panel": "width ~ 0.126 * min(W,H), height ~ 1.18 * width, green progress bar along bottom edge",
            "bottom_center_troop_slider": "width ~ 0.389 * min(W,H) desktop, height = width/12, [-] and [+] caps = 1.5 * height each",
            "top_left_leaderboard": "semi-transparent dark panel anchored top-left with rank, name, land/troops",
        },
        "canvas_typography": {
            "default_font": "bold sans-serif (Arial Black, sans-serif in logo/titles)",
            "number_abbreviations": ["raw integer < 1000", "K (thousands)", "M (millions)"],
        },
    }


if __name__ == "__main__":
    print(json.dumps(extract_ui_palette(), indent=2))
