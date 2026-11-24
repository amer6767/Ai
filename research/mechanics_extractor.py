"""Offline mechanics extractor for Territorial.io (text.html).

ZONE A RESEARCH TOOL ONLY. Never imported by runtime AI packages.
Extracts verified game constants, bot difficulty tables, interest math,
combat equations, and hotkey defaults directly from text.html.
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


def extract_mechanics_summary(html_path: str | Path = "text.html") -> dict[str, Any]:
    raw = load_source(html_path)
    table = extract_string_table(raw)
    src = substitute_string_table(raw, table)

    def _find_int(pattern: str, default: int | None = None) -> int | None:
        m = re.search(pattern, src)
        return int(m.group(1)) if m else default

    def _find_int_list(pattern: str) -> list[int]:
        m = re.search(pattern, src)
        if not m:
            return []
        return [int(x.strip()) for x in m.group(1).split(",") if x.strip().lstrip("-").isdigit()]

    def _find_str_list(pattern: str) -> list[str]:
        m = re.search(pattern, src, re.S)
        if not m:
            return []
        return json.loads("[" + m.group(1) + "]")

    tick_ms = _find_int(r"this\.aCD=(\d+)", 56)
    max_players = _find_int(r"this\.ez=(\d+)", 512)
    max_troops = _find_int(r"this\.a5G=(\d+)", 1_000_000_000)

    # Bot difficulty vectors from c5() (aE)
    bot_table = {
        "difficulties": ["Very Easy", "Easy", "Normal", "Hard", "Very Hard", "Impossible", "H Bot"],
        "kH_passivity": _find_int_list(r"this\.kH=\[([0-9,\s]+)\]"),
        "kY_boat_threshold": _find_int_list(r"this\.kY=\[([0-9,\s]+)\]"),
        "kF_aggression": _find_int_list(r"this\.kF=\[([0-9,\s]+)\]"),
        "kZ_search_budget": _find_int_list(r"this\.kZ=\[([0-9,\s]+)\]"),
        "ka_action_tier": _find_int_list(r"this\.ka=\[([0-9,\s]+)\]"),
        "kb_mountain_pct": _find_int_list(r"this\.kb=\[([0-9,\s]+)\]"),
        "kc_intercept_prob_per_1000": _find_int_list(r"this\.kc=\[([0-9,\s]+)\]"),
    }

    # Hotkey table from dg() (bX)
    raw_keys = _find_str_list(r"var aaD=\[(.*?)\];")
    hotkey_names = [
        "attack_at_mouse",
        "launch_ship_at_mouse",
        "increase_slider",
        "decrease_slider",
        "slightly_increase_slider",
        "slightly_decrease_slider",
        "toggle_ui",
        "zoom_in",
        "zoom_out",
        "camera_left",
        "camera_right",
        "camera_up",
        "camera_down",
        "add_to_weakest_attack",
        "call_peace_vote",
        "intercept_ship_at_mouse",
    ]
    hotkeys = {}
    for i, name in enumerate(hotkey_names):
        if 2 * i < len(raw_keys):
            hotkeys[name] = raw_keys[2 * i]

    return {
        "source_file": str(html_path),
        "tick_interval_ms": tick_ms,
        "ticks_per_second": round(1000.0 / (tick_ms or 56), 2),
        "interest_payout_every_ticks": 10,
        "land_income_every_ticks": 100,
        "opening_interest_decay_ticks": 1920,
        "opening_interest_start_pct": 7.0,
        "opening_interest_end_pct": 1.0,
        "soft_troop_cap_per_pixel": 100,
        "hard_troop_cap_per_pixel": 150,
        "max_players_capacity": max_players,
        "max_troops_absolute": max_troops,
        "combat_laws": {
            "neutral_cost_formula": "fx * gL (border pixels * base pixel tax)",
            "defender_resistance_formula": "gv = floor(fx * defender_troops / (1 + floor(10 * defender_land / 16)))",
            "occupied_attacker_cost_formula": "gz = fx * gL + 2 * gv + counter_attack_troops",
            "defender_troop_loss_formula": "gv (after canceling counter_attack_troops 1-for-1)",
            "defender_efficiency_ratio": "2:1 attacker-to-defender casualty ratio plus border tax",
        },
        "bot_difficulty_table": bot_table,
        "default_hotkeys": hotkeys,
    }


if __name__ == "__main__":
    summary = extract_mechanics_summary()
    print(json.dumps(summary, indent=2))
