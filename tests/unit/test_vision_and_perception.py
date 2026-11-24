"""Unit tests for pure-pixel computer vision, OCR, UI scale invariance, and perception."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.types import BBox, RawFrame, ScreenState
from perception.state_assembler import VisualPerceptionPipeline
from training.pixel_sim_env import SimScenarioConfig, VisualPixelSimEnv
from vision.ocr import parse_clock_string, parse_percent_string, parse_troop_string
from vision.ui_detector import ScaleInvariantUIDetector


def test_ocr_parsers_handle_all_territorial_formats() -> None:
    assert parse_troop_string("927") == 927.0
    assert parse_troop_string("12 450") == 12450.0
    assert parse_troop_string("14.2K") == 14200.0
    assert parse_troop_string("2.05M") == 2050000.0
    assert parse_percent_string("6.77%") == pytest.approx(6.77)
    assert parse_clock_string("2:04") == 124


def test_slider_white_divider_detection_scale_invariant() -> None:
    det = ScaleInvariantUIDetector()
    for (w, h) in ((1280, 720), (1024, 600), (1600, 900)):
        env = VisualPixelSimEnv(SimScenarioConfig(render_w=w, render_h=h, num_bots=3), seed=11, use_fast_perception=True)
        env.reset()
        for target_pct in (0.20, 0.50, 0.80):
            env.slider_pct = target_pct
            rgb = env.render_rgb_screenshot()
            layout = det.detect(rgb)
            assert layout.screen_state == ScreenState.IN_GAME
            assert layout.slider_box is not None
            assert layout.balance_box is not None
            assert abs(layout.estimated_slider_pct - target_pct) < 0.08, (
                f"Resolution {w}x{h}, target {target_pct}, got {layout.estimated_slider_pct}"
            )


def test_perception_pipeline_extracts_entities_and_self_mask() -> None:
    env = VisualPixelSimEnv(SimScenarioConfig(num_bots=5, map_type="ring"), seed=23, use_fast_perception=False)
    env.reset()
    for _ in range(3):
        env.step(1, 0.5, 0.5, 0.30)
    rgb = env.render_rgb_screenshot()
    pipe = VisualPerceptionPipeline()
    pipe.self_id.set_color_hint(env.colors[1])
    raw = RawFrame(rgb=rgb, timestamp=1.0, frame_id=1, window_bbox=BBox(0, 0, rgb.shape[1], rgb.shape[0]))
    pf = pipe.process(raw)
    assert pf.screen_state == ScreenState.IN_GAME
    assert pf.self_entity_id is not None
    assert pf.self_land_px > 0
    assert len(pf.entities) >= 2
    assert pf.water_mask.sum() > 0


def test_golden_real_screenshots_classification_if_present() -> None:
    golden_dir = Path("captures/golden")
    expected = {
        "01_main_menu.png": ScreenState.MAIN_MENU,
        "02_custom_scenario.png": ScreenState.SINGLEPLAYER_LOBBY,
        "04_in_game_live.png": ScreenState.SPAWN_SELECTION,
        "05_in_game_expanded.png": ScreenState.IN_GAME,
    }
    det = ScaleInvariantUIDetector()
    checked = 0
    for fname, exp_state in expected.items():
        p = golden_dir / fname
        if not p.exists():
            continue
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        ui = det.detect(rgb)
        assert ui.screen_state == exp_state, f"{fname}: expected {exp_state}, got {ui.screen_state}"
        if fname == "05_in_game_expanded.png":
            assert ui.slider_box is not None
            assert abs(ui.estimated_slider_pct - 0.50) < 0.08
        checked += 1
    if checked == 0:
        pytest.skip("Golden screenshots not generated in this workspace")
