"""Integration tests for ActionVerifier suppression, Kaggle GPU bundling, and GitHub Pro workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from computer_control.action_verifier import ActionVerifier
from core.types import ActionDecision, ActionType, BBox, PerceivedFrame, RawFrame, ScreenState, UILayout
from training.cloud.kaggle_runner import KaggleGPUOrchestrator


def test_action_verifier_suppresses_repeated_failed_clicks_and_recovers() -> None:
    verifier = ActionVerifier(cooldown_steps=4, grid_bins=16)
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    raw = RawFrame(rgb=img, timestamp=0.0, frame_id=1, window_bbox=BBox(0, 0, 300, 200))
    pf = PerceivedFrame(
        frame_id=1,
        timestamp=0.0,
        width=300,
        height=200,
        screen_state=ScreenState.IN_GAME,
        ui=UILayout(screen_state=ScreenState.IN_GAME),
        label_map=np.zeros((64, 64), dtype=np.int32),
        self_mask=np.zeros((64, 64), dtype=np.uint8),
        neutral_mask=np.zeros((64, 64), dtype=np.uint8),
        water_mask=np.zeros((64, 64), dtype=np.uint8),
        combat_delta_map=np.zeros((64, 64), dtype=np.float32),
    )
    act = ActionDecision(ActionType.ATTACK_ENEMY, target_x_norm=0.25, target_y_norm=0.25, slider_pct=0.30)

    # First identical-frame click fails but isn't cooled down yet
    r1 = verifier.verify(act, raw, raw, pf, pf, latency_ms=25.0)
    assert r1.succeeded is False
    assert verifier.is_suppressed(ActionType.ATTACK_ENEMY, 0.25, 0.25) is False

    # Second consecutive failure on same quantized region activates cooldown mask!
    r2 = verifier.verify(act, raw, raw, pf, pf, latency_ms=25.0)
    assert r2.succeeded is False
    assert verifier.is_suppressed(ActionType.ATTACK_ENEMY, 0.25, 0.25) is True

    mask = verifier.suppression_mask(ActionType.ATTACK_ENEMY, 32, 32)
    assert not bool(mask[8, 8])
    assert bool(mask[24, 24])

    # Cooldown expires cleanly after 4 ticks
    for _ in range(4):
        verifier.tick()
    assert verifier.is_suppressed(ActionType.ATTACK_ENEMY, 0.25, 0.25) is False


def test_kaggle_gpu_kernel_bundle_generation(tmp_path: Path) -> None:
    orch = KaggleGPUOrchestrator(username="demo_user", kernel_slug="tio-cv-ai")
    bundle = orch.build_kernel_bundle(out_dir=tmp_path / "kg_bundle", ppo_updates_per_stage=12)
    meta_path = bundle / "kernel-metadata.json"
    script_path = bundle / "run_kaggle_gpu.py"
    assert meta_path.exists() and script_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["enable_gpu"] == "true"
    assert meta["id"] == "demo_user/tio-cv-ai"
    assert "run_full_9_stage_training" in script_path.read_text(encoding="utf-8")


def test_github_actions_workflows_valid_yaml() -> None:
    wf_dir = Path(".github/workflows")
    for fname in ("ci_firewall_and_tests.yml", "eval_arena_matrix.yml"):
        p = wf_dir / fname
        assert p.exists(), f"Missing workflow {p}"
        parsed = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert "jobs" in parsed and len(parsed["jobs"]) >= 1
