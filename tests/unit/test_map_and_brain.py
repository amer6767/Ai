"""Unit tests for camera odometry, persistent world map, territory graph, strategic planner, and neural models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.types import ActionDecision, ActionType, BBox, PerceivedFrame, ScreenState, TerritoryEntity, UILayout
from map_analysis.camera_odometry import VisualCameraOdometry
from map_analysis.territory_graph import TerritoryGraphBuilder
from map_analysis.world_map import PersistentWorldMap
from models.policy_value_net import TerritorialPolicyValueNet
from strategic_brain.economy_engine import EconomyEngine
from strategic_brain.forward_planner import ForwardStrategicPlanner
from strategic_brain.opponent_modeler import OpponentProfile


def test_camera_odometry_recovers_translation() -> None:
    odom = VisualCameraOdometry(work_size=128)
    grid = np.zeros((128, 128), dtype=np.uint8)
    grid[30:90, 40:100] = 1
    grid[50:70, 60:80] = 0
    odom.update(grid, np.zeros((128, 128), dtype=np.int32))
    shifted = np.roll(np.roll(grid, shift=5, axis=0), shift=-7, axis=1)
    pose = odom.update(shifted, np.zeros((128, 128), dtype=np.int32))
    assert abs(pose.last_dx - (7.0 / 128.0)) < 0.025
    assert abs(pose.last_dy - (-5.0 / 128.0)) < 0.025


def test_forward_planner_avoids_suicidal_attack_and_punishes_vulture_target() -> None:
    lm = np.zeros((64, 64), dtype=np.int32)
    lm[20:40, 20:32] = 1  # self
    lm[20:40, 32:44] = 2  # strong healthy neighbor
    lm[10:20, 20:32] = 3  # weak collapsed neighbor

    self_ent = TerritoryEntity(1, True, False, (50, 150, 50), 240, (0.4, 0.45), BBox(20, 20, 12, 20), 1000.0, 0.95)
    strong_ent = TerritoryEntity(2, False, False, (200, 40, 40), 240, (0.6, 0.45), BBox(32, 20, 12, 20), 2400.0, 0.9, border_contact_with_self_px=20, attack_point_norm=(0.55, 0.45))
    weak_ent = TerritoryEntity(3, False, False, (40, 40, 200), 120, (0.4, 0.25), BBox(20, 10, 12, 10), 120.0, 0.9, border_contact_with_self_px=12, is_under_attack=True, attack_point_norm=(0.4, 0.25))

    pf = PerceivedFrame(
        frame_id=1,
        timestamp=1.0,
        width=640,
        height=480,
        screen_state=ScreenState.IN_GAME,
        ui=UILayout(screen_state=ScreenState.IN_GAME, hud_troops=1000, hud_land=240),
        label_map=lm,
        self_mask=(lm == 1).astype(np.uint8),
        neutral_mask=(lm == 0).astype(np.uint8),
        water_mask=np.zeros((64, 64), dtype=np.uint8),
        combat_delta_map=np.zeros((64, 64), dtype=np.float32),
        entities=[self_ent, strong_ent, weak_ent],
        self_entity_id=1,
        self_troops=1000.0,
        self_land_px=240,
        self_land_ratio=0.1,
    )
    graph = TerritoryGraphBuilder().build(pf)
    econ = EconomyEngine().assess(pf, graph.max_neighbor_threat)
    planner = ForwardStrategicPlanner()
    cands = [
        ActionDecision(ActionType.WAIT),
        ActionDecision(ActionType.ATTACK_ENEMY, target_entity_id=2, slider_pct=0.50),
        ActionDecision(ActionType.ATTACK_ENEMY, target_entity_id=3, slider_pct=0.25),
    ]
    profiles = {3: OpponentProfile((2, 2, 12), vulnerability_score=0.9)}
    scored = planner.score_candidates(cands, pf, graph, econ, profiles)
    assert scored[0].action.target_entity_id == 3, "Planner should prioritize vulturing weak neighbor #3"
    strong_score = next(s.strategic_score for s in scored if s.action.target_entity_id == 2)
    wait_score = next(s.strategic_score for s in scored if s.action.action_type == ActionType.WAIT)
    assert strong_score < wait_score, "Planner must reject suicidal 2:1 attack against stronger healthy neighbor"


def test_policy_value_network_forward_and_onnx_export(tmp_path: Path) -> None:
    net = TerritorialPolicyValueNet(hidden_dim=128, spatial_grid=16)
    b = 2
    obs_map = torch.randn(b, 11, 128, 128)
    nf = torch.randn(b, 32, 16)
    adj = torch.eye(32).unsqueeze(0).repeat(b, 1, 1)
    nm = torch.ones(b, 32, dtype=torch.bool)
    sc = torch.randn(b, 16)
    am = torch.tensor([[True, True, False, False, False], [True, True, True, False, False]])
    sm = torch.ones(b, 5, 256, dtype=torch.bool)

    out = net.forward_step(obs_map, nf, adj, nm, sc, action_mask=am, spatial_masks=sm)
    assert out.action_type.shape == (b,)
    assert out.log_prob.shape == (b,)
    assert out.value.shape == (b,)
    assert out.next_hidden.shape == (1, b, 128)
    assert int(out.action_type[0]) in (0, 1)

    onnx_out = net.export_onnx(tmp_path / "test_policy.onnx")
    assert onnx_out.exists() and onnx_out.stat().st_size > 10_000
