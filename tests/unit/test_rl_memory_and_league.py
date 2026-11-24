"""Unit tests for RL PPO updates, persistent SQLite/Parquet memory, mistake detection, Elo, gating, and dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

from core.config import RLConfig
from dashboard.server import app
from evaluation.gating import should_promote_champion
from evaluation.metrics import compute_evaluation_report
from memory.checkpoint_manager import CheckpointManager
from memory.episode_store import PersistentMemoryStore, StepRecord
from memory.mistake_analyzer import MistakeAnalyzer
from models.policy_value_net import TerritorialPolicyValueNet
from reinforcement_learning.ppo import RecurrentPPOTrainer
from reinforcement_learning.replay_buffer import RolloutBuffer
from self_play.league_manager import SelfPlayLeagueManager
from training.pixel_sim_env import SimScenarioConfig
from training.vec_env import VectorizedPixelEnv


def test_ppo_collect_and_gradient_update_runs_cleanly() -> None:
    scenarios = [SimScenarioConfig(num_bots=3, max_steps=30, map_type="continent") for _ in range(2)]
    vec_env = VectorizedPixelEnv(scenarios, base_seed=5, use_fast_perception=True)
    model = TerritorialPolicyValueNet(hidden_dim=96, spatial_grid=32)
    trainer = RecurrentPPOTrainer(model, rl_cfg=RLConfig(rollout_steps=8, num_envs=2, update_epochs=2, hidden_dim=96, spatial_grid=32))
    buf = RolloutBuffer(num_steps=8, num_envs=2, spatial_cells=1024, hidden_dim=96)

    obs = vec_env.reset()
    hidden = model.rnn.initial_state(2, trainer.device)
    obs, hidden, _ = trainer.collect_rollout(vec_env, obs, hidden, buf)
    metrics = trainer.update(buf)
    assert np.isfinite(metrics.policy_loss)
    assert np.isfinite(metrics.value_loss)
    assert metrics.entropy > 0.0


def test_memory_persistence_mistakes_and_atomic_checkpoints(tmp_path: Path) -> None:
    db_path = tmp_path / "mem.db"
    ep_dir = tmp_path / "eps"
    ckpt_dir = tmp_path / "ckpts"

    store = PersistentMemoryStore(db_path=db_path, episodes_dir=ep_dir)
    ep_id = store.start_episode("pixel_sim", curriculum_stage=4, opponent_tier="Normal", checkpoint_id="c1")

    steps = [
        StepRecord(i, float(i), "IN_GAME", "WAIT" if i < 5 else "ATTACK_ENEMY", 0.5, 0.5, 0.70, i != 2, "ok", 1000.0, 0.25 if i < 6 else 0.10, True, -0.05 if i < 6 else -1.5, 0.5, "test")
        for i in range(12)
    ]
    pq_path = store.finish_episode(ep_id, steps, {"won": False, "peak_land_ratio": 0.25, "decision_accuracy": 0.9})
    assert pq_path is not None and pq_path.exists()

    mistakes = MistakeAnalyzer().analyze_episode(steps)
    assert len(mistakes) >= 2
    store.record_mistakes(ep_id, mistakes)
    assert len(store.recent_episodes()) == 1
    assert len(store.recent_mistakes()) >= 2

    model = TerritorialPolicyValueNet(hidden_dim=64, spatial_grid=16)
    mgr = CheckpointManager(checkpoints_dir=ckpt_dir, db_path=db_path)
    cid, path = mgr.save(model, None, curriculum_stage=4, global_steps=500, is_champion=True)
    assert path.exists()

    model2 = TerritorialPolicyValueNet(hidden_dim=64, spatial_grid=16)
    loaded = mgr.load_latest(model2, prefer_champion=True)
    assert loaded is not None and loaded["checkpoint_id"] == cid


def test_self_play_league_and_champion_gating() -> None:
    league = SelfPlayLeagueManager()
    opp = league.select_opponent()
    new_elo = league.record_match(opp.member_id, main_won=True, main_placement_pct=1.0)
    assert new_elo > 1000.0

    rep_a = compute_evaluation_report([{"won": False, "rank": 4, "total_players": 6, "peak_land_ratio": 0.15, "decision_accuracy": 0.8}] * 4).to_dict()
    rep_b = compute_evaluation_report([{"won": True, "rank": 1, "total_players": 6, "peak_land_ratio": 0.82, "decision_accuracy": 0.92}] * 4).to_dict()
    promote, _ = should_promote_champion(rep_b, rep_a)
    assert promote is True


def test_dashboard_api_health_and_status() -> None:
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert "hardware" in body and "curriculum" in body
