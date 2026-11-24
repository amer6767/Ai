"""Automated multi-difficulty Benchmark Suite (Visual Pixel Arena + Live Unmodified Chrome Arena)."""

from __future__ import annotations

from typing import Any

import torch

from evaluation.metrics import EvaluationReport, compute_evaluation_report
from models.policy_value_net import SLIDER_BINS, TerritorialPolicyValueNet
from training.live_browser_env import LiveX11BrowserEnv
from training.pixel_sim_env import SimScenarioConfig, VisualPixelSimEnv


def evaluate_policy_visual_arena(
    model: TerritorialPolicyValueNet,
    difficulties: tuple[str, ...] = ("Very Easy", "Easy", "Normal", "Hard"),
    episodes_per_tier: int = 3,
    device: str | torch.device = "cpu",
    base_seed: int = 7000,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Benchmark checkpoint across standardized unseen seeds, maps, and escalating bot difficulties."""
    dev = torch.device(device)
    model.to(dev).eval()
    map_types = ("continent", "ring", "islands", "archipelago")
    all_episodes: list[dict[str, Any]] = []
    per_tier: dict[str, dict[str, Any]] = {}
    g = model.spatial_grid

    for tier_idx, diff in enumerate(difficulties):
        tier_eps: list[dict[str, Any]] = []
        for ep in range(episodes_per_tier):
            sc = SimScenarioConfig(
                grid_size=48,
                render_w=384,
                render_h=256,
                num_bots=3 + (tier_idx % 4),
                bot_difficulty=diff,
                map_type=map_types[(tier_idx + ep) % len(map_types)],
                max_steps=95,
                domain_randomize=False,
            )
            env = VisualPixelSimEnv(scenario=sc, seed=base_seed + tier_idx * 100 + ep, use_fast_perception=True)
            obs, _ = env.reset()
            hidden = model.rnn.initial_state(1, dev)
            last_info: dict[str, Any] = {}

            for _ in range(sc.max_steps):
                with torch.no_grad():
                    out = model.forward_step(
                        obs_map=torch.from_numpy(obs.obs_map).unsqueeze(0).to(dev),
                        node_feats=torch.from_numpy(obs.node_feats).unsqueeze(0).to(dev),
                        adj=torch.from_numpy(obs.adj).unsqueeze(0).to(dev),
                        node_mask=torch.from_numpy(obs.node_mask).unsqueeze(0).to(dev),
                        scalars=torch.from_numpy(obs.scalars).unsqueeze(0).to(dev),
                        hidden=hidden,
                        action_mask=torch.from_numpy(obs.action_mask).unsqueeze(0).to(dev),
                        spatial_masks=torch.from_numpy(obs.spatial_masks).unsqueeze(0).to(dev),
                        deterministic=deterministic,
                    )
                hidden = out.next_hidden
                a_idx = int(out.action_type.item())
                sp_idx = int(out.spatial_idx.item())
                sl_idx = int(out.slider_idx.item())
                gy, gx = divmod(sp_idx, g)
                obs, _, term, trunc, last_info = env.step(
                    a_idx,
                    (gx + 0.5) / g,
                    (gy + 0.5) / g,
                    SLIDER_BINS[sl_idx],
                )
                if term or trunc:
                    break
            tier_eps.append(last_info)
            all_episodes.append(last_info)

        per_tier[diff] = compute_evaluation_report(tier_eps).to_dict()

    overall = compute_evaluation_report(all_episodes)
    return {"overall": overall.to_dict(), "by_difficulty": per_tier, "raw_episodes": all_episodes}


def evaluate_policy_live_browser(
    model: TerritorialPolicyValueNet,
    num_matches: int = 2,
    max_steps: int = 45,
    display_num: int = 120,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run real autonomous matches against live Territorial.io client via Xvfb + Chrome + X11 mouse."""
    dev = torch.device(device)
    model.to(dev).eval()
    env = LiveX11BrowserEnv(display_num=display_num, max_steps=max_steps, step_interval_sec=0.22)
    episodes: list[dict[str, Any]] = []
    g = model.spatial_grid

    try:
        for _ in range(num_matches):
            obs, reset_info = env.reset()
            hidden = model.rnn.initial_state(1, dev)
            last_info: dict[str, Any] = dict(reset_info)
            for _ in range(max_steps):
                with torch.no_grad():
                    out = model.forward_step(
                        obs_map=torch.from_numpy(obs.obs_map).unsqueeze(0).to(dev),
                        node_feats=torch.from_numpy(obs.node_feats).unsqueeze(0).to(dev),
                        adj=torch.from_numpy(obs.adj).unsqueeze(0).to(dev),
                        node_mask=torch.from_numpy(obs.node_mask).unsqueeze(0).to(dev),
                        scalars=torch.from_numpy(obs.scalars).unsqueeze(0).to(dev),
                        hidden=hidden,
                        action_mask=torch.from_numpy(obs.action_mask).unsqueeze(0).to(dev),
                        spatial_masks=torch.from_numpy(obs.spatial_masks).unsqueeze(0).to(dev),
                        deterministic=True,
                    )
                hidden = out.next_hidden
                a_idx = int(out.action_type.item())
                sp_idx = int(out.spatial_idx.item())
                sl_idx = int(out.slider_idx.item())
                gy, gx = divmod(sp_idx, g)
                obs, _, term, trunc, last_info = env.step(
                    a_idx,
                    (gx + 0.5) / g,
                    (gy + 0.5) / g,
                    SLIDER_BINS[sl_idx],
                )
                if term or trunc:
                    break
            episodes.append(last_info)
    finally:
        env.close()

    report: EvaluationReport = compute_evaluation_report(episodes)
    return {"overall": report.to_dict(), "episodes": episodes}
