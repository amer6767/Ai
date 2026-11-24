"""Imitation Learning (Behavioral Cloning + DAgger warm-start) strictly from visual observations.

Generates expert demonstrations using the pure-CV `ForwardStrategicPlanner` on `VisualPixelSimEnv`
so the RL actor initializes with sensible neutral expansion, slider discipline, and 2:1 combat
awareness before pure RL self-improvement takes over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.types import ActionType
from models.policy_value_net import SLIDER_BINS, TerritorialPolicyValueNet
from strategic_brain.candidate_generator import CandidateActionGenerator
from strategic_brain.economy_engine import EconomyEngine
from strategic_brain.forward_planner import ForwardStrategicPlanner
from strategic_brain.opponent_modeler import OpponentModeler
from training.pixel_sim_env import SimScenarioConfig, VisualPixelSimEnv

_ATYPE_TO_RL = {
    ActionType.WAIT: 0,
    ActionType.EXPAND_NEUTRAL: 1,
    ActionType.ATTACK_ENEMY: 2,
    ActionType.LAUNCH_BOAT: 3,
    ActionType.PEACE_VOTE: 4,
}


@dataclass
class ImitationSample:
    obs_map: np.ndarray
    node_feats: np.ndarray
    adj: np.ndarray
    node_mask: np.ndarray
    scalars: np.ndarray
    action_type: int
    spatial_idx: int
    slider_idx: int


def _nearest_slider_bin(pct: float) -> int:
    arr = np.asarray(SLIDER_BINS, dtype=np.float32)
    return int(np.argmin(np.abs(arr - pct)))


def collect_visual_planner_demos(
    num_episodes: int = 8,
    max_steps: int = 60,
    seed: int = 100,
    spatial_grid: int = 32,
) -> list[ImitationSample]:
    """Collect demonstrations where the demonstrator sees ONLY PerceivedFrame (no internal sim access)."""
    samples: list[ImitationSample] = []
    econ_eng = EconomyEngine()
    opp_mod = OpponentModeler()
    cand_gen = CandidateActionGenerator()
    planner = ForwardStrategicPlanner()

    map_types = ["continent", "ring", "islands", "archipelago"]
    for ep in range(num_episodes):
        sc = SimScenarioConfig(
            num_bots=3 + (ep % 5),
            bot_difficulty="Normal" if ep % 2 == 0 else "Easy",
            map_type=map_types[ep % len(map_types)],
            max_steps=max_steps,
        )
        env = VisualPixelSimEnv(scenario=sc, seed=seed + ep, use_fast_perception=True)
        obs, _ = env.reset()
        opp_mod.reset()

        for _ in range(max_steps):
            pf = env._last_pf
            graph = env._last_graph
            assert pf is not None and graph is not None
            econ = econ_eng.assess(pf, graph.max_neighbor_threat)
            opps = opp_mod.update(graph.entities, self_losing_troops=not pf.ui.balance_gaining)
            cands = cand_gen.generate(pf, graph, env.world_map, econ)
            scored = planner.score_candidates(cands, pf, graph, econ, opps)

            chosen = scored[0].action if scored else None
            a_idx = _ATYPE_TO_RL.get(chosen.action_type, 0) if chosen else 0
            if not obs.action_mask[a_idx]:
                a_idx = 1 if obs.action_mask[1] else 0

            tx = chosen.target_x_norm if chosen else 0.5
            ty = chosen.target_y_norm if chosen else 0.5
            pct = chosen.slider_pct if chosen else 0.25

            gx = min(spatial_grid - 1, max(0, int(tx * spatial_grid)))
            gy = min(spatial_grid - 1, max(0, int(ty * spatial_grid)))
            sp_idx = gy * spatial_grid + gx
            # Snap to nearest valid masked cell if needed
            if a_idx in (1, 2, 3) and not obs.spatial_masks[a_idx, sp_idx]:
                valid = np.where(obs.spatial_masks[a_idx])[0]
                if len(valid) > 0:
                    sp_idx = int(valid[0])
                    gy, gx = divmod(sp_idx, spatial_grid)
                    tx = (gx + 0.5) / spatial_grid
                    ty = (gy + 0.5) / spatial_grid

            sl_idx = _nearest_slider_bin(pct)
            samples.append(
                ImitationSample(
                    obs_map=obs.obs_map.copy(),
                    node_feats=obs.node_feats.copy(),
                    adj=obs.adj.copy(),
                    node_mask=obs.node_mask.copy(),
                    scalars=obs.scalars.copy(),
                    action_type=a_idx,
                    spatial_idx=sp_idx,
                    slider_idx=sl_idx,
                )
            )
            obs, _, term, trunc, _ = env.step(a_idx, tx, ty, SLIDER_BINS[sl_idx])
            if term or trunc:
                break
    return samples


def train_behavioral_cloning(
    model: TerritorialPolicyValueNet,
    samples: list[ImitationSample],
    epochs: int = 6,
    batch_size: int = 32,
    lr: float = 8e-4,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Supervised Behavioral Cloning warm-start over visual demonstrations."""
    dev = torch.device(device)
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(7)
    n = len(samples)
    if n == 0:
        return {"bc_loss": 0.0, "type_acc": 0.0}

    last_loss = 0.0
    last_acc = 0.0
    for _ in range(epochs):
        order = rng.permutation(n)
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for start in range(0, n, batch_size):
            idxs = order[start : start + batch_size]
            batch = [samples[int(i)] for i in idxs]
            obs_map = torch.from_numpy(np.stack([s.obs_map for s in batch])).to(dev)
            nf = torch.from_numpy(np.stack([s.node_feats for s in batch])).to(dev)
            adj = torch.from_numpy(np.stack([s.adj for s in batch])).to(dev)
            nm = torch.from_numpy(np.stack([s.node_mask for s in batch])).to(dev)
            sc = torch.from_numpy(np.stack([s.scalars for s in batch])).to(dev)
            y_type = torch.tensor([s.action_type for s in batch], dtype=torch.long, device=dev)
            y_sp = torch.tensor([s.spatial_idx for s in batch], dtype=torch.long, device=dev)
            y_sl = torch.tensor([s.slider_idx for s in batch], dtype=torch.long, device=dev)

            out = model.forward_step(
                obs_map, nf, adj, nm, sc, given_actions=(y_type, y_sp, y_sl)
            )
            loss_type = F.cross_entropy(out.type_logits, y_type)
            use_sp = ((y_type == 1) | (y_type == 2) | (y_type == 3)).float()
            loss_sp = (F.cross_entropy(out.spatial_logits, y_sp, reduction="none") * use_sp).sum() / (use_sp.sum() + 1e-6)
            loss_sl = (F.cross_entropy(out.slider_logits, y_sl, reduction="none") * use_sp).sum() / (use_sp.sum() + 1e-6)
            loss = loss_type + 0.5 * loss_sp + 0.35 * loss_sl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += float(loss.item()) * len(batch)
            ep_correct += int((out.type_logits.argmax(dim=-1) == y_type).sum().item())
            ep_total += len(batch)
        last_loss = ep_loss / max(1, ep_total)
        last_acc = ep_correct / max(1, ep_total)

    return {"bc_loss": round(last_loss, 4), "type_acc": round(last_acc, 4), "num_samples": n}
