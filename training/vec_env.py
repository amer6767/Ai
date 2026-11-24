"""Vectorized visual environment runner supporting batched multi-scenario rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from map_analysis.tactical_features import TacticalObservation
from training.pixel_sim_env import SimScenarioConfig, VisualPixelSimEnv


@dataclass
class BatchedObservation:
    obs_map: np.ndarray        # [B, 11, 128, 128]
    node_feats: np.ndarray     # [B, 32, 16]
    adj: np.ndarray            # [B, 32, 32]
    node_mask: np.ndarray      # [B, 32]
    scalars: np.ndarray        # [B, 16]
    action_mask: np.ndarray    # [B, 5]
    spatial_masks: np.ndarray  # [B, 5, 1024]


def stack_observations(obs_list: list[TacticalObservation]) -> BatchedObservation:
    return BatchedObservation(
        obs_map=np.stack([o.obs_map for o in obs_list], axis=0),
        node_feats=np.stack([o.node_feats for o in obs_list], axis=0),
        adj=np.stack([o.adj for o in obs_list], axis=0),
        node_mask=np.stack([o.node_mask for o in obs_list], axis=0),
        scalars=np.stack([o.scalars for o in obs_list], axis=0),
        action_mask=np.stack([o.action_mask for o in obs_list], axis=0),
        spatial_masks=np.stack([o.spatial_masks for o in obs_list], axis=0),
    )


class VectorizedPixelEnv:
    """Runs N diverse pixel-rendered Territorial.io environments synchronously with auto-reset."""

    def __init__(
        self,
        scenarios: list[SimScenarioConfig],
        base_seed: int = 42,
        use_fast_perception: bool = True,
        opponent_policy_fn: Any = None,
    ) -> None:
        self.envs = [
            VisualPixelSimEnv(
                scenario=sc,
                seed=base_seed + idx * 101,
                use_fast_perception=use_fast_perception,
                opponent_policy_fn=opponent_policy_fn,
            )
            for idx, sc in enumerate(scenarios)
        ]
        self.num_envs = len(self.envs)

    def reset(self) -> BatchedObservation:
        obs_list = [env.reset()[0] for env in self.envs]
        return stack_observations(obs_list)

    def step(
        self,
        action_types: np.ndarray,
        target_xs: np.ndarray,
        target_ys: np.ndarray,
        slider_pcts: np.ndarray,
    ) -> tuple[BatchedObservation, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs_list: list[TacticalObservation] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            obs, rew, term, trunc, info = env.step(
                int(action_types[i]),
                float(target_xs[i]),
                float(target_ys[i]),
                float(slider_pcts[i]),
            )
            done = bool(term or trunc)
            rewards[i] = rew
            dones[i] = done
            if done:
                info["terminal"] = True
                obs, _ = env.reset()
            obs_list.append(obs)
            infos.append(info)

        return stack_observations(obs_list), rewards, dones, infos
