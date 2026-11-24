"""On-policy recurrent PPO rollout buffer with GAE and prioritized auxiliary experience replay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from training.vec_env import BatchedObservation


class RolloutBuffer:
    """Stores T steps across B parallel environments and computes GAE(lambda) returns."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_size: int = 128,
        max_nodes: int = 32,
        spatial_cells: int = 1024,
        hidden_dim: int = 192,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_map = np.zeros((num_steps, num_envs, 11, obs_size, obs_size), dtype=np.float32)
        self.node_feats = np.zeros((num_steps, num_envs, max_nodes, 16), dtype=np.float32)
        self.adj = np.zeros((num_steps, num_envs, max_nodes, max_nodes), dtype=np.float32)
        self.node_mask = np.zeros((num_steps, num_envs, max_nodes), dtype=bool)
        self.scalars = np.zeros((num_steps, num_envs, 16), dtype=np.float32)
        self.action_mask = np.zeros((num_steps, num_envs, 5), dtype=bool)
        self.spatial_masks = np.zeros((num_steps, num_envs, 5, spatial_cells), dtype=bool)

        self.action_type = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.spatial_idx = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.slider_idx = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.land_targets = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.init_hidden = np.zeros((1, num_envs, hidden_dim), dtype=np.float32)

        self.advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.ptr = 0

    def reset(self, init_hidden: torch.Tensor) -> None:
        self.ptr = 0
        self.init_hidden = init_hidden.detach().cpu().numpy().astype(np.float32)

    def add(
        self,
        obs: BatchedObservation,
        action_type: np.ndarray,
        spatial_idx: np.ndarray,
        slider_idx: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        land_ratio: np.ndarray,
    ) -> None:
        t = self.ptr
        self.obs_map[t] = obs.obs_map
        self.node_feats[t] = obs.node_feats
        self.adj[t] = obs.adj
        self.node_mask[t] = obs.node_mask
        self.scalars[t] = obs.scalars
        self.action_mask[t] = obs.action_mask
        self.spatial_masks[t] = obs.spatial_masks

        self.action_type[t] = action_type
        self.spatial_idx[t] = spatial_idx
        self.slider_idx[t] = slider_idx
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done.astype(np.float32)
        self.land_targets[t] = land_ratio.astype(np.float32)
        self.ptr += 1

    def compute_gae(self, last_values: np.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.ptr)):
            next_val = last_values if t == self.ptr - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]


@dataclass
class PrioritySample:
    obs_map: np.ndarray
    node_feats: np.ndarray
    adj: np.ndarray
    node_mask: np.ndarray
    scalars: np.ndarray
    target_value: float
    target_land: float
    priority: float


class PrioritizedAuxiliaryReplay:
    """Off-policy prioritized replay for auxiliary value & representation distillation (PPO-compatible)."""

    def __init__(self, capacity: int = 2048) -> None:
        self._buf: deque[PrioritySample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, sample: PrioritySample) -> None:
        self._buf.append(sample)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[PrioritySample]:
        n = len(self._buf)
        if n == 0:
            return []
        prios = np.array([max(1e-3, s.priority) for s in self._buf], dtype=np.float64)
        probs = prios / prios.sum()
        k = min(batch_size, n)
        idxs = rng.choice(n, size=k, replace=False, p=probs)
        items = list(self._buf)
        return [items[int(i)] for i in idxs]
