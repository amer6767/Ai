"""Prioritized Fictitious Self-Play (PFSP) Opponent Sampler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LeagueMember:
    member_id: str
    checkpoint_path: str | None
    elo: float = 1000.0
    games_vs_main: int = 0
    wins_vs_main: int = 0
    role: str = "historical"  # 'historical', 'exploiter', 'native_bot'

    @property
    def main_win_rate(self) -> float:
        if self.games_vs_main == 0:
            return 0.5
        return 1.0 - (self.wins_vs_main / self.games_vs_main)


class PFSPSampler:
    """Samples league opponents inversely proportional to main agent's win rate (hard-matchup focus)."""

    def __init__(self, p_exp: float = 2.0, seed: int = 42) -> None:
        self.p_exp = p_exp
        self.rng = np.random.default_rng(seed)

    def sample(self, members: list[LeagueMember]) -> LeagueMember:
        if not members:
            raise ValueError("Cannot sample from empty league")
        weights = np.array(
            [max(0.05, (1.0 - m.main_win_rate) ** self.p_exp) for m in members],
            dtype=np.float64,
        )
        probs = weights / weights.sum()
        idx = int(self.rng.choice(len(members), p=probs))
        return members[idx]
