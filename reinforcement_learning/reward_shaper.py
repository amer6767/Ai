"""Potential-based reward shaping aligned with Territorial.io survival, economy, and win mastery."""

from __future__ import annotations

import math


class RewardShaper:
    """Computes dense potential-based shaping + sparse win/placement terminal rewards."""

    def __init__(
        self,
        w_land: float = 4.5,
        w_econ: float = 0.28,
        w_red_penalty: float = 0.06,
        w_fail_penalty: float = 0.05,
        w_survival: float = 0.008,
        win_bonus: float = 8.0,
        loss_penalty: float = -3.0,
    ) -> None:
        self.w_land = w_land
        self.w_econ = w_econ
        self.w_red_penalty = w_red_penalty
        self.w_fail_penalty = w_fail_penalty
        self.w_survival = w_survival
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty
        self._prev_land_pot: float | None = None
        self._prev_econ_pot: float | None = None

    def reset(self) -> None:
        self._prev_land_pot = None
        self._prev_econ_pot = None

    def compute(
        self,
        land_ratio: float,
        troops: float,
        soft_cap: float,
        red_interest: bool,
        action_succeeded: bool,
        terminated: bool,
        truncated: bool,
        won: bool,
        rank: int,
        total_players: int,
    ) -> float:
        land_pot = math.log1p(max(0.0, land_ratio) * 100.0)
        eff_troops = min(max(0.0, troops), max(100.0, soft_cap))
        econ_pot = math.log1p(eff_troops)

        if self._prev_land_pot is None:
            d_land = 0.0
            d_econ = 0.0
        else:
            d_land = land_pot - self._prev_land_pot
            d_econ = econ_pot - (self._prev_econ_pot or 0.0)

        self._prev_land_pot = land_pot
        self._prev_econ_pot = econ_pot

        reward = self.w_land * d_land + self.w_econ * d_econ + self.w_survival
        if red_interest:
            reward -= self.w_red_penalty
        if not action_succeeded:
            reward -= self.w_fail_penalty

        if terminated or truncated:
            denom = max(1, total_players - 1)
            placement_pct = max(0.0, min(1.0, 1.0 - (max(1, rank) - 1) / denom))
            if won:
                reward += self.win_bonus
            elif terminated:
                reward += self.loss_penalty * (1.0 - 0.65 * placement_pct)
            else:
                # Truncated alive -> reward based on placement & controlled land
                reward += 2.5 * placement_pct + 3.0 * land_ratio

        return float(reward)
