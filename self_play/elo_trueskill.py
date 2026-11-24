"""Elo and Multiplayer Placement Rating Engine for Territorial.io Self-Play & Checkpoints."""

from __future__ import annotations

import math


class EloRatingEngine:
    """Computes 1v1 and multiplayer pairwise Elo updates."""

    def __init__(self, k_factor: float = 28.0) -> None:
        self.k = k_factor

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    def update_1v1(self, rating_a: float, rating_b: float, score_a: float) -> tuple[float, float]:
        exp_a = self.expected_score(rating_a, rating_b)
        exp_b = 1.0 - exp_a
        new_a = rating_a + self.k * (score_a - exp_a)
        new_b = rating_b + self.k * ((1.0 - score_a) - exp_b)
        return round(new_a, 2), round(new_b, 2)

    def update_multiplayer(self, ratings: list[float], ranks: list[int]) -> list[float]:
        """Pairwise multiplayer Elo update (rank 1 = winner)."""
        n = len(ratings)
        if n < 2:
            return list(ratings)
        deltas = [0.0] * n
        scale = self.k / max(1, n - 1)
        for i in range(n):
            for j in range(i + 1, n):
                exp_i = self.expected_score(ratings[i], ratings[j])
                if ranks[i] < ranks[j]:
                    s_i = 1.0
                elif ranks[i] > ranks[j]:
                    s_i = 0.0
                else:
                    s_i = 0.5
                d = scale * (s_i - exp_i)
                deltas[i] += d
                deltas[j] -= d
        return [round(r + d, 2) for r, d in zip(ratings, deltas)]
