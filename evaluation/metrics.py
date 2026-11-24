"""Seven-metric Territorial.io Evaluation Calculator with 95% Wilson Confidence Intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EvaluationReport:
    num_matches: int
    win_rate: float
    win_rate_ci_low: float
    win_rate_ci_high: float
    mean_survival_sec: float
    mean_placement_pct: float
    mean_peak_land: float
    mean_land_auc: float
    mean_expansion_efficiency: float
    mean_combat_efficiency: float
    mean_decision_accuracy: float
    strategic_consistency: float  # 1 - IQR(placement_pct) across maps/opponents
    composite_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_matches": self.num_matches,
            "win_rate": round(self.win_rate, 4),
            "win_rate_ci_95": [round(self.win_rate_ci_low, 4), round(self.win_rate_ci_high, 4)],
            "mean_survival_sec": round(self.mean_survival_sec, 2),
            "mean_placement_pct": round(self.mean_placement_pct, 4),
            "mean_peak_land": round(self.mean_peak_land, 4),
            "mean_land_auc": round(self.mean_land_auc, 4),
            "mean_expansion_efficiency": round(self.mean_expansion_efficiency, 4),
            "mean_combat_efficiency": round(self.mean_combat_efficiency, 4),
            "mean_decision_accuracy": round(self.mean_decision_accuracy, 4),
            "strategic_consistency": round(self.strategic_consistency, 4),
            "composite_score": round(self.composite_score, 4),
        }


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def compute_evaluation_report(episodes: list[dict[str, Any]]) -> EvaluationReport:
    n = len(episodes)
    if n == 0:
        return EvaluationReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    wins = sum(1 for e in episodes if e.get("won"))
    wr = wins / n
    ci_l, ci_h = wilson_ci(wins, n)

    placements = []
    for e in episodes:
        rank = int(e.get("rank", e.get("placement_rank", 2)) or 2)
        tot = max(2, int(e.get("total_players", 6) or 6))
        placements.append(max(0.0, min(1.0, 1.0 - (rank - 1) / (tot - 1))))

    p_arr = np.asarray(placements, dtype=np.float32)
    q75, q25 = np.percentile(p_arr, [75, 25]) if n >= 2 else (float(p_arr[0]), float(p_arr[0]))
    consistency = float(np.clip(1.0 - (q75 - q25), 0.0, 1.0))

    mean_surv = float(np.mean([float(e.get("survival_sec", 0.0)) for e in episodes]))
    mean_place = float(p_arr.mean())
    mean_peak = float(np.mean([float(e.get("peak_land_ratio", 0.0)) for e in episodes]))
    mean_auc = float(np.mean([float(e.get("land_auc", 0.0)) for e in episodes]))
    mean_exp = float(np.mean([float(e.get("expansion_efficiency", 0.0)) for e in episodes]))
    mean_cmb = float(np.mean([float(e.get("combat_efficiency", 0.0)) for e in episodes]))
    mean_acc = float(np.mean([float(e.get("decision_accuracy", 0.0)) for e in episodes]))

    # Balanced strategic score: rewards actually winning + holding land + surviving + clean verified decisions
    composite = (
        0.40 * wr
        + 0.22 * mean_place
        + 0.18 * min(1.0, mean_peak * 1.4)
        + 0.10 * mean_acc
        + 0.10 * consistency
    )

    return EvaluationReport(
        num_matches=n,
        win_rate=wr,
        win_rate_ci_low=ci_l,
        win_rate_ci_high=ci_h,
        mean_survival_sec=mean_surv,
        mean_placement_pct=mean_place,
        mean_peak_land=mean_peak,
        mean_land_auc=mean_auc,
        mean_expansion_efficiency=mean_exp,
        mean_combat_efficiency=mean_cmb,
        mean_decision_accuracy=mean_acc,
        strategic_consistency=consistency,
        composite_score=float(composite),
    )
