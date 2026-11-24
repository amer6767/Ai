"""Champion Checkpoint Gating: ensures only strictly stronger models replace the active champion."""

from __future__ import annotations

from typing import Any


def should_promote_champion(
    candidate_report: dict[str, Any],
    champion_report: dict[str, Any] | None,
    min_margin: float = 0.015,
) -> tuple[bool, str]:
    """Compare candidate against reigning champion across win rate, composite score, and decision accuracy."""
    cand_score = float(candidate_report.get("composite_score", 0.0))
    cand_wr = float(candidate_report.get("win_rate", 0.0))
    cand_acc = float(candidate_report.get("mean_decision_accuracy", 0.0))

    if champion_report is None:
        return True, f"initial_champion_established(score={cand_score:.3f}, wr={cand_wr:.2%})"

    champ_score = float(champion_report.get("composite_score", 0.0))
    champ_wr = float(champion_report.get("win_rate", 0.0))

    # Guard against models that click accurately without actually winning or holding territory
    if cand_acc < 0.55:
        return False, f"rejected_low_decision_accuracy({cand_acc:.2%})"
    if cand_score >= champ_score + min_margin or (cand_wr >= champ_wr + 0.05 and cand_score >= champ_score):
        return True, f"promoted(score {champ_score:.3f}->{cand_score:.3f}, wr {champ_wr:.2%}->{cand_wr:.2%})"
    return False, f"retained_champion(cand={cand_score:.3f} vs champ={champ_score:.3f})"
