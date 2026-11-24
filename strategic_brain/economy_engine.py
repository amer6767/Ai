"""Economic & compound-interest analyzer grounded in Territorial.io's verified laws."""

from __future__ import annotations

from dataclasses import dataclass

from core.types import PerceivedFrame


@dataclass
class EconomyAssessment:
    estimated_soft_cap: float
    cap_utilization: float          # troops / (100 * land)
    red_interest_risk: bool         # True if near or above soft cap
    optimal_neutral_slider: float   # recommended commitment for neutral land grab
    safe_reserve_ratio: float       # minimum fraction of troops to retain for defense & compounding
    in_opening_compound_phase: bool
    cycle_payout_imminent: bool


class EconomyEngine:
    """Evaluates compound interest, cycle-bar timing, and troop commitment bounds from visual HUD."""

    def assess(self, pf: PerceivedFrame, max_neighbor_threat: float) -> EconomyAssessment:
        land_est = max(10.0, float(pf.ui.hud_land or (pf.self_land_px * 12)))
        soft_cap = 100.0 * land_est
        troops = max(1.0, pf.self_troops)
        cap_util = troops / max(1.0, soft_cap)

        red_risk = pf.ui.red_interest_active or cap_util >= 0.90
        opening = (pf.ui.hud_elapsed_sec or 0) < 95 and pf.neutral_land_ratio > 0.08
        payout_imminent = pf.ui.cycle_progress >= 0.80

        # Base safe reserve grows when dangerous neighbors border us
        safe_reserve = min(0.85, max(0.35, 0.45 + 0.20 * min(1.5, max_neighbor_threat)))
        if opening:
            # During high-interest opening (7% -> 1%), keep steady 50-75% reserve for compounding
            opt_neutral = 0.22 if not red_risk else 0.38
            if payout_imminent and not red_risk:
                # Hold briefly just before land income ticks if reserves are low
                opt_neutral = 0.15
        else:
            opt_neutral = 0.30 if not red_risk else 0.45

        return EconomyAssessment(
            estimated_soft_cap=soft_cap,
            cap_utilization=float(min(1.6, cap_util)),
            red_interest_risk=red_risk,
            optimal_neutral_slider=float(opt_neutral),
            safe_reserve_ratio=float(safe_reserve),
            in_opening_compound_phase=opening,
            cycle_payout_imminent=payout_imminent,
        )
