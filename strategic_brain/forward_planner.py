"""Multi-action strategic lookahead evaluator implementing Territorial.io tactical & economic laws."""

from __future__ import annotations

from dataclasses import dataclass

from core.types import ActionDecision, ActionType, PerceivedFrame, TerritoryEntity
from map_analysis.territory_graph import TerritoryGraphData
from strategic_brain.economy_engine import EconomyAssessment
from strategic_brain.opponent_modeler import OpponentProfile


@dataclass
class ScoredCandidate:
    action: ActionDecision
    strategic_score: float
    expected_net_gain: float
    post_action_safety: float


class ForwardStrategicPlanner:
    """Evaluates multiple candidate actions using 2:1 defender-advantage math,
    compound-interest economics, multi-front survival, and opponent vulture windows.
    """

    def score_candidates(
        self,
        candidates: list[ActionDecision],
        pf: PerceivedFrame,
        graph: TerritoryGraphData,
        econ: EconomyAssessment,
        opponent_profiles: dict[int, OpponentProfile],
    ) -> list[ScoredCandidate]:
        ent_by_id: dict[int, TerritoryEntity] = {e.entity_id: e for e in graph.entities}
        self_troops = max(1.0, pf.self_troops)
        self_area = max(1.0, float(pf.self_land_px))

        # Strongest bordering threat density
        max_border_enemy_troops = max(
            (e.estimated_troops for e in graph.self_border_neighbors),
            default=0.0,
        )
        max_border_enemy_density = max(
            (e.estimated_troops / max(1.0, float(e.area_px)) for e in graph.self_border_neighbors),
            default=0.0,
        )

        scored: list[ScoredCandidate] = []
        for cand in candidates:
            score = 0.0
            net_gain = 0.0
            safety = 1.0

            if cand.action_type == ActionType.WAIT:
                # Waiting compounds interest cleanly UNLESS we are red-capped or wasting open neutral frontier
                if econ.red_interest_risk:
                    score -= 1.2
                elif econ.in_opening_compound_phase and pf.neutral_land_ratio > 0.05:
                    score += 0.25 if econ.cycle_payout_imminent else 0.15
                else:
                    score += 0.35 if graph.multi_front_count >= 2 else 0.20

            elif cand.action_type == ActionType.EXPAND_NEUTRAL:
                send = self_troops * cand.slider_pct
                residual = self_troops - send
                res_density = residual / max(1.0, self_area)
                safety = min(1.5, res_density / max(0.5, max_border_enemy_density + 1e-3))

                # Neutral land has NO 2:1 defender penalty -> huge ROI!
                net_gain = 1.35 + (0.65 if econ.red_interest_risk else 0.0)
                if econ.in_opening_compound_phase:
                    net_gain += 0.45
                if residual < self_troops * 0.30 and max_border_enemy_troops > residual:
                    net_gain -= 0.80  # avoid bankrupting troops next to strong enemy
                score = net_gain + 0.35 * min(1.0, safety)

            elif cand.action_type == ActionType.ATTACK_ENEMY:
                target = ent_by_id.get(cand.target_entity_id or -1)
                if target is None:
                    score = -2.0
                else:
                    send = self_troops * cand.slider_pct
                    residual = self_troops - send
                    t_area = max(1.0, float(target.area_px))
                    t_density = target.estimated_troops / t_area
                    my_density = self_troops / self_area
                    prof = opponent_profiles.get(target.entity_id)
                    vulture = prof.vulnerability_score if prof else 0.0
                    if target.is_under_attack:
                        vulture = max(vulture, 0.7)

                    # Exact 2:1 defender-advantage feasibility check!
                    # Attacking is profitable when send > 2.2 * defender_troops_per_front OR opponent density << ours OR vulturing!
                    advantage_ratio = my_density / max(0.5, t_density)
                    force_ratio = send / max(10.0, target.estimated_troops * 1.8)

                    # Post-action defense safety against OTHER neighbors (anti-multi-front suicide!)
                    other_threat = max(
                        (e.estimated_troops for e in graph.self_border_neighbors if e.entity_id != target.entity_id),
                        default=0.0,
                    )
                    safety = residual / max(20.0, other_threat * 0.85)
                    multi_front_penalty = 0.45 * max(0, graph.multi_front_count - 1)

                    if advantage_ratio >= 1.75 or vulture >= 0.45 or force_ratio >= 0.85:
                        net_gain = (
                            0.45 * min(3.0, advantage_ratio)
                            + 1.30 * vulture
                            + 0.40 * min(2.0, force_ratio)
                            + (0.50 if econ.red_interest_risk else 0.0)
                            + (0.35 if target.has_crown and vulture > 0.3 else 0.0)
                        )
                    else:
                        # Unfavorable 2:1 grind against healthy equal/stronger neighbor!
                        net_gain = -1.10 - 0.50 * cand.slider_pct

                    if safety < 0.65:
                        net_gain -= 1.25 * (0.65 - safety)
                    if cand.slider_pct >= 0.50 and advantage_ratio < 2.2 and vulture < 0.5:
                        net_gain -= 0.70

                    score = net_gain - multi_front_penalty + 0.25 * min(1.0, safety)

            elif cand.action_type == ActionType.LAUNCH_BOAT:
                # Valuable when landlocked/red-capped or claiming open coastal pockets
                if econ.red_interest_risk or pf.neutral_land_ratio > 0.04:
                    score = 0.55 if not any(c.action_type == ActionType.EXPAND_NEUTRAL for c in candidates) else 0.15
                else:
                    score = -0.15

            elif cand.action_type == ActionType.INTERCEPT_BOAT:
                score = 0.95  # cheap high-ROI coastal defense

            elif cand.action_type == ActionType.PEACE_VOTE:
                score = 0.80 if pf.self_land_ratio > 0.72 else -0.5

            elif cand.action_type == ActionType.CAMERA_ZOOM:
                score = 0.30

            scored.append(
                ScoredCandidate(
                    action=cand,
                    strategic_score=float(score),
                    expected_net_gain=float(net_gain),
                    post_action_safety=float(min(2.0, safety)),
                )
            )

        scored.sort(key=lambda s: s.strategic_score, reverse=True)
        return scored
