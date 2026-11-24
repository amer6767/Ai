"""Generates legal macro-action candidates from visual perception and territory adjacency graph."""

from __future__ import annotations

import cv2
import numpy as np

from core.types import ActionDecision, ActionType, PerceivedFrame
from map_analysis.territory_graph import TerritoryGraphData
from map_analysis.world_map import PersistentWorldMap
from strategic_brain.economy_engine import EconomyAssessment


class CandidateActionGenerator:
    """Proposes concrete legal actions (WAIT, EXPAND, ATTACK, BOAT, INTERCEPT, PEACE, SCOUT)."""

    def generate(
        self,
        pf: PerceivedFrame,
        graph: TerritoryGraphData,
        world_map: PersistentWorldMap,
        econ: EconomyAssessment,
    ) -> list[ActionDecision]:
        candidates: list[ActionDecision] = [
            ActionDecision(
                action_type=ActionType.WAIT,
                target_x_norm=0.5,
                target_y_norm=0.5,
                slider_pct=pf.ui.estimated_slider_pct,
                rationale="accumulate_interest_and_defend",
            )
        ]

        h, w = pf.label_map.shape
        self_dil = cv2.dilate((pf.self_mask > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

        # 1. Neutral Expansion candidate (highest land/troop efficiency)
        neutral_frontier = (pf.neutral_mask > 0) & self_dil
        if neutral_frontier.any():
            ys, xs = neutral_frontier.nonzero()
            # Pick a frontier cell with rich adjacent neutral density
            neut_density = cv2.blur(pf.neutral_mask.astype(np.float32), (11, 11))
            scores = neut_density[ys, xs]
            best_i = int(np.argmax(scores))
            tx = float((xs[best_i] + 0.5) / w)
            ty = float((ys[best_i] + 0.5) / h)
            candidates.append(
                ActionDecision(
                    action_type=ActionType.EXPAND_NEUTRAL,
                    target_x_norm=tx,
                    target_y_norm=ty,
                    slider_pct=econ.optimal_neutral_slider,
                    rationale="expand_into_open_neutral_frontier",
                )
            )

        # 2. Bordering Enemy Attack candidates (multiple commitment levels for vulnerable/weak targets)
        for enemy in graph.self_border_neighbors[:8]:
            if enemy.attack_point_norm is None:
                continue
            tx, ty = enemy.attack_point_norm
            for pct in (0.15, 0.30, 0.50):
                candidates.append(
                    ActionDecision(
                        action_type=ActionType.ATTACK_ENEMY,
                        target_x_norm=tx,
                        target_y_norm=ty,
                        slider_pct=pct,
                        target_entity_id=enemy.entity_id,
                        rationale=f"attack_neighbor_{enemy.entity_id}_at_{int(pct*100)}pct",
                    )
                )

        # 3. Naval Boat candidate (when coastal and distant neutral/weak coast exists)
        has_coast = bool(((pf.water_mask > 0) & self_dil).any())
        if has_coast and pf.self_troops >= 100.0:
            distant_neutral = (pf.neutral_mask > 0) & (~self_dil)
            if distant_neutral.any():
                ys, xs = distant_neutral.nonzero()
                idx = len(xs) // 2
                candidates.append(
                    ActionDecision(
                        action_type=ActionType.LAUNCH_BOAT,
                        target_x_norm=float((xs[idx] + 0.5) / w),
                        target_y_norm=float((ys[idx] + 0.5) / h),
                        slider_pct=0.15,
                        rationale="naval_expansion_to_unclaimed_coast",
                    )
                )
            for target in graph.naval_targets[:2]:
                candidates.append(
                    ActionDecision(
                        action_type=ActionType.LAUNCH_BOAT,
                        target_x_norm=target.centroid_norm[0],
                        target_y_norm=target.centroid_norm[1],
                        slider_pct=0.20,
                        target_entity_id=target.entity_id,
                        rationale=f"naval_invasion_target_{target.entity_id}",
                    )
                )

        # 4. Boat interception if hostile ship approaching our coast
        for boat in pf.boats:
            if boat.is_hostile and boat.distance_to_self_coast < 0.14 and pf.self_troops >= 60.0:
                candidates.append(
                    ActionDecision(
                        action_type=ActionType.INTERCEPT_BOAT,
                        target_x_norm=boat.x_norm,
                        target_y_norm=boat.y_norm,
                        slider_pct=0.10,
                        rationale="intercept_incoming_hostile_boat",
                    )
                )
                break

        # 5. Peace vote when clearly dominant in endgame
        if pf.self_land_ratio > 0.65 and pf.alive_count_est <= 4:
            candidates.append(
                ActionDecision(
                    action_type=ActionType.PEACE_VOTE,
                    target_x_norm=0.5,
                    target_y_norm=0.5,
                    slider_pct=0.20,
                    rationale="call_peace_vote_from_dominant_lead",
                )
            )

        # 6. Camera scout if zoomed in and off-screen staleness is high
        dx, dy, max_stale = world_map.stalest_direction()
        if world_map.odometry.pose.zoom > 1.35 and max_stale > 0.72:
            candidates.append(
                ActionDecision(
                    action_type=ActionType.CAMERA_ZOOM,
                    target_x_norm=0.5,
                    target_y_norm=0.5,
                    zoom_delta=-2,
                    camera_dx=dx,
                    camera_dy=dy,
                    rationale="zoom_out_to_refresh_stale_strategic_map",
                )
            )

        return candidates
