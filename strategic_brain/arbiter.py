"""Strategic Arbiter combining the trained Neural RL Policy-Value Network with tactical evaluation."""

from __future__ import annotations

import numpy as np
import torch

from core.types import ActionDecision, ActionType, PerceivedFrame, RawFrame, ScreenState
from map_analysis.tactical_features import TacticalFeatureExtractor, TacticalObservation
from map_analysis.territory_graph import TerritoryGraphBuilder, TerritoryGraphData
from map_analysis.world_map import PersistentWorldMap
from models.policy_value_net import SLIDER_BINS, TerritorialPolicyValueNet
from strategic_brain.candidate_generator import CandidateActionGenerator
from strategic_brain.economy_engine import EconomyEngine
from strategic_brain.forward_planner import ForwardStrategicPlanner
from strategic_brain.opponent_modeler import OpponentModeler

_RL_TO_ACTION_TYPE: tuple[ActionType, ...] = (
    ActionType.WAIT,
    ActionType.EXPAND_NEUTRAL,
    ActionType.ATTACK_ENEMY,
    ActionType.LAUNCH_BOAT,
    ActionType.PEACE_VOTE,
)


class StrategicBrain:
    """Complete independent decision-making brain.

    - Handles Menu / Lobby / Spawn / Endgame transitions visually from detected UI.
    - During active gameplay, runs the PyTorch `TerritorialPolicyValueNet` (Visual CNN + GATv2 + GRU)
      alongside multi-action strategic planning so the agent learns from experience while
      avoiding catastrophic unverified/suicidal blunders.
    """

    def __init__(
        self,
        model: TerritorialPolicyValueNet | None = None,
        device: str | torch.device = "cpu",
        rl_weight: float = 0.75,
        deterministic: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.model = (model or TerritorialPolicyValueNet()).to(self.device)
        self.rl_weight = rl_weight
        self.deterministic = deterministic

        self.world_map = PersistentWorldMap()
        self.graph_builder = TerritoryGraphBuilder()
        self.feature_extractor = TacticalFeatureExtractor()
        self.economy = EconomyEngine()
        self.opponent_modeler = OpponentModeler()
        self.candidate_gen = CandidateActionGenerator()
        self.planner = ForwardStrategicPlanner()
        self._hidden: torch.Tensor | None = None
        self._spawn_clicked_at_step: int | None = None
        self._nav_step = 0

    def reset_episode(self) -> None:
        self.world_map.reset()
        self.opponent_modeler.reset()
        self._hidden = self.model.rnn.initial_state(1, self.device)
        self._spawn_clicked_at_step = None
        self._nav_step = 0

    def _screen_navigation_decision(self, pf: PerceivedFrame) -> ActionDecision:
        """Purely visual navigation for Main Menu, Lobby, Spawn Selection, and Modals."""
        self._nav_step += 1
        h, w = max(1, pf.height), max(1, pf.width)
        if pf.screen_state == ScreenState.MAIN_MENU:
            self._spawn_clicked_at_step = None
            if pf.ui.singleplayer_btn is not None:
                b = pf.ui.singleplayer_btn
                return ActionDecision(
                    action_type=ActionType.UI_CLICK,
                    target_x_norm=b.cx / w,
                    target_y_norm=b.cy / h,
                    rationale="click_detected_custom_scenario_button",
                )
            return ActionDecision(
                action_type=ActionType.UI_CLICK,
                target_x_norm=0.555,
                target_y_norm=0.51,
                rationale="click_custom_scenario_fallback_region",
            )

        if pf.screen_state == ScreenState.SINGLEPLAYER_LOBBY:
            self._spawn_clicked_at_step = None
            if pf.ui.start_match_btn is not None:
                b = pf.ui.start_match_btn
                return ActionDecision(
                    action_type=ActionType.UI_CLICK,
                    target_x_norm=b.cx / w,
                    target_y_norm=b.cy / h,
                    rationale="click_detected_play_footer_button",
                )
            return ActionDecision(
                action_type=ActionType.UI_CLICK,
                target_x_norm=0.75,
                target_y_norm=0.945,
                rationale="click_play_footer_fallback",
            )

        if pf.screen_state == ScreenState.SPAWN_SELECTION:
            # Territorial.io Custom Scenario: 1st click places capital preview, 2nd click confirms & starts match!
            if self._spawn_clicked_at_step is None:
                sx, sy = self.feature_extractor.select_best_spawn_point(pf)
                self._spawn_pos = (sx, sy)
                self._spawn_clicked_at_step = self._nav_step
                return ActionDecision(
                    action_type=ActionType.UI_CLICK,
                    target_x_norm=sx,
                    target_y_norm=sy,
                    rationale="place_optimal_land_spawn_from_heatmap",
                )
            if self._nav_step - self._spawn_clicked_at_step == 1:
                sx, sy = getattr(self, "_spawn_pos", (0.5, 0.5))
                return ActionDecision(
                    action_type=ActionType.UI_CLICK,
                    target_x_norm=sx,
                    target_y_norm=sy,
                    rationale="confirm_spawn_to_launch_match",
                )
            if self._nav_step - self._spawn_clicked_at_step > 12:
                self._spawn_clicked_at_step = None
            return ActionDecision(
                action_type=ActionType.WAIT,
                target_x_norm=0.5,
                target_y_norm=0.5,
                rationale="awaiting_match_start_after_spawn_confirm",
            )

        if pf.screen_state == ScreenState.MODAL_OVERLAY:
            if pf.ui.back_btn is not None:
                b = pf.ui.back_btn
                return ActionDecision(
                    action_type=ActionType.UI_CLICK,
                    target_x_norm=b.cx / w,
                    target_y_norm=b.cy / h,
                    rationale="click_detected_back_button",
                )
            return ActionDecision(action_type=ActionType.WAIT, rationale="awaiting_transition_fade")

        if pf.screen_state in (ScreenState.VICTORY, ScreenState.DEFEAT, ScreenState.SPECTATING):
            return ActionDecision(
                action_type=ActionType.UI_CLICK,
                target_x_norm=0.022,
                target_y_norm=0.955,
                rationale="click_exit_match_icon",
            )

        return ActionDecision(action_type=ActionType.WAIT, rationale="awaiting_recognizable_screen")

    def decide(
        self,
        raw: RawFrame,
        pf: PerceivedFrame,
        suppression_masks: dict[int, np.ndarray] | None = None,
        use_planner_guidance: bool = True,
    ) -> tuple[ActionDecision, TacticalObservation | None, TerritoryGraphData | None]:
        if pf.screen_state != ScreenState.IN_GAME:
            return self._screen_navigation_decision(pf), None, None

        self.world_map.update(pf)
        graph = self.graph_builder.build(pf)
        obs = self.feature_extractor.build_observation(
            raw, pf, self.world_map, graph, suppression_masks=suppression_masks
        )
        econ = self.economy.assess(pf, graph.max_neighbor_threat)
        opp_profiles = self.opponent_modeler.update(graph.entities, self_losing_troops=not pf.ui.balance_gaining)

        # Forward pass through trained Neural RL Policy-Value Network
        if self._hidden is None:
            self._hidden = self.model.rnn.initial_state(1, self.device)

        with torch.no_grad():
            out = self.model.forward_step(
                obs_map=torch.from_numpy(obs.obs_map).unsqueeze(0).to(self.device),
                node_feats=torch.from_numpy(obs.node_feats).unsqueeze(0).to(self.device),
                adj=torch.from_numpy(obs.adj).unsqueeze(0).to(self.device),
                node_mask=torch.from_numpy(obs.node_mask).unsqueeze(0).to(self.device),
                scalars=torch.from_numpy(obs.scalars).unsqueeze(0).to(self.device),
                hidden=self._hidden,
                action_mask=torch.from_numpy(obs.action_mask).unsqueeze(0).to(self.device),
                spatial_masks=torch.from_numpy(obs.spatial_masks).unsqueeze(0).to(self.device),
                deterministic=self.deterministic,
            )
            self._hidden = out.next_hidden

        a_idx = int(out.action_type.item())
        sp_idx = int(out.spatial_idx.item())
        sl_idx = int(out.slider_idx.item())
        g = self.model.spatial_grid
        gy, gx = divmod(sp_idx, g)
        rl_tx = float((gx + 0.5) / g)
        rl_ty = float((gy + 0.5) / g)
        rl_slider = float(SLIDER_BINS[sl_idx])
        rl_type = _RL_TO_ACTION_TYPE[a_idx]
        rl_val = float(out.value.item())
        rl_logp = float(out.log_prob.item())

        rl_decision = ActionDecision(
            action_type=rl_type,
            target_x_norm=rl_tx,
            target_y_norm=rl_ty,
            slider_pct=rl_slider,
            confidence=float(torch.softmax(out.type_logits, dim=-1)[0, a_idx].item()),
            rationale=f"rl_policy_{rl_type.name.lower()}_val={rl_val:.2f}",
            log_prob=rl_logp,
            value_est=rl_val,
            metadata={"rl_action_idx": a_idx, "spatial_idx": sp_idx, "slider_idx": sl_idx},
        )

        if not use_planner_guidance or self.rl_weight >= 0.99:
            return rl_decision, obs, graph

        # Multi-action forward strategic planning & safety arbitration
        candidates = self.candidate_gen.generate(pf, graph, self.world_map, econ)
        scored = self.planner.score_candidates(candidates, pf, graph, econ, opp_profiles)
        if not scored:
            return rl_decision, obs, graph

        best_plan = scored[0]
        # Blend: follow neural RL policy unless the planner detects an urgent high-margin tactical opportunity
        # (e.g. boat intercept, red-cap stall, or severe suicidal overextension)
        rl_type_prob = float(torch.softmax(out.type_logits, dim=-1)[0, a_idx].item())
        use_rl = (self.rl_weight * rl_type_prob >= 0.38) and not (
            rl_type == ActionType.WAIT and econ.red_interest_risk and best_plan.strategic_score > 0.7
        )

        if use_rl:
            return rl_decision, obs, graph

        chosen = best_plan.action
        chosen.value_est = rl_val
        chosen.log_prob = rl_logp
        chosen.confidence = float(np.clip(0.5 + 0.25 * best_plan.strategic_score, 0.1, 0.99))
        chosen.metadata["planner_score"] = round(best_plan.strategic_score, 3)
        chosen.metadata["rl_action_idx"] = a_idx
        return chosen, obs, graph
