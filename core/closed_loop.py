"""Continuous Closed-Loop Autonomous Intelligence Engine.

Implements the full closed loop on the visible external game window:
  OBSERVE -> UNDERSTAND -> ANALYZE -> PLAN -> DECIDE -> CLICK -> VERIFY -> LEARN
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from computer_control.action_verifier import ActionVerifier
from computer_control.mouse_keyboard import ExternalInputController
from computer_control.screen_capture import ScreenCapturer
from core.config import ControlConfig
from core.types import ActionDecision, ActionType, PerceivedFrame, RawFrame, ScreenState, VerificationResult
from memory.episode_store import PersistentMemoryStore, StepRecord
from memory.mistake_analyzer import MistakeAnalyzer
from perception.state_assembler import VisualPerceptionPipeline
from strategic_brain.arbiter import StrategicBrain


@dataclass
class LoopTelemetry:
    step_idx: int
    screen_state: ScreenState
    action: ActionDecision
    verification: VerificationResult
    self_troops: float
    self_land_ratio: float
    red_interest: bool
    alive_est: int
    elapsed_sec: float


class ClosedLoopAgent:
    """Executes OBSERVE -> UNDERSTAND -> ANALYZE -> PLAN -> DECIDE -> CLICK -> VERIFY -> LEARN."""

    def __init__(
        self,
        capturer: ScreenCapturer,
        controller: ExternalInputController,
        perception: VisualPerceptionPipeline,
        brain: StrategicBrain,
        memory: PersistentMemoryStore | None = None,
        control_cfg: ControlConfig | None = None,
        on_step: Callable[[LoopTelemetry, RawFrame, PerceivedFrame], None] | None = None,
    ) -> None:
        self.capturer = capturer
        self.controller = controller
        self.perception = perception
        self.brain = brain
        self.memory = memory
        self.cfg = control_cfg or ControlConfig()
        self.verifier = ActionVerifier(cooldown_steps=self.cfg.failed_action_cooldown_steps)
        self.mistake_analyzer = MistakeAnalyzer()
        self.on_step = on_step

    def _adjust_slider_externally(self, target_pct: float, pf: PerceivedFrame, raw: RawFrame) -> None:
        """Drag/click the visible bottom slider bar to set requested troop commitment."""
        if pf.ui.slider_box is None:
            return
        if abs(pf.ui.estimated_slider_pct - target_pct) <= 0.045:
            return
        x0 = pf.ui.slider_track_x0
        x1 = pf.ui.slider_track_x1
        if x1 <= x0 + 4:
            return
        clamped = max(0.02, min(0.99, target_pct))
        sx = raw.window_bbox.x + int(round(x0 + clamped * (x1 - x0)))
        sy = raw.window_bbox.y + pf.ui.slider_box.cy
        self.controller.click(sx, sy, hold_sec=0.02)
        time.sleep(0.03)

    def _execute_external_action(self, action: ActionDecision, raw: RawFrame, pf: PerceivedFrame) -> None:
        bbox = raw.window_bbox
        tx, ty = self.controller.norm_to_screen(action.target_x_norm, action.target_y_norm, bbox)

        if action.action_type == ActionType.WAIT:
            time.sleep(self.cfg.post_click_settle_sec)
            return

        if action.action_type == ActionType.UI_CLICK:
            self.controller.click(tx, ty, hold_sec=0.03)
            time.sleep(max(0.20, self.cfg.post_click_settle_sec))
            return

        if action.action_type == ActionType.SET_SLIDER:
            self._adjust_slider_externally(action.slider_pct, pf, raw)
            time.sleep(self.cfg.post_click_settle_sec)
            return

        if action.action_type in (ActionType.EXPAND_NEUTRAL, ActionType.ATTACK_ENEMY):
            self._adjust_slider_externally(action.slider_pct, pf, raw)
            self.controller.click(tx, ty, hold_sec=0.025)
            time.sleep(self.cfg.post_click_settle_sec)
            return

        if action.action_type == ActionType.LAUNCH_BOAT:
            self._adjust_slider_externally(action.slider_pct, pf, raw)
            self.controller.move_to(tx, ty)
            self.controller.press_key("KeyB")
            time.sleep(self.cfg.post_click_settle_sec)
            return

        if action.action_type == ActionType.INTERCEPT_BOAT:
            self.controller.move_to(tx, ty)
            self.controller.press_key("KeyI")
            time.sleep(self.cfg.post_click_settle_sec)
            return

        if action.action_type == ActionType.PEACE_VOTE:
            self.controller.press_key("KeyP")
            time.sleep(self.cfg.post_click_settle_sec)
            return

        if action.action_type == ActionType.CAMERA_ZOOM:
            self.controller.scroll(tx, ty, notches=action.zoom_delta or -2)
            self.brain.world_map.odometry.register_action_hint(
                zoom_mult=1.25 if (action.zoom_delta or 0) > 0 else 0.78
            )
            time.sleep(self.cfg.post_click_settle_sec)
            return

        if action.action_type == ActionType.CAMERA_PAN:
            cx, cy = self.controller.norm_to_screen(0.5, 0.5, bbox)
            ex, ey = self.controller.norm_to_screen(
                0.5 - action.camera_dx * 0.4, 0.5 - action.camera_dy * 0.4, bbox
            )
            self.controller.drag(cx, cy, ex, ey)
            self.brain.world_map.odometry.register_action_hint(dx_norm=action.camera_dx, dy_norm=action.camera_dy)
            time.sleep(self.cfg.post_click_settle_sec)

    def run_episode(
        self,
        max_steps: int = 200,
        curriculum_stage: int = 8,
        opponent_tier: str = "live_browser",
        checkpoint_id: str | None = None,
        use_planner_guidance: bool = True,
    ) -> dict[str, Any]:
        """Run one complete closed-loop episode from current window state until match end or max_steps."""
        self.perception.reset()
        self.brain.reset_episode()
        ep_start = time.monotonic()
        episode_id = (
            self.memory.start_episode(
                env_mode="live_browser",
                curriculum_stage=curriculum_stage,
                opponent_tier=opponent_tier,
                checkpoint_id=checkpoint_id,
            )
            if self.memory is not None
            else None
        )

        steps_log: list[StepRecord] = []
        entered_game = False
        won = False
        peak_land = 0.0
        land_sum = 0.0
        in_game_steps = 0
        verified_actions = 0
        total_non_wait = 0
        total_reward = 0.0
        prev_land = 0.0
        prev_troops = 0.0
        final_alive = 512

        for step_idx in range(max_steps):
            self.verifier.tick()

            # 1. OBSERVE + 2. UNDERSTAND
            raw_before = self.capturer.capture()
            pf_before = self.perception.process(raw_before)

            if pf_before.screen_state == ScreenState.IN_GAME:
                entered_game = True
                in_game_steps += 1
                peak_land = max(peak_land, pf_before.self_land_ratio)
                land_sum += pf_before.self_land_ratio
                final_alive = pf_before.alive_count_est
            elif entered_game and pf_before.screen_state in (ScreenState.VICTORY, ScreenState.DEFEAT, ScreenState.SPECTATING):
                won = pf_before.screen_state == ScreenState.VICTORY or pf_before.alive_count_est == 1
                break

            # 3. ANALYZE + 4. PLAN + 5. DECIDE
            supp_masks = {
                1: self.verifier.suppression_mask(ActionType.EXPAND_NEUTRAL, 32, 32),
                2: self.verifier.suppression_mask(ActionType.ATTACK_ENEMY, 32, 32),
                3: self.verifier.suppression_mask(ActionType.LAUNCH_BOAT, 32, 32),
            }
            decision, _, _ = self.brain.decide(
                raw_before,
                pf_before,
                suppression_masks=supp_masks,
                use_planner_guidance=use_planner_guidance,
            )

            # 6. CLICK / EXTERNAL CONTROL
            t0 = time.monotonic()
            self._execute_external_action(decision, raw_before, pf_before)
            latency_ms = (time.monotonic() - t0) * 1000.0

            # 7. VERIFY VISUAL CONSEQUENCE
            raw_after = self.capturer.capture()
            pf_after = self.perception.process(raw_after)
            ver = self.verifier.verify(decision, raw_before, raw_after, pf_before, pf_after, latency_ms)

            if decision.action_type != ActionType.WAIT:
                total_non_wait += 1
                if ver.succeeded:
                    verified_actions += 1

            # 8. LEARN / RECORD REWARD & STEP MEMORY
            step_reward = 0.0
            if pf_after.screen_state == ScreenState.IN_GAME:
                step_reward += 4.0 * (pf_after.self_land_ratio - prev_land)
                if pf_after.self_troops > prev_troops:
                    step_reward += 0.05
                if pf_after.ui.red_interest_active and decision.action_type == ActionType.WAIT:
                    step_reward -= 0.08
                if not ver.succeeded and decision.action_type != ActionType.WAIT:
                    step_reward -= 0.05
                prev_land = pf_after.self_land_ratio
                prev_troops = pf_after.self_troops
            elif entered_game and pf_after.screen_state == ScreenState.VICTORY:
                step_reward += 10.0
                won = True
            elif entered_game and pf_after.screen_state in (ScreenState.DEFEAT, ScreenState.SPECTATING):
                step_reward -= 2.0

            total_reward += step_reward
            steps_log.append(
                StepRecord(
                    step_idx=step_idx,
                    timestamp=time.time(),
                    screen_state=pf_before.screen_state.value,
                    action_type=decision.action_type.name,
                    target_x=decision.target_x_norm,
                    target_y=decision.target_y_norm,
                    slider_pct=decision.slider_pct,
                    verified=ver.succeeded,
                    verify_reason=ver.reason,
                    self_troops=pf_after.self_troops,
                    self_land_ratio=pf_after.self_land_ratio,
                    red_interest=pf_after.ui.red_interest_active,
                    reward=step_reward,
                    value_est=decision.value_est,
                    rationale=decision.rationale,
                )
            )

            if self.on_step is not None:
                telem = LoopTelemetry(
                    step_idx=step_idx,
                    screen_state=pf_after.screen_state,
                    action=decision,
                    verification=ver,
                    self_troops=pf_after.self_troops,
                    self_land_ratio=pf_after.self_land_ratio,
                    red_interest=pf_after.ui.red_interest_active,
                    alive_est=pf_after.alive_count_est,
                    elapsed_sec=time.monotonic() - ep_start,
                )
                self.on_step(telem, raw_after, pf_after)

            if entered_game and pf_after.screen_state in (ScreenState.VICTORY, ScreenState.DEFEAT):
                break

        survival_sec = time.monotonic() - ep_start
        decision_acc = verified_actions / max(1, total_non_wait)
        land_auc = land_sum / max(1, in_game_steps)
        summary = {
            "episode_id": episode_id,
            "won": won,
            "entered_game": entered_game,
            "placement_rank": 1 if won else max(2, final_alive),
            "total_players": 512,
            "survival_sec": round(survival_sec, 2),
            "peak_land_ratio": round(peak_land, 5),
            "land_auc": round(land_auc, 5),
            "expansion_efficiency": round(peak_land * 100.0 / max(1.0, survival_sec / 10.0), 4),
            "combat_efficiency": round(max(0.0, total_reward) / max(1, in_game_steps), 4),
            "decision_accuracy": round(decision_acc, 4),
            "total_reward": round(total_reward, 4),
            "total_steps": len(steps_log),
            "in_game_steps": in_game_steps,
        }

        if self.memory is not None and episode_id is not None:
            self.memory.finish_episode(episode_id, steps_log, summary)
            mistakes = self.mistake_analyzer.analyze_episode(steps_log)
            self.memory.record_mistakes(episode_id, mistakes)
            summary["mistakes_detected"] = len(mistakes)

        return summary
