"""Live unmodified Territorial.io Browser RL Environment via Xvfb + Chrome + External X11 Control.

100% engine-independent: interacts ONLY through external mss screenshots and python-xlib mouse/keys.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from computer_control.action_verifier import ActionVerifier
from computer_control.display_manager import GameDisplaySession
from computer_control.mouse_keyboard import ExternalInputController
from computer_control.screen_capture import ScreenCapturer
from core.config import ControlConfig, VisionConfig
from core.types import ActionDecision, ActionType, PerceivedFrame, RawFrame, ScreenState
from map_analysis.tactical_features import TacticalFeatureExtractor, TacticalObservation
from map_analysis.territory_graph import TerritoryGraphBuilder
from map_analysis.world_map import PersistentWorldMap
from perception.state_assembler import VisualPerceptionPipeline
from reinforcement_learning.reward_shaper import RewardShaper


class LiveX11BrowserEnv:
    """Gym-like wrapper around unmodified `text.html` in an isolated Xvfb + Chrome window."""

    def __init__(
        self,
        game_html_path: str | Path = "text.html",
        display_num: int = 110,
        width: int = 1280,
        height: int = 720,
        max_steps: int = 120,
        step_interval_sec: float = 0.25,
        vision_cfg: VisionConfig | None = None,
    ) -> None:
        self.session = GameDisplaySession(
            game_html_path=game_html_path,
            display_num=display_num,
            width=width,
            height=height,
        )
        self.max_steps = max_steps
        self.step_interval_sec = step_interval_sec
        self.perception = VisualPerceptionPipeline(config=vision_cfg or VisionConfig())
        self.world_map = PersistentWorldMap()
        self.graph_builder = TerritoryGraphBuilder()
        self.feature_extractor = TacticalFeatureExtractor()
        self.verifier = ActionVerifier()
        self.reward_shaper = RewardShaper()

        self.capturer: ScreenCapturer | None = None
        self.controller: ExternalInputController | None = None
        self._started = False
        self._step_idx = 0
        self._peak_land = 0.0
        self._land_sum = 0.0
        self._verified_count = 0
        self._action_count = 0
        self._ep_start_time = 0.0
        self._last_raw: RawFrame | None = None
        self._last_pf: PerceivedFrame | None = None

    def start(self) -> None:
        if self._started:
            return
        self.session.start(startup_wait_sec=2.8)
        self.capturer = ScreenCapturer(display_str=self.session.display_str)
        self.controller = ExternalInputController(
            display_str=self.session.display_str,
            config=ControlConfig(human_like_motion=True, move_steps=6, move_duration_sec=0.04),
        )
        self._started = True

    def _navigate_into_match(self, timeout_sec: float = 22.0) -> tuple[RawFrame, PerceivedFrame]:
        """Visually navigate Main Menu -> Custom Scenario -> Play -> Spawn on Green Land -> IN_GAME."""
        assert self.capturer is not None and self.controller is not None
        deadline = time.monotonic() + timeout_sec
        spawn_clicked = False

        while time.monotonic() < deadline:
            raw = self.capturer.capture()
            pf = self.perception.process(raw)
            if pf.screen_state == ScreenState.IN_GAME:
                return raw, pf

            h, w = raw.height, raw.width
            if pf.screen_state == ScreenState.MAIN_MENU:
                if pf.ui.singleplayer_btn is not None:
                    b = pf.ui.singleplayer_btn
                    self.controller.click(raw.window_bbox.x + b.cx, raw.window_bbox.y + b.cy)
                else:
                    px, py = self.controller.norm_to_screen(0.555, 0.51, raw.window_bbox)
                    self.controller.click(px, py)
                time.sleep(0.9)
            elif pf.screen_state == ScreenState.SINGLEPLAYER_LOBBY:
                if pf.ui.start_match_btn is not None:
                    b = pf.ui.start_match_btn
                    self.controller.click(raw.window_bbox.x + b.cx, raw.window_bbox.y + b.cy)
                else:
                    px, py = self.controller.norm_to_screen(0.75, 0.945, raw.window_bbox)
                    self.controller.click(px, py)
                time.sleep(1.5)
            elif pf.screen_state == ScreenState.SPAWN_SELECTION:
                sx, sy = self.feature_extractor.select_best_spawn_point(pf)
                px, py = self.controller.norm_to_screen(sx, sy, raw.window_bbox)
                self.controller.click(px, py)
                time.sleep(0.35)
                self.controller.click(px, py)
                spawn_clicked = True
                time.sleep(0.8)
            elif pf.screen_state in (ScreenState.SPECTATING, ScreenState.DEFEAT, ScreenState.VICTORY):
                # Click bottom-left exit icon then Escape to return cleanly to menu
                px, py = self.controller.norm_to_screen(0.022, 0.955, raw.window_bbox)
                self.controller.click(px, py)
                time.sleep(0.6)
                self.controller.press_key("Enter")
                time.sleep(0.8)
            elif pf.screen_state == ScreenState.MODAL_OVERLAY:
                if pf.ui.back_btn is not None:
                    b = pf.ui.back_btn
                    self.controller.click(raw.window_bbox.x + b.cx, raw.window_bbox.y + b.cy)
                else:
                    self.controller.press_key("Escape")
                time.sleep(0.7)
            else:
                time.sleep(0.5)

        raw = self.capturer.capture()
        pf = self.perception.process(raw)
        return raw, pf

    def reset(self) -> tuple[TacticalObservation, dict[str, Any]]:
        self.start()
        assert self.capturer is not None and self.controller is not None
        self.perception.reset()
        self.world_map.reset()
        self.reward_shaper.reset()
        self._step_idx = 0
        self._peak_land = 0.0
        self._land_sum = 0.0
        self._verified_count = 0
        self._action_count = 0

        # If currently inside an existing game, exit cleanly first
        cur_raw = self.capturer.capture()
        cur_pf = self.perception.process(cur_raw)
        if cur_pf.screen_state in (ScreenState.IN_GAME, ScreenState.SPECTATING, ScreenState.DEFEAT, ScreenState.VICTORY):
            px, py = self.controller.norm_to_screen(0.022, 0.955, cur_raw.window_bbox)
            self.controller.click(px, py)
            time.sleep(0.5)
            self.controller.press_key("Enter")
            time.sleep(0.8)

        raw, pf = self._navigate_into_match()
        self._ep_start_time = time.monotonic()
        self._last_raw = raw
        self._last_pf = pf
        self.world_map.update(pf)
        graph = self.graph_builder.build(pf)
        obs = self.feature_extractor.build_observation(raw, pf, self.world_map, graph)
        return obs, {"screen_state": pf.screen_state.value, "in_game": pf.screen_state == ScreenState.IN_GAME}

    def step(
        self,
        action_type_idx: int,
        target_x_norm: float,
        target_y_norm: float,
        slider_pct: float,
    ) -> tuple[TacticalObservation, float, bool, bool, dict[str, Any]]:
        assert self.capturer is not None and self.controller is not None
        assert self._last_raw is not None and self._last_pf is not None
        self._step_idx += 1
        self.verifier.tick()

        raw_before = self._last_raw
        pf_before = self._last_pf

        a_map = [
            ActionType.WAIT,
            ActionType.EXPAND_NEUTRAL,
            ActionType.ATTACK_ENEMY,
            ActionType.LAUNCH_BOAT,
            ActionType.PEACE_VOTE,
        ]
        atype = a_map[int(action_type_idx) % len(a_map)]
        decision = ActionDecision(
            action_type=atype,
            target_x_norm=target_x_norm,
            target_y_norm=target_y_norm,
            slider_pct=slider_pct,
        )

        t0 = time.monotonic()
        if atype != ActionType.WAIT:
            self._action_count += 1
            # Adjust bottom slider visibly if needed
            if pf_before.ui.slider_box is not None and abs(pf_before.ui.estimated_slider_pct - slider_pct) > 0.05:
                x0, x1 = pf_before.ui.slider_track_x0, pf_before.ui.slider_track_x1
                if x1 > x0 + 4:
                    sx = raw_before.window_bbox.x + int(x0 + max(0.02, min(0.98, slider_pct)) * (x1 - x0))
                    sy = raw_before.window_bbox.y + pf_before.ui.slider_box.cy
                    self.controller.click(sx, sy, hold_sec=0.02)
            px, py = self.controller.norm_to_screen(target_x_norm, target_y_norm, raw_before.window_bbox)
            if atype in (ActionType.EXPAND_NEUTRAL, ActionType.ATTACK_ENEMY):
                self.controller.click(px, py, hold_sec=0.025)
            elif atype == ActionType.LAUNCH_BOAT:
                self.controller.move_to(px, py)
                self.controller.press_key("KeyB")
            elif atype == ActionType.PEACE_VOTE:
                self.controller.press_key("KeyP")
        time.sleep(self.step_interval_sec)
        latency_ms = (time.monotonic() - t0) * 1000.0

        raw_after = self.capturer.capture()
        pf_after = self.perception.process(raw_after)
        ver = self.verifier.verify(decision, raw_before, raw_after, pf_before, pf_after, latency_ms)
        if atype != ActionType.WAIT and ver.succeeded:
            self._verified_count += 1

        self._last_raw = raw_after
        self._last_pf = pf_after
        self.world_map.update(pf_after)
        graph = self.graph_builder.build(pf_after)
        supp_masks = {
            1: self.verifier.suppression_mask(ActionType.EXPAND_NEUTRAL, 32, 32),
            2: self.verifier.suppression_mask(ActionType.ATTACK_ENEMY, 32, 32),
            3: self.verifier.suppression_mask(ActionType.LAUNCH_BOAT, 32, 32),
        }
        obs = self.feature_extractor.build_observation(raw_after, pf_after, self.world_map, graph, suppression_masks=supp_masks)

        self._peak_land = max(self._peak_land, pf_after.self_land_ratio)
        self._land_sum += pf_after.self_land_ratio

        won = pf_after.screen_state == ScreenState.VICTORY
        lost = pf_after.screen_state in (ScreenState.DEFEAT, ScreenState.SPECTATING)
        terminated = won or lost
        truncated = (not terminated) and (self._step_idx >= self.max_steps)

        soft_cap = max(100.0, 100.0 * max(1.0, float(pf_after.ui.hud_land or (pf_after.self_land_px * 12))))
        reward = self.reward_shaper.compute(
            land_ratio=pf_after.self_land_ratio,
            troops=pf_after.self_troops,
            soft_cap=soft_cap,
            red_interest=pf_after.ui.red_interest_active,
            action_succeeded=ver.succeeded,
            terminated=terminated,
            truncated=truncated,
            won=won,
            rank=1 if won else max(2, pf_after.alive_count_est),
            total_players=512,
        )

        info = {
            "won": won,
            "lost": lost,
            "screen_state": pf_after.screen_state.value,
            "verified": ver.succeeded,
            "verify_reason": ver.reason,
            "decision_accuracy": self._verified_count / max(1, self._action_count),
            "land_ratio": pf_after.self_land_ratio,
            "peak_land_ratio": self._peak_land,
            "land_auc": self._land_sum / max(1, self._step_idx),
            "self_troops": pf_after.self_troops,
            "alive_est": pf_after.alive_count_est,
            "survival_sec": time.monotonic() - self._ep_start_time,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.capturer is not None:
            self.capturer.close()
            self.capturer = None
        if self.controller is not None:
            self.controller.close()
            self.controller = None
        self.session.stop()
        self._started = False
