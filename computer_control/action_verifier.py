"""Post-action visual consequence verification and failed-action suppression."""

from __future__ import annotations

import numpy as np

from core.types import ActionDecision, ActionType, PerceivedFrame, RawFrame, VerificationResult


class ActionVerifier:
    """Verifies via screenshots whether an external input produced its intended visual result.

    Tracks repeated failures on quantized spatial cells so the AI never gets stuck
    repeating a failed click or interacting with an obstructed UI region.
    """

    def __init__(self, cooldown_steps: int = 8, grid_bins: int = 16) -> None:
        self.cooldown_steps = cooldown_steps
        self.grid_bins = grid_bins
        self._step_counter = 0
        self._failure_counts: dict[tuple[int, int, int], int] = {}
        self._cooldown_until: dict[tuple[int, int, int], int] = {}

    def _key(self, action_type: ActionType, x_norm: float, y_norm: float) -> tuple[int, int, int]:
        gx = min(self.grid_bins - 1, max(0, int(x_norm * self.grid_bins)))
        gy = min(self.grid_bins - 1, max(0, int(y_norm * self.grid_bins)))
        return (int(action_type), gx, gy)

    def tick(self) -> None:
        self._step_counter += 1
        expired = [k for k, until in self._cooldown_until.items() if until <= self._step_counter]
        for k in expired:
            del self._cooldown_until[k]
            self._failure_counts.pop(k, None)

    def is_suppressed(self, action_type: ActionType, x_norm: float, y_norm: float) -> bool:
        key = self._key(action_type, x_norm, y_norm)
        return self._cooldown_until.get(key, -1) > self._step_counter

    def suppression_mask(self, action_type: ActionType, grid_h: int, grid_w: int) -> np.ndarray:
        """Return bool mask [grid_h, grid_w] where True = valid, False = cooled-down."""
        mask = np.ones((grid_h, grid_w), dtype=bool)
        for (atype, gx, gy), until in self._cooldown_until.items():
            if atype != int(action_type) or until <= self._step_counter:
                continue
            x0 = int(gx * grid_w / self.grid_bins)
            x1 = max(x0 + 1, int((gx + 1) * grid_w / self.grid_bins))
            y0 = int(gy * grid_h / self.grid_bins)
            y1 = max(y0 + 1, int((gy + 1) * grid_h / self.grid_bins))
            mask[y0:y1, x0:x1] = False
        return mask

    def verify(
        self,
        action: ActionDecision,
        before_raw: RawFrame,
        after_raw: RawFrame,
        before_state: PerceivedFrame,
        after_state: PerceivedFrame,
        latency_ms: float,
    ) -> VerificationResult:
        # Compute normalized global and local pixel delta
        diff = np.abs(after_raw.rgb.astype(np.int16) - before_raw.rgb.astype(np.int16))
        global_diff = float(diff.mean()) / 255.0

        h, w = before_raw.height, before_raw.width
        cx = int(action.target_x_norm * (w - 1))
        cy = int(action.target_y_norm * (h - 1))
        r = max(16, min(h, w) // 16)
        local_patch = diff[max(0, cy - r) : min(h, cy + r), max(0, cx - r) : min(w, cx + r)]
        local_diff = float(local_patch.mean()) / 255.0 if local_patch.size > 0 else global_diff

        troop_delta = float(after_state.self_troops - before_state.self_troops)
        state_changed = after_state.screen_state != before_state.screen_state

        succeeded = False
        reason = "no_visual_change"

        if action.action_type == ActionType.WAIT:
            succeeded = True
            reason = "intentional_wait"
        elif action.action_type == ActionType.UI_CLICK:
            if state_changed or global_diff > 0.008 or local_diff > 0.015:
                succeeded = True
                reason = "ui_transition_confirmed"
            else:
                reason = "ui_click_no_visual_response"
        elif action.action_type == ActionType.SET_SLIDER:
            slider_delta = abs(after_state.ui.estimated_slider_pct - before_state.ui.estimated_slider_pct)
            slider_close = abs(after_state.ui.estimated_slider_pct - action.slider_pct) < 0.09
            if slider_delta > 0.02 or slider_close or local_diff > 0.01:
                succeeded = True
                reason = "slider_adjustment_confirmed"
            else:
                reason = "slider_did_not_move"
        elif action.action_type in (ActionType.EXPAND_NEUTRAL, ActionType.ATTACK_ENEMY, ActionType.LAUNCH_BOAT):
            land_delta = after_state.self_land_px - before_state.self_land_px
            combat_flux = float(after_state.combat_delta_map.sum())
            if troop_delta < -1.0 or land_delta > 0 or combat_flux > 1.5 or local_diff > 0.006 or state_changed:
                succeeded = True
                reason = "offensive_or_expansion_effect_observed"
            else:
                reason = "no_troop_or_border_response"
        elif action.action_type in (ActionType.CAMERA_PAN, ActionType.CAMERA_ZOOM):
            if global_diff > 0.012:
                succeeded = True
                reason = "viewport_transform_observed"
            else:
                reason = "camera_did_not_move"
        else:
            succeeded = global_diff > 0.004 or state_changed
            reason = "generic_visual_change" if succeeded else "no_observed_effect"

        key = self._key(action.action_type, action.target_x_norm, action.target_y_norm)
        if succeeded:
            self._failure_counts.pop(key, None)
        else:
            fails = self._failure_counts.get(key, 0) + 1
            self._failure_counts[key] = fails
            if fails >= 2:
                self._cooldown_until[key] = self._step_counter + self.cooldown_steps

        return VerificationResult(
            succeeded=succeeded,
            reason=reason,
            pixel_diff_score=round(max(global_diff, local_diff), 5),
            troop_delta=troop_delta,
            state_before=before_state.screen_state,
            state_after=after_state.screen_state,
            latency_ms=round(latency_ms, 2),
        )
