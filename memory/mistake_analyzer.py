"""Automated mistake detector extracting blunders and failure patterns from completed episodes."""

from __future__ import annotations

from typing import Any

import numpy as np

from memory.episode_store import StepRecord


class MistakeAnalyzer:
    """Identifies recurring strategic and motor-control errors for hard-scenario retraining."""

    def analyze_episode(self, steps: list[StepRecord]) -> list[dict[str, Any]]:
        if len(steps) < 3:
            return []
        mistakes: list[dict[str, Any]] = []

        values = np.array([s.value_est for s in steps], dtype=np.float32)
        rewards = np.array([s.reward for s in steps], dtype=np.float32)
        td_errors = rewards[:-1] + 0.99 * values[1:] - values[:-1]
        td_std = float(td_errors.std()) + 1e-6
        td_mean = float(td_errors.mean())

        consecutive_red = 0
        for i, s in enumerate(steps):
            if not s.verified and s.action_type != "WAIT":
                mistakes.append(
                    {
                        "step_idx": s.step_idx,
                        "mistake_type": "unverified_action",
                        "severity": 0.6,
                        "action_type": s.action_type,
                        "context": {
                            "reason": s.verify_reason,
                            "xy": [round(s.target_x, 3), round(s.target_y, 3)],
                        },
                    }
                )

            if s.red_interest and s.action_type == "WAIT":
                consecutive_red += 1
                if consecutive_red == 4:
                    mistakes.append(
                        {
                            "step_idx": s.step_idx,
                            "mistake_type": "red_interest_waste",
                            "severity": 0.75,
                            "action_type": s.action_type,
                            "context": {"troops": s.self_troops, "land_ratio": s.self_land_ratio},
                        }
                    )
            else:
                consecutive_red = 0

            # Overextension collapse: large slider attack followed by sharp land loss within 5 steps
            if s.action_type == "ATTACK_ENEMY" and s.slider_pct >= 0.60 and i + 5 < len(steps):
                future_land = steps[i + 5].self_land_ratio
                if s.self_land_ratio > 0.01 and future_land < s.self_land_ratio * 0.72:
                    mistakes.append(
                        {
                            "step_idx": s.step_idx,
                            "mistake_type": "overextension_collapse",
                            "severity": 1.0,
                            "action_type": s.action_type,
                            "context": {
                                "slider_pct": s.slider_pct,
                                "land_before": round(s.self_land_ratio, 4),
                                "land_after_5": round(future_land, 4),
                            },
                        }
                    )

            # Significant TD blunder drop
            if i < len(td_errors) and td_errors[i] < td_mean - 2.2 * td_std and td_errors[i] < -0.25:
                mistakes.append(
                    {
                        "step_idx": s.step_idx,
                        "mistake_type": "blunder_td_drop",
                        "severity": min(1.0, float(abs(td_errors[i]))),
                        "action_type": s.action_type,
                        "context": {"td_error": round(float(td_errors[i]), 4), "rationale": s.rationale},
                    }
                )

        # Deduplicate keeping most severe per step
        return mistakes[:64]
