"""Pure-pixel screen state classifier with temporal hysteresis."""

from __future__ import annotations

from collections import deque

import numpy as np

from core.types import ScreenState, UILayout
from vision.ui_detector import ScaleInvariantUIDetector


class ScreenStateClassifier:
    """Classifies Territorial.io screens purely from pixels and UI layout signatures."""

    def __init__(self, ui_detector: ScaleInvariantUIDetector | None = None, history_len: int = 4) -> None:
        self.ui_detector = ui_detector or ScaleInvariantUIDetector()
        self._history: deque[ScreenState] = deque(maxlen=history_len)
        self._was_in_game = False

    def reset(self) -> None:
        self._history.clear()
        self._was_in_game = False

    def classify(self, rgb: np.ndarray) -> tuple[ScreenState, UILayout]:
        layout = self.ui_detector.detect(rgb)
        state = layout.screen_state

        if state == ScreenState.IN_GAME:
            self._was_in_game = True
        elif state == ScreenState.SPECTATING and self._was_in_game:
            # Distinguish VICTORY (only 1 player alive or large green victory badge) vs DEFEAT
            if layout.hud_players_alive == 1:
                state = ScreenState.VICTORY
            else:
                state = ScreenState.DEFEAT
            layout.screen_state = state
        elif state in (ScreenState.MAIN_MENU, ScreenState.SINGLEPLAYER_LOBBY, ScreenState.SPAWN_SELECTION):
            self._was_in_game = False

        self._history.append(state)
        return state, layout
