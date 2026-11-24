"""End-to-End Autonomous Live Browser Test on Unmodified text.html.

Verifies the entire closed loop:
  Xvfb + plain Chrome (zero CDP) -> mss RGB screenshot -> pure-CV perception ->
  Main Menu -> Custom Scenario -> Play -> Spawn on Green Land -> IN_GAME ->
  external X11 mouse clicks + visual ActionVerifier -> persistent SQLite/Parquet memory.
"""

from __future__ import annotations

from pathlib import Path

from computer_control.display_manager import GameDisplaySession
from computer_control.mouse_keyboard import ExternalInputController
from computer_control.screen_capture import ScreenCapturer
from core.closed_loop import ClosedLoopAgent
from core.config import ControlConfig, VisionConfig
from memory.episode_store import PersistentMemoryStore
from models.policy_value_net import TerritorialPolicyValueNet
from perception.state_assembler import VisualPerceptionPipeline
from strategic_brain.arbiter import StrategicBrain


def test_end_to_end_live_territorial_closed_loop(tmp_path: Path) -> None:
    memory = PersistentMemoryStore(
        db_path=tmp_path / "e2e_mem.db",
        episodes_dir=tmp_path / "e2e_eps",
    )
    model = TerritorialPolicyValueNet()
    perception = VisualPerceptionPipeline(config=VisionConfig(use_neural_seg=False, use_neural_ocr=False))
    brain = StrategicBrain(model=model, device="cpu", deterministic=True)

    with GameDisplaySession("text.html", display_num=140, width=1280, height=720) as sess:
        capturer = ScreenCapturer(display_str=sess.display_str)
        controller = ExternalInputController(
            display_str=sess.display_str,
            config=ControlConfig(human_like_motion=True, move_steps=5, move_duration_sec=0.035, post_click_settle_sec=0.30),
        )
        try:
            agent = ClosedLoopAgent(
                capturer=capturer,
                controller=controller,
                perception=perception,
                brain=brain,
                memory=memory,
                control_cfg=ControlConfig(post_click_settle_sec=0.30),
            )
            summary = agent.run_episode(
                max_steps=45,
                curriculum_stage=8,
                opponent_tier="e2e_live_test",
                checkpoint_id="e2e_smoke",
            )
            assert summary["entered_game"] is True, f"AI failed to navigate into live IN_GAME state: {summary}"
            assert summary["in_game_steps"] >= 5, f"Too few active in-game steps: {summary}"
            assert summary["decision_accuracy"] >= 0.50, f"Low verified external action accuracy: {summary}"
            assert len(memory.recent_episodes()) == 1
        finally:
            capturer.close()
            controller.close()
