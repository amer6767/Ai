"""9-Stage Curriculum Learning Controller with exact scenario specifications and promotion gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from training.pixel_sim_env import SimScenarioConfig


@dataclass(frozen=True)
class CurriculumStageSpec:
    stage: int
    name: str
    description: str
    env_mode: str  # 'vision_supervised', 'map_geography', 'pixel_sim', 'live_browser', 'self_play'
    num_bots: int
    bot_difficulty: str
    map_types: tuple[str, ...]
    max_steps: int
    gate_metric: str
    gate_threshold: float


STAGES: tuple[CurriculumStageSpec, ...] = (
    CurriculumStageSpec(
        stage=1,
        name="Recognize Game Elements",
        description="Supervised training of CanvasOCRNet & VisionUNetMobile on real + synthetic Territorial.io frames.",
        env_mode="vision_supervised",
        num_bots=0,
        bot_difficulty="None",
        map_types=("continent",),
        max_steps=1,
        gate_metric="ocr_and_seg_score",
        gate_threshold=0.88,
    ),
    CurriculumStageSpec(
        stage=2,
        name="Understand Map Geography",
        description="Verify coastline camera odometry, persistent map stitching, and spawn heatmap advantage.",
        env_mode="map_geography",
        num_bots=6,
        bot_difficulty="Easy",
        map_types=("continent", "ring", "islands", "archipelago"),
        max_steps=40,
        gate_metric="spawn_advantage_ratio",
        gate_threshold=1.10,
    ),
    CurriculumStageSpec(
        stage=3,
        name="Learn Basic Expansion",
        description="Sparse opening maps: master fast neutral perimeter expansion without zeroing troops.",
        env_mode="pixel_sim",
        num_bots=2,
        bot_difficulty="Very Easy",
        map_types=("continent", "ring"),
        max_steps=65,
        gate_metric="mean_peak_land",
        gate_threshold=0.32,
    ),
    CurriculumStageSpec(
        stage=4,
        name="Learn Troop Allocation",
        description="Optimize slider commitment percentages for compound interest while avoiding red-interest caps.",
        env_mode="pixel_sim",
        num_bots=3,
        bot_difficulty="Easy",
        map_types=("continent", "ring", "islands"),
        max_steps=80,
        gate_metric="mean_peak_land",
        gate_threshold=0.30,
    ),
    CurriculumStageSpec(
        stage=5,
        name="Learn Attacking & Defending",
        description="4-to-6 player skirmishes mastering 2:1 defender advantage, border defense, and naval raids.",
        env_mode="pixel_sim",
        num_bots=4,
        bot_difficulty="Normal",
        map_types=("continent", "islands", "ring"),
        max_steps=95,
        gate_metric="win_rate_or_top2",
        gate_threshold=0.35,
    ),
    CurriculumStageSpec(
        stage=6,
        name="Learn Opponent Behavior",
        description="Recurrent GRU memory against mixed aggressive/passive bots and full-send punishment.",
        env_mode="pixel_sim",
        num_bots=6,
        bot_difficulty="Hard",
        map_types=("continent", "ring", "islands", "archipelago"),
        max_steps=105,
        gate_metric="survival_percentile",
        gate_threshold=0.55,
    ),
    CurriculumStageSpec(
        stage=7,
        name="Learn Advanced Strategies",
        description="High-density 10-player arenas (Very Hard / Impossible): encirclement, vulturing & endgame closing.",
        env_mode="pixel_sim",
        num_bots=9,
        bot_difficulty="Very Hard",
        map_types=("continent", "ring", "islands", "archipelago"),
        max_steps=115,
        gate_metric="survival_percentile",
        gate_threshold=0.55,
    ),
    CurriculumStageSpec(
        stage=8,
        name="Learn Complete Matches (Live Browser)",
        description="End-to-end closed-loop gameplay and domain adaptation on live Territorial.io via Xvfb + X11 mouse.",
        env_mode="live_browser",
        num_bots=511,
        bot_difficulty="Live",
        map_types=("live_browser",),
        max_steps=80,
        gate_metric="decision_accuracy",
        gate_threshold=0.70,
    ),
    CurriculumStageSpec(
        stage=9,
        name="Optimize Win Rate via Self-Play",
        description="Prioritized Fictitious Self-Play league vs past checkpoints + Hard/Impossible bots with Elo gating.",
        env_mode="self_play",
        num_bots=5,
        bot_difficulty="Hard",
        map_types=("continent", "ring", "islands", "archipelago"),
        max_steps=110,
        gate_metric="elo_gain",
        gate_threshold=10.0,
    ),
)


class CurriculumController:
    """Builds per-stage vectorized scenarios and evaluates promotion gates."""

    def __init__(self, current_stage: int = 1) -> None:
        self.current_stage = max(1, min(9, current_stage))
        self.history: dict[int, dict[str, Any]] = {}

    def spec(self, stage: int | None = None) -> CurriculumStageSpec:
        idx = (stage or self.current_stage) - 1
        return STAGES[max(0, min(len(STAGES) - 1, idx))]

    def build_scenarios(self, stage: int, num_envs: int) -> list[SimScenarioConfig]:
        sp = self.spec(stage)
        scenarios: list[SimScenarioConfig] = []
        for i in range(num_envs):
            scenarios.append(
                SimScenarioConfig(
                    grid_size=48,
                    render_w=384,
                    render_h=256,
                    num_bots=sp.num_bots,
                    bot_difficulty=sp.bot_difficulty,
                    map_type=sp.map_types[i % len(sp.map_types)],
                    max_steps=sp.max_steps,
                    domain_randomize=True,
                )
            )
        return scenarios

    def check_gate(self, stage: int, metrics: dict[str, Any]) -> bool:
        sp = self.spec(stage)
        val = float(metrics.get(sp.gate_metric, 0.0))
        passed = val >= sp.gate_threshold
        self.history[stage] = {"passed": passed, "metric": sp.gate_metric, "value": val, "threshold": sp.gate_threshold, **metrics}
        if passed and stage == self.current_stage and self.current_stage < 9:
            self.current_stage += 1
        return passed
