"""Typed runtime configuration models for Territorial.io Autonomous CV AI."""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


class DisplayConfig(BaseModel):
    display_num: int = 99
    width: int = 1280
    height: int = 720
    color_depth: int = 24
    http_port: int = 18080
    game_html_path: Path = Path("text.html")


class VisionConfig(BaseModel):
    obs_grid_size: int = 128
    world_map_size: int = 256
    max_entities: int = 32
    use_neural_seg: bool = True
    use_neural_ocr: bool = True
    ocr_checkpoint: Path = Path("models/checkpoints/ocr_net.pt")
    seg_checkpoint: Path = Path("models/checkpoints/vision_unet.pt")


class ControlConfig(BaseModel):
    human_like_motion: bool = True
    move_steps: int = 8
    move_duration_sec: float = 0.055
    post_click_settle_sec: float = 0.14
    verify_timeout_sec: float = 0.35
    failed_action_cooldown_steps: int = 8


class RLConfig(BaseModel):
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    update_epochs: int = 4
    rollout_steps: int = 64
    num_envs: int = 4
    hidden_dim: int = 192
    spatial_grid: int = 32
    slider_bins: int = 11


class MemoryConfig(BaseModel):
    db_path: Path = Path("memory/ai_memory.db")
    episodes_dir: Path = Path("memory/episodes")
    checkpoints_dir: Path = Path("models/checkpoints")
    max_kept_checkpoints: int = 15


class ProjectConfig(BaseModel):
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    seed: int = 42
