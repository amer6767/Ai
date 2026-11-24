"""Unified CLI entrypoint for Territorial.io Autonomous Computer-Vision AI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import uvicorn

from computer_control.display_manager import GameDisplaySession
from computer_control.mouse_keyboard import ExternalInputController
from computer_control.screen_capture import ScreenCapturer
from core.closed_loop import ClosedLoopAgent
from core.config import ProjectConfig
from core.hardware import detect_hardware
from evaluation.benchmark_suite import evaluate_policy_live_browser, evaluate_policy_visual_arena
from memory.checkpoint_manager import CheckpointManager
from memory.episode_store import PersistentMemoryStore
from models.policy_value_net import TerritorialPolicyValueNet
from perception.state_assembler import VisualPerceptionPipeline
from strategic_brain.arbiter import StrategicBrain
from training.cloud.kaggle_runner import KaggleGPUOrchestrator
from training.train_pipeline import run_full_9_stage_training


def _load_model(prefer_champion: bool = True) -> tuple[TerritorialPolicyValueNet, str]:
    hw = detect_hardware()
    model = TerritorialPolicyValueNet().to(hw.torch_device)
    ckpt_mgr = CheckpointManager()
    meta = ckpt_mgr.load_latest(model, prefer_champion=prefer_champion, device=hw.torch_device)
    ckpt_id = meta["checkpoint_id"] if meta else "untrained_init"
    return model, ckpt_id


def cmd_train(args: argparse.Namespace) -> None:
    res = run_full_9_stage_training(
        ppo_updates_per_stage=args.updates_per_stage,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        run_live_stage8=not args.skip_live_stage8,
        live_stage8_matches=args.live_matches,
        live_stage8_steps=args.live_steps,
        resume=not args.fresh,
    )
    print(json.dumps(res, indent=2))


def cmd_play(args: argparse.Namespace) -> None:
    cfg = ProjectConfig()
    hw = detect_hardware()
    model, ckpt_id = _load_model(prefer_champion=True)
    memory = PersistentMemoryStore()
    perception = VisualPerceptionPipeline(config=cfg.vision, device=hw.device_str)
    brain = StrategicBrain(model=model, device=hw.torch_device, deterministic=True)

    with GameDisplaySession(
        game_html_path=args.html,
        display_num=args.display,
        width=args.width,
        height=args.height,
    ) as sess:
        capturer = ScreenCapturer(display_str=sess.display_str)
        controller = ExternalInputController(display_str=sess.display_str, config=cfg.control)
        agent = ClosedLoopAgent(
            capturer=capturer,
            controller=controller,
            perception=perception,
            brain=brain,
            memory=memory,
            control_cfg=cfg.control,
        )
        try:
            summary = agent.run_episode(
                max_steps=args.max_steps,
                curriculum_stage=9,
                opponent_tier="live_play",
                checkpoint_id=ckpt_id,
            )
            print(json.dumps(summary, indent=2))
        finally:
            capturer.close()
            controller.close()


def cmd_eval(args: argparse.Namespace) -> None:
    hw = detect_hardware()
    model, ckpt_id = _load_model(prefer_champion=True)
    if args.mode == "live":
        res = evaluate_policy_live_browser(
            model=model,
            num_matches=args.matches,
            max_steps=args.max_steps,
            display_num=args.display,
            device=hw.torch_device,
        )
    else:
        res = evaluate_policy_visual_arena(
            model=model,
            difficulties=tuple(args.difficulties.split(",")),
            episodes_per_tier=args.matches,
            device=hw.torch_device,
        )
    res["checkpoint_id"] = ckpt_id
    print(json.dumps(res, indent=2))


def cmd_kaggle_bundle(args: argparse.Namespace) -> None:
    orch = KaggleGPUOrchestrator(username=args.username, kernel_slug=args.slug)
    bundle = orch.build_kernel_bundle(
        out_dir=args.out_dir,
        ppo_updates_per_stage=args.updates_per_stage,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
    )
    print(f"Kaggle GPU kernel bundle created at: {bundle}")
    if args.push:
        res = orch.push_and_monitor(bundle)
        print(json.dumps(res, indent=2))


def cmd_dashboard(args: argparse.Namespace) -> None:
    uvicorn.run("dashboard.server:app", host=args.host, port=args.port, reload=False)


def main() -> None:
    parser = argparse.ArgumentParser(prog="territorial-ai", description="Territorial.io Autonomous Computer-Vision AI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Run full 9-stage autonomous training pipeline")
    p_train.add_argument("--updates-per-stage", type=int, default=4)
    p_train.add_argument("--rollout-steps", type=int, default=48)
    p_train.add_argument("--num-envs", type=int, default=4)
    p_train.add_argument("--live-matches", type=int, default=2)
    p_train.add_argument("--live-steps", type=int, default=36)
    p_train.add_argument("--skip-live-stage8", action="store_true")
    p_train.add_argument("--fresh", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_play = sub.add_parser("play", help="Play live unmodified Territorial.io autonomously via external X11 controls")
    p_play.add_argument("--html", type=str, default="text.html")
    p_play.add_argument("--display", type=int, default=130)
    p_play.add_argument("--width", type=int, default=1280)
    p_play.add_argument("--height", type=int, default=720)
    p_play.add_argument("--max-steps", type=int, default=120)
    p_play.set_defaults(func=cmd_play)

    p_eval = sub.add_parser("eval", help="Benchmark current champion model")
    p_eval.add_argument("--mode", choices=("arena", "live"), default="arena")
    p_eval.add_argument("--difficulties", type=str, default="Very Easy,Easy,Normal,Hard")
    p_eval.add_argument("--matches", type=int, default=3)
    p_eval.add_argument("--max-steps", type=int, default=45)
    p_eval.add_argument("--display", type=int, default=135)
    p_eval.set_defaults(func=cmd_eval)

    p_kg = sub.add_parser("kaggle", help="Generate or push Kaggle GPU training kernel")
    p_kg.add_argument("--username", type=str, required=True)
    p_kg.add_argument("--slug", type=str, default="territorial-vision-ai-train")
    p_kg.add_argument("--out-dir", type=str, default="runs/kaggle_kernel")
    p_kg.add_argument("--updates-per-stage", type=int, default=24)
    p_kg.add_argument("--rollout-steps", type=int, default=96)
    p_kg.add_argument("--num-envs", type=int, default=8)
    p_kg.add_argument("--push", action="store_true")
    p_kg.set_defaults(func=cmd_kaggle_bundle)

    p_dash = sub.add_parser("dashboard", help="Launch live telemetry dashboard")
    p_dash.add_argument("--host", type=str, default="0.0.0.0")
    p_dash.add_argument("--port", type=int, default=8500)
    p_dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
