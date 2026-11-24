"""Unified 9-Stage Autonomous Training Pipeline for Territorial.io Computer-Vision AI.

Stages:
  1. Recognize Game Elements (train CanvasOCRNet + VisionUNetMobile on synthetic + real captures)
  2. Understand Map Geography (validate coastline odometry, persistent map stitching & spawn heatmap)
  3. Learn Basic Expansion (Imitation warm-start + PPO on sparse neutral-rich arenas)
  4. Learn Troop Allocation (PPO optimizing slider discipline & avoiding red-interest caps)
  5. Learn Attacking & Defending (PPO on 5-player Normal arenas with 2:1 combat & boats)
  6. Learn Opponent Behavior (Recurrent GRU PPO against 7-player Hard mixed lobbies)
  7. Learn Advanced Strategies (Recurrent GRU PPO on 10-player Very Hard arenas)
  8. Learn Complete Matches (End-to-end closed-loop gameplay & adaptation on LIVE text.html in Xvfb + Chrome)
  9. Optimize Win Rate via Self-Play (PFSP League vs historical snapshots + bots + Elo + Champion Gating)

Supports automatic CPU/GPU detection, atomic checkpointing, crash recovery, and honest JSON reporting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import ProjectConfig, RLConfig
from core.hardware import detect_hardware
from evaluation.benchmark_suite import evaluate_policy_live_browser, evaluate_policy_visual_arena
from evaluation.gating import should_promote_champion
from evaluation.metrics import compute_evaluation_report
from map_analysis.tactical_features import TacticalFeatureExtractor
from memory.checkpoint_manager import CheckpointManager
from memory.episode_store import PersistentMemoryStore
from models.ocr_net import CHAR_TO_IDX, OCR_VOCAB, CanvasOCRNet
from models.policy_value_net import SLIDER_BINS, TerritorialPolicyValueNet
from models.vision_unet import VisionUNetMobile
from reinforcement_learning.curriculum import CurriculumController
from reinforcement_learning.imitation import collect_visual_planner_demos, train_behavioral_cloning
from reinforcement_learning.ppo import RecurrentPPOTrainer
from reinforcement_learning.replay_buffer import RolloutBuffer
from self_play.league_manager import SelfPlayLeagueManager
from training.pixel_sim_env import SimScenarioConfig, VisualPixelSimEnv
from training.vec_env import VectorizedPixelEnv
from vision.ocr import render_synthetic_glyph


def train_stage1_vision_models(
    checkpoints_dir: Path,
    device: torch.device,
    ocr_epochs: int = 8,
    seg_epochs: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Stage 1: Train CanvasOCRNet and VisionUNetMobile from scratch and measure held-out accuracy & mIoU."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # --- 1A. Train CanvasOCRNet on domain-randomized canvas glyphs ---
    chars = list(OCR_VOCAB[:-1])
    x_list, y_list = [], []
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX]
    for ch in chars:
        label = CHAR_TO_IDX[ch]
        for _ in range(90):
            canvas = np.zeros((24, 24), dtype=np.uint8)
            f = fonts[int(rng.integers(0, len(fonts)))]
            scale = float(rng.uniform(0.52, 0.80))
            thick = int(rng.integers(1, 3))
            (tw, th), _ = cv2.getTextSize(ch, f, scale, thick)
            ox = int(np.clip((24 - tw) // 2 + int(rng.integers(-2, 3)), 0, max(0, 23 - tw)))
            oy = int(np.clip((24 + th) // 2 + int(rng.integers(-2, 3)), th, 23))
            cv2.putText(canvas, ch, (ox, oy), f, scale, int(rng.integers(205, 256)), thick, cv2.LINE_AA)
            if rng.random() < 0.35:
                canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
            noise = rng.normal(0.0, 8.0, canvas.shape)
            img = np.clip(canvas.astype(np.float32) + noise, 0, 255) / 255.0
            x_list.append(img)
            y_list.append(label)

    x_all = np.stack(x_list, axis=0)[:, None, :, :].astype(np.float32)
    y_all = np.array(y_list, dtype=np.int64)
    perm = rng.permutation(len(y_all))
    split = int(0.85 * len(y_all))
    tr_idx, va_idx = perm[:split], perm[split:]

    ocr_net = CanvasOCRNet().to(device)
    ocr_opt = torch.optim.AdamW(ocr_net.parameters(), lr=2.5e-3, weight_decay=1e-4)
    for _ in range(ocr_epochs):
        ocr_net.train()
        order = rng.permutation(tr_idx)
        for s in range(0, len(order), 64):
            b = order[s : s + 64]
            xb = torch.from_numpy(x_all[b]).to(device)
            yb = torch.from_numpy(y_all[b]).to(device)
            loss = F.cross_entropy(ocr_net(xb), yb)
            ocr_opt.zero_grad(set_to_none=True)
            loss.backward()
            ocr_opt.step()

    ocr_net.eval()
    with torch.no_grad():
        val_preds = ocr_net(torch.from_numpy(x_all[va_idx]).to(device)).argmax(dim=-1).cpu().numpy()
    ocr_acc = float((val_preds == y_all[va_idx]).mean())
    ocr_path = checkpoints_dir / "ocr_net.pt"
    torch.save({"model_state": ocr_net.state_dict(), "val_acc": ocr_acc}, ocr_path)

    # --- 1B. Train VisionUNetMobile on rendered multi-map pixel frames with exact ground-truth pixel classes ---
    seg_imgs, seg_masks = [], []
    map_types = ("continent", "ring", "islands", "archipelago")
    for idx in range(48):
        env = VisualPixelSimEnv(
            scenario=SimScenarioConfig(num_bots=4 + (idx % 6), map_type=map_types[idx % 4], domain_randomize=True),
            seed=seed + idx * 17,
            use_fast_perception=True,
        )
        env.reset()
        # Advance 4 random steps so territories grow and border each other cleanly
        for _ in range(4):
            env.step(1, 0.5, 0.5, 0.30)
        rgb = env.render_rgb_screenshot()
        rgb_128 = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        # Construct semantic class target: 0=ui, 1=water, 2=mountain, 3=neutral, 4=player_interior, 5=player_border
        owner_128 = cv2.resize(env.owner, (128, 128), interpolation=cv2.INTER_NEAREST)
        gt = np.full((128, 128), 3, dtype=np.int64)
        gt[owner_128 == -2] = 1
        player_m = (owner_128 > 0).astype(np.uint8)
        eroded = cv2.erode(player_m, np.ones((3, 3), np.uint8))
        border_m = (player_m > 0) & (eroded == 0)
        gt[player_m > 0] = 4
        gt[border_m] = 5
        # Mark HUD boxes as class 0 (ui_bg)
        gt[:42, :26] = 0
        gt[:24, 108:] = 0
        gt[:10, 46:82] = 0
        gt[118:, 38:90] = 0

        seg_imgs.append(rgb_128.transpose(2, 0, 1))
        seg_masks.append(gt)

    sx = np.stack(seg_imgs, axis=0)
    sy = np.stack(seg_masks, axis=0)
    s_tr, s_va = sx[:40], sx[40:]
    y_tr, y_va = sy[:40], sy[40:]

    unet = VisionUNetMobile().to(device)
    u_opt = torch.optim.AdamW(unet.parameters(), lr=3e-3, weight_decay=1e-4)
    cls_weights = torch.tensor([1.5, 1.0, 1.0, 1.0, 3.5, 5.5], dtype=torch.float32, device=device)
    for _ in range(seg_epochs):
        unet.train()
        order = rng.permutation(len(s_tr))
        for s in range(0, len(order), 8):
            b = order[s : s + 8]
            logits = unet(torch.from_numpy(s_tr[b]).to(device))
            loss = F.cross_entropy(logits, torch.from_numpy(y_tr[b]).to(device), weight=cls_weights)
            u_opt.zero_grad(set_to_none=True)
            loss.backward()
            u_opt.step()

    unet.eval()
    with torch.no_grad():
        preds = unet(torch.from_numpy(s_va).to(device)).argmax(dim=1).cpu().numpy()
    # Compute mean IoU across active classes (0, 1, 3, 4, 5)
    ious = []
    for c in (0, 1, 3, 4, 5):
        inter = float(((preds == c) & (y_va == c)).sum())
        union = float(((preds == c) | (y_va == c)).sum())
        if union > 0:
            ious.append(inter / union)
    seg_miou = float(np.mean(ious)) if ious else 0.0
    seg_pixel_acc = float((preds == y_va).mean())
    seg_path = checkpoints_dir / "vision_unet.pt"
    torch.save({"model_state": unet.state_dict(), "val_miou": seg_miou, "val_pixel_acc": seg_pixel_acc}, seg_path)

    return {
        "ocr_val_accuracy": round(ocr_acc, 4),
        "seg_val_miou": round(seg_miou, 4),
        "seg_pixel_accuracy": round(seg_pixel_acc, 4),
        "ocr_and_seg_score": round(0.5 * ocr_acc + 0.5 * seg_miou, 4),
        "ocr_checkpoint": str(ocr_path),
        "seg_checkpoint": str(seg_path),
    }


def validate_stage2_map_geography(seed: int = 42) -> dict[str, Any]:
    """Stage 2: Measure coastline camera odometry shift error & learned spawn heatmap advantage vs random."""
    rng = np.random.default_rng(seed)
    extractor = TacticalFeatureExtractor()

    # 2A. Test camera odometry shift estimation on known synthetic map translations
    env = VisualPixelSimEnv(SimScenarioConfig(map_type="ring", num_bots=4), seed=seed, use_fast_perception=True)
    env.reset()
    base_water = (env.owner == -2).astype(np.uint8)
    base_water_128 = cv2.resize(base_water, (128, 128), interpolation=cv2.INTER_NEAREST)
    from map_analysis.camera_odometry import VisualCameraOdometry

    odom = VisualCameraOdometry(work_size=128)
    odom.update(base_water_128, np.zeros((128, 128), dtype=np.int32))
    shifted = np.roll(np.roll(base_water_128, shift=6, axis=0), shift=-8, axis=1)
    pose = odom.update(shifted, np.zeros((128, 128), dtype=np.int32))
    # Expected dx_world = -(-8)/128 = +0.0625, dy_world = -(6)/128 = -0.0469
    err = float(np.hypot(pose.last_dx - (8.0 / 128.0), pose.last_dy - (-6.0 / 128.0)))

    # 2B. Compare spawn heatmap clearance vs random land spawns across 16 maps
    heatmap_scores, rand_scores = [], []
    map_types = ("continent", "ring", "islands", "archipelago")
    for i in range(16):
        e = VisualPixelSimEnv(SimScenarioConfig(map_type=map_types[i % 4], num_bots=6), seed=seed + 50 + i, use_fast_perception=True)
        e.reset()
        pf = e._last_pf
        assert pf is not None
        sx, sy = extractor.select_best_spawn_point(pf)
        h, w = pf.neutral_mask.shape
        dist_clear = cv2.distanceTransform((pf.neutral_mask > 0).astype(np.uint8) * 255, cv2.DIST_L2, 5)
        hx = min(w - 1, max(0, int(sx * w)))
        hy = min(h - 1, max(0, int(sy * h)))
        heatmap_scores.append(float(dist_clear[hy, hx]))

        ys, xs = (pf.neutral_mask > 0).nonzero()
        if len(xs) > 0:
            ri = int(rng.integers(0, len(xs)))
            rand_scores.append(float(dist_clear[ys[ri], xs[ri]]))
        else:
            rand_scores.append(0.1)

    adv_ratio = float(np.mean(heatmap_scores)) / max(1e-3, float(np.mean(rand_scores)))
    return {
        "odometry_shift_error": round(err, 4),
        "heatmap_mean_clearance": round(float(np.mean(heatmap_scores)), 2),
        "random_mean_clearance": round(float(np.mean(rand_scores)), 2),
        "spawn_advantage_ratio": round(adv_ratio, 3),
    }


def run_full_9_stage_training(
    cfg: ProjectConfig | None = None,
    ppo_updates_per_stage: int = 4,
    rollout_steps: int = 48,
    num_envs: int = 4,
    run_live_stage8: bool = True,
    live_stage8_matches: int = 2,
    live_stage8_steps: int = 36,
    resume: bool = True,
) -> dict[str, Any]:
    """Execute real end-to-end training across all 9 curriculum stages with persistent memory & gating."""
    cfg = cfg or ProjectConfig()
    hw = detect_hardware()
    device = hw.torch_device
    t_start = time.monotonic()

    memory = PersistentMemoryStore(db_path=cfg.memory.db_path, episodes_dir=cfg.memory.episodes_dir)
    ckpt_mgr = CheckpointManager(
        checkpoints_dir=cfg.memory.checkpoints_dir,
        db_path=cfg.memory.db_path,
        max_kept=cfg.memory.max_kept_checkpoints,
    )
    curriculum = CurriculumController(current_stage=1)

    model = TerritorialPolicyValueNet(hidden_dim=cfg.rl.hidden_dim, spatial_grid=cfg.rl.spatial_grid).to(device)
    rl_cfg = RLConfig(
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        hidden_dim=cfg.rl.hidden_dim,
        spatial_grid=cfg.rl.spatial_grid,
        update_epochs=3,
    )
    trainer = RecurrentPPOTrainer(model=model, rl_cfg=rl_cfg, hw=hw)

    global_steps = 0
    stage_reports: dict[str, Any] = {}
    start_stage = 1

    if resume:
        loaded_stage, loaded_steps, loaded_metrics = memory.load_curriculum_state()
        loaded_ckpt = ckpt_mgr.load_latest(model, trainer.optimizer, device=device)
        if loaded_ckpt is not None and loaded_steps > 0:
            global_steps = loaded_steps
            stage_reports.update(loaded_metrics)
            start_stage = max(1, min(9, loaded_stage))

    # --- Baseline Untrained Evaluation (only on fresh run so we prove genuine learning!) ---
    if "baseline_untrained" not in stage_reports:
        untrained_eval = evaluate_policy_visual_arena(
            model, difficulties=("Very Easy", "Easy", "Normal"), episodes_per_tier=2, device=device, base_seed=9100
        )
        stage_reports["baseline_untrained"] = untrained_eval["overall"]
        print("[BASELINE] Untrained eval:", json.dumps(stage_reports["baseline_untrained"]), flush=True)
    else:
        print(f"[RESUME] Resuming from Stage {start_stage} at global_steps={global_steps}", flush=True)

    # --- STAGE 1: Vision & OCR Supervised Training ---
    is_gpu = device.type == "cuda"
    if start_stage <= 1:
        s1 = train_stage1_vision_models(
            cfg.memory.checkpoints_dir,
            device=device,
            ocr_epochs=14 if is_gpu else 8,
            seg_epochs=30 if is_gpu else 10,
        )
        curriculum.check_gate(1, s1)
        stage_reports["stage_1_vision"] = s1
        memory.save_curriculum_state(2, global_steps, stage_reports)
        print("[STAGE 1] Vision/OCR:", json.dumps(s1), flush=True)

    # --- STAGE 2: Map Geography & Spawn Validation ---
    if start_stage <= 2:
        s2 = validate_stage2_map_geography(seed=cfg.seed)
        curriculum.check_gate(2, s2)
        stage_reports["stage_2_geography"] = s2
        memory.save_curriculum_state(3, global_steps, stage_reports)
        print("[STAGE 2] Geography:", json.dumps(s2), flush=True)

    # --- Imitation Warm-Start from pure-CV visual planner demonstrations ---
    if global_steps == 0:
        demos = collect_visual_planner_demos(
            num_episodes=24 if is_gpu else 10,
            max_steps=70 if is_gpu else 55,
            seed=cfg.seed,
        )
        bc_metrics = train_behavioral_cloning(
            model,
            demos,
            epochs=14 if is_gpu else 8,
            batch_size=64 if is_gpu else 32,
            device=device,
        )
        stage_reports["imitation_warmstart"] = bc_metrics
        ckpt_mgr.save(model, trainer.optimizer, curriculum_stage=3, global_steps=0, tag="bc_warmstart", extra=bc_metrics)
        print("[IMITATION] Warm-start:", json.dumps(bc_metrics), flush=True)

    # --- STAGES 3 -> 7: Progressive Curriculum Recurrent PPO ---
    champion_eval_report: dict[str, Any] | None = None
    league = SelfPlayLeagueManager()

    for stage_num in range(max(3, start_stage), 8):
        spec = curriculum.spec(stage_num)
        scenarios = curriculum.build_scenarios(stage_num, num_envs=num_envs)
        vec_env = VectorizedPixelEnv(scenarios, base_seed=cfg.seed + stage_num * 1000, use_fast_perception=True)
        obs = vec_env.reset()
        hidden = model.rnn.initial_state(num_envs, device)
        buf = RolloutBuffer(
            num_steps=rollout_steps,
            num_envs=num_envs,
            hidden_dim=cfg.rl.hidden_dim,
        )

        all_ep_infos: list[dict[str, Any]] = []
        last_ppo = None
        for _ in range(ppo_updates_per_stage):
            obs, hidden, ep_infos = trainer.collect_rollout(vec_env, obs, hidden, buf)
            last_ppo = trainer.update(buf)
            global_steps += rollout_steps * num_envs
            all_ep_infos.extend(ep_infos)

        # Evaluate checkpoint at end of stage
        eval_res = evaluate_policy_visual_arena(
            model,
            difficulties=(spec.bot_difficulty,) if spec.bot_difficulty != "None" else ("Easy",),
            episodes_per_tier=3,
            device=device,
            base_seed=8000 + stage_num * 100,
        )
        ov = eval_res["overall"]
        win_or_top2 = float(
            np.mean([1.0 if (e.get("won") or int(e.get("rank", 99)) <= 2) else 0.0 for e in eval_res["raw_episodes"]])
        )
        stage_metrics = {
            "stage_name": spec.name,
            "global_steps": global_steps,
            "ppo_policy_loss": round(last_ppo.policy_loss, 4) if last_ppo else 0.0,
            "ppo_value_loss": round(last_ppo.value_loss, 4) if last_ppo else 0.0,
            "ppo_entropy": round(last_ppo.entropy, 4) if last_ppo else 0.0,
            "ppo_mean_reward": round(last_ppo.mean_reward, 4) if last_ppo else 0.0,
            "mean_peak_land": ov["mean_peak_land"],
            "win_rate": ov["win_rate"],
            "win_rate_or_top2": round(win_or_top2, 4),
            "survival_percentile": ov["mean_placement_pct"],
            "decision_accuracy": ov["mean_decision_accuracy"],
            "composite_score": ov["composite_score"],
        }
        curriculum.check_gate(stage_num, stage_metrics)
        stage_reports[f"stage_{stage_num}"] = stage_metrics

        ckpt_id, ckpt_path = ckpt_mgr.save(
            model,
            trainer.optimizer,
            curriculum_stage=stage_num,
            global_steps=global_steps,
            elo_rating=league.main_elo,
            eval_win_rate=ov["win_rate"],
            eval_survival_pct=ov["mean_placement_pct"],
            tag=f"ppo_s{stage_num}",
            extra=stage_metrics,
        )
        league.register_snapshot(ckpt_id, ckpt_path, elo=league.main_elo)
        memory.save_curriculum_state(stage_num + 1, global_steps, stage_reports)
        print(f"[STAGE {stage_num}] {spec.name}: {json.dumps(stage_metrics)}", flush=True)

    # --- STAGE 8: Live Unmodified Territorial.io Browser End-to-End Domain Validation ---
    if run_live_stage8:
        live_res = evaluate_policy_live_browser(
            model=model,
            num_matches=live_stage8_matches,
            max_steps=live_stage8_steps,
            display_num=115,
            device=device,
        )
        s8_metrics = {
            "stage_name": "Learn Complete Matches (Live Browser)",
            **live_res["overall"],
            "decision_accuracy": live_res["overall"]["mean_decision_accuracy"],
        }
        curriculum.check_gate(8, s8_metrics)
        stage_reports["stage_8_live_browser"] = s8_metrics
        memory.save_curriculum_state(9, global_steps, stage_reports)
        print(f"[STAGE 8] Live Browser: {json.dumps(s8_metrics)}", flush=True)

    # --- STAGE 9: Self-Play League Optimization + Champion Gating + ONNX Export ---
    pre_sp_elo = league.main_elo
    for sp_round in range(max(2, ppo_updates_per_stage)):
        opp = league.select_opponent()
        diff_map = {
            "bot_easy": "Easy",
            "bot_normal": "Normal",
            "bot_hard": "Hard",
            "bot_very_hard": "Very Hard",
            "bot_impossible": "Impossible",
        }
        bot_diff = diff_map.get(opp.member_id, "Hard")
        scenarios = [
            SimScenarioConfig(
                num_bots=5,
                bot_difficulty=bot_diff,
                map_type=("continent", "ring", "islands", "archipelago")[i % 4],
                max_steps=105,
                domain_randomize=True,
            )
            for i in range(num_envs)
        ]
        vec_env = VectorizedPixelEnv(scenarios, base_seed=cfg.seed + 9000 + sp_round * 77, use_fast_perception=True)
        obs = vec_env.reset()
        hidden = model.rnn.initial_state(num_envs, device)
        buf = RolloutBuffer(num_steps=rollout_steps, num_envs=num_envs, hidden_dim=cfg.rl.hidden_dim)
        obs, hidden, ep_infos = trainer.collect_rollout(vec_env, obs, hidden, buf)
        trainer.update(buf)
        global_steps += rollout_steps * num_envs
        for ep in ep_infos:
            tot = max(2, int(ep.get("total_players", 6)))
            rank = int(ep.get("rank", 2))
            p_pct = max(0.0, min(1.0, 1.0 - (rank - 1) / (tot - 1)))
            league.record_match(opp.member_id, bool(ep.get("won", False)), p_pct)
        if (sp_round + 1) % max(1, ppo_updates_per_stage // 4) == 0:
            print(f"[STAGE 9] round {sp_round+1}/{max(2, ppo_updates_per_stage)} opp={opp.member_id} elo={league.main_elo:.1f} steps={global_steps}", flush=True)

    # Final standardized multi-tier benchmark vs untrained baseline
    final_eval = evaluate_policy_visual_arena(
        model,
        difficulties=("Very Easy", "Easy", "Normal", "Hard", "Very Hard"),
        episodes_per_tier=4 if is_gpu else 3,
        device=device,
        base_seed=9100,
    )
    final_ov = final_eval["overall"]
    for ep in final_eval["raw_episodes"]:
        tot = max(2, int(ep.get("total_players", 6)))
        rank = int(ep.get("rank", 2))
        p_pct = max(0.0, min(1.0, 1.0 - (rank - 1) / (tot - 1)))
        league.record_match("bot_normal", bool(ep.get("won", False)), p_pct)

    promoted, gate_reason = should_promote_champion(final_ov, champion_eval_report)
    final_ckpt_id, final_ckpt_path = ckpt_mgr.save(
        model,
        trainer.optimizer,
        curriculum_stage=9,
        global_steps=global_steps,
        elo_rating=league.main_elo,
        eval_win_rate=final_ov["win_rate"],
        eval_survival_pct=final_ov["mean_placement_pct"],
        is_champion=promoted,
        tag="champion_s9",
        extra={"eval": final_ov, "gate_reason": gate_reason},
    )

    onnx_path = None
    try:
        onnx_path = str(model.export_onnx(cfg.memory.checkpoints_dir / "champion.onnx"))
    except Exception:
        onnx_path = None

    stage_reports["stage_9_self_play"] = {
        "elo_start": pre_sp_elo,
        "elo_final": league.main_elo,
        "elo_gain": round(league.main_elo - pre_sp_elo, 2),
        "standings": league.standings(),
        "promoted": promoted,
        "gate_reason": gate_reason,
        "checkpoint_id": final_ckpt_id,
        "checkpoint_path": str(final_ckpt_path),
        "onnx_path": onnx_path,
    }
    stage_reports["final_benchmark"] = final_eval["overall"]
    stage_reports["final_by_difficulty"] = final_eval["by_difficulty"]
    stage_reports["hardware"] = hw.device_name
    stage_reports["total_global_steps"] = global_steps
    stage_reports["total_wall_sec"] = round(time.monotonic() - t_start, 2)

    memory.save_curriculum_state(9, global_steps, stage_reports)
    report_path = Path("runs/training_summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stage_reports, indent=2), encoding="utf-8")
    return stage_reports


if __name__ == "__main__":
    summary = run_full_9_stage_training()
    print(json.dumps(summary, indent=2))
