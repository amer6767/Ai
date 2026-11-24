"""Clipped Recurrent PPO Trainer with Factorized Hybrid Actions, Masking, Auxiliary Losses & AMP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import RLConfig
from core.hardware import HardwareProfile, detect_hardware
from models.policy_value_net import SLIDER_BINS, TerritorialPolicyValueNet
from reinforcement_learning.replay_buffer import PrioritizedAuxiliaryReplay, PrioritySample, RolloutBuffer
from training.vec_env import BatchedObservation, VectorizedPixelEnv


@dataclass
class PPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    aux_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    mean_reward: float


class RecurrentPPOTrainer:
    """Trains TerritorialPolicyValueNet via clipped PPO + GRU BPTT + auxiliary replay distillation."""

    def __init__(
        self,
        model: TerritorialPolicyValueNet,
        rl_cfg: RLConfig | None = None,
        hw: HardwareProfile | None = None,
    ) -> None:
        self.cfg = rl_cfg or RLConfig()
        self.hw = hw or detect_hardware()
        self.device = self.hw.torch_device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate, eps=1e-5, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.hw.supports_amp)
        self.aux_replay = PrioritizedAuxiliaryReplay(capacity=2048)
        self._rng = np.random.default_rng(42)

    def collect_rollout(
        self,
        vec_env: VectorizedPixelEnv,
        obs: BatchedObservation,
        hidden: torch.Tensor,
        buffer: RolloutBuffer,
    ) -> tuple[BatchedObservation, torch.Tensor, list[dict[str, Any]]]:
        self.model.eval()
        buffer.reset(hidden)
        ep_summaries: list[dict[str, Any]] = []
        g = self.model.spatial_grid

        prev_dones = torch.zeros(vec_env.num_envs, device=self.device)
        for _ in range(buffer.num_steps):
            with torch.no_grad():
                out = self.model.forward_step(
                    obs_map=torch.from_numpy(obs.obs_map).to(self.device),
                    node_feats=torch.from_numpy(obs.node_feats).to(self.device),
                    adj=torch.from_numpy(obs.adj).to(self.device),
                    node_mask=torch.from_numpy(obs.node_mask).to(self.device),
                    scalars=torch.from_numpy(obs.scalars).to(self.device),
                    hidden=hidden,
                    dones=prev_dones,
                    action_mask=torch.from_numpy(obs.action_mask).to(self.device),
                    spatial_masks=torch.from_numpy(obs.spatial_masks).to(self.device),
                    deterministic=False,
                )
            hidden = out.next_hidden
            a_np = out.action_type.cpu().numpy()
            sp_np = out.spatial_idx.cpu().numpy()
            sl_np = out.slider_idx.cpu().numpy()
            lp_np = out.log_prob.cpu().numpy()
            val_np = out.value.cpu().numpy()

            tx_np = ((sp_np % g) + 0.5) / float(g)
            ty_np = ((sp_np // g) + 0.5) / float(g)
            pct_np = np.array([SLIDER_BINS[int(i)] for i in sl_np], dtype=np.float32)

            next_obs, rewards, dones, infos = vec_env.step(a_np, tx_np, ty_np, pct_np)
            land_ratios = np.array([float(inf.get("land_ratio", 0.0)) for inf in infos], dtype=np.float32)

            buffer.add(
                obs=obs,
                action_type=a_np,
                spatial_idx=sp_np,
                slider_idx=sl_np,
                log_prob=lp_np,
                value=val_np,
                reward=rewards,
                done=dones,
                land_ratio=land_ratios,
            )

            for inf in infos:
                if inf.get("terminal"):
                    ep_summaries.append(inf)

            obs = next_obs
            prev_dones = torch.from_numpy(dones.astype(np.float32)).to(self.device)

        # Bootstrap last values
        with torch.no_grad():
            last_out = self.model.forward_step(
                obs_map=torch.from_numpy(obs.obs_map).to(self.device),
                node_feats=torch.from_numpy(obs.node_feats).to(self.device),
                adj=torch.from_numpy(obs.adj).to(self.device),
                node_mask=torch.from_numpy(obs.node_mask).to(self.device),
                scalars=torch.from_numpy(obs.scalars).to(self.device),
                hidden=hidden,
                dones=prev_dones,
                action_mask=torch.from_numpy(obs.action_mask).to(self.device),
                spatial_masks=torch.from_numpy(obs.spatial_masks).to(self.device),
                deterministic=True,
            )
            last_vals = last_out.value.cpu().numpy()

        buffer.compute_gae(last_vals, gamma=self.cfg.gamma, gae_lambda=self.cfg.gae_lambda)

        # Push highest-TD-surprise transitions to prioritized auxiliary replay for representation stability
        td_abs = np.abs(buffer.advantages[: buffer.ptr])
        flat_idx = np.argsort(td_abs.ravel())[::-1][:16]
        for fi in flat_idx:
            t_i, e_i = divmod(int(fi), buffer.num_envs)
            self.aux_replay.push(
                PrioritySample(
                    obs_map=buffer.obs_map[t_i, e_i].copy(),
                    node_feats=buffer.node_feats[t_i, e_i].copy(),
                    adj=buffer.adj[t_i, e_i].copy(),
                    node_mask=buffer.node_mask[t_i, e_i].copy(),
                    scalars=buffer.scalars[t_i, e_i].copy(),
                    target_value=float(buffer.returns[t_i, e_i]),
                    target_land=float(buffer.land_targets[t_i, e_i]),
                    priority=float(td_abs[t_i, e_i]) + 0.05,
                )
            )

        return obs, hidden, ep_summaries

    def update(self, buffer: RolloutBuffer) -> PPOUpdateMetrics:
        self.model.train()
        t_len = buffer.ptr
        b_envs = buffer.num_envs

        adv = buffer.advantages[:t_len]
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_map_t = torch.from_numpy(buffer.obs_map[:t_len]).to(self.device)
        node_feats_t = torch.from_numpy(buffer.node_feats[:t_len]).to(self.device)
        adj_t = torch.from_numpy(buffer.adj[:t_len]).to(self.device)
        node_mask_t = torch.from_numpy(buffer.node_mask[:t_len]).to(self.device)
        scalars_t = torch.from_numpy(buffer.scalars[:t_len]).to(self.device)
        act_mask_t = torch.from_numpy(buffer.action_mask[:t_len]).to(self.device)
        sp_masks_t = torch.from_numpy(buffer.spatial_masks[:t_len]).to(self.device)

        a_type_t = torch.from_numpy(buffer.action_type[:t_len]).to(self.device)
        sp_idx_t = torch.from_numpy(buffer.spatial_idx[:t_len]).to(self.device)
        sl_idx_t = torch.from_numpy(buffer.slider_idx[:t_len]).to(self.device)
        old_lp_t = torch.from_numpy(buffer.log_probs[:t_len]).to(self.device)
        old_val_t = torch.from_numpy(buffer.values[:t_len]).to(self.device)
        ret_t = torch.from_numpy(buffer.returns[:t_len]).to(self.device)
        adv_t = torch.from_numpy(adv_norm).to(self.device)
        dones_t = torch.from_numpy(buffer.dones[:t_len]).to(self.device)
        land_t = torch.from_numpy(buffer.land_targets[:t_len]).to(self.device)
        init_h_t = torch.from_numpy(buffer.init_hidden).to(self.device)

        pg_losses, vf_losses, aux_losses, entropies, kls, clips = [], [], [], [], [], []

        for _ in range(self.cfg.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            tb = t_len * b_envs

            with torch.amp.autocast("cuda", enabled=self.hw.supports_amp):
                # 1. Batched Visual CNN + GATv2 + Scalar encoding over [T*B, ...] (8-15x faster!)
                fused_flat, vis_sp_flat = self.model._encode_features(
                    obs_map_t.reshape(tb, *obs_map_t.shape[2:]),
                    node_feats_t.reshape(tb, *node_feats_t.shape[2:]),
                    adj_t.reshape(tb, *adj_t.shape[2:]),
                    node_mask_t.reshape(tb, *node_mask_t.shape[2:]),
                    scalars_t.reshape(tb, *scalars_t.shape[2:]),
                )

                # 2. Exact recurrent GRU unroll resetting hidden state at episode boundaries
                dones_prev = torch.zeros_like(dones_t)
                if t_len > 1:
                    dones_prev[1:] = dones_t[:-1]
                core_seq, _ = self.model.rnn.forward_sequence(
                    fused_flat.view(t_len, b_envs, -1),
                    init_h_t,
                    dones_prev,
                )
                core_flat = core_seq.reshape(tb, -1)

                # 3. Batched Factorized Hybrid Actor & Multi-Head Critic
                a_flat = a_type_t.reshape(tb)
                sp_flat = sp_idx_t.reshape(tb)
                sl_flat = sl_idx_t.reshape(tb)
                am_flat = act_mask_t.reshape(tb, -1)
                sm_flat = sp_masks_t.reshape(tb, sp_masks_t.shape[2], sp_masks_t.shape[3])

                type_logits = self.model.action_type_head(core_flat)
                safe_am = torch.where(am_flat.any(dim=-1, keepdim=True), am_flat, torch.ones_like(am_flat))
                type_dist = torch.distributions.Categorical(logits=type_logits.masked_fill(~safe_am, -1e8))

                sp_logits = self.model._decode_spatial(core_flat, vis_sp_flat, a_flat)
                row_idx = torch.arange(tb, device=self.device)
                sp_m = sm_flat[row_idx, a_flat]
                safe_sp = torch.where(sp_m.any(dim=-1, keepdim=True), sp_m, torch.ones_like(sp_m))
                sp_dist = torch.distributions.Categorical(logits=sp_logits.masked_fill(~safe_sp, -1e8))

                a_emb = self.model.action_embed(a_flat)
                sl_logits = self.model.slider_head(torch.cat([core_flat, a_emb], dim=-1))
                sl_dist = torch.distributions.Categorical(logits=sl_logits)

                uses_sp = ((a_flat == 1) | (a_flat == 2) | (a_flat == 3)).to(type_logits.dtype)
                lp_seq = (
                    type_dist.log_prob(a_flat)
                    + uses_sp * sp_dist.log_prob(sp_flat)
                    + uses_sp * sl_dist.log_prob(sl_flat)
                ).view(t_len, b_envs)
                ent_seq = (
                    type_dist.entropy()
                    + 0.35 * uses_sp * sp_dist.entropy()
                    + 0.25 * uses_sp * sl_dist.entropy()
                ).view(t_len, b_envs)
                val_seq = self.model.value_head(core_flat).view(t_len, b_envs)
                aux_seq = self.model.aux_land_head(core_flat).view(t_len, b_envs)

                log_ratio = torch.clamp(lp_seq - old_lp_t, -10.0, 10.0)
                ratio = torch.exp(log_ratio)

                pg1 = -adv_t * ratio
                pg2 = -adv_t * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # Clipped value loss
                v_clipped = old_val_t + torch.clamp(val_seq - old_val_t, -self.cfg.clip_coef, self.cfg.clip_coef)
                vf_loss = 0.5 * torch.max((val_seq - ret_t) ** 2, (v_clipped - ret_t) ** 2).mean()

                ent_loss = ent_seq.mean()
                aux_loss = F.mse_loss(torch.sigmoid(aux_seq), land_t)

                # Off-policy prioritized auxiliary value/land replay distillation
                if len(self.aux_replay) >= 16:
                    samples = self.aux_replay.sample(16, self._rng)
                    s_map = torch.from_numpy(np.stack([s.obs_map for s in samples])).to(self.device)
                    s_nf = torch.from_numpy(np.stack([s.node_feats for s in samples])).to(self.device)
                    s_adj = torch.from_numpy(np.stack([s.adj for s in samples])).to(self.device)
                    s_nm = torch.from_numpy(np.stack([s.node_mask for s in samples])).to(self.device)
                    s_sc = torch.from_numpy(np.stack([s.scalars for s in samples])).to(self.device)
                    s_v = torch.tensor([s.target_value for s in samples], dtype=torch.float32, device=self.device)
                    s_l = torch.tensor([s.target_land for s in samples], dtype=torch.float32, device=self.device)
                    r_out = self.model.forward_step(s_map, s_nf, s_adj, s_nm, s_sc, deterministic=True)
                    aux_loss = aux_loss + 0.15 * F.smooth_l1_loss(r_out.value, s_v) + 0.2 * F.mse_loss(torch.sigmoid(r_out.aux_land_pred), s_l)

                loss = pg_loss + self.cfg.vf_coef * vf_loss - self.cfg.ent_coef * ent_loss + 0.25 * aux_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()

            pg_losses.append(float(pg_loss.item()))
            vf_losses.append(float(vf_loss.item()))
            aux_losses.append(float(aux_loss.item()))
            entropies.append(float(ent_loss.item()))
            kls.append(float(approx_kl))
            clips.append(float(clip_frac))

        return PPOUpdateMetrics(
            policy_loss=float(np.mean(pg_losses)),
            value_loss=float(np.mean(vf_losses)),
            aux_loss=float(np.mean(aux_losses)),
            entropy=float(np.mean(entropies)),
            approx_kl=float(np.mean(kls)),
            clip_fraction=float(np.mean(clips)),
            mean_reward=float(buffer.rewards[:t_len].mean()),
        )
