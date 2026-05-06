# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""VWBC PPO subclass with privileged-encoder regularization + DAgger updates.

Differences vs. the upstream rsl-rl-lib 3.1.x ``PPO``:

* ``priv_reg_loss`` term added to the policy objective. The student
  (history encoder) latent is regressed against the teacher (priv encoder)
  latent. Coefficient follows VWBC's two-knot schedule
  ``priv_reg_coef_schedual = [start, end, anchor_iter, ramp_iters]``: the
  coefficient ramps linearly from ``start`` (at counter==anchor_iter) to
  ``end`` over ``ramp_iters`` iterations.
* **DAgger pass** — every ``dagger_update_freq`` iterations, after the PPO
  step but before clearing storage, run a separate generator that updates
  ONLY ``policy.history_encoder`` parameters via L2 between
  ``priv_latent.detach()`` and ``hist_latent``.
* ``enforce_min_std`` — after each update, clamp ``policy.std`` to the
  per-dim floor specified by ``min_policy_std``.

**Deviations from upstream VWBC** (deliberate, documented):

* ``mixing_schedule`` — VWBC stacks per-component arm/leg rewards into a
  2D tensor and cross-mixes advantages. IsaacLab rewards are scalar; would
  require subclassing ``RolloutStorage`` + ``Transition``. Skipped.
* ``torque_supervision`` — disabled in b1z1 (``torque_supervision=False``). Skipped.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms import PPO


class VWBCPPO(PPO):
    """PPO with priv-encoder DAgger reg + per-dim min-std + dagger encoder pass."""

    def __init__(
        self,
        policy,
        *,
        # VWBC-specific
        priv_reg_coef_schedual: tuple[float, float, int, int] = (0.0, 0.1, 3000, 7000),
        dagger_update_freq: int = 20,
        min_policy_std: list[float] | None = None,
        # Standard PPO kwargs (passed straight through)
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        self.priv_reg_coef_schedual = tuple(priv_reg_coef_schedual)
        self.dagger_update_freq = int(dagger_update_freq)
        if min_policy_std is None:
            self.min_policy_std = None
        else:
            self.min_policy_std = torch.as_tensor(min_policy_std, dtype=torch.float32, device=self.device)
            if self.min_policy_std.numel() != self.policy.std.numel():
                raise ValueError(
                    f"min_policy_std length {self.min_policy_std.numel()} != policy.std length {self.policy.std.numel()}"
                )
        # DAgger optimizer trains only the history encoder
        self.hist_encoder_optimizer = optim.Adam(
            self.policy.history_encoder.parameters(), lr=self.learning_rate
        )
        self._iter_counter: int = 0

    # ---- priv reg coefficient schedule (matches VWBC manip_loco/ppo.py) ----
    def _priv_reg_coef(self) -> float:
        start, end, anchor, ramp = self.priv_reg_coef_schedual
        ramp = max(int(ramp), 1)
        stage = min(max((self._iter_counter - int(anchor)), 0) / ramp, 1.0)
        return stage * (end - start) + start

    def _enforce_min_std(self) -> None:
        if self.min_policy_std is None:
            return
        with torch.no_grad():
            cur = self.policy.std.detach()
            self.policy.std.data.copy_(torch.maximum(cur, self.min_policy_std))

    # ------------------------------------------------------------------------
    # update() — copies the upstream loop and adds priv_reg_loss + DAgger
    # ------------------------------------------------------------------------
    def update(self):  # noqa: C901
        if self.symmetry is not None:
            raise NotImplementedError("VWBCPPO does not implement symmetry — drop symmetry_cfg from cfg.")
        if self.rnd is not None:
            raise NotImplementedError("VWBCPPO does not implement RND — drop rnd_cfg from cfg.")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_priv_reg_loss = 0.0
        priv_reg_coef = self._priv_reg_coef()

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Recompute log-prob / value for current params
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # --- priv-reg DAgger term (teacher: priv encoder, student: hist encoder) ---
            x_actor = self.policy._flat(obs_batch, "policy")
            priv_latent = self.policy.infer_priv_latent(x_actor)
            with torch.inference_mode():
                hist_latent = self.policy.infer_hist_latent(x_actor)
            # priv_reg_loss treats hist as the *target*; it pulls priv toward
            # hist. (This matches VWBC line 177; counter-intuitive but
            # deliberate — encourages the priv encoder to stay decodable
            # from history.) Detach the hist side so only priv updates.
            priv_reg_loss = (priv_latent - hist_latent.detach()).norm(p=2, dim=1).mean()

            # --- KL-adaptive LR ---
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            # --- surrogate ---
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # --- value loss ---
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + priv_reg_coef * priv_reg_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_priv_reg_loss += priv_reg_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_priv_reg_loss /= num_updates

        # ---- DAgger pass (history encoder distillation) --------------------
        mean_hist_latent_loss = 0.0
        ran_dagger = False
        if self.dagger_update_freq > 0 and (self._iter_counter % self.dagger_update_freq == 0):
            ran_dagger = True
            if self.policy.is_recurrent:
                gen2 = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            else:
                gen2 = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            d_count = 0
            for (
                obs_batch,
                _ab,
                _tv,
                _adv,
                _ret,
                _olp,
                _omu,
                _osigma,
                _hid,
                _mask,
            ) in gen2:
                x_actor = self.policy._flat(obs_batch, "policy")
                with torch.inference_mode():
                    priv_lat = self.policy.infer_priv_latent(x_actor)
                hist_lat = self.policy.infer_hist_latent(x_actor)
                hist_loss = (priv_lat.detach() - hist_lat).norm(p=2, dim=1).mean()
                self.hist_encoder_optimizer.zero_grad()
                hist_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.history_encoder.parameters(), self.max_grad_norm)
                self.hist_encoder_optimizer.step()
                mean_hist_latent_loss += hist_loss.item()
                d_count += 1
            if d_count > 0:
                mean_hist_latent_loss /= d_count

        # ---- bookkeeping ---------------------------------------------------
        self.storage.clear()
        self._iter_counter += 1
        self._enforce_min_std()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "priv_reg": mean_priv_reg_loss,
            "priv_reg_coef": float(priv_reg_coef),
        }
        if ran_dagger:
            loss_dict["hist_latent"] = mean_hist_latent_loss
        return loss_dict
