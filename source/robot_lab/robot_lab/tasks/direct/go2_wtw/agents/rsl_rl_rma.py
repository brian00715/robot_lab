# SPDX-License-Identifier: Apache-2.0
"""RSL-RL modules matching walk-these-ways ``ppo_cse`` RMA training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from rsl_rl.algorithms import PPO


def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name in ("relu", "crelu"):
        return nn.ReLU()
    if name == "lrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


def _make_mlp(in_dim: int, out_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
    act = _activation(activation)
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dims[0]), act]
    for i, hidden_dim in enumerate(hidden_dims):
        if i == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            layers.append(nn.Linear(hidden_dim, hidden_dims[i + 1]))
            layers.append(_activation(activation))
    return nn.Sequential(*layers)


class _StudentActor(nn.Module):
    def __init__(self, adaptation_module: nn.Module, actor_body: nn.Module):
        super().__init__()
        self.adaptation_module = adaptation_module
        self.actor_body = actor_body

    def forward(self, observation_history: torch.Tensor):
        latent = self.adaptation_module(observation_history)
        return self.actor_body(torch.cat((observation_history, latent), dim=-1))


class Go2WTWActorCritic(nn.Module):
    """Actor-critic with history encoder and privileged latent teacher target."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str,
        init_noise_std: float,
        adaptation_module_branch_hidden_dims: list[int] | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "Go2WTWActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.obs_groups = obs_groups
        self.num_obs_history = obs["obs_history"].shape[-1]
        self.num_privileged_obs = obs["privileged"].shape[-1]
        adaptation_hidden = adaptation_module_branch_hidden_dims or [256, 128]

        self.adaptation_module = _make_mlp(
            self.num_obs_history, self.num_privileged_obs, adaptation_hidden, activation
        )
        actor_in = self.num_obs_history + self.num_privileged_obs
        self.actor_body = _make_mlp(actor_in, num_actions, actor_hidden_dims, activation)
        self.critic_body = _make_mlp(actor_in, 1, critic_hidden_dims, activation)

        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def actor(self):
        return _StudentActor(self.adaptation_module, self.actor_body)

    def update_distribution(self, observation_history: torch.Tensor):
        latent = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs["obs_history"])
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.act_student(obs["obs_history"])

    def act_student(self, observation_history: torch.Tensor, policy_info: dict | None = None):
        latent = self.adaptation_module(observation_history)
        if policy_info is not None:
            policy_info["latents"] = latent.detach().cpu().numpy()
        return self.actor_body(torch.cat((observation_history, latent), dim=-1))

    def act_teacher(self, observation_history: torch.Tensor, privileged_obs: torch.Tensor, policy_info: dict | None = None):
        if policy_info is not None:
            policy_info["latents"] = privileged_obs.detach().cpu().numpy()
        return self.actor_body(torch.cat((observation_history, privileged_obs), dim=-1))

    def evaluate(self, obs, **kwargs):
        return self.critic_body(torch.cat((obs["obs_history"], obs["privileged"]), dim=-1))

    def get_student_latent(self, obs):
        return self.adaptation_module(obs["obs_history"])

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        pass

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


class Go2WTWPPO(PPO):
    """PPO update plus the original walk-these-ways adaptation-module MSE step."""

    def __init__(
        self,
        policy,
        adaptation_module_learning_rate: float = 1.0e-3,
        num_adaptation_module_substeps: int = 1,
        selective_adaptation_module_loss: bool = False,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        self.num_adaptation_module_substeps = num_adaptation_module_substeps
        self.selective_adaptation_module_loss = selective_adaptation_module_loss
        self.adaptation_module_optimizer = optim.Adam(
            self.policy.parameters(), lr=adaptation_module_learning_rate
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_adaptation_module_loss = 0.0
        mean_adaptation_module_test_loss = 0.0

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
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

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
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

            data_size = obs_batch["privileged"].shape[0]
            num_train = int(data_size // 5 * 4)
            for _ in range(self.num_adaptation_module_substeps):
                adaptation_pred = self.policy.get_student_latent(obs_batch)
                adaptation_target = obs_batch["privileged"]
                selection_indices = torch.arange(
                    adaptation_pred.shape[1], device=adaptation_pred.device, dtype=torch.long
                )
                if self.selective_adaptation_module_loss:
                    selection_indices = torch.tensor([0], device=adaptation_pred.device, dtype=torch.long)
                adaptation_loss = F.mse_loss(
                    adaptation_pred[:num_train, selection_indices],
                    adaptation_target[:num_train, selection_indices],
                )
                adaptation_test_loss = F.mse_loss(
                    adaptation_pred[num_train:, selection_indices],
                    adaptation_target[num_train:, selection_indices],
                )
                self.adaptation_module_optimizer.zero_grad()
                adaptation_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                self.adaptation_module_optimizer.step()

                mean_adaptation_module_loss += adaptation_loss.item()
                mean_adaptation_module_test_loss += adaptation_test_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        num_adaptation_updates = num_updates * self.num_adaptation_module_substeps
        self.storage.clear()

        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "adaptation_module": mean_adaptation_module_loss / num_adaptation_updates,
            "adaptation_module_test": mean_adaptation_module_test_loss / num_adaptation_updates,
        }
