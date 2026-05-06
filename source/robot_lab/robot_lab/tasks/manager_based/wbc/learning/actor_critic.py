# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""VWBC custom ActorCritic for rsl-rl-lib 3.1.x.

Architectural parity with ``visual_wholebody/third_party/rsl_rl/.../actor_critic.py``:

* **Privileged encoder** — MLP ``num_priv -> 64 -> priv_latent_dim (20)``
  (b1z1_config.policy.priv_encoder_dims = [64, 20]).
* **History encoder** — :class:`StateHistoryEncoder` consuming
  ``num_hist`` proprio steps and producing the same latent dimension as the
  priv encoder. Trained via DAgger to mimic the priv latent (handled in
  :class:`VWBCPPO`).
* **Two-stage actor** — shared backbone MLP (`actor_hidden_dims=[128]`) on
  ``[proprio, latent]``, then a control head (``leg_control_head_hidden_dims
  = [128, 128]``) producing 12 leg actions. The arm head from the original
  VWBC is *omitted* on purpose: arm joints are env-driven via IK in this
  port (Batch 1 decision), so `num_actions = 12`.
* **Critic** — symmetric: shared backbone (`critic_hidden_dims=[128]`) on
  full obs (proprio + priv + hist), then a value head (`leg_control_head`
  shape) producing scalar value.
* **Per-dim init noise std** — each leg action dim gets its own initial std,
  matching ``init_std=[0.8, 1.0, 1.0]*4`` for the dog joints.

Observation slicing — input is a single flat tensor (one obs group "policy")
shaped ``[B, num_prop + num_priv + num_hist*num_prop]``:

* ``[:, :num_prop]``                       → proprio (66)
* ``[:, num_prop:num_prop+num_priv]``      → privileged (18)
* ``[:, -num_hist*num_prop:]``             → history (10×66 = 660)

This matches the layout produced by :func:`mdp.vwbc_full_observation`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _get_activation(name: str) -> nn.Module:
    return {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "selu": nn.SELU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }[name]


class StateHistoryEncoder(nn.Module):
    """Conv-1D temporal encoder over a fixed-length proprio history window.

    Mirrors the ``tsteps == 10`` branch of the VWBC StateHistoryEncoder:
    Linear projection per timestep → two 1-D convs over time → linear head.
    """

    def __init__(self, activation: nn.Module, num_prop: int, num_hist: int, output_size: int):
        super().__init__()
        if num_hist != 10:
            raise ValueError(
                f"VWBC StateHistoryEncoder is calibrated for tsteps=10; got {num_hist}"
            )
        self.tsteps = num_hist
        self.num_prop = num_prop
        channel_size = 10
        self.encoder = nn.Sequential(nn.Linear(num_prop, 3 * channel_size), activation)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=4, stride=2),
            activation,
            nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=2, stride=1),
            activation,
            nn.Flatten(),
        )
        self.linear_output = nn.Sequential(nn.Linear(channel_size * 3, output_size), activation)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        # hist: [B, T, P]  (already reshaped by caller)
        nd = hist.shape[0]
        T = self.tsteps
        projection = self.encoder(hist.reshape(nd * T, -1))
        out = self.conv_layers(projection.reshape(nd, T, -1).permute(0, 2, 1))
        return self.linear_output(out)


class VWBCDeployActor(nn.Module):
    """Standalone deployment actor (history-encoder path) for JIT / ONNX export.

    Takes the full flat obs tensor ``(B, num_prop + num_priv + num_hist*num_prop)``
    and returns leg actions ``(B, num_actions)``.  Privileged info is discarded —
    only proprio and history are used, matching the deployed policy's inference path.

    Build via :meth:`VWBCActorCritic.build_deploy_actor` rather than constructing
    directly; that method deep-copies the sub-modules so this object is independent.
    """

    def __init__(
        self,
        num_prop: int,
        num_hist: int,
        history_encoder: nn.Module,
        actor_backbone: nn.Module,
        leg_head: nn.Module,
    ):
        super().__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.history_encoder = history_encoder
        self.actor_backbone = actor_backbone
        self.leg_head = leg_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prop = x[:, : self.num_prop]
        hist = x[:, -(self.num_hist * self.num_prop) :].view(-1, self.num_hist, self.num_prop)
        latent = self.history_encoder(hist)
        return self.leg_head(self.actor_backbone(torch.cat([prop, latent], dim=-1)))


class _Backbone(nn.Module):
    """Sequential MLP that returns either the requested hidden activations or identity."""

    def __init__(self, in_dim: int, hidden_dims: list[int], activation: nn.Module):
        super().__init__()
        if not hidden_dims:
            self.net = nn.Identity()
            self.out_dim = in_dim
            return
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dims[0]), activation]
        for k in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[k], hidden_dims[k + 1]), activation]
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Head(nn.Module):
    """Stacked-MLP head with a final un-activated linear projection to ``out_dim``."""

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, activation: nn.Module):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dims[0]), activation]
        for k in range(len(hidden_dims)):
            if k == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[k], out_dim))
            else:
                layers += [nn.Linear(hidden_dims[k], hidden_dims[k + 1]), activation]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VWBCActorCritic(nn.Module):
    """Two-stage actor + symmetric critic with VWBC priv/hist encoders.

    Constructor signature matches modern rsl-rl ``ActorCritic``:
    ``(obs, obs_groups, num_actions, **kwargs)``.

    Required kwargs:

    * ``num_prop``, ``num_priv``, ``num_hist`` — observation segment lengths.
    * ``actor_hidden_dims``, ``critic_hidden_dims`` (lists, may be empty).
    * ``leg_control_head_hidden_dims`` — head MLP for both actor & critic.
    * ``priv_encoder_dims`` — MLP shape for priv encoder; last dim is latent
      size shared with history encoder.
    * ``init_noise_std`` — scalar OR list of length num_actions (per-dim).
    * ``activation`` — string.
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [128],
        critic_hidden_dims: list[int] = [128],
        leg_control_head_hidden_dims: list[int] = [128, 128],
        priv_encoder_dims: list[int] = [64, 20],
        activation: str = "elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        num_prop: int = 66,
        num_priv: int = 18,
        num_hist: int = 10,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(
                "VWBCActorCritic.__init__ got unexpected kwargs (ignored): "
                + str(list(kwargs.keys()))
            )
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError(
                "VWBCActorCritic does not support obs normalization (priv block must"
                " stay un-normalized so the encoder learns the right mapping)."
            )

        # ---- record obs-group routing ---------------------------------------
        self.obs_groups = obs_groups
        # We only support the single-group layout produced by vwbc_full_observation.
        # Verify the obs group sizes line up with num_prop+num_priv+num_hist*num_prop.
        expected_dim = num_prop + num_priv + num_hist * num_prop
        actor_dim = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        critic_dim = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        if actor_dim != expected_dim or critic_dim != expected_dim:
            diag = {k: tuple(v.shape) for k, v in obs.items()}
            raise ValueError(
                f"VWBCActorCritic obs dim mismatch: expected {expected_dim}, "
                f"got actor={actor_dim} critic={critic_dim}. "
                f"obs keys+shapes: {diag}. obs_groups={obs_groups}."
            )

        self.num_prop = num_prop
        self.num_priv = num_priv
        self.num_hist = num_hist
        self.num_actions = num_actions

        act = _get_activation(activation)

        # ---- priv encoder ----------------------------------------------------
        priv_layers: list[nn.Module] = [nn.Linear(num_priv, priv_encoder_dims[0]), act]
        for k in range(len(priv_encoder_dims) - 1):
            priv_layers += [nn.Linear(priv_encoder_dims[k], priv_encoder_dims[k + 1]), act]
        self.priv_encoder = nn.Sequential(*priv_layers)
        latent_dim = priv_encoder_dims[-1]

        # ---- history encoder (DAgger student) -------------------------------
        self.history_encoder = StateHistoryEncoder(act, num_prop, num_hist, latent_dim)

        # ---- actor backbone + leg head --------------------------------------
        self.actor_backbone = _Backbone(num_prop + latent_dim, actor_hidden_dims, act)
        self.leg_head = _Head(self.actor_backbone.out_dim, leg_control_head_hidden_dims, num_actions, act)

        # ---- critic backbone + value head -----------------------------------
        # Critic sees proprio + priv (no history needed — priv is ground truth).
        self.critic_backbone = _Backbone(num_prop + num_priv, critic_hidden_dims, act)
        self.value_head = _Head(self.critic_backbone.out_dim, leg_control_head_hidden_dims, 1, act)

        # ---- action noise (per-dim or scalar) -------------------------------
        if noise_std_type != "scalar":
            raise ValueError(f"VWBCActorCritic only supports noise_std_type='scalar', got {noise_std_type}")
        if isinstance(init_noise_std, (list, tuple)):
            std0 = torch.as_tensor(init_noise_std, dtype=torch.float32)
            if std0.numel() != num_actions:
                raise ValueError(
                    f"init_noise_std length {std0.numel()} != num_actions {num_actions}"
                )
        else:
            std0 = torch.full((num_actions,), float(init_noise_std))
        self.std = nn.Parameter(std0)

        # ---- distribution ---------------------------------------------------
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        # The "obs normalizer" hooks expected by OnPolicyRunner — no-ops here.
        self.actor_obs_normalization = False
        self.critic_obs_normalization = False
        self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()

        print(f"VWBC priv_encoder: {self.priv_encoder}")
        print(f"VWBC history_encoder: {self.history_encoder}")
        print(f"VWBC actor_backbone+head: {self.actor_backbone.net} | {self.leg_head.net}")
        print(f"VWBC critic_backbone+head: {self.critic_backbone.net} | {self.value_head.net}")

    # ---------- obs handling -------------------------------------------------
    def _flat(self, obs, group: str) -> torch.Tensor:
        parts = [obs[g] for g in self.obs_groups[group]]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _slice(self, x: torch.Tensor):
        prop = x[:, : self.num_prop]
        priv = x[:, self.num_prop : self.num_prop + self.num_priv]
        hist = x[:, -self.num_hist * self.num_prop :].view(-1, self.num_hist, self.num_prop)
        return prop, priv, hist

    # ---------- actor forward (with optional hist encoding) ------------------
    def _actor(self, x: torch.Tensor, hist_encoding: bool = False) -> torch.Tensor:
        prop, priv, hist = self._slice(x)
        latent = self.history_encoder(hist) if hist_encoding else self.priv_encoder(priv)
        backbone_out = self.actor_backbone(torch.cat([prop, latent], dim=-1))
        return self.leg_head(backbone_out)

    def infer_priv_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.priv_encoder(x[:, self.num_prop : self.num_prop + self.num_priv])

    def infer_hist_latent(self, x: torch.Tensor) -> torch.Tensor:
        hist = x[:, -self.num_hist * self.num_prop :].view(-1, self.num_hist, self.num_prop)
        return self.history_encoder(hist)

    # ---------- critic forward ----------------------------------------------
    def _critic(self, x: torch.Tensor) -> torch.Tensor:
        prop_priv = x[:, : self.num_prop + self.num_priv]
        return self.value_head(self.critic_backbone(prop_priv))

    # ---------- rsl-rl required API -----------------------------------------
    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, x: torch.Tensor, hist_encoding: bool = False):
        mean = self._actor(x, hist_encoding=hist_encoding)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, hist_encoding: bool = False, **_):
        x = self._flat(obs, "policy")
        self.update_distribution(x, hist_encoding=hist_encoding)
        return self.distribution.sample()

    def act_inference(self, obs):
        # Default to priv encoder at inference unless explicitly configured otherwise
        # (during deployment the user can switch to hist_encoding via a method).
        x = self._flat(obs, "policy")
        return self._actor(x, hist_encoding=self._inference_hist_encoding)

    _inference_hist_encoding: bool = False  # toggle externally for deployment

    def evaluate(self, obs, **_):
        x = self._flat(obs, "critic")
        return self._critic(x)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        # No-op: priv block must stay un-normalized.
        return

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    def build_deploy_actor(self) -> VWBCDeployActor:
        """Return a standalone :class:`VWBCDeployActor` ready for JIT/ONNX export.

        Deep-copies the three sub-modules so the returned object is fully
        independent of this actor-critic (safe to script, save, move to CPU).
        """
        import copy
        return VWBCDeployActor(
            num_prop=self.num_prop,
            num_hist=self.num_hist,
            history_encoder=copy.deepcopy(self.history_encoder),
            actor_backbone=copy.deepcopy(self.actor_backbone),
            leg_head=copy.deepcopy(self.leg_head),
        )
