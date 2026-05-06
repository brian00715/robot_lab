# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL PPO configuration for the GO2 + ARX5 VWBC port (Batch 2).

Wires the custom :class:`VWBCActorCritic` and :class:`VWBCPPO` (registered in
``robot_lab.tasks.manager_based.wbc.learning``) into the runner cfg.

Hyperparameter alignment vs. ``b1z1_config.B1Z1RoughCfgPPO``:

* Policy
    - ``actor_hidden_dims=[128]``, ``critic_hidden_dims=[128]``
    - ``leg_control_head_hidden_dims=[128, 128]``
    - ``priv_encoder_dims=[64, 20]``
    - ``init_noise_std=[0.8, 1.0, 1.0]*4``  (12 leg dims; arm is env-driven)
    - VWBC obs split: ``num_prop=66``, ``num_priv=18``, ``num_hist=10``
* Algorithm
    - ``learning_rate=2e-4``, ``schedule="fixed"``, ``desired_kl=None``,
      ``entropy_coef=0.0``, ``num_learning_epochs=5``,
      ``num_mini_batches=4``, ``gamma=0.99``, ``lam=0.95``,
      ``clip_param=0.2``, ``max_grad_norm=1.0``,
      ``value_loss_coef=1.0``, ``use_clipped_value_loss=True``
    - ``priv_reg_coef_schedual=[0.0, 0.1, 3000, 7000]``
    - ``dagger_update_freq=20``
    - ``min_policy_std=[0.15, 0.25, 0.25]*4``  (12 leg dims)

Deviations from b1z1 (deliberate, documented in VWBCPPO docstring):

* ``mixing_schedule`` skipped — IsaacLab reward is scalar, not 2-D [leg, arm].
* ``torque_supervision`` skipped — disabled in b1z1 (``torque_supervision=False``).
* ``tracking_contacts_shaped_force/vel`` skipped — return 0 when
  ``observe_gait_commands=False`` (the b1z1 setting); effectively zero reward.
"""

from __future__ import annotations

from dataclasses import field

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


# ---------------------------------------------------------------------------
# Policy / algorithm cfg subclasses with VWBC-specific fields
# ---------------------------------------------------------------------------


@configclass
class VWBCActorCriticCfg(RslRlPpoActorCriticCfg):
    """Adds VWBC-specific architecture knobs."""

    class_name: str = "VWBCActorCritic"

    # Note: init_noise_std on the parent class is typed ``float``; the runner
    # passes the field straight through to ``VWBCActorCritic.__init__`` which
    # accepts list-or-scalar. We populate the list in
    # :meth:`ArxX5WbcRoughPPORunnerCfg.__post_init__`.
    init_noise_std: float = 1.0

    leg_control_head_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    priv_encoder_dims: list[int] = field(default_factory=lambda: [64, 20])

    # Observation split (must match mdp.vwbc_full_observation output layout)
    num_prop: int = 66
    num_priv: int = 18
    num_hist: int = 10


@configclass
class VWBCAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Adds VWBC-specific algorithm knobs (DAgger, priv-reg, min-std)."""

    class_name: str = "VWBCPPO"

    priv_reg_coef_schedual: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 3000.0, 7000.0]
    )
    dagger_update_freq: int = 20
    min_policy_std: list[float] = field(
        default_factory=lambda: [0.15, 0.25, 0.25] * 4
    )


# ---------------------------------------------------------------------------
# Runner cfgs
# ---------------------------------------------------------------------------


@configclass
class ArxX5WbcRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """VWBC PPO runner cfg — declares custom VWBC classes and hyperparameters."""

    num_steps_per_env = 24
    max_iterations = 45000
    save_interval = 200
    experiment_name = "unitree_go2_x5_wbc_rough"

    # Single composite obs group; critic uses the same priv-bearing tensor.
    obs_groups: dict[str, list[str]] = field(
        default_factory=lambda: {"policy": ["policy"], "critic": ["policy"]}
    )

    policy: VWBCActorCriticCfg = VWBCActorCriticCfg(
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128],
        critic_hidden_dims=[128],
        activation="elu",
    )

    algorithm: VWBCAlgorithmCfg = VWBCAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        # Populate per-dim init noise std (12 leg dims, arm is env-driven).
        self.policy.init_noise_std = [0.8, 1.0, 1.0] * 4


@configclass
class ArxX5WbcFlatPPORunnerCfg(ArxX5WbcRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_x5_wbc_flat"
