# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Domain randomization events for the VWBC port.

Implements VWBC-specific events not provided by Isaac Lab:

* ``randomize_motor_strength``: per-env multiplicative leg & arm motor-strength
  scaling. Stored on the action term so ``process_actions`` multiplies actions
  by this factor (matches ``manip_loco._compute_torques``: ``actions_scaled =
  actions * motor_strength * action_scale``). Also writes ``env._vwbc_leg_motor_strength``
  for the privileged obs.

* ``store_mass_params``: copies the per-env added base mass + COM offset +
  gripper mass produced by ``randomize_rigid_body_mass``-style events into a
  single 5-D tensor on ``env._vwbc_mass_params``.

* ``store_friction_coeff``: copies the average per-env friction into
  ``env._vwbc_friction_coeffs`` for the privileged obs.

* ``push_by_setting_velocity_with_zero_cmd_boost``: like the standard push
  event but multiplies the impulse by 2.5 when the velocity command is zero
  (matches ``manip_loco._push_robots`` lines 912-916).

The standard IsaacLab events (``randomize_rigid_body_mass``,
``randomize_rigid_body_com``, ``randomize_rigid_body_material``,
``apply_external_force_torque``, ``reset_root_state_uniform``,
``reset_joints_by_scale``) are reused unchanged from
``isaaclab.envs.mdp.events``.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ---------------------------------------------------------------------------
# Motor-strength randomization (multiplicative on policy actions)
# ---------------------------------------------------------------------------


def randomize_motor_strength(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    leg_motor_strength_range: tuple[float, float] = (0.7, 1.3),
    action_term_name: str = "joint_pos",
):
    """Sample per-env, per-leg-joint motor-strength multipliers in
    ``leg_motor_strength_range`` and write them onto the VWBC action term.

    Matches ``manip_loco`` motor-strength randomization (b1z1.py:201-202):
    ``leg_motor_strength_range = arm_motor_strength_range = [0.7, 1.3]``.
    Because in our port the policy only actuates the 12 leg DOFs (the arm
    follows env-driven IK), only the leg branch of the source randomization
    is replicated here.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    n = env_ids.numel()
    lo, hi = leg_motor_strength_range
    sample = torch.empty(n, 12, device=env.device).uniform_(lo, hi)

    # write to the action term
    try:
        action_term = env.action_manager.get_term(action_term_name)
        if hasattr(action_term, "set_motor_strength"):
            action_term.set_motor_strength(sample, env_ids)
    except Exception:
        pass

    # mirror buffer for privileged obs
    if not hasattr(env, "_vwbc_leg_motor_strength"):
        env._vwbc_leg_motor_strength = torch.ones(env.num_envs, 12, device=env.device)
    env._vwbc_leg_motor_strength[env_ids] = sample


# ---------------------------------------------------------------------------
# Mass parameter snapshot (for the privileged obs)
# ---------------------------------------------------------------------------


def randomize_base_mass_and_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    base_mass_add_range: tuple[float, float] = (0.0, 15.0),
    base_com_range: tuple[float, float, float] = (0.15, 0.15, 0.15),
    gripper_mass_add_range: tuple[float, float] = (0.0, 0.1),
    gripper_body_name: str = "ee",
):
    """Add random mass to the base, random COM offset to the base, and random
    mass to the gripper. Records the sampled values into
    ``env._vwbc_mass_params`` for the privileged obs.

    Works at startup (``env_ids = all envs``).

    Layout of ``_vwbc_mass_params`` (matches VWBC ``mass_params_tensor``):

        [0]   base mass added (kg)
        [1:4] base COM offset (m)
        [4]   gripper mass added (kg)
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    asset: Articulation = env.scene[asset_cfg.name]

    # buffer
    if not hasattr(env, "_vwbc_mass_params"):
        env._vwbc_mass_params = torch.zeros(env.num_envs, 5, device=env.device)

    # ---- base mass ----
    base_body_ids = asset_cfg.body_ids
    if isinstance(base_body_ids, slice):
        base_body_ids_t = torch.tensor([0], dtype=torch.int)
    else:
        base_body_ids_t = torch.as_tensor(base_body_ids, dtype=torch.int)

    masses = asset.root_physx_view.get_masses()
    n = env_ids.numel()
    rand_mass = torch.empty(n, 1, device="cpu").uniform_(*base_mass_add_range)
    masses[env_ids[:, None], base_body_ids_t] = (
        masses[env_ids[:, None], base_body_ids_t] + rand_mass
    )
    asset.root_physx_view.set_masses(masses, env_ids)
    env._vwbc_mass_params[env_ids.to(env.device), 0:1] = rand_mass.to(env.device)

    # ---- base COM offset ----
    cx, cy, cz = base_com_range
    rand_com = torch.empty(n, 3, device="cpu")
    rand_com[:, 0].uniform_(-cx, cx)
    rand_com[:, 1].uniform_(-cy, cy)
    rand_com[:, 2].uniform_(-cz, cz)

    coms = asset.root_physx_view.get_coms()  # (n_envs, n_bodies, 7) or (n_envs, n_bodies, 3)
    if coms.dim() == 3 and coms.shape[-1] == 7:
        coms[env_ids[:, None], base_body_ids_t, 0:3] = (
            coms[env_ids[:, None], base_body_ids_t, 0:3] + rand_com.unsqueeze(1)
        )
    else:
        coms[env_ids[:, None], base_body_ids_t, :] = (
            coms[env_ids[:, None], base_body_ids_t, :] + rand_com.unsqueeze(1)
        )
    asset.root_physx_view.set_coms(coms, env_ids)
    env._vwbc_mass_params[env_ids.to(env.device), 1:4] = rand_com.to(env.device)

    # ---- gripper mass ----
    if gripper_body_name in asset.body_names:
        gripper_idx = asset.body_names.index(gripper_body_name)
        gripper_idx_t = torch.tensor([gripper_idx], dtype=torch.int)
        masses = asset.root_physx_view.get_masses()
        rand_g = torch.empty(n, 1, device="cpu").uniform_(*gripper_mass_add_range)
        masses[env_ids[:, None], gripper_idx_t] = masses[env_ids[:, None], gripper_idx_t] + rand_g
        asset.root_physx_view.set_masses(masses, env_ids)
        env._vwbc_mass_params[env_ids.to(env.device), 4:5] = rand_g.to(env.device)


# ---------------------------------------------------------------------------
# Friction snapshot (for the privileged obs)
# ---------------------------------------------------------------------------


def randomize_friction_record(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    friction_range: tuple[float, float] = (0.3, 3.0),
):
    """Sample per-env friction in ``friction_range`` (1000-bucket-style sampling),
    apply to all rigid shapes of the robot, and record the sample into
    ``env._vwbc_friction_coeffs``.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    asset: Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties()  # (n_envs, n_shapes, 3)

    n = env_ids.numel()
    sample = torch.empty(n, 1, device="cpu").uniform_(*friction_range)

    materials[env_ids, :, 0] = sample  # static friction
    materials[env_ids, :, 1] = sample  # dynamic friction (same as static, matches IG buckets)
    asset.root_physx_view.set_material_properties(materials, env_ids)

    if not hasattr(env, "_vwbc_friction_coeffs"):
        env._vwbc_friction_coeffs = torch.ones(env.num_envs, 1, device=env.device)
    env._vwbc_friction_coeffs[env_ids.to(env.device)] = sample.to(env.device)


# ---------------------------------------------------------------------------
# Push with zero-cmd boost
# ---------------------------------------------------------------------------


def push_robot_zero_cmd_boost(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    zero_cmd_boost: float = 2.5,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the robot. Matches ``manip_loco._push_robots`` (lines 907-917).

    The horizontal velocity is multiplied by ``zero_cmd_boost`` for envs whose
    velocity command sums to exactly zero (i.e. clipped to "stand still"
    by ``VWBCVelocityCommand``).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    asset: RigidObject = env.scene[asset_cfg.name]
    vel_w = asset.data.root_vel_w[env_ids].clone()

    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    ranges = torch.tensor(
        [velocity_range.get(k, (0.0, 0.0)) for k in keys], device=asset.device
    )
    n = env_ids.numel()
    deltas = torch.empty(n, 6, device=asset.device)
    for i in range(6):
        deltas[:, i].uniform_(ranges[i, 0].item(), ranges[i, 1].item())

    vel_w[:, :6] = vel_w[:, :6] + deltas

    # boost xy when command sums to zero
    try:
        cmd = env.command_manager.get_command(command_name)
        is_zero_cmd = (cmd[env_ids].abs().sum(dim=-1) < 1e-6)
        boost = torch.where(is_zero_cmd, zero_cmd_boost, 1.0).unsqueeze(-1)
        vel_w[:, 0:2] = vel_w[:, 0:2] * boost
    except Exception:
        pass

    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


__all__ = [
    "randomize_motor_strength",
    "randomize_base_mass_and_com",
    "randomize_friction_record",
    "push_robot_zero_cmd_boost",
]
