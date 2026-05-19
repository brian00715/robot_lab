# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Observation functions for the VWBC port.

The complete VWBC observation vector (``manip_loco.compute_observations``) is:

    proprio (66 dims, for action_dim=18 in source) =
        body_orientation (roll, pitch)                    2
        base_ang_vel * obs_scales.ang_vel                 3
        (dof_pos - default_pos)[:18, reindexed]          18
        dof_vel * 0.05 [:18, reindexed]                  18
        last_action[:12, reindexed]                      12
        foot_contacts_from_sensor [reindexed]             4
        commands[:3] * commands_scale                     3
        ee_goal_local_cart                                3
        zeros for ee_goal_orn                             3
    priv (18 dims) =
        mass_params (1 base-mass + 3 base-com + 1 gripper-mass)   5
        friction_coeff                                             1
        leg_motor_strength[:12] - 1                               12
    history buffer (10 frames of proprio)                       660

In the IsaacLab port we drop the ``reindex_all`` / ``reindex_feet`` remapping
because the joint- and body-name ordering is already defined by the asset; the
semantic quantities (each leg's hip/thigh/calf; FL/FR/RL/RR feet) are what
matter, and the policy learns whichever ordering the asset uses. Everything
else is preserved verbatim.

Privileged fields (``mass_params``, ``friction``, ``motor_strength``) are
populated by the event terms in ``wbc.mdp.events`` onto ``env._vwbc_*``
attributes; if unavailable they default to zero.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# VWBC proprioceptive obs terms (ordered as in manip_loco.compute_observations)
# -----------------------------------------------------------------------------


def body_orientation_rp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body roll, pitch (2D) — matches ``_get_body_orientation(return_yaw=False)``."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    # euler_xyz_from_quat returns angles in [0, 2pi); wrap to [-pi, pi] to match VWBC.
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    return torch.stack([roll, pitch], dim=-1)


def last_leg_action(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Last 12-D leg action (no arm). Matches ``action_history_buf[:, -1][:12]``."""
    try:
        term = env.action_manager.get_term(action_name)
        raw = term.raw_actions  # shape (N, 12) for VisualWholeBodyAction
    except Exception:
        # Fallback to the full action vector, taking the first 12 dims.
        raw = env.action_manager.action[:, :12]
    return raw[:, :12]


def foot_contacts_from_sensor(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    threshold: float = 1.5,
) -> torch.Tensor:
    """Per-foot binary contact flag (4D), matches ``foot_contacts_from_sensor``.

    VWBC uses the force-sensor norm > 1.5 N. We map to the contact sensor:
    latest contact force magnitude > threshold.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    # pick the latest time-step slot (index 0 is most recent in IsaacLab ContactSensor)
    mag = torch.norm(forces[:, 0], dim=-1)
    return (mag > threshold).float()


def ee_goal_local_cart(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_goal",
) -> torch.Tensor:
    """EE goal expressed in the robot base frame (3D).

    Mirrors ``manip_loco.compute_observations`` lines 217-218: take the
    world-frame goal published by the EE command, subtract the arm base
    position, rotate by the inverse base quaternion.

    For simplicity and numerical stability in the port we approximate "arm
    base" as the dog root position (the sphere-center offset is already
    baked into the world-frame goal by ``EEGoalSphereCommand``). The rotation
    from world to base-yaw frame is equivalent.
    """
    ee_cmd = env.command_manager.get_term(command_name)
    ee_goal_world = ee_cmd.curr_ee_goal_cart_world  # (N, 3)
    asset: Articulation = env.scene["robot"]
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w  # wxyz
    delta_w = ee_goal_world - root_pos
    return quat_apply_inverse(root_quat, delta_w)


def ee_goal_orn_zero(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """3-D zero placeholder to match VWBC's ``0*curr_ee_goal_sphere`` obs slot."""
    return torch.zeros(env.num_envs, 3, device=env.device)


# -----------------------------------------------------------------------------
# VWBC privileged obs block (mass_params + friction + leg motor strength)
# -----------------------------------------------------------------------------


def vwbc_mass_params(env: ManagerBasedRLEnv) -> torch.Tensor:
    """5-D privileged tensor ``[rand_mass, rand_com(3), gripper_rand_mass]``.

    Populated by ``wbc.mdp.events.randomize_base_mass_and_com``.
    Zero fallback when domain randomization is off.
    """
    buf = getattr(env, "_vwbc_mass_params", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, 5, device=env.device)
        env._vwbc_mass_params = buf
    return buf


def vwbc_friction_coeffs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1-D privileged tensor: per-env friction coefficient."""
    buf = getattr(env, "_vwbc_friction_coeffs", None)
    if buf is None:
        buf = torch.ones(env.num_envs, 1, device=env.device)
        env._vwbc_friction_coeffs = buf
    return buf


def vwbc_leg_motor_strength_minus_one(env: ManagerBasedRLEnv) -> torch.Tensor:
    """12-D privileged tensor: leg motor-strength multiplier minus 1.

    Matches ``motor_strength[:, :12] - 1`` in ``manip_loco.compute_observations``.
    """
    # Preferred path: read directly from the action term (so it is always
    # consistent with what process_actions() multiplied this step).
    try:
        action_term = env.action_manager.get_term("joint_pos")
        if hasattr(action_term, "motor_strength"):
            return action_term.motor_strength - 1.0
    except Exception:
        pass
    buf = getattr(env, "_vwbc_leg_motor_strength", None)
    if buf is None:
        buf = torch.ones(env.num_envs, 12, device=env.device)
        env._vwbc_leg_motor_strength = buf
    return buf - 1.0


# -----------------------------------------------------------------------------
# Composite VWBC observation (single obs term).
# -----------------------------------------------------------------------------


def vwbc_full_observation(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    ee_goal_command_name: str = "ee_goal",
    contact_sensor_name: str = "contact_forces",
    foot_body_pattern: str = ".*_foot",
    action_name: str = "joint_pos",
    dog_joint_pattern: str = "(FL|FR|RL|RR)_(hip|thigh|calf)_joint",
    arm_joint_pattern: str = "joint[1-6]",
    history_len: int = 10,
    contact_threshold: float = 1.5,
    obs_scale_ang_vel: float = 1.0,
    obs_scale_dof_pos: float = 1.0,
    obs_scale_dof_vel: float = 0.05,
    obs_scale_lin_vel: float = 1.0,
    add_noise: bool = False,
    noise_scales: dict[str, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-shot VWBC observation builder.

    Produces the full ``[proprio_now, priv, history(10 x proprio)]`` vector
    used by the source policy. Per-frame proprio layout (66 dims):

        2  body_orientation_rp
        3  base_ang_vel * scale_ang
       12  (dog_joint_pos - default) * scale_dof
        6  (arm_joint_pos - default) * scale_dof
       12  dog_joint_vel * scale_dof_vel
        6  arm_joint_vel * scale_dof_vel
       12  last_leg_action
        4  foot_contacts (bool)
        3  command[:3] * commands_scale
        3  ee_goal_local_cart
        3  zeros (orn placeholder)
       --
       66

    Privileged block (18 dims): mass_params(5) + friction(1) + leg motor
    strength minus 1 (12).

    Total = 66 + 18 + 10 * 66 = **744** when history_len=10.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    N = env.num_envs
    d = env.device

    # ---- proprio frame ----------------------------------------------------
    body_rp = body_orientation_rp(env, asset_cfg)  # (N, 2)
    ang_vel = asset.data.root_ang_vel_b * obs_scale_ang_vel  # (N, 3)

    # joint pos / vel
    joint_pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos
    joint_vel = asset.data.joint_vel

    dog_idx = _resolve_joint_indices(asset, dog_joint_pattern)
    arm_idx = _resolve_joint_indices(asset, arm_joint_pattern)
    dog_pos_rel = (joint_pos[:, dog_idx] - default_pos[:, dog_idx]) * obs_scale_dof_pos
    arm_pos_rel = (joint_pos[:, arm_idx] - default_pos[:, arm_idx]) * obs_scale_dof_pos
    dog_vel = joint_vel[:, dog_idx] * obs_scale_dof_vel
    arm_vel = joint_vel[:, arm_idx] * obs_scale_dof_vel

    # last leg action (12)
    last_act = last_leg_action(env, action_name)  # (N, 12)

    # foot contacts (4) — resolve body IDs directly to avoid unresolved SceneEntityCfg.body_ids
    _sensor: ContactSensor = env.scene[contact_sensor_name]
    if not hasattr(env, "_vwbc_foot_body_ids"):
        import re as _re
        env._vwbc_foot_body_ids = [
            i for i, n in enumerate(_sensor.body_names) if _re.fullmatch(foot_body_pattern, n)
        ]
    _foot_ids = env._vwbc_foot_body_ids
    _forces = _sensor.data.net_forces_w_history[:, 0, _foot_ids, :]  # (N, 4, 3)
    foot_contacts = (torch.norm(_forces, dim=-1) > contact_threshold).float()  # (N, 4)

    # commands (3)
    base_cmd = env.command_manager.get_command(command_name)  # (N, 3)
    commands_scaled = base_cmd.clone()
    commands_scaled[:, 0] *= obs_scale_lin_vel
    commands_scaled[:, 1] *= obs_scale_lin_vel
    commands_scaled[:, 2] *= obs_scale_ang_vel

    # EE goal local cart + orn placeholder
    ee_local = ee_goal_local_cart(env, ee_goal_command_name)  # (N, 3)
    ee_orn = torch.zeros(N, 3, device=d)

    proprio = torch.cat(
        [
            body_rp,
            ang_vel,
            dog_pos_rel,
            arm_pos_rel,
            dog_vel,
            arm_vel,
            last_act,
            foot_contacts,
            commands_scaled,
            ee_local,
            ee_orn,
        ],
        dim=-1,
    )

    # Optional Gaussian-style noise (b1z1_config has add_noise=False by default).
    if add_noise and noise_scales is not None:
        proprio = _apply_proprio_noise(
            proprio,
            noise_scales=noise_scales,
            obs_scale_ang_vel=obs_scale_ang_vel,
            obs_scale_dof_pos=obs_scale_dof_pos,
            obs_scale_dof_vel=obs_scale_dof_vel,
        )

    # ---- privileged block -------------------------------------------------
    priv = torch.cat(
        [
            vwbc_mass_params(env),
            vwbc_friction_coeffs(env),
            vwbc_leg_motor_strength_minus_one(env),
        ],
        dim=-1,
    )

    # ---- history buffer ---------------------------------------------------
    # Allocated lazily on env so it survives across calls and resets.
    if not hasattr(env, "_vwbc_obs_history"):
        env._vwbc_obs_history = torch.zeros(N, history_len, proprio.shape[-1], device=d)
    hist = env._vwbc_obs_history

    # On episode reset, replicate the current proprio across the history slots
    # (matches `obs_history_buf = [obs_buf]*history_len` when episode_length<=1).
    just_reset = (env.episode_length_buf <= 1).view(N, 1, 1)
    hist = torch.where(
        just_reset,
        proprio.unsqueeze(1).expand(-1, history_len, -1),
        torch.cat([hist[:, 1:], proprio.unsqueeze(1)], dim=1),
    )
    env._vwbc_obs_history = hist

    obs = torch.cat([proprio, priv, hist.reshape(N, -1)], dim=-1)
    return obs


def _resolve_joint_indices(asset: Articulation, pattern) -> list[int]:
    """Resolve joint indices.

    If *pattern* is a list/tuple of joint names, indices are returned in that
    explicit order (preserving the requested ordering).  If it is a regex
    string, indices are returned in simulator-natural order.
    """
    if isinstance(pattern, (list, tuple)):
        name_to_idx = {n: i for i, n in enumerate(asset.joint_names)}
        return [name_to_idx[n] for n in pattern if n in name_to_idx]
    import re as _re
    return [i for i, n in enumerate(asset.joint_names) if _re.fullmatch(pattern, n)]


def _apply_proprio_noise(
    proprio: torch.Tensor,
    noise_scales: dict[str, float],
    obs_scale_ang_vel: float,
    obs_scale_dof_pos: float,
    obs_scale_dof_vel: float,
) -> torch.Tensor:
    """Add per-channel uniform noise. Matches ``manip_loco._get_noise_scale_vec``.

    Layout assumed: [body_rp(2), ang_vel(3), dog_pos(12), arm_pos(6), dog_vel(12),
    arm_vel(6), last_act(12), foot(4), cmd(3), ee_local(3), ee_orn(3)].
    The arm pos/vel slots receive zero noise (matching VWBC).
    """
    n = torch.empty_like(proprio).uniform_(-1.0, 1.0)
    scale = torch.zeros_like(proprio[0])
    i = 0
    scale[i : i + 2] = 0.0; i += 2
    scale[i : i + 3] = noise_scales.get("ang_vel", 0.0) * obs_scale_ang_vel; i += 3
    scale[i : i + 12] = noise_scales.get("dof_pos", 0.0) * obs_scale_dof_pos; i += 12
    scale[i : i + 6] = 0.0; i += 6  # arm dof_pos noise = 0 in VWBC
    scale[i : i + 12] = noise_scales.get("dof_vel", 0.0) * obs_scale_dof_vel; i += 12
    scale[i : i + 6] = 0.0; i += 6  # arm dof_vel noise = 0 in VWBC
    scale[i : i + 12] = 0.0; i += 12  # last_action
    scale[i : i + 4] = 0.0; i += 4    # foot contacts
    scale[i : i + 3] = 0.0; i += 3    # commands
    scale[i : i + 3] = 0.0; i += 3    # ee_local
    scale[i : i + 3] = 0.0; i += 3    # ee_orn
    return proprio + n * scale


# -----------------------------------------------------------------------------
# __all__
# -----------------------------------------------------------------------------


__all__ = [
    "body_orientation_rp",
    "last_leg_action",
    "foot_contacts_from_sensor",
    "ee_goal_local_cart",
    "ee_goal_orn_zero",
    "vwbc_mass_params",
    "vwbc_friction_coeffs",
    "vwbc_leg_motor_strength_minus_one",
    "vwbc_full_observation",
]
