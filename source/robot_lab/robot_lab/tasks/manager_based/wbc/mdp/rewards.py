# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reward functions for the VWBC port (visual_wholebody / B1Z1).

Each function below mirrors the corresponding ``ManipLoco_rewards._reward_*``
in ``visual_wholebody/low-level/legged_gym/envs/rewards/maniploco_rewards.py``.
Sign conventions follow IsaacLab — the **weight** carries the sign, the
function returns a non-negative magnitude when the source treats the term as
a *penalty* and a non-negative magnitude when the source treats it as a
*reward*. Comments cite the source file/line for traceability.

VWBC divides the total leg reward and total arm reward by 100 in
``compute_reward``; we replicate that by multiplying every weight by 0.01 in
the env_cfg, so the per-step magnitudes match the source's
``rew_buf /= 100`` exactly.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walking_mask(env: ManagerBasedRLEnv, command_name: str, lin_clip: float, ang_clip: float) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.abs(cmd[:, 0]) > lin_clip) | (torch.abs(cmd[:, 2]) > ang_clip)


def _resolve_dog_joint_ids(asset: Articulation, dog_pattern: str) -> list[int]:
    import re
    return [i for i, n in enumerate(asset.joint_names) if re.fullmatch(dog_pattern, n)]


def _resolve_hip_joint_ids(asset: Articulation, hip_pattern: str = ".*_hip_joint") -> list[int]:
    import re
    return [i for i, n in enumerate(asset.joint_names) if re.fullmatch(hip_pattern, n)]


# ---------------------------------------------------------------------------
# Velocity tracking
# ---------------------------------------------------------------------------


def tracking_lin_vel_max(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    lin_vel_x_clip: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Asymmetric ratio reward (mlr.py:298-303).

    For ``|cmd_x| >= lin_vel_x_clip``::

        rew = min(v_x, cmd_x) / cmd_x        (cmd_x > 0)
            = min(-v_x, -cmd_x) / -cmd_x     (cmd_x < 0)

    For zero-cmd: rew = exp(-|v_x|).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    v_x = asset.data.root_lin_vel_b[:, 0]
    pos_branch = torch.minimum(v_x, cmd_x) / (cmd_x + 1e-5)
    neg_branch = torch.minimum(-v_x, -cmd_x) / (-cmd_x + 1e-5)
    rew = torch.where(cmd_x > 0, pos_branch, neg_branch)
    zero_mask = torch.abs(cmd_x) < lin_vel_x_clip
    rew = torch.where(zero_mask, torch.exp(-torch.abs(v_x)), rew)
    return rew


def tracking_ang_vel_yaw(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    sigma: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-(wz_cmd - wz_act)^2 / sigma) — mlr.py:142-144."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err2 = torch.square(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-err2 / sigma)


# ---------------------------------------------------------------------------
# Stability penalties
# ---------------------------------------------------------------------------


def lin_vel_z_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``base_lin_vel[:, 2]**2`` — mlr.py:134-136."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``sum(base_ang_vel[:, :2]**2)`` — mlr.py:138-140."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=-1)


def roll_abs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``|roll|`` — mlr.py:221-225."""
    from isaaclab.utils.math import euler_xyz_from_quat
    asset: Articulation = env.scene[asset_cfg.name]
    roll, _, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    return torch.abs(roll)


def base_height_l1(
    env: ManagerBasedRLEnv,
    target_height: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``|root_z - target|`` — mlr.py:227-230 (note: source uses L1 not L2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_pos_w[:, 2] - target_height)


# ---------------------------------------------------------------------------
# Joint-state penalties (leg only — apply over dog_joint_ids)
# ---------------------------------------------------------------------------


def torques_l2_full(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``sum(torques^2)`` over all joints — mlr.py:119-122 (no slicing)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=-1)


def dof_acc_leg(
    env: ManagerBasedRLEnv,
    dog_joint_pattern: str = "(FL|FR|RL|RR)_(hip|thigh|calf)_joint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``sum(((vel_t - vel_{t-1})/dt)^2)`` over leg joints — mlr.py:151-153."""
    asset: Articulation = env.scene[asset_cfg.name]
    dog_idx = _resolve_dog_joint_ids(asset, dog_joint_pattern)
    # IsaacLab exposes joint_acc directly
    acc = asset.data.joint_acc[:, dog_idx]
    return torch.sum(torch.square(acc), dim=-1)


def delta_torques_leg(
    env: ManagerBasedRLEnv,
    dog_joint_pattern: str = "(FL|FR|RL|RR)_(hip|thigh|calf)_joint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``sum((tau_t - tau_{t-1})^2)`` over leg joints — mlr.py:165-167.

    Maintains a per-env cache of the previous step's torque on the env object.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    dog_idx = _resolve_dog_joint_ids(asset, dog_joint_pattern)
    cur = asset.data.applied_torque[:, dog_idx]
    last = getattr(env, "_vwbc_last_leg_torque", None)
    if last is None or last.shape != cur.shape:
        last = torch.zeros_like(cur)
        env._vwbc_last_leg_torque = last
    out = torch.sum(torch.square(cur - last), dim=-1)
    env._vwbc_last_leg_torque = cur.detach().clone()
    return out


def action_rate_leg(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """``sum((a_t - a_{t-1})^2)`` over leg actions — mlr.py:155-157.

    Source slices actions[:12]; in our port the policy already only outputs the
    12 leg actions, so we use the raw action vector unchanged.
    """
    try:
        term = env.action_manager.get_term(action_name)
        cur = term.raw_actions
    except Exception:
        cur = env.action_manager.action
    last = getattr(env, "_vwbc_last_leg_action", None)
    if last is None or last.shape != cur.shape:
        last = torch.zeros_like(cur)
        env._vwbc_last_leg_action = last
    out = torch.sum(torch.square(cur - last), dim=-1)
    env._vwbc_last_leg_action = cur.detach().clone()
    return out


def dof_pos_limits_leg(
    env: ManagerBasedRLEnv,
    dog_joint_pattern: str = "(FL|FR|RL|RR)_(hip|thigh|calf)_joint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Out-of-soft-limits L1 over leg joints — mlr.py:159-163."""
    asset: Articulation = env.scene[asset_cfg.name]
    dog_idx = _resolve_dog_joint_ids(asset, dog_joint_pattern)
    pos = asset.data.joint_pos[:, dog_idx]
    limits = asset.data.soft_joint_pos_limits[:, dog_idx]
    over = (-(pos - limits[..., 0]).clip(max=0.0)) + (pos - limits[..., 1]).clip(min=0.0)
    return torch.sum(over, dim=-1)


def hip_pos_l2(
    env: ManagerBasedRLEnv,
    hip_joint_pattern: str = ".*_hip_joint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``sum((q_hip - default_hip)^2)`` — mlr.py:187-189."""
    asset: Articulation = env.scene[asset_cfg.name]
    hip_idx = _resolve_hip_joint_ids(asset, hip_joint_pattern)
    err = asset.data.joint_pos[:, hip_idx] - asset.data.default_joint_pos[:, hip_idx]
    return torch.sum(torch.square(err), dim=-1)


def work_leg(
    env: ManagerBasedRLEnv,
    dog_joint_pattern: str = "(FL|FR|RL|RR)_(hip|thigh|calf)_joint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``|sum(tau * qdot)|`` over leg joints — mlr.py:146-149."""
    asset: Articulation = env.scene[asset_cfg.name]
    dog_idx = _resolve_dog_joint_ids(asset, dog_joint_pattern)
    p = asset.data.applied_torque[:, dog_idx] * asset.data.joint_vel[:, dog_idx]
    return torch.abs(torch.sum(p, dim=-1))


def dof_vel_leg(
    env: ManagerBasedRLEnv,
    dog_joint_pattern: str = "(FL|FR|RL|RR)_(hip|thigh|calf)_joint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``sum(qdot^2)`` over leg joints.

    Penalises high joint velocities to suppress rapid/nervous stepping.
    Complements action_rate (which penalises *changes* in commands) by
    directly penalising the resulting joint motion.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    dog_idx = _resolve_dog_joint_ids(asset, dog_joint_pattern)
    return torch.sum(torch.square(asset.data.joint_vel[:, dog_idx]), dim=-1)


# ---------------------------------------------------------------------------
# Stand still / walking-conditioned terms
# ---------------------------------------------------------------------------


def stand_still_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    lin_vel_x_clip: float = 0.2,
    ang_vel_yaw_clip: float = 0.5,
    dog_joint_pattern: str = "(FL|FR|RL|RR)_(hip|thigh|calf)_joint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``exp(-sum|q_leg - q_default| * 0.05)`` when NOT walking — mlr.py:173-178."""
    asset: Articulation = env.scene[asset_cfg.name]
    dog_idx = _resolve_dog_joint_ids(asset, dog_joint_pattern)
    err = torch.sum(
        torch.abs(asset.data.joint_pos[:, dog_idx] - asset.data.default_joint_pos[:, dog_idx]),
        dim=-1,
    )
    rew = torch.exp(-err * 0.05)
    walking = _walking_mask(env, command_name, lin_vel_x_clip, ang_vel_yaw_clip)
    rew = torch.where(walking, torch.zeros_like(rew), rew)
    return rew


def walking_dof_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    lin_vel_x_clip: float = 0.2,
    ang_vel_yaw_clip: float = 0.5,
    dog_joint_pattern: str = "(FL|FR|RL|RR)_(hip|thigh|calf)_joint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Same kernel as ``stand_still_exp`` but active when walking — mlr.py:180-185."""
    asset: Articulation = env.scene[asset_cfg.name]
    dog_idx = _resolve_dog_joint_ids(asset, dog_joint_pattern)
    err = torch.sum(
        torch.abs(asset.data.joint_pos[:, dog_idx] - asset.data.default_joint_pos[:, dog_idx]),
        dim=-1,
    )
    rew = torch.exp(-err * 0.05)
    walking = _walking_mask(env, command_name, lin_vel_x_clip, ang_vel_yaw_clip)
    rew = torch.where(walking, rew, torch.zeros_like(rew))
    return rew


def alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant 1.0 — mlr.py:201-202."""
    return torch.ones(env.num_envs, device=env.device)


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float = 0.33,
    sigma: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for being at target height.

    Returns exp(-((z - target) / sigma)^2), which is 1.0 at the target and
    decays smoothly.  Unlike the L1 *penalty* (base_height_l1), this is a
    *positive* reward that provides a strong gradient toward the correct
    standing height and is zero-safe at episode start.

    sigma=0.05 m → reward > 0.5 within ±3.4 cm of target.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    h_err = asset.data.root_pos_w[:, 2] - target_height
    return torch.exp(-(h_err / sigma) ** 2)


def upright_bonus(
    env: ManagerBasedRLEnv,
    roll_threshold: float = 0.2,
    pitch_threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Binary bonus when the robot is approximately upright.

    Returns 1.0 when |roll| < roll_threshold AND |pitch| < pitch_threshold,
    else 0.0.  Provides a clear "standing upright" signal independent of
    forward velocity, complementing tracking_lin_vel_max which is zero when
    the command is zero.

    Default thresholds ±0.2 rad (≈11°) require the body to be roughly level.
    """
    from isaaclab.utils.math import euler_xyz_from_quat
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    return ((torch.abs(roll) < roll_threshold) & (torch.abs(pitch) < pitch_threshold)).float()


# ---------------------------------------------------------------------------
# Contact / foot terms
# ---------------------------------------------------------------------------


def collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*"),
    threshold: float = 0.1,
) -> torch.Tensor:
    """``sum( ||F|| > threshold )`` over penalized links — mlr.py:169-171."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    mag = torch.norm(forces, dim=-1).max(dim=1)[0]
    return torch.sum((mag > threshold).float(), dim=-1)


def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    max_force: float = 40.0,
    warmup_seconds: float = 2.0,
) -> torch.Tensor:
    """``relu(||F|| - max_force)`` summed over feet, gated after 2 s warmup — mlr.py:210-214."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    mag = torch.norm(forces, dim=-1).max(dim=1)[0]  # (N, num_feet)
    over = torch.sum(torch.clamp(mag - max_force, min=0.0), dim=-1)
    warmup_steps = warmup_seconds / env.step_dt
    flag = (env.episode_length_buf > warmup_steps).float()
    return flag * over


def feet_drag(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    contact_threshold: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
) -> torch.Tensor:
    """``sum( |foot_xyz_vel| * is_contact )`` — mlr.py:204-208."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    is_contact = (torch.norm(forces[:, 0], dim=-1) > contact_threshold).float()
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, F, 3)
    speed = torch.sum(torch.abs(foot_vel_w), dim=-1)  # (N, F)
    return torch.sum(speed * is_contact, dim=-1)


def feet_jerk(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    warmup_steps: int = 50,
) -> torch.Tensor:
    """``sum(||F_t - F_{t-1}||)`` over feet — mlr.py:191-199.

    Maintains a per-env cache of the previous step's foot contact forces.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :][:, 0]
    last = getattr(env, "_vwbc_last_foot_force", None)
    if last is None or last.shape != forces.shape:
        last = torch.zeros_like(forces)
        env._vwbc_last_foot_force = last
        result = torch.zeros(env.num_envs, device=env.device)
    else:
        result = torch.sum(torch.norm(forces - last, dim=-1), dim=-1)
    env._vwbc_last_foot_force = forces.detach().clone()
    result = torch.where(
        env.episode_length_buf < warmup_steps,
        torch.zeros_like(result),
        result,
    )
    return result


# ---------------------------------------------------------------------------
# Feet air time + height  (missing from original port; b1z1 scale: 2.0 / 1.0)
# ---------------------------------------------------------------------------


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    threshold: float = 0.5,
    lin_vel_clip: float = 0.2,
    ang_vel_clip: float = 0.5,
) -> torch.Tensor:
    """Reward long foot air times. Matches manip_loco._reward_feet_air_time (threshold=0.5 s).

    b1z1: feet_air_time = 2.0, threshold = 0.5 s, all-feet (we use all 4;
    b1z1 used only front 2 due to B1 gait bias — not meaningful for GO2).
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    rew = torch.sum((last_air - threshold) * first_contact, dim=1)
    rew *= _walking_mask(env, command_name, lin_vel_clip, ang_vel_clip)
    return rew


def feet_height_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    target_norm: float = 0.3,
    lin_vel_clip: float = 0.2,
    ang_vel_clip: float = 0.5,
) -> torch.Tensor:
    """Penalise when L2 norm of foot z-heights is below target.

    Matches manip_loco._reward_feet_height: ``clamp(norm(feet_z) - target, max=0)``.
    b1z1: feet_height = 1.0, target = 0.3 m.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (N, K)
    rew = torch.clamp(torch.norm(feet_z, dim=-1) - target_norm, max=0.0)
    rew *= _walking_mask(env, command_name, lin_vel_clip, ang_vel_clip)
    return rew


# ---------------------------------------------------------------------------
# EE goal tracking
# ---------------------------------------------------------------------------


def tracking_ee_world(
    env: ManagerBasedRLEnv,
    ee_goal_command_name: str = "ee_goal",
    sigma: float = 1.0,
    ee_body_name: str = "ee",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``exp(-sum(|ee_pos - ee_goal_world|) / sigma * 2)`` — mlr.py:17-20.

    Note the factor-of-two inside the exponent (matches source exactly).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset.body_names.index(ee_body_name)
    ee_pos_w = asset.data.body_pos_w[:, body_id, :]
    ee_cmd = env.command_manager.get_term(ee_goal_command_name)
    err = torch.sum(torch.abs(ee_pos_w - ee_cmd.curr_ee_goal_cart_world), dim=-1)
    return torch.exp(-err / sigma * 2.0)


# ---------------------------------------------------------------------------
# H2: FPG-style IK feasibility reward (geometric prior)
# ---------------------------------------------------------------------------


def ik_feasibility_dls(
    env: ManagerBasedRLEnv,
    ee_goal_command_name: str = "ee_goal",
    ee_body_name: str = "ee",
    arm_joint_pattern: str = "joint[1-6]",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ik_damping: float = 0.05,
    sigma_q: float = 0.35,
    residual_tol: float = 0.08,
    pos_tol: float = 0.20,
) -> torch.Tensor:
    """FPG-style IK feasibility reward (H2 geometric prior).

    Uses the same DLS Jacobian IK already running in ``VisualWholeBodyAction``
    to measure how close the current arm configuration is to the IK solution
    for the commanded EE goal.  Returns:

    .. math::

        r = \\exp\\!\\left(-\\frac{\\|\\Delta q_{norm}\\|_2^2}{\\sigma_q^2}\\right)
            \\quad \\text{if solvable, else } 0

    where ``delta_q_norm`` is the DLS joint increment normalised by each
    joint's soft range.  The "solvable" gate fires when the linearisation
    residual is below ``residual_tol`` **and** the position error is within
    ``pos_tol``.  Both thresholds are intentionally generous so the reward is
    not too sparse early in training.
    """
    import re
    from isaaclab.utils.math import compute_pose_error

    asset: Articulation = env.scene[asset_cfg.name]

    # --- one-time index caching (avoids per-step string search) -----------
    if not hasattr(env, "_h2_arm_joint_ids"):
        env._h2_arm_joint_ids = [
            i for i, n in enumerate(asset.joint_names) if re.fullmatch(arm_joint_pattern, n)
        ]
    if not hasattr(env, "_h2_ee_body_id"):
        env._h2_ee_body_id = asset.body_names.index(ee_body_name)

    arm_ids = env._h2_arm_joint_ids
    ee_id = env._h2_ee_body_id

    # --- current EE pose --------------------------------------------------
    ee_pos_w = asset.data.body_pos_w[:, ee_id, :]
    ee_quat_w = asset.data.body_quat_w[:, ee_id, :]
    ee_quat_w = ee_quat_w / (ee_quat_w.norm(dim=-1, keepdim=True) + 1e-8)

    # --- desired EE pose from command term --------------------------------
    cmd = env.command_manager.get_term(ee_goal_command_name)
    p_des = cmd.curr_ee_goal_cart_world
    q_des = cmd.ee_goal_orn_quat

    # --- pose error -------------------------------------------------------
    pos_err, rot_err = compute_pose_error(
        ee_pos_w, ee_quat_w, p_des, q_des, rot_error_type="axis_angle"
    )
    delta_x = torch.cat([pos_err, rot_err], dim=-1)  # (N, 6)

    # --- Jacobian (same extraction logic as VisualWholeBodyAction) --------
    jacobians = asset.root_physx_view.get_jacobians()
    body_index = ee_id
    if jacobians.shape[1] == len(asset.body_names) - 1:
        body_index -= 1
    dof_offset = 6 if jacobians.shape[-1] == asset.num_joints + 6 else 0
    arm_cols = torch.as_tensor(arm_ids, device=env.device, dtype=torch.long) + dof_offset
    J = jacobians[:, body_index, :6, :].index_select(-1, arm_cols)  # (N, 6, 6)

    # --- DLS solve --------------------------------------------------------
    Jt = J.transpose(1, 2)
    damp = (ik_damping ** 2) * torch.eye(6, device=env.device).unsqueeze(0)
    A = J @ Jt + damp
    delta_q = (Jt @ torch.linalg.solve(A, delta_x.unsqueeze(-1))).squeeze(-1)  # (N, 6)

    # --- normalise by joint soft range ------------------------------------
    joint_limits = asset.data.soft_joint_pos_limits[:, arm_ids, :]  # (N, 6, 2)
    joint_range = joint_limits[..., 1] - joint_limits[..., 0]       # (N, 6)
    dq_norm = delta_q / (joint_range + 1e-6)                        # (N, 6)

    # --- solvability gate -------------------------------------------------
    lin_res = torch.norm(
        (J @ delta_q.unsqueeze(-1)).squeeze(-1) - delta_x, dim=-1
    )  # (N,)
    pos_norm = torch.norm(pos_err, dim=-1)  # (N,)
    solvable = (lin_res <= residual_tol) & (pos_norm <= pos_tol)

    # --- feasibility score ------------------------------------------------
    f = torch.exp(-torch.sum(dq_norm ** 2, dim=-1) / (sigma_q ** 2))
    return torch.where(solvable, f, torch.zeros_like(f))


__all__ = [
    "tracking_lin_vel_max",
    "tracking_ang_vel_yaw",
    "lin_vel_z_square",
    "ang_vel_xy_square",
    "roll_abs",
    "base_height_l1",
    "base_height_exp",
    "upright_bonus",
    "torques_l2_full",
    "dof_acc_leg",
    "delta_torques_leg",
    "action_rate_leg",
    "dof_pos_limits_leg",
    "hip_pos_l2",
    "work_leg",
    "dof_vel_leg",
    "stand_still_exp",
    "walking_dof_exp",
    "alive",
    "collision",
    "feet_contact_forces",
    "feet_drag",
    "feet_jerk",
    "feet_air_time",
    "feet_height_l2",
    "tracking_ee_world",
    "ik_feasibility_dls",
]
