"""UMI on Legs Direct RL Environment.

Exact algorithm port from IsaacGym UMI-on-Legs (mani-centric-wbc/legged_gym/env/isaacgym/).

MDP:
- 18D continuous actions -> PD position control -> 18 joints (12 dog + 6 arm)
- 96D noisy observations (actor) / 143D privileged (critic)
- Multi-time-horizon end-effector pose tracking with curriculum
- Domain randomization (friction, mass, PD gains, pushes)
- Constraint-based penalty rewards
- Contact sensor for termination and force-based constraints
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply_inverse as quat_rotate_inverse

from .umi_on_legs_env_cfg import UmiOnLegsEnvCfg


# ---- Quaternion/matrix conversion helpers (pure PyTorch, matching pt3d semantics) ----

def quaternion_to_matrix(quat):
    """Convert quaternion (wxyz) to 3x3 rotation matrix."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, tx * y, tx * z
    tyy, tyz, tzz = ty * y, ty * z, tz * z
    return torch.stack([
        torch.stack([1.0 - (tyy + tzz), txy - twz, txz + twy], dim=-1),
        torch.stack([txy + twz, 1.0 - (txx + tzz), tyz - twx], dim=-1),
        torch.stack([txz - twy, tyz + twx, 1.0 - (txx + tyy)], dim=-1),
    ], dim=-2)


def matrix_to_rotation_6d(mat):
    """Convert 3x3 rotation matrix to 6D representation (first two columns)."""
    return mat[..., :2].reshape(*mat.shape[:-2], 6)


def matrix_to_quaternion(matrix):
    """Convert 3x3 rotation matrix to quaternion (wxyz)."""
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    q = torch.zeros(*matrix.shape[:-2], 4, device=matrix.device, dtype=matrix.dtype)
    mask = trace > 0
    s = 2.0 * torch.sqrt(trace[mask] + 1.0)
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (matrix[mask, 2, 1] - matrix[mask, 1, 2]) / s
    q[mask, 2] = (matrix[mask, 0, 2] - matrix[mask, 2, 0]) / s
    q[mask, 3] = (matrix[mask, 1, 0] - matrix[mask, 0, 1]) / s
    mask2 = (~mask) & (matrix[..., 0, 0] > matrix[..., 1, 1]) & (matrix[..., 0, 0] > matrix[..., 2, 2])
    s = 2.0 * torch.sqrt(1.0 + matrix[mask2, 0, 0] - matrix[mask2, 1, 1] - matrix[mask2, 2, 2])
    q[mask2, 0] = (matrix[mask2, 2, 1] - matrix[mask2, 1, 2]) / s
    q[mask2, 1] = 0.25 * s
    q[mask2, 2] = (matrix[mask2, 0, 1] + matrix[mask2, 1, 0]) / s
    q[mask2, 3] = (matrix[mask2, 0, 2] + matrix[mask2, 2, 0]) / s
    mask3 = (~mask) & (~mask2)
    s = 2.0 * torch.sqrt(1.0 + matrix[mask3, 1, 1] - matrix[mask3, 0, 0] - matrix[mask3, 2, 2])
    q[mask3, 0] = (matrix[mask3, 0, 2] - matrix[mask3, 2, 0]) / s
    q[mask3, 1] = (matrix[mask3, 0, 1] + matrix[mask3, 1, 0]) / s
    q[mask3, 2] = 0.25 * s
    q[mask3, 3] = (matrix[mask3, 1, 2] + matrix[mask3, 2, 1]) / s
    return q


def euler_angles_to_matrix(euler, convention="XYZ"):
    """Convert Euler angles to 3x3 rotation matrix. Convention: 'XYZ'."""
    rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    return torch.stack([
        torch.stack([cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz], dim=-1),
        torch.stack([cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz], dim=-1),
        torch.stack([-sy, cy * sx, cx * cy], dim=-1),
    ], dim=-2)


def quat_mul(q1, q2):
    """Multiply two quaternions (wxyz * wxyz -> wxyz)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


# ---- Environment ----

class UmiOnLegsEnv(DirectRLEnv):
    cfg: UmiOnLegsEnvCfg

    def __init__(self, cfg: UmiOnLegsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- Convert config lists to tensors ----
        self.action_scale = torch.tensor(self.cfg.action_scale, device=self.device).float()
        self.dof_offset = torch.tensor(self.cfg.dof_offset, device=self.device).float()

        kp_1d = torch.tensor(self.cfg.kp[:18], device=self.device).float()
        kd_1d = torch.tensor(self.cfg.kd[:18], device=self.device).float()
        torque_1d = torch.tensor(self.cfg.torque_limit[:18], device=self.device).float()
        self._init_kp = kp_1d.clone()
        self._init_kd = kd_1d.clone()
        self.kp = kp_1d[None, :].repeat(self.num_envs, 1)
        self.kd = kd_1d[None, :].repeat(self.num_envs, 1)
        self.torque_limit = torque_1d[None, :].repeat(self.num_envs, 1)

        # ---- Control buffer for delay simulation ----
        self._ctrl_buffer = torch.zeros(
            self.num_envs, self.cfg.decimation + 1, self.cfg.num_actions, device=self.device,
        )
        self._prev_action = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._computed_torque = torch.zeros(self.num_envs, 18, device=self.device)

        # ---- Resolve indices ----
        self._resolve_indices()

        # ---- Domain randomization state ----
        self._push_interval_steps = int(np.ceil(self.cfg.push_interval_s / self.step_dt))

        # ---- Task state: reaching task curriculum ----
        self._past_pos_err = torch.ones(self.num_envs, device=self.device)
        self._past_orn_err = torch.ones(self.num_envs, device=self.device)
        self._pos_err_sigma = self.cfg.pos_err_sigma
        self._orn_err_sigma = self.cfg.orn_err_sigma
        self._pos_curriculum_level = self.cfg.init_pos_curriculum_level
        self._orn_curriculum_level = self.cfg.init_orn_curriculum_level

        if self.cfg.pos_sigma_curriculum:
            pos_items = sorted(self.cfg.pos_sigma_curriculum, key=lambda x: x[0], reverse=True)
            self._pos_sigma_thresholds = torch.tensor([t for t, _ in pos_items], device=self.device)
            self._pos_sigma_values = [s for _, s in pos_items]
            idx = min(self._pos_curriculum_level, len(self._pos_sigma_values) - 1)
            self._pos_err_sigma = self._pos_sigma_values[idx]
            self._past_pos_err *= self._pos_sigma_thresholds[idx]

        if self.cfg.orn_sigma_curriculum:
            orn_items = sorted(self.cfg.orn_sigma_curriculum, key=lambda x: x[0], reverse=True)
            self._orn_sigma_thresholds = torch.tensor([t for t, _ in orn_items], device=self.device)
            self._orn_sigma_values = [s for _, s in orn_items]
            idx = min(self._orn_curriculum_level, len(self._orn_sigma_values) - 1)
            self._orn_err_sigma = self._orn_sigma_values[idx]
            self._past_orn_err *= self._orn_sigma_thresholds[idx]

        # Target trajectory storage (CPU for memory)
        self._target_pos_seq = torch.zeros(self.num_envs, self.max_episode_length, 3, device="cpu")
        self._target_rot_mat_seq = torch.zeros(self.num_envs, self.max_episode_length, 3, 3, device="cpu")

        print(f"[UMI-on-Legs] num_envs={self.num_envs}, device={self.device}")
        print(f"[UMI-on-Legs] step_dt={self.step_dt}, physics_dt={self.physics_dt}")
        print(f"[UMI-on-Legs] observation_space={self.cfg.observation_space}, state_space={self.cfg.state_space}")
        print(f"[UMI-on-Legs] DOF count={self._num_dof}, body count={self._num_bodies}")

        # Enable debug visualization if configured
        if self.cfg.debug_vis:
            self.set_debug_vis(True)

    # ========== Index Resolution ==========

    def _resolve_indices(self):
        robot = self.scene["robot"]
        all_joint_names = robot.data.joint_names
        self._num_dof = len(all_joint_names)
        all_body_names = robot.data.body_names
        self._num_bodies = len(all_body_names)

        # Policy joint names (in original order: FR, FL, RR, RL then arm)
        dog_patterns = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        arm_patterns = [f"joint{i}" for i in range(1, 7)]
        policy_joint_names = dog_patterns + arm_patterns

        self._policy_joint_indices = []
        for name in policy_joint_names:
            for idx, jn in enumerate(all_joint_names):
                if jn == name or name in jn:
                    self._policy_joint_indices.append(idx)
                    break
        self._policy_joint_indices = torch.tensor(self._policy_joint_indices, device=self.device, dtype=torch.long)

        # Termination body indices
        self._termination_body_indices = []
        for pattern in self.cfg.termination_contact_body_patterns:
            for idx, bn in enumerate(all_body_names):
                if pattern in bn:
                    self._termination_body_indices.append(idx)
        self._termination_body_indices = torch.tensor(
            list(set(self._termination_body_indices)), device=self.device, dtype=torch.long
        )

        # End-effector: "ee" or "end_effector" or "link6" or "x5_link6"
        self._ee_body_idx = None
        for candidate in ["ee", "end_effector", "x5_link6", "link6", "gripper", "wrist"]:
            for idx, bn in enumerate(all_body_names):
                if bn == candidate or candidate in bn.lower():
                    self._ee_body_idx = idx
                    break
            if self._ee_body_idx is not None:
                break
        if self._ee_body_idx is None:
            body_list = "\n  ".join([f"[{i}] {bn}" for i, bn in enumerate(all_body_names)])
            print(f"WARNING: end-effector not found. Body names:\n  {body_list}")
            print("Using last rigid body as fallback.")
            self._ee_body_idx = self._num_bodies - 1

        # Feet body indices
        feet_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        thigh_names = ["FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"]
        self._feet_body_indices = []
        self._thigh_body_indices = []
        for fn in feet_names:
            for idx, bn in enumerate(all_body_names):
                if fn in bn:
                    self._feet_body_indices.append(idx)
                    break
        for tn in thigh_names:
            for idx, bn in enumerate(all_body_names):
                if tn in bn:
                    self._thigh_body_indices.append(idx)
                    break

        print(f"[UMI-on-Legs] Policy joints: {len(self._policy_joint_indices)}")
        print(f"[UMI-on-Legs] Termination bodies: {len(self._termination_body_indices)}")
        print(f"[UMI-on-Legs] EE body index: {self._ee_body_idx} ({all_body_names[self._ee_body_idx]})")

    # ========== Scene Setup ==========

    def _setup_scene(self):
        self._robot = self.scene["robot"]
        # Contact sensor is available as self.scene["contact_forces"]
        # Dome light is configured in the scene config

    # ========== Pre-physics Step ==========

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.to(self.device)
        actions = torch.clamp(actions, -self.cfg.max_action_value, self.cfg.max_action_value)
        self._ctrl_buffer = torch.cat((actions[:, None, :], self._ctrl_buffer[:, :-1, :]), dim=1)
        self._prev_action = self._ctrl_buffer[:, 0, :]

    # ========== Apply Action (per physics step) ==========

    def _apply_action(self):
        decimation_step = self._sim_step_counter % self.cfg.decimation
        curr_target_idx = int(torch.ceil(
            torch.tensor((self.cfg.ctrl_delay_steps - decimation_step) / self.cfg.decimation)
        ).item())
        curr_target_idx = max(0, min(curr_target_idx, self._ctrl_buffer.shape[1] - 1))

        delayed_action = self._ctrl_buffer[:, curr_target_idx, :]
        target = delayed_action * self.action_scale[None, :] + self.dof_offset[None, :]

        curr_pos = self._robot.data.joint_pos[:, self._policy_joint_indices[:18]]
        curr_vel = self._robot.data.joint_vel[:, self._policy_joint_indices[:18]]

        torques = self.kp * (target - curr_pos) - self.kd * curr_vel
        torques = torch.clamp(torques, -self.torque_limit, self.torque_limit)
        self._computed_torque = torques

        all_torques = torch.zeros(self.num_envs, self._num_dof, device=self.device)
        all_torques[:, self._policy_joint_indices[:18]] = torques
        self._robot.set_joint_effort_target(all_torques)

    # ========== Observations ==========

    def _get_observations(self) -> dict[str, torch.Tensor]:
        robot = self._robot
        root_quat_w = robot.data.root_quat_w
        gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device).repeat(self.num_envs, 1)
        gravity_dir = gravity / torch.linalg.norm(gravity, dim=1, keepdims=True)

        local_root_ang_vel = quat_rotate_inverse(root_quat_w, robot.data.root_ang_vel_w)
        local_root_gravity = quat_rotate_inverse(root_quat_w, gravity_dir)
        local_root_lin_vel = quat_rotate_inverse(root_quat_w, robot.data.root_lin_vel_w)

        policy_dof_pos = robot.data.joint_pos[:, self._policy_joint_indices[:18]]
        policy_dof_vel = robot.data.joint_vel[:, self._policy_joint_indices[:18]]

        add_noise = self._sim_step_counter > 0

        # Actor (noisy)
        obs_terms = []
        term = local_root_ang_vel * self.cfg.root_ang_vel_scale
        if add_noise:
            term += torch.randn_like(term) * self.cfg.root_ang_vel_noise
        obs_terms.append(term)
        term = local_root_gravity
        if add_noise:
            term += torch.randn_like(term) * self.cfg.root_gravity_noise
        obs_terms.append(term)
        term = (policy_dof_pos - self.dof_offset[None, :18]) * self.cfg.dof_pos_scale
        if add_noise:
            term += torch.randn_like(term) * self.cfg.dof_pos_noise
        obs_terms.append(term)
        term = policy_dof_vel * self.cfg.dof_vel_scale
        if add_noise:
            term += torch.randn_like(term) * self.cfg.dof_vel_noise
        obs_terms.append(term)
        state_obs = torch.cat(obs_terms, dim=1)  # 42D
        task_obs = self._compute_task_observations()  # 36D
        policy_obs = torch.cat([state_obs, task_obs, self._prev_action], dim=1)  # 96D

        # Critic (privileged, noiseless)
        priv_terms = [
            local_root_lin_vel * self.cfg.root_lin_vel_scale,
            local_root_ang_vel * self.cfg.root_ang_vel_scale,
            local_root_gravity,
            (policy_dof_pos - self.dof_offset[None, :18]) * self.cfg.dof_pos_scale,
            policy_dof_vel * self.cfg.dof_vel_scale,
        ]
        private_state_obs = torch.cat(priv_terms, dim=1)  # 45D
        critic_obs = torch.cat([private_state_obs, task_obs, self._prev_action], dim=1)  # 99D
        critic_obs_padded = torch.zeros(self.num_envs, self.cfg.state_space, device=self.device)
        critic_obs_padded[:, :critic_obs.shape[1]] = critic_obs

        return {"policy": policy_obs, "critic": critic_obs_padded}

    # ========== Rewards ==========

    def _get_rewards(self) -> torch.Tensor:
        robot = self._robot
        root_quat_w = robot.data.root_quat_w
        gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device).repeat(self.num_envs, 1)
        gravity_dir = gravity / torch.linalg.norm(gravity, dim=1, keepdims=True)

        local_root_lin_vel = quat_rotate_inverse(root_quat_w, robot.data.root_lin_vel_w)
        local_root_ang_vel = quat_rotate_inverse(root_quat_w, robot.data.root_ang_vel_w)
        local_root_gravity = quat_rotate_inverse(root_quat_w, gravity_dir)

        # --- Individual reward components ---
        rew_components = {}

        # Env stabilization
        rew_lin_vel_z = -torch.square(local_root_lin_vel[:, 2])
        rew_ang_vel_xy = -torch.sum(torch.square(local_root_ang_vel[:, :2]), dim=1)
        rew_orientation = -torch.sum(torch.square(local_root_gravity[:, :2]), dim=1)
        rew_components["env/lin_vel_z"] = rew_lin_vel_z
        rew_components["env/ang_vel_xy"] = rew_ang_vel_xy
        rew_components["env/orientation"] = rew_orientation

        # Task: pose tracking
        pos_err = self._get_pos_err()
        orn_err = self._get_orn_err()
        rew_pos = torch.exp(-(pos_err ** 2) / self._pos_err_sigma)
        rew_orn = torch.exp(-orn_err / self._orn_err_sigma)
        rew_pose = rew_pos * rew_orn * self.cfg.pose_reward_scale
        rew_components["task/pose"] = rew_pose

        # Constraint: action_rate
        delta = (self._ctrl_buffer[:, 0, :] - self._ctrl_buffer[:, 1, :]).abs().sum(dim=1)
        rew_action_rate = self.cfg.action_rate_weight * delta
        rew_components["constraint/action_rate"] = rew_action_rate

        # Constraint: torque
        rew_torque = self.cfg.torque_weight * (self._computed_torque.abs() ** self.cfg.torque_power).sum(dim=1)
        rew_components["constraint/torque"] = rew_torque

        # Constraint: even_mass_distribution
        cs = getattr(self.scene, "contact_forces", None)
        rew_even_mass = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.even_mass_distribution_weight != 0 and cs is not None and len(self._feet_body_indices) >= 4:
            net_forces = cs.data.net_forces_w
            foot_forces = net_forces[:, self._feet_body_indices, 2].clamp(min=0.0)
            total_force = foot_forces.sum(dim=1, keepdim=True) + 1e-8
            mass_dist = foot_forces / total_force
            rew_even_mass = self.cfg.even_mass_distribution_weight * mass_dist.std(dim=1)
        rew_components["constraint/even_mass"] = rew_even_mass

        # Constraint: feet_under_hips
        rew_feet_under = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.feet_under_hips_weight != 0 and len(self._feet_body_indices) >= 4:
            for foot_idx, thigh_idx in zip(self._feet_body_indices, self._thigh_body_indices):
                foot_pos = robot.data.body_pos_w[:, foot_idx, :2]
                thigh_pos = robot.data.body_pos_w[:, thigh_idx, :2]
                dist = torch.norm(foot_pos - thigh_pos, dim=1)
                rew_feet_under += self.cfg.feet_under_hips_weight * (1.0 - torch.exp(-dist / 0.5))
        rew_components["constraint/feet_under_hips"] = rew_feet_under

        # Constraint: aligned_body_ee
        rew_aligned = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.aligned_body_ee_weight != 0:
            j1_idx = self._policy_joint_indices[12:13]
            j5_idx = self._policy_joint_indices[16:17]
            rew_aligned = self.cfg.aligned_body_ee_weight * (
                robot.data.joint_pos[:, j1_idx].abs() + robot.data.joint_pos[:, j5_idx].abs()
            ).sum(dim=1)
        rew_components["constraint/aligned_ee"] = rew_aligned

        # Constraint: root_height
        rew_height = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.root_height_weight != 0:
            rew_height = self.cfg.root_height_weight * torch.square(
                robot.data.root_pos_w[:, 2] - self.cfg.root_height_target
            )
        rew_components["constraint/root_height"] = rew_height

        # Store reward components in extras["log"] for rsl_rl logging.
        # CRITICAL: must copy dict each step — rsl_rl runner appends references to
        # ep_infos[], so all entries would point to the same last-step dict otherwise.
        log_dict = {}
        for name, rew in rew_components.items():
            log_dict[f"reward/{name}"] = rew
        log_dict["task/pos_err"] = pos_err
        log_dict["task/orn_err"] = orn_err
        log_dict["task/pos_sigma"] = torch.full_like(pos_err, self._pos_err_sigma)
        log_dict["task/orn_sigma"] = torch.full_like(pos_err, self._orn_err_sigma)
        self.extras["log"] = log_dict.copy()  # shallow copy — new dict, same tensors

        # Sum and positive-only clip
        total_reward = sum(rew_components.values())
        total_reward = torch.clamp(total_reward, min=0.0)

        return total_reward

    # ========== Done/Termination ==========

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cs = getattr(self.scene, "contact_forces", None)
        contact_termination = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if cs is not None:
            net_forces = cs.data.net_forces_w  # (num_envs, num_bodies, 3)
            termination_forces = net_forces[:, self._termination_body_indices, :]
            contact_termination = torch.any(
                torch.norm(termination_forces, dim=-1) > self.cfg.termination_contact_force_threshold,
                dim=1,
            )

        # Timeout
        timeout = self.episode_length_buf >= self.max_episode_length

        # Out of bounds
        root_xy = self._robot.data.root_pos_w[:, :2]
        walked_off = torch.logical_or(
            (root_xy < -self.cfg.safe_bounds_xy).any(dim=1),
            (root_xy > self.cfg.safe_bounds_xy).any(dim=1),
        )
        timeout = timeout | walked_off

        return contact_termination, timeout

    # ========== Reset ==========

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        if self.cfg.randomize_pd_params:
            self._randomize_pd_gains(env_ids)

        self._reset_task(env_ids)
        self._ctrl_buffer[env_ids] = 0.0

        if self.cfg.push_robots and self.common_step_counter % self._push_interval_steps == 0:
            root_state = self._robot.data.root_state_w[env_ids]
            push_vel = (torch.rand(len(env_ids), 6, device=self.device) * 2 - 1) * self.cfg.max_push_vel_lin
            root_state[:, 7:13] = push_vel
            self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        self.scene.write_data_to_sim()

    def _reset_dofs(self, env_ids):
        robot = self._robot
        policy_indices = self._policy_joint_indices[:18]
        dof_pos_default = self.dof_offset[None, :18].repeat(len(env_ids), 1)
        noise = self.cfg.dof_pos_reset_range_scale * torch.randn(len(env_ids), 18, device=self.device)
        new_dof_pos = (dof_pos_default + noise).clamp(-3.0, 5.0)
        robot.write_joint_state_to_sim(new_dof_pos, torch.zeros_like(new_dof_pos), env_ids=env_ids, joint_ids=policy_indices)

    def _reset_root_states(self, env_ids):
        root_pos = torch.tensor([[-0.5, 0.0, 0.35]], device=self.device).repeat(len(env_ids), 1)
        root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(len(env_ids), 1)
        root_lin_vel = torch.zeros(len(env_ids), 3, device=self.device)
        root_ang_vel = torch.zeros(len(env_ids), 3, device=self.device)

        init_pos_noise = torch.tensor(self.cfg.init_pos_noise, device=self.device)
        if (init_pos_noise > 0).any():
            root_pos += torch.randn_like(root_pos) * init_pos_noise[None, :]
        init_euler_noise = torch.tensor(self.cfg.init_euler_noise, device=self.device)
        if (init_euler_noise > 0).any():
            euler_noise_tensor = torch.randn(len(env_ids), 3, device=self.device) * init_euler_noise[None, :]
            noise_quat = matrix_to_quaternion(euler_angles_to_matrix(euler_noise_tensor, "XYZ"))
            root_quat = quat_mul(root_quat, noise_quat)

        self._robot.write_root_state_to_sim(
            torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=1),
            env_ids=env_ids,
        )

    def _randomize_pd_gains(self, env_ids):
        kp_ratio = (torch.rand(len(env_ids), 18, device=self.device)
                    * (self.cfg.kp_ratio_range[1] - self.cfg.kp_ratio_range[0])
                    + self.cfg.kp_ratio_range[0])
        kd_ratio = (torch.rand(len(env_ids), 18, device=self.device)
                    * (self.cfg.kd_ratio_range[1] - self.cfg.kd_ratio_range[0])
                    + self.cfg.kd_ratio_range[0])
        self.kp[env_ids] = self._init_kp[None, :] * kp_ratio
        self.kd[env_ids] = self._init_kd[None, :] * kd_ratio

    def _reset_task(self, env_ids):
        n = len(env_ids)

        # Update curriculum
        if self.cfg.pos_sigma_curriculum:
            avg_pos_err = self._past_pos_err.mean().item()
            for level, (threshold, sigma) in enumerate(
                sorted(self.cfg.pos_sigma_curriculum, key=lambda x: x[0], reverse=True)
            ):
                if avg_pos_err < threshold:
                    self._pos_err_sigma = sigma
                    self._pos_curriculum_level = level
                    break
        if self.cfg.orn_sigma_curriculum:
            avg_orn_err = self._past_orn_err.mean().item()
            for level, (threshold, sigma) in enumerate(
                sorted(self.cfg.orn_sigma_curriculum, key=lambda x: x[0], reverse=True)
            ):
                if avg_orn_err < threshold:
                    self._orn_err_sigma = sigma
                    self._orn_curriculum_level = level
                    break

        self._past_pos_err[env_ids] = 1.0
        self._past_orn_err[env_ids] = 1.0
        self._generate_random_trajectories(env_ids)

    # ========== Task: Reaching with Multi-Time-Horizon Observations ==========

    def _generate_random_trajectories(self, env_ids):
        n = len(env_ids)
        ep_len = self.max_episode_length

        base_pos = torch.tensor([[0.3, 0.0, 0.3]], device="cpu").repeat(n, 1)

        pos_seq = torch.zeros(n, ep_len, 3)
        rot_seq = torch.zeros(n, ep_len, 3, 3)

        for i in range(n):
            num_waypoints = torch.randint(2, 4, (1,)).item()
            waypoint_positions = base_pos[i:i+1] + (torch.rand(num_waypoints, 3) * 0.6 - 0.3)
            waypoint_rots = torch.eye(3).unsqueeze(0).repeat(num_waypoints, 1, 1)

            for t in range(ep_len):
                frac = t / max(ep_len - 1, 1)
                wp_idx = min(int(frac * (num_waypoints - 1)), num_waypoints - 2)
                wp_frac = frac * (num_waypoints - 1) - wp_idx
                pos_seq[i, t] = (
                    waypoint_positions[wp_idx] * (1 - wp_frac) + waypoint_positions[wp_idx + 1] * wp_frac
                )
                rot_seq[i, t] = waypoint_rots[wp_idx]

        self._target_pos_seq[env_ids.cpu()] = pos_seq
        self._target_rot_mat_seq[env_ids.cpu()] = rot_seq

    def _get_targets_at_times(self, times):
        dt = self.step_dt
        episode_step = (times / dt).long().clamp(0, self.max_episode_length - 1)
        env_idx = torch.arange(self.num_envs, device=self.device)
        pos = self._target_pos_seq[env_idx.cpu(), episode_step.cpu()].to(self.device)
        rot_mat = self._target_rot_mat_seq[env_idx.cpu(), episode_step.cpu()].to(self.device)
        return pos, rot_mat

    def _compute_task_observations(self):
        robot = self._robot
        episode_time = self.episode_length_buf * self.step_dt
        ee_pos = robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_wxyz = robot.data.body_quat_w[:, self._ee_body_idx]
        ee_rot_mat = quaternion_to_matrix(ee_quat_wxyz)

        obs_terms = []
        for t_offset in self.cfg.target_obs_times:
            t_tensor = torch.full((self.num_envs,), t_offset, device=self.device)
            target_pos, target_rot = self._get_targets_at_times(episode_time + t_tensor)

            pos_diff = target_pos - ee_pos
            rel_pos = torch.bmm(ee_rot_mat.transpose(1, 2), pos_diff.unsqueeze(-1)).squeeze(-1)

            if self.cfg.position_obs_encoding == "linear":
                pos_obs = rel_pos * self.cfg.pos_obs_scale
            else:
                distance = torch.linalg.norm(rel_pos, dim=-1, keepdim=True) + 1e-8
                direction = rel_pos / distance
                pos_obs = torch.cat([torch.log(distance * self.cfg.pos_obs_scale), direction], dim=-1)

            if self.cfg.pos_obs_clip is not None:
                pos_obs = torch.clamp(pos_obs, -self.cfg.pos_obs_clip, self.cfg.pos_obs_clip)

            rel_rot = torch.bmm(ee_rot_mat.transpose(1, 2), target_rot)
            orn_obs = matrix_to_rotation_6d(rel_rot) * self.cfg.orn_obs_scale

            obs_terms.append(torch.cat([pos_obs, orn_obs], dim=1))

        return torch.cat(obs_terms, dim=1)

    def _get_pos_err(self):
        ee_pos = self._robot.data.body_pos_w[:, self._ee_body_idx]
        episode_time = self.episode_length_buf * self.step_dt
        target_pos, _ = self._get_targets_at_times(episode_time)
        return torch.sqrt(torch.sum(torch.square(target_pos - ee_pos), dim=1))

    def _get_orn_err(self):
        ee_quat_wxyz = self._robot.data.body_quat_w[:, self._ee_body_idx]
        ee_rot_mat = quaternion_to_matrix(ee_quat_wxyz)
        episode_time = self.episode_length_buf * self.step_dt
        _, target_rot = self._get_targets_at_times(episode_time)
        rot_err_mat = target_rot @ ee_rot_mat.transpose(1, 2)
        trace = torch.diagonal(rot_err_mat, dim1=-2, dim2=-1).sum(dim=-1)
        trace = torch.clamp(trace, min=-1 + 1e-8, max=3 - 1e-8)
        rotation_magnitude = torch.arccos((trace - 1) / 2)
        rotation_magnitude = rotation_magnitude % (2 * math.pi)
        return torch.min(rotation_magnitude, 2 * math.pi - rotation_magnitude)

    # ========== Debug Visualization ==========

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create/update visualization markers for target and current EE poses."""
        if debug_vis:
            if not hasattr(self, "_target_visualizer"):
                import isaaclab.sim as sim_utils
                from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
                from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

                # Markers for current EE pose (green frame) and target poses (spheres)
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/UMIonLegs",
                    markers={
                        "ee_frame": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                            scale=(0.15, 0.15, 0.15),
                        ),
                        "target_sphere": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.0)),
                        ),
                    },
                )
                self._target_visualizer = VisualizationMarkers(marker_cfg)
            self._target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "_target_visualizer"):
                self._target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update marker positions each frame (all environments)."""
        if not hasattr(self, "_target_visualizer"):
            return
        robot = self._robot
        n = self.num_envs

        # Current EE positions (all envs)
        ee_pos = robot.data.body_pos_w[:, self._ee_body_idx]  # (n, 3)
        ee_quat = robot.data.body_quat_w[:, self._ee_body_idx]  # (n, 4) wxyz
        ee_quat_xyzw = ee_quat[:, [1, 2, 3, 0]]  # wxyz -> xyzw

        # Target positions at observation time horizons (all envs)
        episode_time = self.episode_length_buf * self.step_dt
        target_positions = []
        for t_offset in self.cfg.target_obs_times:
            t_tensor = torch.full((n,), t_offset, device=self.device)
            t_pos, _ = self._get_targets_at_times(episode_time + t_tensor)
            target_positions.append(t_pos)  # (n, 3)

        num_targets = len(target_positions)

        # Interleave: [ee_0, targets_0, ee_1, targets_1, ...]
        all_pos_list = []
        all_quat_list = []
        marker_indices_list = []
        id_quat_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device)

        for i in range(n):
            all_pos_list.append(ee_pos[i:i+1])
            all_quat_list.append(ee_quat_xyzw[i:i+1])
            marker_indices_list.append(0)  # EE frame
            for tp in target_positions:
                all_pos_list.append(tp[i:i+1])
                all_quat_list.append(id_quat_xyzw)
                marker_indices_list.append(1)  # target sphere

        all_pos = torch.cat(all_pos_list, dim=0)  # (n * (1+num_targets), 3)
        all_quat = torch.cat(all_quat_list, dim=0)  # (n * (1+num_targets), 4)
        marker_indices = torch.tensor(marker_indices_list, dtype=torch.long, device=self.device)

        self._target_visualizer.visualize(all_pos, all_quat, marker_indices=marker_indices)
