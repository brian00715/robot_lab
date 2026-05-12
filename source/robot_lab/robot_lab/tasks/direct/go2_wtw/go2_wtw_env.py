# SPDX-License-Identifier: Apache-2.0
# Ported from walk-these-ways-go2 (IsaacGym) to IsaacLab 2.3.0
# Original: go2_gym/envs/base/legged_robot.py + go2_gym/envs/rewards/corl_rewards.py

from __future__ import annotations

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    RED_ARROW_X_MARKER_CFG,
    SPHERE_MARKER_CFG,
)
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import (
    quat_apply_yaw,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_apply_inverse,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
)

from .go2_wtw_curriculum import RewardThresholdCurriculum
from .go2_wtw_env_cfg import Go2WalkTheseWaysEnvCfg


class Go2WalkTheseWaysEnv(DirectRLEnv):
    """Go2 Walk-These-Ways locomotion environment (Direct RL).

    Faithful port of walk-these-ways-go2 from IsaacGym to IsaacLab 2.3.0.
    Preserves MDP semantics: 15-command gait control, CoRL rewards,
    RewardThresholdCurriculum, and domain randomisation.
    """

    cfg: Go2WalkTheseWaysEnvCfg

    # hip joints (0,3,6,9) get hip_scale_reduction applied in go2_config
    HIP_JOINT_INDICES = [0, 3, 6, 9]

    def __init__(self, cfg: Go2WalkTheseWaysEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- body/joint indices -----------------------------------------------
        # IMPORTANT: contact_sensor and robot.data use DIFFERENT body orderings.
        # Use contact_sensor.find_bodies() for force lookups (contact_forces tensor).
        # Use robot.find_bodies() for position/velocity lookups (body_pos_w, body_lin_vel_w).
        cs_feet_idx, _ = self.contact_sensor.find_bodies(".*foot")
        self.cs_feet_indices = sorted(cs_feet_idx)  # for contact_forces
        rob_feet_idx, _ = self.robot.find_bodies(".*foot")
        self.robot_feet_indices = sorted(rob_feet_idx)  # for body_pos_w / body_lin_vel_w
        # Alias for backward compat with reward fns that use feet forces
        self.feet_indices = self.cs_feet_indices

        self.penalised_contact_indices, _ = self.contact_sensor.find_bodies(".*thigh|.*calf")
        self.termination_contact_indices, _ = self.contact_sensor.find_bodies("base")
        self.penalised_contact_indices = sorted(self.penalised_contact_indices)
        self.termination_contact_indices = sorted(self.termination_contact_indices)
        base_idx, _ = self.robot.find_bodies("base")
        self.robot_base_index = int(base_idx[0])

        # ---- dof limits (soft, already factored by soft_joint_pos_limit_factor) -----
        self.dof_pos_limits = self.robot.data.soft_joint_pos_limits[0]  # [12, 2]

        # ---- default joint positions -----------------------------------------
        self.default_dof_pos = self.robot.data.default_joint_pos[0].clone()  # [12]

        # ---- gravity vector --------------------------------------------------
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # ---- buffers ----------------------------------------------------------
        num_dof = self.cfg.action_space  # 12

        self.actions = torch.zeros(self.num_envs, num_dof, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros(self.num_envs, num_dof, device=self.device)
        self.joint_pos_target = torch.zeros(self.num_envs, num_dof, device=self.device)
        self.last_joint_pos_target = torch.zeros_like(self.joint_pos_target)
        self.last_last_joint_pos_target = torch.zeros_like(self.joint_pos_target)
        self.last_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        # foot velocities: prev is updated at END of each step so rewards can use both current and prev
        self.foot_velocities = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.prev_foot_velocities = torch.zeros_like(self.foot_velocities)

        # ---- gait / clock buffers --------------------------------------------
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.foot_indices = torch.zeros(self.num_envs, 4, device=self.device)
        self.desired_footswing_height = torch.zeros(self.num_envs, device=self.device)

        # ---- commands --------------------------------------------------------
        self.commands = torch.zeros(self.num_envs, self.cfg.num_commands, device=self.device)
        self.commands_scale = self._build_commands_scale()

        # ---- cached state (updated in _refresh_reward_state) -----------------
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros_like(self.base_lin_vel)
        self.projected_gravity = torch.zeros_like(self.base_lin_vel)
        self.base_pos = torch.zeros_like(self.base_lin_vel)
        self.base_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.dof_pos = torch.zeros(self.num_envs, num_dof, device=self.device)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.torques = torch.zeros_like(self.dof_pos)
        self.contact_forces = torch.zeros(self.num_envs, self.robot.num_bodies, 3, device=self.device)
        self.foot_positions = torch.zeros(self.num_envs, 4, 3, device=self.device)

        # ---- domain randomization buffers ------------------------------------
        self.friction_coeffs = torch.ones(self.num_envs, 1, device=self.device)
        self.restitutions = torch.zeros(self.num_envs, 1, device=self.device)
        self.payloads = torch.zeros(self.num_envs, device=self.device)
        self.com_displacements = torch.zeros(self.num_envs, 3, device=self.device)
        self.motor_strengths = torch.ones(self.num_envs, num_dof, device=self.device)
        self.motor_offsets = torch.zeros(self.num_envs, num_dof, device=self.device)
        self.Kp_factors = torch.ones(self.num_envs, num_dof, device=self.device)
        self.Kd_factors = torch.ones(self.num_envs, num_dof, device=self.device)
        self.gravities = torch.zeros(self.num_envs, 3, device=self.device)

        # ---- action lag buffer -----------------------------------------------
        lag_n = self.cfg.lag_timesteps + 1
        self.lag_buffer = [torch.zeros(self.num_envs, num_dof, device=self.device) for _ in range(lag_n)]

        # ---- noise scale vector ----------------------------------------------
        self.noise_scale_vec = self._build_noise_scale_vec()
        self.obs_history = torch.zeros(
            self.num_envs,
            self.cfg.num_observation_history * self.cfg.num_scalar_observations,
            device=self.device,
        )

        # ---- command curriculum ----------------------------------------------
        self._init_command_distribution()

        # ---- reward bookkeeping ----------------------------------------------
        self.reward_names = [
            "tracking_lin_vel",
            "tracking_ang_vel",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "orientation_control",
            "torques",
            "dof_vel",
            "dof_acc",
            "action_rate",
            "action_smoothness_1",
            "action_smoothness_2",
            "collision",
            "dof_pos_limits",
            "jump",
            "tracking_contacts_shaped_force",
            "tracking_contacts_shaped_vel",
            "feet_clearance_cmd_linear",
            "feet_slip",
            "feet_impact_vel",
            "raibert_heuristic",
            "feet_contact_forces",
            "base_height",
            "dof_pos",
            "feet_air_time",
        ]
        self.reward_scales = {
            "tracking_lin_vel": self.cfg.rew_tracking_lin_vel,
            "tracking_ang_vel": self.cfg.rew_tracking_ang_vel,
            "lin_vel_z": self.cfg.rew_lin_vel_z,
            "ang_vel_xy": self.cfg.rew_ang_vel_xy,
            "orientation": self.cfg.rew_orientation,
            "orientation_control": self.cfg.rew_orientation_control,
            "torques": self.cfg.rew_torques,
            "dof_vel": self.cfg.rew_dof_vel,
            "dof_acc": self.cfg.rew_dof_acc,
            "action_rate": self.cfg.rew_action_rate,
            "action_smoothness_1": self.cfg.rew_action_smoothness_1,
            "action_smoothness_2": self.cfg.rew_action_smoothness_2,
            "collision": self.cfg.rew_collision,
            "dof_pos_limits": self.cfg.rew_dof_pos_limits,
            "jump": self.cfg.rew_jump,
            "tracking_contacts_shaped_force": self.cfg.rew_tracking_contacts_shaped_force,
            "tracking_contacts_shaped_vel": self.cfg.rew_tracking_contacts_shaped_vel,
            "feet_clearance_cmd_linear": self.cfg.rew_feet_clearance_cmd_linear,
            "feet_slip": self.cfg.rew_feet_slip,
            "feet_impact_vel": self.cfg.rew_feet_impact_vel,
            "raibert_heuristic": self.cfg.rew_raibert_heuristic,
            "feet_contact_forces": self.cfg.rew_feet_contact_forces,
            "base_height": self.cfg.rew_base_height,
            "dof_pos": self.cfg.rew_dof_pos,
            "feet_air_time": self.cfg.rew_feet_air_time,
        }
        self.reward_scales = {name: scale * self.step_dt for name, scale in self.reward_scales.items()}
        self.episode_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in self.reward_names}
        self.episode_sums["total"] = torch.zeros(self.num_envs, device=self.device)
        self.command_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in self.reward_names}
        self.command_sums["lin_vel_raw"] = torch.zeros(self.num_envs, device=self.device)
        self.command_sums["ang_vel_raw"] = torch.zeros(self.num_envs, device=self.device)
        self.command_sums["lin_vel_residual"] = torch.zeros(self.num_envs, device=self.device)
        self.command_sums["ang_vel_residual"] = torch.zeros(self.num_envs, device=self.device)
        self.command_sums["ep_timesteps"] = torch.zeros(self.num_envs, device=self.device)

        # ---- startup DR -------------------------------------------------------
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._cache_default_rigid_body_props()
        self._randomize_rigid_body_props(env_ids)
        self._refresh_actor_rigid_body_props(env_ids)
        self._refresh_actor_material_props(env_ids)
        self._randomize_dof_props(env_ids)

        if self.cfg.enable_debug_vis:
            self.set_debug_vis(True)

    # ==========================================================================
    # Scene setup
    # ==========================================================================

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self.robot

        # Contact sensor for ALL robot bodies (needed for feet + termination + collision)
        contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=2,
            update_period=0.0,
            track_air_time=True,
        )
        self.contact_sensor = ContactSensor(contact_sensor_cfg)
        self.scene.sensors["contact_forces"] = self.contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ==========================================================================
    # Step callbacks (called by DirectRLEnv.step() in order):
    # _pre_physics_step → [physics loop] → _get_dones → _get_rewards → _reset_idx → _get_observations
    # ==========================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        # Save foot velocity state BEFORE physics for prev_foot_velocities reward
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.actions = torch.clip(actions, -self.cfg.clip_actions, self.cfg.clip_actions)

    def _apply_action(self):
        # Scale actions; hip joints get additional reduction
        actions_scaled = self.actions * self.cfg.action_scale
        actions_scaled[:, self.HIP_JOINT_INDICES] *= self.cfg.hip_scale_reduction
        # Motor strength DR: scales effective torque by scaling the position offset
        actions_scaled = actions_scaled * self.motor_strengths

        # Lag buffer simulation
        if self.cfg.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        else:
            self.joint_pos_target = actions_scaled + self.default_dof_pos

        # Motor offset (simulates calibration error)
        target = self.joint_pos_target + self.motor_offsets
        self.robot.set_joint_position_target(target)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Refresh physics state first
        self._refresh_reward_state()

        # Advance gait clock and compute desired_contact_states
        self._step_contact_targets()

        # Periodic updates (command resample, DR, push)
        self._periodic_updates()

        # Contact-based termination (base body)
        base_contact = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )

        # Body height termination
        height_fail = (
            self.base_pos[:, 2] < self.cfg.terminal_body_height
            if self.cfg.use_terminal_body_height
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )

        # Roll/pitch termination
        if self.cfg.use_terminal_roll_pitch:
            ori_fail = torch.norm(self.projected_gravity[:, :2], dim=-1) > self.cfg.terminal_body_ori
        else:
            ori_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        terminated = base_contact | height_fail | ori_fail
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        # state was refreshed in _get_dones (called first), so we can compute rewards directly
        return self._compute_reward()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        if len(env_ids) == 0:
            return

        self._log_episode(env_ids)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Randomize physics properties on reset
        self._randomize_rigid_body_props(env_ids)
        self._refresh_actor_rigid_body_props(env_ids)
        self._refresh_actor_material_props(env_ids)
        self._randomize_dof_props(env_ids)

        # Reset joint states with random scaling [0.5, 1.5]
        joint_pos = self.robot.data.default_joint_pos[env_ids] * torch.empty(
            len(env_ids), self.cfg.action_space, device=self.device
        ).uniform_(0.5, 1.5)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset root state
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 0] += torch.empty(len(env_ids), device=self.device).uniform_(
            -self.cfg.init_x_range, self.cfg.init_x_range
        )
        root_state[:, 1] += torch.empty(len(env_ids), device=self.device).uniform_(
            -self.cfg.init_y_range, self.cfg.init_y_range
        )
        root_state[:, 2] = self.cfg.init_pos_z + self.scene.env_origins[env_ids, 2]

        # Random yaw
        if self.cfg.init_yaw_range > 0.0:
            yaws = torch.empty(len(env_ids), device=self.device).uniform_(
                -self.cfg.init_yaw_range, self.cfg.init_yaw_range
            )
            quats = quat_from_angle_axis(yaws, torch.tensor([0.0, 0.0, 1.0], device=self.device))
            root_state[:, 3:7] = quats
        else:
            root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # identity in WXYZ

        # Random base velocity
        root_state[:, 7:13] = torch.empty(len(env_ids), 6, device=self.device).uniform_(
            -self.cfg.init_vel_range, self.cfg.init_vel_range
        )
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        # Reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_joint_pos_target[env_ids] = 0.0
        self.last_last_joint_pos_target[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.gait_indices[env_ids] = 0.0
        self.obs_history[env_ids] = 0.0
        for buf in self.lag_buffer:
            buf[env_ids] = 0.0

        self._resample_commands(env_ids)

    def _get_observations(self) -> dict:
        # Update action/vel history at END of step (used by next step's rewards)
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        # Update foot velocities for next step's prev_foot_velocities
        self.foot_velocities = self.robot.data.body_lin_vel_w[:, self.robot_feet_indices, :]

        obs = torch.cat(
            [
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 15
                (self.dof_pos - self.default_dof_pos) * self.cfg.obs_dof_pos_scale,  # 12
                self.dof_vel * self.cfg.obs_dof_vel_scale,  # 12
                self.actions,  # 12
            ],
            dim=-1,
        )

        if self.cfg.observe_two_prev_actions:
            obs = torch.cat([obs, self.last_actions], dim=-1)  # 12

        if self.cfg.observe_clock_inputs:
            obs = torch.cat([obs, self.clock_inputs], dim=-1)  # 4

        if self.cfg.add_noise:
            obs = obs + (2.0 * torch.rand_like(obs) - 1.0) * self.noise_scale_vec

        obs = torch.clip(obs, -self.cfg.clip_observations, self.cfg.clip_observations)
        self.obs_history = torch.cat([self.obs_history[:, self.cfg.num_scalar_observations :], obs], dim=-1)

        # Privileged observations for teacher (friction + restitution per env)
        priv_obs = torch.cat(
            [
                self._scale_shift(self.friction_coeffs, self.cfg.friction_obs_range),
                self._scale_shift(self.restitutions, self.cfg.restitution_obs_range),
            ],
            dim=-1,
        )

        return {"policy": obs, "obs_history": self.obs_history, "privileged": priv_obs}

    # ==========================================================================
    # Periodic in-episode updates
    # ==========================================================================

    def _periodic_updates(self):
        """Handle mid-episode command resampling, DR, and robot pushes."""
        step_dt = self.step_dt

        # Command resampling
        sample_interval = max(1, round(self.cfg.resampling_time / step_dt))
        resample_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

        # Gravity randomization (per sim-step counter)
        gravity_interval = max(1, round(self.cfg.gravity_rand_interval_s / step_dt))
        if self.common_step_counter % gravity_interval == 0:
            self._randomize_gravity()
        gravity_dur = max(1, round(self.cfg.gravity_impulse_duration / step_dt))
        if (self.common_step_counter - gravity_dur) % gravity_interval == 0:
            self._randomize_gravity(torch.zeros(3, device=self.device))

        # DR interval
        rand_interval = max(1, round(self.cfg.rand_interval_s / step_dt))
        rand_ids = (self.episode_length_buf % rand_interval == 0).nonzero(as_tuple=False).flatten()
        if len(rand_ids) > 0:
            if self.cfg.randomize_rigids_after_start:
                self._randomize_rigid_body_props(rand_ids)
                self._refresh_actor_rigid_body_props(rand_ids)
                self._refresh_actor_material_props(rand_ids)
            self._randomize_dof_props(rand_ids)

        # Push robots
        if self.cfg.push_robots:
            push_interval = max(1, round(self.cfg.push_interval_s / step_dt))
            push_ids = (self.episode_length_buf % push_interval == 0).nonzero(as_tuple=False).flatten()
            if len(push_ids) > 0:
                self._push_robots(push_ids)

    # ==========================================================================
    # Physics state refresh
    # ==========================================================================

    def _refresh_reward_state(self):
        """Cache physics tensors for reward computation."""
        self.base_lin_vel = self.robot.data.root_lin_vel_b  # body frame
        self.base_ang_vel = self.robot.data.root_ang_vel_b
        self.projected_gravity = self.robot.data.projected_gravity_b
        self.base_pos = self.robot.data.root_pos_w
        self.base_quat = self.robot.data.root_quat_w
        self.dof_pos = self.robot.data.joint_pos
        self.dof_vel = self.robot.data.joint_vel
        self.contact_forces = self.contact_sensor.data.net_forces_w  # [N, n_body, 3]
        self.foot_positions = self.robot.data.body_pos_w[:, self.robot_feet_indices, :]  # [N, 4, 3]
        self.foot_velocities = self.robot.data.body_lin_vel_w[:, self.robot_feet_indices, :]  # [N, 4, 3]
        self.torques = self.robot.data.applied_torque[:, : self.cfg.action_space]

    # ==========================================================================
    # Reward computation (CoRL reward functions from corl_rewards.py)
    # ==========================================================================

    def _compute_reward(self) -> torch.Tensor:
        rew_buf = torch.zeros(self.num_envs, device=self.device)
        rew_buf_pos = torch.zeros_like(rew_buf)
        rew_buf_neg = torch.zeros_like(rew_buf)

        rew_fns = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "orientation": self._reward_orientation,
            "orientation_control": self._reward_orientation_control,
            "torques": self._reward_torques,
            "dof_vel": self._reward_dof_vel,
            "dof_acc": self._reward_dof_acc,
            "action_rate": self._reward_action_rate,
            "action_smoothness_1": self._reward_action_smoothness_1,
            "action_smoothness_2": self._reward_action_smoothness_2,
            "collision": self._reward_collision,
            "dof_pos_limits": self._reward_dof_pos_limits,
            "jump": self._reward_jump,
            "tracking_contacts_shaped_force": self._reward_tracking_contacts_shaped_force,
            "tracking_contacts_shaped_vel": self._reward_tracking_contacts_shaped_vel,
            "feet_clearance_cmd_linear": self._reward_feet_clearance_cmd_linear,
            "feet_slip": self._reward_feet_slip,
            "feet_impact_vel": self._reward_feet_impact_vel,
            "raibert_heuristic": self._reward_raibert_heuristic,
            "feet_contact_forces": self._reward_feet_contact_forces,
            "base_height": self._reward_base_height,
            "dof_pos": self._reward_dof_pos,
            "feet_air_time": self._reward_feet_air_time,
        }

        for name, fn in rew_fns.items():
            scale = self.reward_scales[name]
            if scale == 0.0:
                continue
            rew = fn() * scale
            rew_buf += rew
            rew_buf_pos += torch.clip(rew, min=0.0)
            rew_buf_neg += torch.clip(rew, max=0.0)
            self.episode_sums[name] += rew
            if name in ("tracking_contacts_shaped_force", "tracking_contacts_shaped_vel"):
                self.command_sums[name] += scale + rew
            else:
                self.command_sums[name] += rew

        if self.cfg.only_positive_rewards:
            rew_buf = torch.clip(rew_buf, min=0.0)
        elif self.cfg.only_positive_rewards_ji22_style:
            rew_buf = rew_buf_pos * torch.exp(rew_buf_neg / self.cfg.sigma_rew_neg)

        self.episode_sums["total"] += rew_buf
        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1

        return rew_buf

    # ------------ CoRL reward functions ----------------------------------------

    def _reward_tracking_lin_vel(self):
        err = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-err / self.cfg.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        err = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-err / self.cfg.tracking_sigma_yaw)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_orientation_control(self):
        rp = self.commands[:, 10:12]  # [roll_cmd, pitch_cmd]
        quat_roll = quat_from_angle_axis(
            -rp[:, 1],
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1),
        )
        quat_pitch = quat_from_angle_axis(
            -rp[:, 0],
            torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1),
        )
        desired_quat = quat_mul(quat_roll, quat_pitch)
        desired_gravity = quat_apply_inverse(desired_quat, self.gravity_vec)
        return torch.sum(torch.square(self.projected_gravity[:, :2] - desired_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.step_dt), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness_1(self):
        diff = torch.square(self.joint_pos_target - self.last_joint_pos_target)
        diff = diff * (self.last_actions != 0).float()
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        diff = torch.square(self.joint_pos_target - 2.0 * self.last_joint_pos_target + self.last_last_joint_pos_target)
        diff = diff * (self.last_actions != 0).float()
        diff = diff * (self.last_last_actions != 0).float()
        return torch.sum(diff, dim=1)

    def _reward_collision(self):
        return torch.sum(
            (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1).float(),
            dim=1,
        )

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_jump(self):
        jump_target = self.commands[:, 3] + self.cfg.base_height_target
        return -torch.square(self.base_pos[:, 2] - jump_target)

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)  # [N, 4]
        desired = self.desired_contact_states
        rew = torch.zeros(self.num_envs, device=self.device)
        for i in range(4):
            rew += -(1 - desired[:, i]) * (1 - torch.exp(-(foot_forces[:, i] ** 2) / self.cfg.gait_force_sigma))
        return rew / 4

    def _reward_tracking_contacts_shaped_vel(self):
        foot_vels = torch.norm(self.foot_velocities, dim=2)  # [N, 4]
        desired = self.desired_contact_states
        rew = torch.zeros(self.num_envs, device=self.device)
        for i in range(4):
            rew += -(desired[:, i] * (1 - torch.exp(-(foot_vels[:, i] ** 2) / self.cfg.gait_vel_sigma)))
        return rew / 4

    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = self.foot_positions[:, :, 2]
        target_height = self.commands[:, 9].unsqueeze(1) * phases + 0.02
        rew = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        return torch.sum(rew, dim=1)

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        foot_vels_sq = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2))
        return torch.sum(contact_filt.float() * foot_vels_sq, dim=1)

    def _reward_feet_impact_vel(self):
        prev_fvz = self.prev_foot_velocities[:, :, 2]
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0
        rew = contact.float() * torch.square(torch.clip(prev_fvz, -100.0, 0.0))
        return torch.sum(rew, dim=1)

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_body = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_body[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat), cur_footsteps_translated[:, i, :])

        if self.cfg.num_commands >= 13:
            stance_w = self.commands[:, 12:13]
            ys_nom = torch.cat([stance_w / 2, -stance_w / 2, stance_w / 2, -stance_w / 2], dim=1)
        else:
            sw = 0.3
            ys_nom = (
                torch.tensor([sw / 2, -sw / 2, sw / 2, -sw / 2], device=self.device)
                .unsqueeze(0)
                .expand(self.num_envs, -1)
            )

        if self.cfg.num_commands >= 14:
            stance_l = self.commands[:, 13:14]
            xs_nom = torch.cat([stance_l / 2, stance_l / 2, -stance_l / 2, -stance_l / 2], dim=1)
        else:
            sl = 0.45
            xs_nom = (
                torch.tensor([sl / 2, sl / 2, -sl / 2, -sl / 2], device=self.device)
                .unsqueeze(0)
                .expand(self.num_envs, -1)
            )

        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.commands[:, 4]
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]

        if self.cfg.num_commands >= 14:
            sl_val = self.commands[:, 13:14]
        else:
            sl_val = torch.full((self.num_envs, 1), 0.45, device=self.device)
        y_vel_des = yaw_vel_des * sl_val / 2

        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        ys_nom = ys_nom + desired_ys_offset
        xs_nom = xs_nom + desired_xs_offset

        desired_footsteps = torch.cat([xs_nom.unsqueeze(2), ys_nom.unsqueeze(2)], dim=2)
        err = torch.abs(desired_footsteps - footsteps_body[:, :, 0:2])
        return torch.sum(torch.square(err), dim=(1, 2))

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.max_contact_force).clip(
                min=0.0
            ),
            dim=1,
        )

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.cfg.base_height_target)

    def _reward_dof_pos(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        first_contact = (self.feet_air_time > 0) * contact
        self.feet_air_time += self.step_dt
        self.feet_air_time *= (~contact).float()
        rew = torch.sum((self.feet_air_time - 0.5) * first_contact.float(), dim=1)
        rew = rew * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
        return rew

    # ==========================================================================
    # Gait clock
    # ==========================================================================

    def _step_contact_targets(self):
        """Advance gait clock and compute desired_contact_states and clock_inputs."""
        if not self.cfg.observe_gait_commands:
            return

        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        offsets = self.commands[:, 6]
        bounds = self.commands[:, 7]
        durations = self.commands[:, 8]

        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        # Raw foot phase indices (FL/FR/RL/RR)
        fi0 = self.gait_indices + phases + offsets + bounds
        fi1 = self.gait_indices + offsets
        fi2 = self.gait_indices + bounds
        fi3 = self.gait_indices + phases

        self.foot_indices = torch.remainder(torch.stack([fi0, fi1, fi2, fi3], dim=1), 1.0)

        # Remap foot phases for clock signal (stance=0..0.5, swing=0.5..1)
        def remap_phase(fi):
            r = torch.remainder(fi, 1.0)
            fi_out = torch.zeros_like(fi)
            stance = r < durations
            swing = ~stance
            fi_out[stance] = r[stance] * (0.5 / durations[stance])
            fi_out[swing] = 0.5 + (r[swing] - durations[swing]) * (0.5 / (1.0 - durations[swing]))
            return fi_out

        self.clock_inputs[:, 0] = torch.sin(2 * math.pi * remap_phase(fi0))
        self.clock_inputs[:, 1] = torch.sin(2 * math.pi * remap_phase(fi1))
        self.clock_inputs[:, 2] = torch.sin(2 * math.pi * remap_phase(fi2))
        self.clock_inputs[:, 3] = torch.sin(2 * math.pi * remap_phase(fi3))

        # Von Mises smoothing for desired contact states (kappa controls transition sharpness)
        kappa = self.cfg.kappa_gait_probs
        cdf = torch.distributions.normal.Normal(0.0, kappa).cdf

        def smooth_contact(fi):
            r = torch.remainder(fi, 1.0)
            return cdf(r) * (1 - cdf(r - 0.5)) + cdf(r - 1) * (1 - cdf(r - 0.5 - 1))

        self.desired_contact_states[:, 0] = smooth_contact(fi0)
        self.desired_contact_states[:, 1] = smooth_contact(fi1)
        self.desired_contact_states[:, 2] = smooth_contact(fi2)
        self.desired_contact_states[:, 3] = smooth_contact(fi3)

        if self.cfg.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]

    # ==========================================================================
    # Command curriculum
    # ==========================================================================

    def _init_command_distribution(self):
        self.category_names = ["pronk", "trot", "pace", "bound"]

        kw = dict(
            x_vel=(self.cfg.limit_vel_x[0], self.cfg.limit_vel_x[1], self.cfg.num_bins_vel_x),
            y_vel=(self.cfg.limit_vel_y[0], self.cfg.limit_vel_y[1], self.cfg.num_bins_vel_y),
            yaw_vel=(self.cfg.limit_vel_yaw[0], self.cfg.limit_vel_yaw[1], self.cfg.num_bins_vel_yaw),
            body_height=(self.cfg.limit_body_height[0], self.cfg.limit_body_height[1], self.cfg.num_bins_body_height),
            gait_frequency=(
                self.cfg.limit_gait_frequency[0],
                self.cfg.limit_gait_frequency[1],
                self.cfg.num_bins_gait_frequency,
            ),
            gait_phase=(
                self.cfg.limit_gait_phase[0],
                self.cfg.limit_gait_phase[1],
                self.cfg.num_bins_gait_phase,
            ),
            gait_offset=(
                self.cfg.limit_gait_offset[0],
                self.cfg.limit_gait_offset[1],
                self.cfg.num_bins_gait_offset,
            ),
            gait_bounds=(
                self.cfg.limit_gait_bound[0],
                self.cfg.limit_gait_bound[1],
                self.cfg.num_bins_gait_bound,
            ),
            gait_duration=(
                self.cfg.limit_gait_duration[0],
                self.cfg.limit_gait_duration[1],
                self.cfg.num_bins_gait_duration,
            ),
            footswing_height=(
                self.cfg.limit_footswing_height[0],
                self.cfg.limit_footswing_height[1],
                self.cfg.num_bins_footswing_height,
            ),
            body_pitch=(self.cfg.limit_body_pitch[0], self.cfg.limit_body_pitch[1], self.cfg.num_bins_body_pitch),
            body_roll=(self.cfg.limit_body_roll[0], self.cfg.limit_body_roll[1], self.cfg.num_bins_body_roll),
            stance_width=(
                self.cfg.limit_stance_width[0],
                self.cfg.limit_stance_width[1],
                self.cfg.num_bins_stance_width,
            ),
            stance_length=(
                self.cfg.limit_stance_length[0],
                self.cfg.limit_stance_length[1],
                self.cfg.num_bins_stance_length,
            ),
            aux_reward_coef=(
                self.cfg.limit_aux_reward_coef[0],
                self.cfg.limit_aux_reward_coef[1],
                self.cfg.num_bins_aux_reward_coef,
            ),
        )

        low = np.array(
            [
                self.cfg.lin_vel_x[0],
                self.cfg.lin_vel_y[0],
                self.cfg.ang_vel_yaw[0],
                self.cfg.body_height_cmd[0],
                self.cfg.gait_frequency_cmd_range[0],
                self.cfg.gait_phase_cmd_range[0],
                self.cfg.gait_offset_cmd_range[0],
                self.cfg.gait_bound_cmd_range[0],
                self.cfg.gait_duration_cmd_range[0],
                self.cfg.footswing_height_range[0],
                self.cfg.body_pitch_range[0],
                self.cfg.body_roll_range[0],
                self.cfg.stance_width_range[0],
                self.cfg.stance_length_range[0],
                self.cfg.aux_reward_coef_range[0],
            ]
        )
        high = np.array(
            [
                self.cfg.lin_vel_x[1],
                self.cfg.lin_vel_y[1],
                self.cfg.ang_vel_yaw[1],
                self.cfg.body_height_cmd[1],
                self.cfg.gait_frequency_cmd_range[1],
                self.cfg.gait_phase_cmd_range[1],
                self.cfg.gait_offset_cmd_range[1],
                self.cfg.gait_bound_cmd_range[1],
                self.cfg.gait_duration_cmd_range[1],
                self.cfg.footswing_height_range[1],
                self.cfg.body_pitch_range[1],
                self.cfg.body_roll_range[1],
                self.cfg.stance_width_range[1],
                self.cfg.stance_length_range[1],
                self.cfg.aux_reward_coef_range[1],
            ]
        )

        self.curricula = []
        for i, _ in enumerate(self.category_names):
            c = RewardThresholdCurriculum(seed=self.cfg.curriculum_seed + i, **kw)
            c.set_to(low=low, high=high)
            self.curricula.append(c)

        self.env_command_bins = np.zeros(self.num_envs, dtype=int)
        self.env_command_categories = np.zeros(self.num_envs, dtype=int)

        self.curriculum_thresholds = {
            "tracking_lin_vel": self.cfg.curriculum_tracking_lin_vel,
            "tracking_ang_vel": self.cfg.curriculum_tracking_ang_vel,
            "tracking_contacts_shaped_force": self.cfg.curriculum_tracking_contacts_shaped_force,
            "tracking_contacts_shaped_vel": self.cfg.curriculum_tracking_contacts_shaped_vel,
        }

    def _resample_commands(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        ep_len = min(
            self.max_episode_length,
            max(1, round(self.cfg.resampling_time / self.step_dt)),
        )

        # Update curriculum from episode rewards
        for i, (_, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            mask = self.env_command_categories[env_ids.cpu().numpy()] == i
            cat_ids = env_ids[torch.tensor(mask, device=self.device, dtype=torch.bool)]
            if len(cat_ids) == 0:
                continue
            task_rewards, thresholds = [], []
            for key in [
                "tracking_lin_vel",
                "tracking_ang_vel",
                "tracking_contacts_shaped_force",
                "tracking_contacts_shaped_vel",
            ]:
                if self.reward_scales.get(key, 0.0) != 0.0:
                    task_rewards.append(self.command_sums[key][cat_ids] / ep_len)
                    thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])
            old_bins = self.env_command_bins[cat_ids.cpu().numpy()]
            if thresholds:
                curriculum.update(
                    old_bins,
                    task_rewards,
                    thresholds,
                    local_range=np.array(
                        [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    ),
                )

        # Assign envs to gait categories
        rand_f = torch.rand(len(env_ids), device=self.device)
        p = 1.0 / len(self.category_names)
        cat_env_ids = [
            env_ids[torch.logical_and(p * i <= rand_f, rand_f < p * (i + 1))] for i in range(len(self.category_names))
        ]

        for i, (category, ids, curriculum) in enumerate(zip(self.category_names, cat_env_ids, self.curricula)):
            bs = len(ids)
            if bs == 0:
                continue
            new_cmds, new_bins = curriculum.sample(batch_size=bs)
            self.env_command_bins[ids.cpu().numpy()] = new_bins
            self.env_command_categories[ids.cpu().numpy()] = i
            self.commands[ids] = torch.tensor(
                new_cmds[:, : self.cfg.num_commands], dtype=torch.float32, device=self.device
            )

        # Enforce gait-specific phase structure
        for category, ids in zip(self.category_names, cat_env_ids):
            if len(ids) == 0:
                continue
            if category == "trot":
                self.commands[ids, 5] = self.commands[ids, 5] / 2 + 0.25
                self.commands[ids, 6] = 0
                self.commands[ids, 7] = 0
            elif category == "pace":
                self.commands[ids, 5] = 0
                self.commands[ids, 6] = self.commands[ids, 6] / 2 + 0.25
                self.commands[ids, 7] = 0
            elif category == "bound":
                self.commands[ids, 5] = 0
                self.commands[ids, 6] = 0
                self.commands[ids, 7] = self.commands[ids, 7] / 2 + 0.25
            elif category == "pronk":
                self.commands[ids, 5] = (self.commands[ids, 5] / 2 - 0.25) % 1
                self.commands[ids, 6] = (self.commands[ids, 6] / 2 - 0.25) % 1
                self.commands[ids, 7] = (self.commands[ids, 7] / 2 - 0.25) % 1

        if self.cfg.binary_phases:
            self.commands[env_ids, 5] = torch.round(2 * self.commands[env_ids, 5]) / 2.0 % 1
            self.commands[env_ids, 6] = torch.round(2 * self.commands[env_ids, 6]) / 2.0 % 1
            self.commands[env_ids, 7] = torch.round(2 * self.commands[env_ids, 7]) / 2.0 % 1

        # Zero small xy velocity commands (dead zone)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # Reset command sums
        for key in self.command_sums:
            self.command_sums[key][env_ids] = 0.0

    # ==========================================================================
    # Domain randomization
    # ==========================================================================

    def _cache_default_rigid_body_props(self):
        self.default_base_mass = None
        self.default_base_com = None
        view = getattr(self.robot, "root_physx_view", None)
        if view is None:
            return
        try:
            masses = view.get_masses()
            self.default_base_mass = masses[:, self.robot_base_index].clone()
        except Exception:
            self.default_base_mass = None
        try:
            coms = view.get_coms()
            self.default_base_com = coms[:, self.robot_base_index].clone()
        except Exception:
            self.default_base_com = None

    def _randomize_rigid_body_props(self, env_ids: torch.Tensor):
        if self.cfg.randomize_base_mass:
            lo, hi = self.cfg.added_mass_range
            self.payloads[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(lo, hi)

        if self.cfg.randomize_com_displacement:
            lo, hi = self.cfg.com_displacement_range
            self.com_displacements[env_ids] = torch.empty(len(env_ids), 3, device=self.device).uniform_(lo, hi)

        if self.cfg.randomize_friction:
            lo, hi = self.cfg.friction_range
            self.friction_coeffs[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(lo, hi)

        if self.cfg.randomize_restitution:
            lo, hi = self.cfg.restitution_range
            self.restitutions[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(lo, hi)

    def _refresh_actor_rigid_body_props(self, env_ids: torch.Tensor):
        """Push randomized base mass and COM to PhysX when tensor views are available."""
        view = getattr(self.robot, "root_physx_view", None)
        if view is None:
            return
        env_ids_cpu = env_ids.cpu()
        if self.default_base_mass is not None:
            try:
                masses = view.get_masses()
                masses[env_ids, self.robot_base_index] = self.default_base_mass[env_ids] + self.payloads[env_ids]
                view.set_masses(masses, env_ids_cpu)
            except Exception:
                pass
        if self.default_base_com is not None:
            try:
                coms = view.get_coms()
                coms[env_ids, self.robot_base_index] = self.default_base_com[env_ids]
                coms[env_ids, self.robot_base_index, :3] += self.com_displacements[env_ids]
                view.set_coms(coms, env_ids_cpu)
            except Exception:
                pass

    def _refresh_actor_material_props(self, env_ids: torch.Tensor):
        """Push randomized friction/restitution to simulation."""
        if not hasattr(self.robot, "root_physx_view"):
            return
        try:
            # props: [num_envs, n_shapes, 3]  cols: static_friction, dynamic_friction, restitution
            # Must pass full tensor and env_ids; API indexes into it at env_ids rows
            props = self.robot.root_physx_view.get_material_properties()
            if props is None:
                return
            props[env_ids, :, 0] = self.friction_coeffs[env_ids, 0:1]
            props[env_ids, :, 1] = self.friction_coeffs[env_ids, 0:1]
            props[env_ids, :, 2] = self.restitutions[env_ids, 0:1]
            self.robot.root_physx_view.set_material_properties(props, env_ids.cpu())
        except Exception:
            pass  # material props API not always available

    def _randomize_dof_props(self, env_ids: torch.Tensor):
        n = len(env_ids)
        if self.cfg.randomize_motor_strength:
            lo, hi = self.cfg.motor_strength_range
            self.motor_strengths[env_ids] = (
                torch.empty(n, 1, device=self.device).uniform_(lo, hi).repeat(1, self.cfg.action_space)
            )

        if self.cfg.randomize_motor_offset:
            lo, hi = self.cfg.motor_offset_range
            self.motor_offsets[env_ids] = torch.empty(n, self.cfg.action_space, device=self.device).uniform_(lo, hi)

        if self.cfg.randomize_Kp_factor:
            lo, hi = self.cfg.Kp_factor_range
            self.Kp_factors[env_ids] = (
                torch.empty(n, 1, device=self.device).uniform_(lo, hi).repeat(1, self.cfg.action_space)
            )

        if self.cfg.randomize_Kd_factor:
            lo, hi = self.cfg.Kd_factor_range
            self.Kd_factors[env_ids] = (
                torch.empty(n, 1, device=self.device).uniform_(lo, hi).repeat(1, self.cfg.action_space)
            )

    def _randomize_gravity(self, external_force: torch.Tensor | None = None):
        if external_force is not None:
            self.gravities[:] = external_force.unsqueeze(0)
        elif self.cfg.randomize_gravity:
            lo, hi = self.cfg.gravity_range
            self.gravities[:] = torch.empty(3, device=self.device).uniform_(lo, hi).unsqueeze(0)
        g = self.gravities[0] + torch.tensor([0.0, 0.0, -9.8], device=self.device)
        self.cfg.sim.gravity = tuple(float(v) for v in g.detach().cpu())
        try:
            from pxr import Gf, UsdPhysics

            stage = self.sim.stage
            scene_prim = stage.GetPrimAtPath("/physicsScene")
            if not scene_prim.IsValid():
                scene_prim = stage.GetPrimAtPath("/World/physicsScene")
            if scene_prim.IsValid():
                physics_scene = UsdPhysics.Scene(scene_prim)
                gravity = g.detach().cpu().numpy()
                gravity_mag = float(np.linalg.norm(gravity))
                gravity_dir = gravity / gravity_mag if gravity_mag > 0.0 else gravity
                physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(*gravity_dir.tolist())).Set(
                    Gf.Vec3f(*gravity_dir.tolist())
                )
                physics_scene.CreateGravityMagnitudeAttr(gravity_mag).Set(gravity_mag)
        except Exception:
            pass
        # Update gravity_vec for orientation reward.
        self.gravity_vec[:] = (g / g.norm()).unsqueeze(0)

    def _push_robots(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        max_vel = self.cfg.max_push_vel_xy
        root_lin = self.robot.data.root_lin_vel_w[env_ids].clone()
        root_lin[:, :2] = torch.empty(len(env_ids), 2, device=self.device).uniform_(-max_vel, max_vel)
        root_ang = self.robot.data.root_ang_vel_w[env_ids]
        self.robot.write_root_velocity_to_sim(torch.cat([root_lin, root_ang], dim=1), env_ids)

    # ==========================================================================
    # Helpers
    # ==========================================================================

    @staticmethod
    def _scale_shift(value: torch.Tensor, value_range: tuple[float, float]) -> torch.Tensor:
        lo, hi = value_range
        scale = 2.0 / (hi - lo)
        shift = (hi + lo) / 2.0
        return (value - shift) * scale

    def _build_commands_scale(self) -> torch.Tensor:
        s = self.cfg
        scale = [
            s.cmd_scale_lin_vel,  # vx
            s.cmd_scale_lin_vel,  # vy
            s.cmd_scale_ang_vel,  # yaw
            s.cmd_scale_body_height,  # body height
            s.cmd_scale_gait_freq,  # gait freq
            s.cmd_scale_gait_phase,  # gait phase
            s.cmd_scale_gait_phase,  # gait offset
            s.cmd_scale_gait_phase,  # gait bound
            s.cmd_scale_gait_phase,  # gait duration
            s.cmd_scale_footswing_height,
            s.cmd_scale_body_pitch,
            s.cmd_scale_body_roll,
            s.cmd_scale_stance_width,
            s.cmd_scale_stance_length,
            s.cmd_scale_aux_reward,
        ]
        return torch.tensor(scale[: self.cfg.num_commands], dtype=torch.float32, device=self.device)

    def _build_noise_scale_vec(self) -> torch.Tensor:
        s = self.cfg
        nl = s.noise_level
        vec = torch.cat(
            [
                torch.full((3,), s.noise_gravity * nl, device=self.device),
                torch.zeros(s.num_commands, device=self.device),
                torch.full((12,), s.noise_dof_pos * nl * s.obs_dof_pos_scale, device=self.device),
                torch.full((12,), s.noise_dof_vel * nl * s.obs_dof_vel_scale, device=self.device),
                torch.zeros(12, device=self.device),  # actions have no noise
            ]
        )
        if s.observe_two_prev_actions:
            vec = torch.cat([vec, torch.zeros(12, device=self.device)])
        if s.observe_clock_inputs:
            vec = torch.cat([vec, torch.zeros(4, device=self.device)])
        return vec

    def _log_episode(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        ep_info = {}
        for key in self.episode_sums:
            ep_info[f"rew_{key}"] = torch.mean(self.episode_sums[key][env_ids]).item()
            self.episode_sums[key][env_ids] = 0.0

        # Curriculum stats
        if self.cfg.command_curriculum:
            ep_info["command_area"] = float(np.mean([np.sum(c.weights) / c.weights.shape[0] for c in self.curricula]))

        self.extras["episode"] = ep_info
        self.extras["time_outs"] = self.reset_time_outs

    # ==========================================================================
    # Debug visualization
    # ==========================================================================

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_vis_cmd_vel"):
                cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/WTW/cmd_vel"
                cfg.markers["arrow"].scale = (0.7, 0.3, 0.3) # length, thickness, head_thickness
                self._vis_cmd_vel = VisualizationMarkers(cfg)

                cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/WTW/cur_vel"
                cfg.markers["arrow"].scale = (0.7, 0.3, 0.3)
                self._vis_cur_vel = VisualizationMarkers(cfg)

                cfg = FRAME_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/WTW/body_frame"
                cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
                del cfg.markers["connecting_line"]
                self._vis_body_frame = VisualizationMarkers(cfg)

                # Command pose frame: 3 cylinders (red=X, green=Y, blue=Z)
                # CylinderCfg axis="X/Y/Z" sets long axis; all 3 share the same cmd_pose_quat
                _arm, _r = 0.8, 0.012
                self._vis_cmd_pose_x = VisualizationMarkers(VisualizationMarkersCfg(
                    prim_path="/Visuals/WTW/cmd_pose_x",
                    markers={"cyl": sim_utils.CylinderCfg(
                        radius=_r, height=_arm, axis="X",
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
                    )},
                ))
                self._vis_cmd_pose_y = VisualizationMarkers(VisualizationMarkersCfg(
                    prim_path="/Visuals/WTW/cmd_pose_y",
                    markers={"cyl": sim_utils.CylinderCfg(
                        radius=_r, height=_arm, axis="Y",
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1)),
                    )},
                ))
                self._vis_cmd_pose_z = VisualizationMarkers(VisualizationMarkersCfg(
                    prim_path="/Visuals/WTW/cmd_pose_z",
                    markers={"cyl": sim_utils.CylinderCfg(
                        radius=_r, height=_arm, axis="Z",
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 1.0)),
                    )},
                ))

                cfg = SPHERE_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/WTW/height_target"
                cfg.markers["sphere"].radius = 0.04
                cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.0))
                self._vis_height_target = VisualizationMarkers(cfg)

            self._vis_cmd_vel.set_visibility(True)
            self._vis_cur_vel.set_visibility(True)
            self._vis_body_frame.set_visibility(True)
            self._vis_cmd_pose_x.set_visibility(True)
            self._vis_cmd_pose_y.set_visibility(True)
            self._vis_cmd_pose_z.set_visibility(True)
            self._vis_height_target.set_visibility(True)
        else:
            if hasattr(self, "_vis_cmd_vel"):
                self._vis_cmd_vel.set_visibility(False)
                self._vis_cur_vel.set_visibility(False)
                self._vis_body_frame.set_visibility(False)
                self._vis_cmd_pose_x.set_visibility(False)
                self._vis_cmd_pose_y.set_visibility(False)
                self._vis_cmd_pose_z.set_visibility(False)
                self._vis_height_target.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        base_pos = self.robot.data.root_pos_w.clone()  # [N, 3]
        base_quat = self.robot.data.root_quat_w  # [N, 4] wxyz
        _, _, cur_yaw = euler_xyz_from_quat(base_quat)

        zeros = torch.zeros(self.num_envs, device=self.device)

        # Both arrows use the same reference so lengths are directly comparable:
        #   length = speed * VIS_SCALE  (metres per m/s)
        VIS_SCALE = 2.5
        default_scale = self._vis_cmd_vel.cfg.markers["arrow"].scale

        # --- commanded XY velocity arrow (green) ---
        # Commands are body-frame velocities. Rotate by base yaw so the marker is drawn in world frame.
        cmd_vxy_body = self.commands[:, :2]
        yaw_cos = torch.cos(cur_yaw)
        yaw_sin = torch.sin(cur_yaw)
        cmd_vxy = torch.stack(
            [
                yaw_cos * cmd_vxy_body[:, 0] - yaw_sin * cmd_vxy_body[:, 1],
                yaw_sin * cmd_vxy_body[:, 0] + yaw_cos * cmd_vxy_body[:, 1],
            ],
            dim=1,
        )
        cmd_speed = torch.linalg.norm(cmd_vxy_body, dim=1)
        cmd_heading = torch.atan2(cmd_vxy[:, 1], cmd_vxy[:, 0])
        cmd_vel_quat = quat_from_euler_xyz(zeros, zeros, cmd_heading)
        cmd_vel_scale = torch.tensor(default_scale, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        cmd_vel_scale[:, 0] = cmd_speed.clamp(min=0.05) * VIS_SCALE
        arrow_pos = base_pos.clone()
        arrow_pos[:, 2] += 0.55
        self._vis_cmd_vel.visualize(arrow_pos, cmd_vel_quat, cmd_vel_scale)

        # --- current XY velocity arrow (blue) — same position, same scale formula ---
        cur_vxy = self.robot.data.root_lin_vel_w[:, :2]
        cur_speed = torch.linalg.norm(cur_vxy, dim=1)
        cur_heading = torch.atan2(cur_vxy[:, 1], cur_vxy[:, 0])
        cur_vel_quat = quat_from_euler_xyz(zeros, zeros, cur_heading)
        cur_vel_scale = torch.tensor(default_scale, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        cur_vel_scale[:, 0] = cur_speed.clamp(min=0.05) * VIS_SCALE
        self._vis_cur_vel.visualize(arrow_pos, cur_vel_quat, cur_vel_scale)

        # --- body orientation frame (actual roll/pitch) ---
        frame_pos = base_pos.clone()
        frame_pos[:, 2] += 0.05
        self._vis_body_frame.visualize(frame_pos, base_quat)

        # --- commanded pose frame: 3 cylinders at commanded height ---
        # CylinderCfg axis="X/Y/Z" bakes the long-axis direction into the prim;
        # cmd_pose_quat rotates the entire frame in world space — no per-axis offset needed.
        cmd_roll = self.commands[:, 11]
        cmd_pitch = self.commands[:, 10]
        cmd_pose_quat = quat_from_euler_xyz(cmd_roll, cmd_pitch, cur_yaw)
        target_h = self.commands[:, 3] + self.cfg.base_height_target
        cmd_pose_pos = base_pos.clone()
        cmd_pose_pos[:, 2] = target_h + 0.05
        self._vis_cmd_pose_x.visualize(cmd_pose_pos, cmd_pose_quat)
        self._vis_cmd_pose_y.visualize(cmd_pose_pos, cmd_pose_quat)
        self._vis_cmd_pose_z.visualize(cmd_pose_pos, cmd_pose_quat)

        # --- height target sphere (yellow) ---
        height_pos = base_pos.clone()
        height_pos[:, 2] = target_h
        identity_quat = torch.zeros(self.num_envs, 4, device=self.device)
        identity_quat[:, 0] = 1.0
        self._vis_height_target.visualize(height_pos, identity_quat)
