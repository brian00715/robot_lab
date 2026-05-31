# SPDX-License-Identifier: Apache-2.0
"""Go2 + ARX5 arm Walk-These-Ways environment with arm disturbance curriculum.

This environment extends Go2WalkTheseWaysEnv with:
- A 6-DOF ARX5 arm that follows predefined trajectories (not policy-controlled)
- 5-stage curriculum that gradually increases arm motion intensity
- Extended observations (82D per step) including arm joint states
- Dog policy still outputs 12D actions; arm is an autonomous disturbance source
"""

from __future__ import annotations

import torch

from .go2_wtw_env import Go2WalkTheseWaysEnv
from .go2_wtw_x5_env_cfg import Go2X5WalkTheseWaysEnvCfg
from .arm_controller import create_arm_controller


class Go2X5WalkTheseWaysEnv(Go2WalkTheseWaysEnv):
    """Go2 + ARX5 Walk-These-Ways locomotion environment with arm disturbance.

    The arm follows predefined trajectories to create diverse perturbations.
    The dog policy must learn to maintain stable locomotion despite these
    disturbances. A 5-stage curriculum gradually increases arm motion intensity.

    Architecture:
        - Dog policy:  12D action output → dog joint position targets
        - Arm control: ARX5TrajectoryController → arm joint position targets
        - Observations: 82D = 70D (WTW base) + 6D arm pos + 6D arm vel
        - obs_history:  82 × 30 = 2460D (for RMA adaptation module)
    """

    cfg: Go2X5WalkTheseWaysEnvCfg

    def __init__(self, cfg: Go2X5WalkTheseWaysEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- arm joint indices --------------------------------------------------
        self.arm_joint_indices, self.arm_joint_names_resolved = self.robot.find_joints(
            list(self.cfg.arm_joint_names), preserve_order=True
        )
        if len(self.arm_joint_indices) != len(self.cfg.arm_joint_names):
            raise RuntimeError(
                f"Expected {len(self.cfg.arm_joint_names)} arm joints, "
                f"got {len(self.arm_joint_indices)}: {self.arm_joint_names_resolved}"
            )
        self.num_arm_dof = len(self.arm_joint_indices)

        # ---- arm state buffers --------------------------------------------------
        self.arm_dof_pos = torch.zeros(self.num_envs, self.num_arm_dof, device=self.device)
        self.arm_dof_vel = torch.zeros(self.num_envs, self.num_arm_dof, device=self.device)
        # Default targets: all zeros (arm extends backward), matches original implementation
        self.arm_joint_targets = torch.zeros(self.num_envs, len(self.cfg.arm_joint_names), device=self.device)

        # ---- arm curriculum state -----------------------------------------------
        self._arm_stage: int = self.cfg.arm_curriculum_initial_stage - 1   # 0-indexed
        self._arm_motion_scale: float = self.cfg.arm_stage_motion_scales[self._arm_stage]
        self._arm_stage_reward_buf: list[float] = []
        print(
            f"[Go2X5WTW] Initialized arm curriculum at stage {self._arm_stage + 1} "
            f"(motion_scale={self._arm_motion_scale})"
        )

        # ---- arm controller -----------------------------------------------------
        self._arm_controller = create_arm_controller(
            num_envs=self.num_envs,
            device=self.device,
            stage=self.cfg.arm_curriculum_initial_stage,
        )
        print(f"[Go2X5WTW] ARX5TrajectoryController ready ({self.num_arm_dof} arm joints)")

        # ---- arm noise scale vector (appended to base noise_scale_vec) ----------
        # noise_scale_vec from parent covers the first 70D; we extend for arm dims
        noise_arm_pos = self.cfg.noise_arm_joint_pos * self.cfg.noise_level
        noise_arm_vel = self.cfg.noise_arm_joint_vel * self.cfg.noise_level
        arm_noise = torch.cat([
            torch.full((self.num_arm_dof,), noise_arm_pos, device=self.device),
            torch.full((self.num_arm_dof,), noise_arm_vel, device=self.device),
        ])  # 12D
        # Extend parent noise_scale_vec (70D) → 82D
        self.noise_scale_vec = torch.cat([self.noise_scale_vec, arm_noise], dim=-1)

        # ---- episode reward buffer for arm curriculum tracking ------------------
        self._ep_total_reward = torch.zeros(self.num_envs, device=self.device)

    # ==========================================================================
    # Action application (override to add arm control)
    # ==========================================================================

    def _apply_action(self):
        """Apply dog policy actions (12D) + arm trajectory actions (6D)."""
        # ---- Dog joints (identical to parent) -----------------------------------
        actions_scaled = self.actions * self.cfg.action_scale
        actions_scaled[:, self.hip_joint_indices] *= self.cfg.hip_scale_reduction

        if self.cfg.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        else:
            self.joint_pos_target = actions_scaled + self.default_dof_pos

        dog_target = self.joint_pos_target + self.motor_offsets
        self.robot.set_joint_position_target(dog_target, joint_ids=self.joint_indices)

        # ---- Arm joints (ARX5TrajectoryController) ------------------------------
        arm_targets = self._arm_controller.generate_arm_action(self)
        self.arm_joint_targets = arm_targets.clone()
        self.robot.set_joint_position_target(arm_targets, joint_ids=self.arm_joint_indices)

    # ==========================================================================
    # Observations (override to include arm state)
    # ==========================================================================

    def _get_observations(self) -> dict:
        # ---- Refresh arm joint state from physics -------------------------------
        self.arm_dof_pos = self.robot.data.joint_pos[:, self.arm_joint_indices]
        self.arm_dof_vel = self.robot.data.joint_vel[:, self.arm_joint_indices]

        # ---- Base 70D WTW obs (gravity, cmd, dof_pos/vel, actions, clock) ------
        obs_base = torch.cat(
            [
                self.projected_gravity,                                         # 3
                self.commands * self.commands_scale,                            # 15
                (self.dof_pos - self.default_dof_pos) * self.cfg.obs_dof_pos_scale,  # 12
                self.dof_vel * self.cfg.obs_dof_vel_scale,                      # 12
                self.actions,                                                   # 12
            ],
            dim=-1,
        )

        if self.cfg.observe_two_prev_actions:
            obs_base = torch.cat([obs_base, self.last_actions], dim=-1)         # +12

        if self.cfg.observe_clock_inputs:
            obs_base = torch.cat([obs_base, self.clock_inputs], dim=-1)         # +4

        # ---- Arm state (12D) -----------------------------------------------
        # arm_joint_pos(6) + arm_joint_vel(6)
        arm_obs = torch.cat(
            [
                self.arm_dof_pos,                                               # 6
                self.arm_dof_vel * 0.05,                                        # 6 (same vel scale as dog)
            ],
            dim=-1,
        )

        # ---- Combine: 82D total ------------------------------------------------
        obs = torch.cat([obs_base, arm_obs], dim=-1)

        # ---- Noise + clip -------------------------------------------------------
        if self.cfg.add_noise:
            obs = obs + (2.0 * torch.rand_like(obs) - 1.0) * self.noise_scale_vec

        obs = torch.clip(obs, -self.cfg.clip_observations, self.cfg.clip_observations)

        # ---- Rolling history (obs_history: 82 × 30 = 2460D) --------------------
        self.obs_history = torch.cat(
            [self.obs_history[:, self.cfg.num_scalar_observations:], obs], dim=-1
        )

        # ---- Privileged obs (friction + restitution, 2D) -------------------------
        priv_obs = torch.cat(
            [
                self._scale_shift(self.friction_coeffs, self.cfg.friction_obs_range),
                self._scale_shift(self.restitutions, self.cfg.restitution_obs_range),
            ],
            dim=-1,
        )

        # ---- Advance action buffers (same timing as parent) ---------------------
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.foot_velocities = self.robot.data.body_lin_vel_w[:, self.robot_feet_indices, :]

        return {"policy": obs, "obs_history": self.obs_history, "privileged": priv_obs}

    # ==========================================================================
    # Reset (override to also reset arm controller)
    # ==========================================================================

    def _reset_idx(self, env_ids: torch.Tensor):
        # ----- Arm curriculum: advance based on just-finished episode rewards ----
        if len(env_ids) > 0:
            mean_rew = float(self._ep_total_reward[env_ids].mean().item())
            self._step_arm_curriculum(mean_rew)
            self._ep_total_reward[env_ids] = 0.0

        super()._reset_idx(env_ids)
        # Reset arm controller trajectories for reset environments
        self._arm_controller.reset_idx(env_ids)

        # ---- Write arm joint state to simulation (critical!) --------------------
        # The parent _reset_idx only resets the 12 dog joints via write_joint_state_to_sim.
        # Arm joints retain their end-of-episode simulation state unless explicitly reset.
        # Without this, the arm snaps from a random position to home_pose on the first
        # step of each episode, creating large impulse forces that destabilize the dog.
        arm_home = self._arm_controller.home_pose.unsqueeze(0).expand(len(env_ids), -1).clone()
        arm_zero_vel = torch.zeros(len(env_ids), self.num_arm_dof, device=self.device)
        self.robot.write_joint_state_to_sim(arm_home, arm_zero_vel, self.arm_joint_indices, env_ids)

        # Sync internal buffers with the now-reset sim state
        self.arm_dof_pos[env_ids] = arm_home
        self.arm_dof_vel[env_ids] = 0.0
        self.arm_joint_targets[env_ids] = self._arm_controller.home_pose.clone()
        # Inject arm curriculum info so RSL-RL logs it each iteration
        if "episode" in self.extras:
            self.extras["episode"]["arm_stage"] = float(self._arm_stage + 1)
            self.extras["episode"]["arm_motion_scale"] = float(self._arm_motion_scale)

    # ==========================================================================
    # Reward accumulation for automatic arm curriculum advancement
    # ==========================================================================

    def _get_rewards(self) -> torch.Tensor:
        rewards = super()._get_rewards()
        self._ep_total_reward.add_(rewards)
        return rewards

    def _resample_commands_if_due(self):
        """Resample gait commands and, if enabled via cfg flag, arm mode/stage too."""
        super()._resample_commands_if_due()

        if not getattr(self.cfg, "play_resample_arm", False):
            return

        import math as _math
        step_dt = self.step_dt
        sample_interval = max(1, round(self.cfg.resampling_time / step_dt))
        # Only act on the very first env's counter as a proxy (all envs share same interval)
        if self.episode_length_buf[0].item() % sample_interval != 0:
            return

        # Randomly pick a new arm motion mode for every env
        modes = [
            "circular", "figure_eight", "sinusoidal", "random_walk", "reach_points",
            "fishing", "grasping", "swinging", "probing",
        ]
        import random
        self._arm_controller.motion_modes = [random.choice(modes) for _ in range(self.num_envs)]
        # Re-randomize per-env trajectory parameters so motion looks fresh
        self._arm_controller.frequencies = (
            torch.rand(self.num_envs, device=self.device) * 1.5 + 0.3
        )
        self._arm_controller.amplitudes = (
            torch.rand(self.num_envs, device=self.device) * 0.4 + 0.1
        )
        self._arm_controller.phase_offsets = (
            torch.rand(self.num_envs, device=self.device) * 2 * _math.pi
        )
        self._arm_controller.timesteps.zero_()

        # Randomly pick an arm stage (1 … max_stage, skip stage-0 which is locked)
        max_stage = len(self.cfg.arm_stage_motion_scales) - 1
        self._arm_stage = random.randint(1, max_stage)
        self._arm_motion_scale = self.cfg.arm_stage_motion_scales[self._arm_stage]

    # ==========================================================================
    # Arm curriculum (auto-driven from _reset_idx via episode reward tracking)
    # ==========================================================================

    def _step_arm_curriculum(self, mean_episode_reward: float | None = None):
        """Advance arm curriculum stage when performance threshold is met.

        Call this from outside (e.g., training runner) or from _periodic_updates.
        Alternatively pass ``mean_episode_reward`` directly.

        The curriculum manages ``env._arm_motion_scale`` which is read by
        ``ARX5TrajectoryController.generate_arm_action()``.
        """
        max_stage = len(self.cfg.arm_stage_motion_scales) - 1
        if self._arm_stage >= max_stage:
            return  # Already at max stage

        if mean_episode_reward is not None:
            self._arm_stage_reward_buf.append(mean_episode_reward)

        if len(self._arm_stage_reward_buf) < self.cfg.arm_stage_eval_window:
            return

        mean_rew = sum(self._arm_stage_reward_buf[-self.cfg.arm_stage_eval_window:]) / self.cfg.arm_stage_eval_window
        if mean_rew >= self.cfg.arm_stage_advance_threshold:
            self._arm_stage = min(self._arm_stage + 1, max_stage)
            self._arm_motion_scale = self.cfg.arm_stage_motion_scales[self._arm_stage]
            self._arm_stage_reward_buf.clear()
            print(
                f"[Go2X5WTW] Arm curriculum advanced to stage {self._arm_stage + 1} "
                f"(motion_scale={self._arm_motion_scale:.2f}, triggered by mean_rew={mean_rew:.3f})"
            )

    def get_arm_curriculum_info(self) -> dict:
        """Return current arm curriculum state for logging."""
        return {
            "arm_stage": self._arm_stage + 1,
            "arm_motion_scale": self._arm_motion_scale,
            "arm_stage_reward_buf_len": len(self._arm_stage_reward_buf),
        }
