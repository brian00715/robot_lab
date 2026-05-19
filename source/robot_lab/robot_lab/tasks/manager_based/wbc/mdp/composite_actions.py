# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Composite action term ported from visual_wholebody.

The action space matches ``manip_loco`` precisely:

* ``action_dim = 12`` (dog joints only). The arm is **not** policy-controlled.
* The arm follows the **environment-driven** EE goal (see
  :class:`EEGoalSphereCommand`) via damped least-squares Jacobian IK
  (``lambda^2 = 0.05^2``, no per-step delta clamp).
* Per-joint dog action scale ``[0.4, 0.45, 0.45]`` (hip / thigh / calf), exactly
  as in ``b1z1_config.control.action_scale``.
* A ``action_delay = 3`` frame action history buffer is maintained. Before
  ``delay_curriculum_switch_steps`` global env steps the policy reads the
  ``-1``-th history slot (1-step delay); afterwards it switches to ``-2``
  (2-step delay) — exact replica of ``manip_loco.step``.

Per the user's "ignore physical-asset differences" directive we reuse Isaac
Lab's actuator model (Kp/Kd from the asset config) instead of re-implementing
``b1z1_config.control.stiffness/damping``. The motor-strength multiplicative
randomization from VWBC is approximated by scaling the *action targets*
themselves (action × motor_strength), which yields the same effective torque
under PD control.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class VisualWholeBodyAction(ActionTerm):
    """12-D dog action + env-driven arm IK targets.

    On every ``process_actions`` call:
      1. The raw 12-D action enters a circular history buffer of length
         ``action_delay + 2``.
      2. A curriculum-gated delayed action is read out (``-1`` then ``-2``).
      3. ``dog_target = default_pos + scale * delayed_action * motor_strength``.
      4. ``arm_target = dof_pos + IK(dpos, drot)`` using the EE goal published
         by the ``EEGoalSphereCommand`` term.
    """

    cfg: VisualWholeBodyActionCfg

    def __init__(self, cfg: VisualWholeBodyActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._env = env
        self._asset = env.scene[cfg.asset_name]
        self._joint_names = self._asset.joint_names
        self._body_names = self._asset.body_names

        self._dog_joint_ids = self._resolve_joint_ids(cfg.dog_joint_names)
        self._arm_joint_ids = self._resolve_joint_ids(cfg.arm_joint_names)
        if len(self._dog_joint_ids) != 12:
            raise ValueError(f"VisualWholeBodyAction expects 12 dog joints, got {len(self._dog_joint_ids)}")
        if len(self._arm_joint_ids) != 6:
            raise ValueError(f"VisualWholeBodyAction expects 6 arm joints, got {len(self._arm_joint_ids)}")
        self._ee_body_id = self._resolve_body_id(cfg.ee_body_name)

        # per-joint dog action scale tensor (shape: [12])
        self._dog_scale = self._make_dog_scale(cfg.scale)
        self._dog_offset = self._asset.data.default_joint_pos[0, self._dog_joint_ids].clone()

        # action delay buffer: shape (N, action_delay + 2, action_dim)
        self._action_delay = int(cfg.action_delay)
        self._delay_buf_len = self._action_delay + 2
        self._action_history = torch.zeros(
            env.num_envs, self._delay_buf_len, self.action_dim, device=env.device
        )

        # motor strength multiplier (set by event term `randomize_motor_strength`)
        self._motor_strength = torch.ones(env.num_envs, len(self._dog_joint_ids), device=env.device)

        # Cached EE goal command term reference (resolved lazily, after manager init)
        self._ee_goal_command_name = cfg.ee_goal_command_name
        self._ee_command = None
        self._ik_damping = float(cfg.ik_damping)
        self._ik_delta_clamp = float(cfg.ik_delta_clamp)
        self._clip_action = float(cfg.clip_actions)
        self._delay_switch_steps = int(cfg.delay_curriculum_switch_steps)

        self._arm_warmup_steps = int(cfg.arm_warmup_steps)

        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=env.device)
        self._processed_actions = self._asset.data.default_joint_pos.clone()
        self._dog_target_buf = self._dog_offset.unsqueeze(0).repeat(env.num_envs, 1).clone()

    @property
    def action_dim(self) -> int:
        return 12  # dog joints only — arm is env-driven

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def dog_joint_ids(self) -> list[int]:
        return list(self._dog_joint_ids)

    @property
    def arm_joint_ids(self) -> list[int]:
        return list(self._arm_joint_ids)

    @property
    def motor_strength(self) -> torch.Tensor:
        return self._motor_strength

    def set_motor_strength(self, motor_strength: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Set per-env motor strength multipliers (used by event term)."""
        if env_ids is None:
            self._motor_strength[:] = motor_strength
        else:
            self._motor_strength[env_ids] = motor_strength

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long)
        self._action_history[env_ids] = 0.0
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._asset.data.default_joint_pos[env_ids]

    def process_actions(self, actions: torch.Tensor):
        # Clip raw actions (matches `actions = torch.clip(actions, -clip, clip)`)
        clipped = torch.clamp(actions, -self._clip_action, self._clip_action)
        self._raw_actions[:] = clipped

        # Push into history buffer (most recent is index -1)
        self._action_history = torch.cat(
            [self._action_history[:, 1:, :], clipped.unsqueeze(1)], dim=1
        )

        # Curriculum-gated read-out (matches manip_loco.step lines 77-80)
        global_steps = int(getattr(self._env, "common_step_counter", 0))
        delay_idx = -1 if global_steps < self._delay_switch_steps else -2
        delayed = self._action_history[:, delay_idx, :]

        # Per-joint dog target: scale * delayed * motor_strength + default_pos
        dog_targets = self._dog_offset + self._dog_scale * delayed * self._motor_strength
        self._dog_target_buf[:] = dog_targets

        # Arm IK targets from env-driven EE goal
        # During warmup, hold arm at current joint positions (no IK motion).
        # This lets the dog learn to stand before the arm begins moving.
        global_steps_arm = int(getattr(self._env, "common_step_counter", 0))
        if global_steps_arm < self._arm_warmup_steps:
            arm_targets = self._asset.data.joint_pos[:, self._arm_joint_ids].clone()
        else:
            arm_targets = self._compute_arm_targets()

        # Build full target vector
        full_targets = self._asset.data.default_joint_pos.clone()
        full_targets[:, self._dog_joint_ids] = dog_targets
        full_targets[:, self._arm_joint_ids] = arm_targets
        self._processed_actions = full_targets

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions)

    # ----- arm IK ---------------------------------------------------------

    def _compute_arm_targets(self) -> torch.Tensor:
        if self._ee_command is None:
            self._ee_command = self._env.command_manager.get_term(self._ee_goal_command_name)

        # Goal pose in world frame (published by the EE goal command term)
        ee_goal_pos_w = self._ee_command.curr_ee_goal_cart_world
        ee_goal_quat_w = self._ee_command.ee_goal_orn_quat

        # Current EE pose in world frame
        ee_pos_w = self._asset.data.body_pos_w[:, self._ee_body_id, :]
        ee_quat_w = self._asset.data.body_quat_w[:, self._ee_body_id, :]
        # Normalize current quat (parity with manip_loco._reward_tracking_ee_world neighbourhood)
        ee_quat_w = ee_quat_w / (ee_quat_w.norm(dim=-1, keepdim=True) + 1e-8)

        pos_error, rot_error = math_utils.compute_pose_error(
            ee_pos_w, ee_quat_w, ee_goal_pos_w, ee_goal_quat_w, rot_error_type="axis_angle"
        )
        delta_pose = torch.cat((pos_error, rot_error), dim=-1).unsqueeze(-1)  # (N, 6, 1)

        jacobian = self._get_arm_jacobian()  # (N, 6, 6)
        jt = torch.transpose(jacobian, 1, 2)
        damp = (self._ik_damping ** 2) * torch.eye(6, device=self._env.device)
        A = jacobian @ jt + damp.unsqueeze(0)
        delta_q = jt @ torch.linalg.solve(A, delta_pose)  # (N, 6, 1)
        delta_q = delta_q.squeeze(-1)
        # Clamp per-step delta to prevent large arm swings that destabilise the
        # light GO2 base (B1 was 4x heavier and tolerated unclamped IK).
        delta_q = delta_q.clamp(-self._ik_delta_clamp, self._ik_delta_clamp)

        arm_pos = self._asset.data.joint_pos[:, self._arm_joint_ids]
        arm_targets = arm_pos + delta_q
        # Clip to joint limits to avoid solver drift outside the asset limits
        arm_limits = self._asset.data.soft_joint_pos_limits[:, self._arm_joint_ids]
        return torch.max(torch.min(arm_targets, arm_limits[..., 1]), arm_limits[..., 0])

    def _get_arm_jacobian(self) -> torch.Tensor:
        jacobians = self._asset.root_physx_view.get_jacobians()
        body_index = self._ee_body_id
        if jacobians.shape[1] == len(self._body_names) - 1:
            body_index -= 1
        if body_index < 0:
            raise RuntimeError("End-effector body cannot be the articulation root for IK.")

        dof_offset = 6 if jacobians.shape[-1] == self._asset.num_joints + 6 else 0
        joint_columns = torch.as_tensor(self._arm_joint_ids, device=self._env.device, dtype=torch.long) + dof_offset
        return jacobians[:, body_index, :6, :].index_select(-1, joint_columns)

    # ----- name resolution / scale construction ---------------------------

    def _resolve_joint_ids(self, joint_names) -> list[int]:
        if isinstance(joint_names, str):
            joint_names = [joint_names]
        out: list[int] = []
        for pattern in joint_names:
            matches = [i for i, name in enumerate(self._joint_names) if re.fullmatch(pattern, name)]
            if not matches and pattern in self._joint_names:
                matches = [self._joint_names.index(pattern)]
            if not matches:
                raise ValueError(f"Joint pattern '{pattern}' did not match any robot joint.")
            out.extend(matches)
        return out

    def _resolve_body_id(self, body_name: str) -> int:
        if body_name in self._body_names:
            return self._body_names.index(body_name)
        matches = [i for i, name in enumerate(self._body_names) if re.fullmatch(body_name, name)]
        if not matches:
            raise ValueError(f"Body '{body_name}' did not match any robot body.")
        return matches[0]

    def _make_dog_scale(self, scale_cfg) -> torch.Tensor:
        names = [self._joint_names[i] for i in self._dog_joint_ids]
        if isinstance(scale_cfg, dict):
            values = []
            for jn in names:
                v = 0.45  # default to thigh/calf
                for pat, val in scale_cfg.items():
                    if re.fullmatch(pat, jn) or re.match(pat, jn):
                        v = val
                        break
                values.append(v)
            return torch.tensor(values, device=self._env.device, dtype=torch.float32)
        return torch.full((len(self._dog_joint_ids),), float(scale_cfg), device=self._env.device)


@configclass
class VisualWholeBodyActionCfg(ActionTermCfg):
    """Config for VWBC dog-only joint actions + env-driven arm IK."""

    class_type: type[ActionTerm] = VisualWholeBodyAction

    asset_name: str = "robot"
    dog_joint_names: list[str] | tuple[str, ...] | str = ()
    arm_joint_names: list[str] | tuple[str, ...] | str = ()
    ee_body_name: str = "ee"
    ee_goal_command_name: str = "ee_goal"

    # Per-joint dog scale: hip 0.4, thigh/calf 0.45 — matches b1z1_config.action_scale.
    scale: float | dict[str, float] = None  # set in __post_init__ of env

    ik_damping: float = 0.05
    # Max arm joint change per policy step (rad).  B1 could tolerate unclamped
    # IK because it is ~4x heavier than GO2; the clamp prevents the arm from
    # jolting the light base on the first step of every episode.
    ik_delta_clamp: float = 0.05
    # Global env steps before arm IK activates.  Set to a non-zero value to
    # let the dog policy learn to stand before the arm begins moving.
    arm_warmup_steps: int = 0
    clip_actions: float = 100.0

    # Action delay buffer length (3 in VWBC: action_history_buf carries 5 = 3+2 slots).
    action_delay: int = 3

    # Global step at which the policy switches from reading [-1] to [-2] of the buffer.
    # Equivalent to the `if self.global_steps < 10000 * 24` switch in manip_loco.step.
    delay_curriculum_switch_steps: int = 10000 * 24
