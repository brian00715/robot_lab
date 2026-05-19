# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unitree GO2 + ARX5 — VWBC port (rough terrain)."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm

from robot_lab.tasks.manager_based.wbc.wbc_env_cfg import WbcRoughEnvCfg
from robot_lab.assets.go2_x5 import GO2_X5_CFG
import robot_lab.tasks.manager_based.wbc.mdp as mdp


DOG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


@configclass
class ArxX5WbcRoughEnvCfg(WbcRoughEnvCfg):
    """GO2 + ARX5 rough-terrain env."""

    base_link_name: str = "base"
    foot_link_name: str = ".*_foot"
    ee_body_name: str = "ee"

    def __post_init__(self):
        super().__post_init__()

        # ---- scene ----
        self.scene.robot = GO2_X5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        if self.scene.height_scanner_base is not None:
            self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ---- actions: pin joint patterns + per-joint scales (b1z1: 0.4 hip, 0.45 thigh/calf) ----
        self.actions.joint_pos.dog_joint_names = DOG_JOINT_NAMES
        self.actions.joint_pos.arm_joint_names = ARM_JOINT_NAMES
        self.actions.joint_pos.ee_body_name = self.ee_body_name
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.4,
            ".*_thigh_joint": 0.45,
            ".*_calf_joint": 0.45,
        }

        # ---- events: pin asset_cfg body names ----
        self.events.randomize_friction.params["asset_cfg"] = SceneEntityCfg("robot")
        self.events.randomize_base_mass_and_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=self.base_link_name
        )
        self.events.randomize_base_mass_and_com.params["gripper_body_name"] = self.ee_body_name

        # ---- observations: fix joint ordering ----
        # The simulator groups joints by type (all hips, then all thighs, then
        # calfs), so the default regex-based resolver returns joints in type-
        # grouped order.  The action term uses DOG_JOINT_NAMES (per-leg groups)
        # as its reference.  Passing DOG_JOINT_NAMES as the explicit list here
        # ensures the observation and action spaces share the same joint order.
        self.observations.policy.full.params["dog_joint_pattern"] = list(DOG_JOINT_NAMES)

        # ---- EE goal sphere: recalibrate for GO2 geometry ----
        # The default parameters (sphere_center_x=0.3, z_invariant=0.7) were
        # calibrated for B1 (body height ~0.75 m).  For GO2 (body height 0.33 m)
        # the arm mount is at world z ≈ 0.48 m; the original sphere center at
        # z=0.7 puts all EE goals 1.0–1.4 m from the ARX5 mount (reach ~0.65 m),
        # so IK is always saturated and applies constant maximum torques that
        # immediately flip the 15 kg GO2.  Lower the sphere center to z=0.50
        # and move it back (x=0.0) so EE goals are within the arm's workspace.
        import math as _math
        self.commands.ee_goal.sphere_center_x_offset = 0.0
        self.commands.ee_goal.sphere_center_z_invariant_offset = 0.50
        self.commands.ee_goal.collision_lower_limits = (-0.35, -0.15, -0.45)
        self.commands.ee_goal.collision_upper_limits = (0.35, 0.15, -0.05)
        self.commands.ee_goal.underground_limit = -0.45
        self.commands.ee_goal.ranges.init_pos_start = (0.25, 0.0, 0.0)
        self.commands.ee_goal.ranges.init_pos_end = (0.30, 0.0, 0.0)
        self.commands.ee_goal.ranges.pos_l = (0.15, 0.45)
        self.commands.ee_goal.ranges.pos_p = (-_math.pi / 4.0, _math.pi / 4.0)

        # ---- actions: GO2-safe IK params ----
        # ik_delta_clamp=0.05 (3× smaller than default) to limit arm reaction
        # torque to ≈1.25 Nm/joint on GO2's light base.
        # No arm_warmup_steps: let IK run from step 0 so the arm immediately
        # moves toward the forward EE goal, reducing the arm's left-biased
        # gravity torque that causes systematic left-leg collapse.
        self.actions.joint_pos.ik_delta_clamp = 0.05

        # ---- rewards: GO2-specific overrides ----
        # GO2 stands at ~0.33 m base height (not B1's 0.55 m).
        self.rewards.base_height.params["target_height"] = 0.33
        # Keep the original L1 penalty weight (−5.0/100) — the positive
        # base_height_exp and upright_bonus below provide the standing incentive.
        # Aggressive weight scaling alone caused crouching-local-minimum issues
        # when the termination threshold was also raised.

        # dof_acc: GO2 (15 kg) has much smaller joint inertia than B1 (55 kg),
        # so the same random policy produces ~13× more acceleration penalty than
        # alive reward, driving the policy into the "die immediately" local
        # minimum.  Reduce weight 15× so alive > dof_acc at the random-policy
        # operating point, giving the policy gradient incentive to stay alive.
        self.rewards.dof_acc.weight = -5e-10  # was -7.5e-9 in wbc_env_cfg.py

        # action_rate: increase 3× to suppress rapid leg action oscillations
        # default -0.015 * 0.01 = -1.5e-4; new -0.05 * 0.01 = -5e-4
        self.rewards.action_rate.weight = -0.05 / 100.0

        # dof_vel_leg: penalise leg joint velocities directly.
        # At typical rapid stepping (5 rad/s × 12 joints): sum(vel²)≈300.
        # Weight -1e-3/100 → penalty ≈0.003 per step (30% of alive=0.01),
        # enough to deter high-frequency shuffling without killing locomotion.
        self.rewards.dof_vel_leg = RewTerm(
            func=mdp.dof_vel_leg,
            weight=-1e-3 / 100.0,
        )

        # ---- new: positive standing rewards ----
        # base_height_exp: Gaussian reward (σ=5 cm) centered on target height.
        # At h_err=0 → +2.0/100/step; at h_err=5 cm → +0.74/100/step.
        # This positive gradient dominates any "crouch to avoid penalties"
        # local minimum without requiring overly strict termination thresholds.
        self.rewards.base_height_exp = RewTerm(
            func=mdp.base_height_exp,
            weight=2.0 / 100.0,
            params={"target_height": 0.33, "sigma": 0.05},
        )
        # upright_bonus: binary reward when |roll|<0.2 rad AND |pitch|<0.2 rad.
        # At standing: +1.5/100/step.  Any significant lean → 0.
        # Together with base_height_exp these rewards make proper upright
        # standing worth +3.5/100/step vs alive alone (+1.0/100/step),
        # giving a 3.5× incentive to stand over simply surviving crouched.
        self.rewards.upright_bonus = RewTerm(
            func=mdp.upright_bonus,
            weight=1.5 / 100.0,
            params={"roll_threshold": 0.2, "pitch_threshold": 0.2},
        )

        # ---- rewards: pin contact body names ----
        self.rewards.collision.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[f"^(?!.*{self.foot_link_name}).*_(thigh|calf)$", "base"],
        )
        # b1z1 penalises contacts on thigh/trunk/calf — on GO2+X5 the trunk
        # body is named ``base`` (no separate ``trunk`` link).
