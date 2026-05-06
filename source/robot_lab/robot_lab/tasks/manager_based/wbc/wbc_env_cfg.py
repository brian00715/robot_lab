# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Top-level env configuration for the visual_wholebody port.

This file wires the new VWBC-aligned MDP terms into a manager-based env. The
robot-specific config (joint names, defaults, spawn) lives in
``config/<robot>/rough_env_cfg.py``.

Key choices:

* Two command terms — ``base_velocity`` (3-D, ``vy=0``, 3 s resample, with
  threshold clip + global-step positive-only gate) and ``ee_goal``
  (env-driven spherical EE goal trajectory).
* One observation term — :func:`mdp.vwbc_full_observation` produces the full
  ``[proprio, priv, history(10)]`` flat vector exactly matching VWBC.
* One action term — :class:`mdp.VisualWholeBodyActionCfg` (12-D dog actions +
  arm IK driven by the env's EE goal command, with action_delay=3 + curriculum).
* Reward set — full b1z1 leg + ``tracking_ee_world`` arm reward, all weights
  multiplied by ``1/100`` to replicate VWBC's ``rew_buf /= 100``.
* Terminations — bad orientation (|roll| or |pitch| > 0.8) + low base height
  (z < 0.1) + timeout. ``illegal_contact`` is left disabled (b1z1 also
  disables it).
* Episode length — 10 s (matches VWBC).
"""

from __future__ import annotations

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm  # noqa: F401  (kept for types)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)
import robot_lab.tasks.manager_based.wbc.mdp as mdp


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@configclass
class WbcCommandsCfg:
    """Two VWBC commands: base velocity (3-D) + EE goal (sphere)."""

    base_velocity = mdp.VWBCVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 3.0),
        lin_vel_x_clip=0.2,
        ang_vel_yaw_clip=0.5,
        positive_only_until_steps=5000 * 24,
        ranges=mdp.VWBCVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 0.8),
            ang_vel_z=(-1.0, 1.0),
        ),
    )

    ee_goal = mdp.EEGoalSphereCommandCfg(
        asset_name="robot",
        traj_time=(1.0, 3.0),
        hold_time=(0.5, 2.0),
        collision_lower_limits=(-0.8, -0.2, -0.7),
        collision_upper_limits=(0.1, 0.2, -0.05),
        underground_limit=-0.7,
        num_collision_check_samples=10,
        sphere_center_x_offset=0.3,
        sphere_center_y_offset=0.0,
        sphere_center_z_invariant_offset=0.7,
        arm_induced_pitch=0.38,
        ranges=mdp.EEGoalSphereCommandCfg.Ranges(
            init_pos_start=(0.5, math.pi / 8.0, 0.0),
            init_pos_end=(0.7, 0.0, 0.0),
            pos_l=(0.4, 0.95),
            pos_p=(-math.pi / 2.5, math.pi / 3.0),
            pos_y=(-1.2, 1.2),
            delta_orn_r=(-0.5, 0.5),
            delta_orn_p=(-0.5, 0.5),
            delta_orn_y=(-0.5, 0.5),
        ),
    )


# ---------------------------------------------------------------------------
# Actions  (action_dim = 12 — robot-specific defaults set in subclass)
# ---------------------------------------------------------------------------


@configclass
class WbcActionsCfg:
    """Single composite action term. Robot-specific joint names set in subclass."""

    joint_pos = mdp.VisualWholeBodyActionCfg(
        asset_name="robot",
        # populated by config subclasses:
        dog_joint_names=(),
        arm_joint_names=(),
        ee_body_name="ee",
        ee_goal_command_name="ee_goal",
        scale={".*_hip_joint": 0.4, "^(?!.*_hip_joint).*_(thigh|calf)_joint": 0.45},
        ik_damping=0.05,
        clip_actions=100.0,
        action_delay=3,
        delay_curriculum_switch_steps=10000 * 24,
    )


# ---------------------------------------------------------------------------
# Observations  (single composite term)
# ---------------------------------------------------------------------------


@configclass
class WbcObservationsCfg:
    """One policy obs group with a single composite VWBC observation term."""

    @configclass
    class PolicyCfg(ObsGroup):
        full = ObsTerm(
            func=mdp.vwbc_full_observation,
            params={
                "command_name": "base_velocity",
                "ee_goal_command_name": "ee_goal",
                "contact_sensor_name": "contact_forces",
                "foot_body_pattern": ".*_foot",
                "action_name": "joint_pos",
                "dog_joint_pattern": "(FL|FR|RL|RR)_(hip|thigh|calf)_joint",
                "arm_joint_pattern": "joint[1-6]",
                "history_len": 10,
                "contact_threshold": 1.5,
                "obs_scale_ang_vel": 1.0,
                "obs_scale_dof_pos": 1.0,
                "obs_scale_dof_vel": 0.05,
                "obs_scale_lin_vel": 1.0,
                "add_noise": False,  # b1z1_config: noise.add_noise = False
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Rewards (b1z1 leg set + tracking_ee_world; all weights /= 100)
# ---------------------------------------------------------------------------


_VWBC_REW_NORM = 1.0 / 100.0  # parity with manip_loco.compute_reward 'rew_buf /= 100'


@configclass
class WbcRewardsCfg:
    """Reward terms following ``b1z1_config.rewards.scales`` and ``arm_scales``."""

    # ---- velocity tracking ----
    tracking_lin_vel_max = RewTerm(
        func=mdp.tracking_lin_vel_max,
        weight=2.0 * _VWBC_REW_NORM,
        params={"command_name": "base_velocity", "lin_vel_x_clip": 0.2},
    )
    tracking_ang_vel_yaw = RewTerm(
        func=mdp.tracking_ang_vel_yaw,
        weight=0.5 * _VWBC_REW_NORM,
        params={"command_name": "base_velocity", "sigma": 0.2},
    )

    # ---- stability ----
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_square, weight=-1.5 * _VWBC_REW_NORM)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_square, weight=-0.2 * _VWBC_REW_NORM)
    roll = RewTerm(func=mdp.roll_abs, weight=-2.0 * _VWBC_REW_NORM)
    base_height = RewTerm(
        func=mdp.base_height_l1,
        weight=-5.0 * _VWBC_REW_NORM,
        params={"target_height": 0.55},
    )

    # ---- joint penalties (leg-only where applicable) ----
    torques = RewTerm(func=mdp.torques_l2_full, weight=-2.5e-5 * _VWBC_REW_NORM)
    dof_acc = RewTerm(func=mdp.dof_acc_leg, weight=-7.5e-7 * _VWBC_REW_NORM)
    delta_torques = RewTerm(func=mdp.delta_torques_leg, weight=-1.0e-7 * _VWBC_REW_NORM)
    action_rate = RewTerm(func=mdp.action_rate_leg, weight=-0.015 * _VWBC_REW_NORM)
    dof_pos_limits = RewTerm(func=mdp.dof_pos_limits_leg, weight=-10.0 * _VWBC_REW_NORM)
    hip_pos = RewTerm(func=mdp.hip_pos_l2, weight=-0.3 * _VWBC_REW_NORM)
    work = RewTerm(func=mdp.work_leg, weight=-0.003 * _VWBC_REW_NORM)

    # ---- stand still / walking-conditioned ----
    stand_still = RewTerm(
        func=mdp.stand_still_exp,
        weight=1.0 * _VWBC_REW_NORM,
        params={"command_name": "base_velocity", "lin_vel_x_clip": 0.2, "ang_vel_yaw_clip": 0.5},
    )
    walking_dof = RewTerm(
        func=mdp.walking_dof_exp,
        weight=1.5 * _VWBC_REW_NORM,
        params={"command_name": "base_velocity", "lin_vel_x_clip": 0.2, "ang_vel_yaw_clip": 0.5},
    )
    alive = RewTerm(func=mdp.alive, weight=1.0 * _VWBC_REW_NORM)

    # ---- contact / feet ----
    collision = RewTerm(
        func=mdp.collision,
        weight=-10.0 * _VWBC_REW_NORM,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=()),
            "threshold": 0.1,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=2.0 * _VWBC_REW_NORM,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.5,
            "lin_vel_clip": 0.2,
            "ang_vel_clip": 0.5,
        },
    )
    feet_height = RewTerm(
        func=mdp.feet_height_l2,
        weight=1.0 * _VWBC_REW_NORM,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_norm": 0.3,
            "lin_vel_clip": 0.2,
            "ang_vel_clip": 0.5,
        },
    )
    feet_contact_forces = RewTerm(
        func=mdp.feet_contact_forces,
        weight=-0.001 * _VWBC_REW_NORM,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "max_force": 40.0,
            "warmup_seconds": 2.0,
        },
    )
    feet_drag = RewTerm(
        func=mdp.feet_drag,
        weight=-0.08 * _VWBC_REW_NORM,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "contact_threshold": 1.5,
        },
    )
    feet_jerk = RewTerm(
        func=mdp.feet_jerk,
        weight=-2.0e-4 * _VWBC_REW_NORM,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )

    # ---- EE tracking (arm reward) ----
    tracking_ee_world = RewTerm(
        func=mdp.tracking_ee_world,
        weight=0.8 * _VWBC_REW_NORM,
        params={"ee_goal_command_name": "ee_goal", "sigma": 1.0, "ee_body_name": "ee"},
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class WbcTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.base_roll_pitch_too_large,
        params={"roll_threshold": 0.8, "pitch_threshold": 0.8},
    )
    base_height_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.1},
    )


# ---------------------------------------------------------------------------
# Curriculum (empty — VWBC has no manager-driven curricula)
# ---------------------------------------------------------------------------


@configclass
class WbcCurriculumCfg:
    pass


# ---------------------------------------------------------------------------
# Events (domain randomization)
# ---------------------------------------------------------------------------


@configclass
class WbcEventCfg:
    """VWBC-aligned event set.

    Ranges follow ``b1z1_config.domain_rand``: friction [0.3, 3.0]; base mass
    add [0, 15]; COM ±0.15 each axis; gripper mass [0, 0.1]; motor strength
    [0.7, 1.3]; reset xy ±0.5 m, yaw ±π/2; reset velocity ±0.1; reset DOF
    ``× [0.8, 1.2]``; push every 8 s, max ±0.5 m/s, with 2.5× boost when
    velocity command is zero.
    """

    # ---- startup ----
    randomize_friction = EventTerm(
        func=mdp.randomize_friction_record,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_range": (0.3, 3.0),
        },
    )

    randomize_base_mass_and_com = EventTerm(
        func=mdp.randomize_base_mass_and_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "base_mass_add_range": (0.0, 15.0),
            "base_com_range": (0.15, 0.15, 0.15),
            "gripper_mass_add_range": (0.0, 0.1),
            "gripper_body_name": "ee",
        },
    )

    randomize_motor_strength = EventTerm(
        func=mdp.randomize_motor_strength,
        mode="startup",
        params={
            "leg_motor_strength_range": (0.7, 1.3),
            "action_term_name": "joint_pos",
        },
    )

    # ---- reset ----
    reset_root_state = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi / 2, math.pi / 2),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    # ---- interval (push every ~8 s, with zero-cmd boost) ----
    push_robot = EventTerm(
        func=mdp.push_robot_zero_cmd_boost,
        mode="interval",
        interval_range_s=(8.0, 8.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            "zero_cmd_boost": 2.5,
            "command_name": "base_velocity",
        },
    )


# ---------------------------------------------------------------------------
# Top-level env config
# ---------------------------------------------------------------------------


@configclass
class WbcRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Top-level VWBC env. Robot-specific subclasses override scene.robot etc."""

    commands: WbcCommandsCfg = WbcCommandsCfg()
    actions: WbcActionsCfg = WbcActionsCfg()
    observations: WbcObservationsCfg = WbcObservationsCfg()
    rewards: WbcRewardsCfg = WbcRewardsCfg()
    terminations: WbcTerminationsCfg = WbcTerminationsCfg()
    events: WbcEventCfg = WbcEventCfg()
    curriculum: WbcCurriculumCfg = WbcCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Episode length — VWBC uses 10 s (b1z1_config.env.episode_length_s = 10).
        self.episode_length_s = 10.0
