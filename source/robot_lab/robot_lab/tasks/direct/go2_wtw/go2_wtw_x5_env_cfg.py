# SPDX-License-Identifier: Apache-2.0
"""Config for Go2 + ARX5 arm Walk-These-Ways environment (Direct RL, arm disturbance training)."""

from __future__ import annotations

from isaaclab.utils import configclass

from .go2_wtw_env_cfg import Go2WalkTheseWaysEnvCfg
from robot_lab.assets.go2_x5_wtw import GO2_X5_WTW_CFG


# ---- arm joint names (ARX5, 6 DOF) -----------------------------------------------
ARM_JOINT_NAMES: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)

# ---- dog joint names (matches parent cfg) ----------------------------------------
DOG_JOINT_NAMES: tuple[str, ...] = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)


@configclass
class Go2X5WalkTheseWaysEnvCfg(Go2WalkTheseWaysEnvCfg):
    """Config for Go2 + ARX5 Walk-These-Ways env with arm disturbance curriculum.

    Extends the base WTW config with:
    - Go2+X5 combined asset (18+ DOF: 12 dog + 6 arm + 2 gripper)
    - Extended observation space (adds arm joint pos/vel + EE position)
    - Policy still outputs 12D (dog joints only)
    - Arm follows ARX5TrajectoryController (predefined trajectories)
    - 5-stage curriculum that gradually increases arm motion intensity

    Observation layout (82D per timestep):
        gravity(3) + cmd(15) + dof_pos(12) + dof_vel(12) + actions(12) +
        last_actions(12) + clock(4) + arm_joint_pos(6) + arm_joint_vel(6) = 82D

    obs_history shape: 82 × 30 (history_length) = 2460D
    privileged_obs: 2D (friction, restitution) – unchanged
    """

    # ---- robot asset: Go2 + ARX5 ------------------------------------------------
    robot = GO2_X5_WTW_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ---- joint names -------------------------------------------------------------
    joint_names: tuple[str, ...] = DOG_JOINT_NAMES   # 12D policy output (dog only)
    arm_joint_names: tuple[str, ...] = ARM_JOINT_NAMES  # 6D arm (trajectory-controlled)

    # ---- observation / action spaces -------------------------------------------
    # Base obs: 82D (70 + 12 arm joints)
    observation_space: int = 82
    num_scalar_observations: int = 82
    # obs_history: 82 × 30 = 2460D
    num_observation_history: int = 30
    num_privileged_obs: int = 2
    # Policy outputs 12D for dog joints only
    action_space: int = 12
    state_space: int = 2

    # ---- arm curriculum stages --------------------------------------------------
    # Stage 1: arm static (scale 0.0) – just learn to walk stably
    # Stage 2: light arm motion (scale 1.2)
    # Stage 3: medium arm motion (scale 1.8)
    # Stage 4: strong arm motion (scale 2.5)
    # Stage 5: full arm motion (scale 3.75)
    arm_curriculum_initial_stage: int = 1
    arm_stage_motion_scales: tuple[float, ...] = (0.0, 1.2, 1.8, 2.5, 3.75)
    # Episode reward threshold to advance to next arm stage.
    # Episode total reward (sum over ~1000 steps) for a good walking policy is ~60+.
    # Crawling policy gets ~40. Set threshold to 50 to require proper locomotion.
    arm_stage_advance_threshold: float = 50.0
    # How many curriculum steps to average reward over before evaluating threshold
    arm_stage_eval_window: int = 200

    # ---- arm observation noise -------------------------------------------------
    noise_arm_joint_pos: float = 0.01
    noise_arm_joint_vel: float = 0.5
