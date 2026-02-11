# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Configuration for Unitree GO2 + ARX5 Arm - Stage 1 Training

This configuration implements the Stage 1 training plan from the technical document:
- Robust locomotion tracking under arm disturbance
- 76D observation space (dog + arm states)
- 18D action space (12 dog joints + 6 arm joints)
- Policy outputs 12D (dog only), arm follows predefined trajectories
- Anti-flip safety mechanisms
- Combined CoM stability rewards
"""

import math

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from robot_lab.tasks.manager_based.locomotion.velocity_pose.velocity_pose_env_cfg import (
    LocomotionVelocityPoseRoughEnvCfg,
)
from robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp.composite_actions import DogArmCompositeAction
import robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp as mdp

##
# Pre-defined configs
##
from robot_lab.assets.go2_x5 import GO2_X5_CFG  # Combined GO2 + ARX5 asset

# Define joint names at module level for use in observation configs
DOG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


@configclass
class UnitreeGo2X5VelocityPoseRoughEnvCfg(LocomotionVelocityPoseRoughEnvCfg):
    """Configuration for Unitree GO2 + ARX5 velocity and pose tracking (Stage 1)."""
    
    base_link_name = "base"
    foot_link_name = ".*_foot"
    
    # Dog joint names (12 DOF)
    dog_joint_names = DOG_JOINT_NAMES
    
    # Arm joint names (6 DOF)
    arm_joint_names = ARM_JOINT_NAMES
    
    # All joint names (18 DOF total)
    all_joint_names = DOG_JOINT_NAMES + ARM_JOINT_NAMES

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ========================================
        # Scene Configuration
        # ========================================
        self.scene.robot = GO2_X5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ========================================
        # Commands Configuration
        # ========================================
        # Set default height for Go2 (approximately 0.33m)
        self.commands.base_velocity_pose.default_height = 0.33
        
        # Stage 1 Curriculum: Start conservative, gradually expand
        # These will be adjusted by curriculum learning
        self.commands.base_velocity_pose.ranges.height = (0.30, 0.36)  # Stage 1: ±3cm
        self.commands.base_velocity_pose.ranges.roll = (-0.2, 0.2)     # Stage 1: ±11.5°
        self.commands.base_velocity_pose.ranges.pitch = (-0.15, 0.15)  # Stage 1: ±8.6°
        
        # Velocity ranges (can be more aggressive)
        self.commands.base_velocity_pose.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity_pose.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity_pose.ranges.ang_vel_z = (-1.0, 1.0)

        # ========================================
        # Observations Configuration (76D total)
        # ========================================
        @configclass
        class GO2X5ObservationsCfg:
            """Unified 76D observation space for Stage 1."""
            
            @configclass
            class PolicyCfg(ObsGroup):
                """Policy observations (76D)."""
                
                # Dog base state (6D)
                base_lin_vel = ObsTerm(
                    func=mdp.base_lin_vel,
                    noise=Unoise(n_min=-0.1, n_max=0.1),
                    scale=2.0
                )  # 3D
                
                base_ang_vel = ObsTerm(
                    func=mdp.base_ang_vel,
                    noise=Unoise(n_min=-0.2, n_max=0.2),
                    scale=0.25
                )  # 3D
                
                # Gravity projection (3D)
                projected_gravity = ObsTerm(
                    func=mdp.projected_gravity,
                    noise=Unoise(n_min=-0.05, n_max=0.05)
                )  # 3D
                
                # Commands (7D)
                velocity_commands = ObsTerm(
                    func=mdp.generated_commands,
                    params={"command_name": "base_velocity_pose"}
                )  # 7D: vx, vy, ωz, h, r, p, yaw=0
                
                # Dog joint states (24D)
                joint_pos = ObsTerm(
                    func=mdp.joint_pos_rel,
                    noise=Unoise(n_min=-0.01, n_max=0.01),
                    params={"asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_NAMES)},
                    scale=1.0
                )  # 12D
                
                joint_vel = ObsTerm(
                    func=mdp.joint_vel_rel,
                    noise=Unoise(n_min=-1.5, n_max=1.5),
                    params={"asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_NAMES)},
                    scale=0.05
                )  # 12D
                
                # Last actions (12D: policy controls dog joints only)
                actions = ObsTerm(func=mdp.last_action)  # 12D
                
                # ========== ARM OBSERVATIONS (NEW) ==========
                
                # Arm joint states (12D)
                arm_joint_pos = ObsTerm(
                    func=mdp.arm_joint_pos_rel,
                    noise=Unoise(n_min=-0.01, n_max=0.01),
                    params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)}
                )  # 6D
                
                arm_joint_vel = ObsTerm(
                    func=mdp.arm_joint_vel_rel,
                    noise=Unoise(n_min=-0.5, n_max=0.5),
                    params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)}
                )  # 6D
                
                # Arm end effector position relative to base (3D)
                arm_ee_pos_relative = ObsTerm(
                    func=mdp.arm_end_effector_position_relative,
                    params={
                        "asset_cfg": SceneEntityCfg("robot"),
                        "ee_body_name": "link6"
                    }
                )  # 3D
                
                # Combined center of mass offset (3D)
                combined_com_offset = ObsTerm(
                    func=mdp.combined_center_of_mass_offset,
                    params={
                        "dog_cfg": SceneEntityCfg("robot"),
                        "dog_mass": 15.0,
                        "arm_mass": 3.0,
                        "arm_body_names": ["link1", "link2", "link3", "link4", "link5", "link6"]
                    }
                )  # 3D
                
                def __post_init__(self):
                    self.enable_corruption = True
                    self.concatenate_terms = True
            
            # Policy and critic use same observations
            policy: PolicyCfg = PolicyCfg()
        
        # Total observation dimension: 3+3+3+7+12+12+12+6+6+3+3 = 70D
        # Note: 12D actions = policy controls dog joints only, arm controlled by trajectory
        self.observations = GO2X5ObservationsCfg()

        # ========================================
        # Actions Configuration
        # ========================================
        # Use composite action term that combines:
        # - Policy outputs: 12D (dog joints only)
        # - Arm trajectory: 6D (generated internally)
        # - Total applied: 18D (combined to robot)
        from isaaclab.managers import ActionTermCfg
        self.actions.joint_pos = ActionTermCfg(
            class_type=DogArmCompositeAction,
            asset_name="robot",
        )

        # ========================================
        # Events Configuration
        # ========================================
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ========================================
        # Rewards Configuration (Stage 1)
        # ========================================
        
        # Disable rewards with empty body_names from parent class (would cause ValueError)
        self.rewards.wheel_vel_penalty = None
        self.rewards.feet_distance_y_exp = None
        
        # General
        self.rewards.is_terminated.weight = 0
        self.rewards.lin_vel_z_l2.weight = -0.0001
        self.rewards.ang_vel_xy_l2.weight = -0.0001
        
        # Keep disabled rewards
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-5
        
        # Command-aware penalties
        self.rewards.stand_still = RewTerm(
            func=mdp.stand_still_full_cmd,
            weight=-2.0,
            params={
                "command_name": "base_velocity_pose",
                "velocity_threshold": 0.1,
                "height_threshold": 0.02,
                "angle_threshold": 0.05,
            }
        )
        
        self.rewards.joint_pos_penalty = RewTerm(
            func=mdp.joint_pos_penalty_full_cmd,
            weight=-1.0,
            params={
                "command_name": "base_velocity_pose",
                "asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_NAMES),
                "stand_still_scale": 5.0,
                "velocity_threshold": 0.5,
                "velocity_cmd_threshold": 0.1,
                "height_threshold": 0.02,
                "angle_threshold": 0.05,
            }
        )
        
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.20

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 6.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        
        self.rewards.track_ang_vel_z_exp.weight = 6.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.707
        
        self.rewards.track_height_exp = RewTerm(
            func=mdp.track_height_exp,
            weight=4.0,
            params={
                "command_name": "base_velocity_pose",
                "std": 0.5,
                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            }
        )
        
        self.rewards.track_orientation_exp = RewTerm(
            func=mdp.track_orientation_exp_without_yaw,
            weight=0.0,
            params={
                "command_name": "base_velocity_pose",
                "std": 0.707,
            }
        )

        # Others - feet rewards
        self.rewards.feet_air_time.weight = 0.3
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -1.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.3
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 1.0
        
        # ========================================
        # ARM-Specific Stability Rewards (NEW for Stage 1)
        # ========================================
        
        # Combined CoM stability - penalize CoM offset caused by arm movement
        self.rewards.combined_com_stability = RewTerm(
            func=mdp.combined_com_stability_reward,
            weight=3.0,
            params={
                "dog_cfg": SceneEntityCfg("robot"),
                "dog_mass": 15.0,
                "arm_mass": 3.0,
                "target_com_offset": (0.0, 0.0, 0.0),
                "std": 0.10,
                "arm_body_names": ["link1", "link2", "link3", "link4", "link5", "link6"]
            }
        )
        
        # Base stability - penalize excessive linear/angular acceleration
        self.rewards.base_stability = RewTerm(
            func=mdp.base_stability_reward,
            weight=2.0,
            params={
                "lin_acc_std": 3.0,
                "ang_acc_std": 5.0,
                "asset_cfg": SceneEntityCfg("robot")
            }
        )
        
        # Feet contact force balance - encourage even weight distribution
        self.rewards.feet_contact_balance = RewTerm(
            func=mdp.feet_contact_force_balance,
            weight=1.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "target_distribution": [0.25, 0.25, 0.25, 0.25]
            }
        )
        
        # Anti-flip reward - CRITICAL for preventing rollover
        self.rewards.anti_flip_reward = RewTerm(
            func=mdp.anti_flip_orientation_reward,
            weight=5.0,
            params={
                "roll_threshold": 0.785,    # 45°
                "pitch_threshold": 0.524,   # 30°
                "penalize_flip_attempt": True,
                "asset_cfg": SceneEntityCfg("robot")
            }
        )
        
        # Upright bonus - reward staying upright
        self.rewards.upright_bonus = RewTerm(
            func=mdp.upright_bonus_reward,
            weight=2.0,
            params={
                "target_gravity_z": -1.0,
                "tolerance": 0.1,
                "asset_cfg": SceneEntityCfg("robot")
            }
        )
        
        # ========================================
        # Terminations Configuration (Safety-Critical)
        # ========================================
        
        # User requirement: "不要实现终止条件，我需要机器狗学会在侧翻后自己爬起来"
        # Only keep time_out and terrain_out_of_bounds, disable illegal_contact
        self.terminations.illegal_contact = None
        
        # ========================================
        # Curriculum Configuration
        # ========================================
        
        # Disable velocity-based curriculums (use stage-based instead)
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        
        # Enable height and pose curriculum for VelocityPose task
        self.curriculum.command_curriculum_height_pose = CurrTerm(
            func=mdp.command_curriculum_height_pose,
            params={"command_name": "base_velocity_pose"}
        )


@configclass
class UnitreeGo2X5VelocityPoseFlatEnvCfg(UnitreeGo2X5VelocityPoseRoughEnvCfg):
    """Configuration for flat terrain."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Flat terrain - no height scanner needed
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
