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
from robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp.low_level.composite_actions import (
    DogArmCompositeAction,
)
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
class GO2X5ObservationsCfg:
    """Unified observation space: 84D (Low-Level & High-Level)."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations (84D unified)."""
        
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale=2.0
        )
        
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25
        )
        
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity_pose"}
        )
        
        actions = ObsTerm(func=mdp.last_action)
        
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=DOG_JOINT_NAMES)},
            scale=1.0
        )
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=DOG_JOINT_NAMES)},
            scale=0.05
        )
        
        arm_joint_pos = ObsTerm(
            func=mdp.arm_joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=ARM_JOINT_NAMES)}
        )
        
        arm_joint_vel = ObsTerm(
            func=mdp.arm_joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=ARM_JOINT_NAMES)}
        )
        
        arm_ee_pos_relative = ObsTerm(
            func=mdp.arm_end_effector_position_relative,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "ee_body_name": "link6"
            }
        )
        
        combined_com_offset = ObsTerm(
            func=mdp.combined_center_of_mass_offset,
            params={
                "dog_cfg": SceneEntityCfg("robot"),
                "dog_mass": 15.0,
                "arm_mass": 3.0,
                "arm_body_names": [
                    "link1", "link2", "link3",
                    "link4", "link5", "link6"
                ]
            }
        )
        
        placeholder_world_pos = ObsTerm(
            func=mdp.placeholder_world_position,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        placeholder_world_yaw = ObsTerm(
            func=mdp.placeholder_world_yaw,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        placeholder_ee_target = ObsTerm(
            func=mdp.placeholder_ee_target_world,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        placeholder_ee_error = ObsTerm(
            func=mdp.placeholder_ee_position_error,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        placeholder_traj_progress = ObsTerm(
            func=mdp.placeholder_trajectory_progress,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        placeholder_pose_cmds = ObsTerm(
            func=mdp.placeholder_current_pose_commands,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()


@configclass
class UnitreeGo2X5VelocityPoseRoughEnvCfg(LocomotionVelocityPoseRoughEnvCfg):
    """Configuration for Unitree GO2 + ARX5 velocity and pose tracking (Stage 1)."""
    
    base_link_name = "base"
    foot_link_name = ".*_foot"
    
    dog_joint_names = DOG_JOINT_NAMES
    arm_joint_names = ARM_JOINT_NAMES
    all_joint_names = DOG_JOINT_NAMES + ARM_JOINT_NAMES

    def __post_init__(self):
        super().__post_init__()
        
        # Replace policy with 84D unified observation space
        self.observations.policy = GO2X5ObservationsCfg.PolicyCfg()
        
        # Disable height_scan in critic (not used in our config)
        if hasattr(self.observations.critic, 'height_scan'):
            self.observations.critic.height_scan = None

        # Scene
        self.scene.robot = GO2_X5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # Commands
        self.commands.base_velocity_pose.default_height = 0.33
        self.commands.base_velocity_pose.ranges.height = (0.23, 0.43)
        self.commands.base_velocity_pose.ranges.roll = (-0.785, 0.785)
        self.commands.base_velocity_pose.ranges.pitch = (-0.436, 0.436)
        self.commands.base_velocity_pose.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity_pose.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity_pose.ranges.ang_vel_z = (-1.0, 1.0)

        # Actions
        # ========================================
        # Use composite action term that combines:
        # - Policy outputs: 12D (dog joints only)
        # - Arm trajectory: 6D (generated internally)
        # - Total applied: 18D (combined to robot)
        
        # Create custom ActionTermCfg for composite action with scale settings
        from isaaclab.managers import ActionTermCfg as BaseActionTermCfg
        
        @configclass
        class DogArmActionCfg(BaseActionTermCfg):
            """Configuration for dog-arm composite action with scale settings."""
            # These will be accessed by DogArmCompositeAction.__init__
            scale: dict = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
            clip: dict = {".*": (-100.0, 100.0)}
        
        self.actions.joint_pos = DogArmActionCfg(
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
        # Rewards Configuration
        # ========================================
        
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
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
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
        self.rewards.track_ang_vel_z_exp.weight = 6.0
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.05)
        
        self.rewards.track_height_exp = RewTerm(
            func=mdp.track_height_exp,
            weight=2.0,
            params={
                "command_name": "base_velocity_pose",
                "std": math.sqrt(0.25),
                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            }
        )
        
        self.rewards.track_orientation_exp = RewTerm(
            func=mdp.track_orientation_exp_without_yaw,
            weight=1.0,
            params={
                "command_name": "base_velocity_pose",
                "std": math.sqrt(0.5),
            }
        )

        # Anti-spinning reward: accumulated angular velocity penalty (deployable with IMU-only)
        self.rewards.accumulated_ang_vel_standing = RewTerm(
            func=mdp.accumulated_ang_vel_penalty_when_standing,
            weight=4.0,
            params={
                "command_name": "base_velocity_pose",
                "velocity_threshold": 0.05,  
                "angle_std": math.radians(10), 
            }
        )

        # Others - feet rewards - EXACT SAME AS GO2
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.1
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
        
        if self.__class__.__name__ in ["UnitreeGo2X5VelocityPoseRoughEnvCfg", "UnitreeGo2X5VelocityPoseFlatEnvCfg"]:
            self.disable_zero_weight_rewards()
        
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
