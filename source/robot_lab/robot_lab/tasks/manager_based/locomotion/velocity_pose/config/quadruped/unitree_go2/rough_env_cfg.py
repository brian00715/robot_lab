# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from robot_lab.tasks.manager_based.locomotion.velocity_pose.velocity_pose_env_cfg import LocomotionVelocityPoseRoughEnvCfg
import robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp as mdp

##
# Pre-defined configs
##
from robot_lab.assets.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2VelocityPoseRoughEnvCfg(LocomotionVelocityPoseRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Commands------------------------------
        # Set default height for Go2 (approximately 0.33m)
        self.commands.base_velocity_pose.default_height = 0.33
        # Stage 3-4 ranges for inference/play mode: test full pose control capability
        # Use wider ranges to demonstrate learned pose tracking ability
        self.commands.base_velocity_pose.ranges.height = (0.23, 0.43)  # ±10cm (Stage 3 range)
        self.commands.base_velocity_pose.ranges.roll = (-0.349, 0.349)   # ±20° (Stage 3 range)
        self.commands.base_velocity_pose.ranges.pitch = (-0.21, 0.21)    # ±12° (Stage 3 range) - ENABLED!

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
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

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        self.rewards.lin_vel_z_l2.weight = -0.0001  # Placeholder, controlled by curriculum
        self.rewards.ang_vel_xy_l2.weight = -0.0001  # Placeholder, controlled by curriculum
        
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
        
        # Replace: Command-aware stand still penalty (considering 6D command)
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
        
        # Replace: Command-aware joint position penalty (considering 6D command)
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
        self.rewards.action_rate_l2.weight = -0.13

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards (existing) - Doubled for stronger tracking signal
        self.rewards.track_lin_vel_xy_exp.weight = 6.0  
        self.rewards.track_ang_vel_z_exp.weight = 3.0   
        
        # New: Height tracking reward with exponential growth
        # NOTE: Parameters (std, weight) will be dynamically adjusted by curriculum learning
        # Initial values are placeholders - curriculum will override based on stage:
        #   Stage 3 (0-15k): weight=3.0, std=sqrt(0.05)≈0.22m
        #   Stage 4 (15k+):  weight=4.0, std=sqrt(0.05)≈0.22m
        self.rewards.track_height_exp = RewTerm(
            func=mdp.track_height_exp,
            weight=2.0,  # Placeholder - will be updated by curriculum
            params={
                "command_name": "base_velocity_pose",
                "std": math.sqrt(0.25),  # Placeholder - will be updated by curriculum
                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            }
        )
        
        # New: Orientation tracking reward (roll and pitch only, yaw ignored) with exponential growth
        # NOTE: This uses track_orientation_exp_without_yaw to decouple from localization systems
        # Only tracks roll and pitch (yaw is hardcoded to 0°) - can be determined from IMU gravity projection
        # Parameters (std, weight) will be dynamically adjusted by curriculum learning
        # Initial values are placeholders - curriculum will override based on stage:
        #   Stage 3 (0-15k): weight=1.5, std=sqrt(0.10)≈0.316rad (18°)
        #   Stage 4 (15k+):  weight=2.0, std=sqrt(0.10)≈0.316rad (18°)
        self.rewards.track_orientation_exp = RewTerm(
            func=mdp.track_orientation_exp_without_yaw,  # CHANGED: Use no-yaw version
            weight=1.0,  # Placeholder - will be updated by curriculum
            params={
                "command_name": "base_velocity_pose",
                "std": math.sqrt(0.5),  # Placeholder - will be updated by curriculum
            }
        )

        # Others
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
        self.rewards.upward.weight = 1.0  # Placeholder - controlled by curriculum (Stage 1: 1.0, Stage 2-4: 0.0)

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2VelocityPoseRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        # Disable velocity-based curriculums (use stage-based instead)
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        
        # Enable stage-based curriculum for height and pose commands
        # This curriculum automatically switches between 3 stages based on training iterations:
        # - Stage 1 (0-20k): Fixed at default (no height/pose variation)
        # - Stage 2 (20k-40k): Small ranges (±3cm height, ±8° roll, 0° pitch)
        # - Stage 3 (40k+): Large ranges (±7cm height, ±10° roll, ±8° pitch)
        from isaaclab.managers import CurriculumTermCfg as CurrTerm
        self.curriculum.command_curriculum_height_pose = CurrTerm(
            func=mdp.command_curriculum_height_pose,
            params={"command_name": "base_velocity_pose"}
        )
