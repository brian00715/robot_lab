# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

# Import base velocity environment configuration
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    ActionsCfg,
    ObservationsCfg,
    EventCfg,
    RewardsCfg,
    TerminationsCfg,
    CurriculumCfg,
)

import robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp as mdp


@configclass
class VelocityPoseCommandsCfg:
    """Command specifications for the MDP with velocity and pose control."""

    base_velocity_pose = mdp.UniformVelocityPoseCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        default_height=0.35,  # Will be overridden in specific robot configs
        ranges=mdp.UniformVelocityPoseCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
            # For curriculum learning, set height and pose ranges to zero initially
            height=(0.0, 0.0),  # Will use default_height
            roll=(0.0, 0.0),    # Will use 0.0
            pitch=(0.0, 0.0),   # Will use 0.0
        ),
    )


@configclass
class LocomotionVelocityPoseRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the locomotion velocity and pose tracking environment.
    
    This environment extends velocity tracking with height and orientation control.
    For curriculum learning, height, roll, and pitch commands start at default values.
    """

    # Override commands with velocity_pose commands
    commands: VelocityPoseCommandsCfg = VelocityPoseCommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        # Call parent post init
        super().__post_init__()
        
        # Update observations to include the new command dimensions
        # The velocity_commands observation now includes [lin_vel_x, lin_vel_y, ang_vel_z, height, roll, pitch]
        if self.observations.policy.velocity_commands is not None:
            self.observations.policy.velocity_commands.params["command_name"] = "base_velocity_pose"
        if self.observations.critic.velocity_commands is not None:
            self.observations.critic.velocity_commands.params["command_name"] = "base_velocity_pose"
        
        # Update reward terms that reference commands
        if hasattr(self.rewards, "track_lin_vel_xy_exp") and self.rewards.track_lin_vel_xy_exp is not None:
            self.rewards.track_lin_vel_xy_exp.params["command_name"] = "base_velocity_pose"
        if hasattr(self.rewards, "track_ang_vel_z_exp") and self.rewards.track_ang_vel_z_exp is not None:
            self.rewards.track_ang_vel_z_exp.params["command_name"] = "base_velocity_pose"
        if hasattr(self.rewards, "stand_still") and self.rewards.stand_still is not None:
            self.rewards.stand_still.params["command_name"] = "base_velocity_pose"
        if hasattr(self.rewards, "joint_pos_penalty") and self.rewards.joint_pos_penalty is not None:
            self.rewards.joint_pos_penalty.params["command_name"] = "base_velocity_pose"
        if hasattr(self.rewards, "wheel_vel_penalty") and self.rewards.wheel_vel_penalty is not None:
            self.rewards.wheel_vel_penalty.params["command_name"] = "base_velocity_pose"
        if hasattr(self.rewards, "feet_air_time") and self.rewards.feet_air_time is not None:
            self.rewards.feet_air_time.params["command_name"] = "base_velocity_pose"
        if hasattr(self.rewards, "feet_gait") and self.rewards.feet_gait is not None:
            self.rewards.feet_gait.params["command_name"] = "base_velocity_pose"
        if hasattr(self.rewards, "feet_contact") and self.rewards.feet_contact is not None:
            self.rewards.feet_contact.params["command_name"] = "base_velocity_pose"
        if hasattr(self.rewards, "feet_contact_without_cmd") and self.rewards.feet_contact_without_cmd is not None:
            self.rewards.feet_contact_without_cmd.params["command_name"] = "base_velocity_pose"
        if hasattr(self.rewards, "feet_height") and self.rewards.feet_height is not None:
            self.rewards.feet_height.params["command_name"] = "base_velocity_pose"
        if hasattr(self.rewards, "feet_height_body") and self.rewards.feet_height_body is not None:
            self.rewards.feet_height_body.params["command_name"] = "base_velocity_pose"
