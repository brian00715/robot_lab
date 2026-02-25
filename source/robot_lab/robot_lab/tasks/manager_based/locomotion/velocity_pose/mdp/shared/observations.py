# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Observation functions for VelocityPose task.

This module extends velocity task observations with additional terms
specific to height and pose control, such as:
- Height error relative to command
- Orientation error (roll, pitch) relative to command
- Vertical velocity
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

# Import common observations from velocity task
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations import (
    joint_pos_rel_without_wheel,
    phase,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def base_height_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Height command from the command manager.
    
    Returns:
        Height command tensor (num_envs, 1) in meters.
    """
    command = env.command_manager.get_command("base_velocity_pose")
    # Command format: [lin_vel_x, lin_vel_y, ang_vel_z, height, roll, pitch]
    return command[:, 3:4]  # Extract height (keep 2D shape)


def base_orientation_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Orientation command (roll, pitch, yaw) from the command manager.
    
    Returns:
        Orientation command tensor (num_envs, 3) in radians [roll, pitch, yaw].
        These angles define the desired Base Frame C orientation relative to Point Frame B.
    """
    command = env.command_manager.get_command("base_velocity_pose")
    # Command format: [lin_vel_x, lin_vel_y, ang_vel_z, height, roll, pitch, yaw]
    return command[:, 4:7]  # Extract roll, pitch, yaw


def base_height_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Height error between current base height and commanded height.
    
    This provides the policy with direct feedback on height tracking performance.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Asset configuration.
    
    Returns:
        Height error tensor (num_envs, 1) in meters.
    """
    # Get current base height (z-coordinate of root position)
    asset: Articulation = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2:3]
    
    # Get commanded height
    commanded_height = base_height_command(env)
    
    # Calculate error
    height_error = commanded_height - current_height
    
    return height_error


def base_height_normalized(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_range: tuple[float, float] = (0.22, 0.40),
) -> torch.Tensor:
    """Normalized base height relative to valid range.
    
    Normalization helps the policy understand how close the robot is to
    physical height limits.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Asset configuration.
        height_range: Physical height limits (min, max) in meters.
    
    Returns:
        Normalized height tensor (num_envs, 1) in range [-1, 1].
    """
    # Get current base height
    asset: Articulation = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2:3]
    
    # Normalize to [-1, 1]
    height_min, height_max = height_range
    height_center = (height_min + height_max) / 2.0
    height_scale = (height_max - height_min) / 2.0
    
    normalized_height = (current_height - height_center) / height_scale
    
    return normalized_height


def base_orientation_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Orientation error (roll, pitch) between current and commanded orientation.
    
    This provides direct feedback on pose tracking performance.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Asset configuration.
    
    Returns:
        Orientation error tensor (num_envs, 2) in radians [roll_error, pitch_error].
    """
    # Get current orientation from projected gravity
    asset: Articulation = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    
    # Calculate current roll and pitch from projected gravity
    # projected_gravity is in base frame: [gx, gy, gz]
    # When robot is upright: gravity = [0, 0, -1], we want roll=0, pitch=0
    # FIXED: Correct formula - roll = atan2(gy, -gz), pitch = atan2(-gx, -gz)
    current_roll = torch.atan2(projected_gravity[:, 1], -projected_gravity[:, 2])
    current_pitch = torch.atan2(-projected_gravity[:, 0], -projected_gravity[:, 2])
    current_orientation = torch.stack([current_roll, current_pitch], dim=1)
    
    # Get commanded orientation
    commanded_orientation = base_orientation_command(env)
    
    # Calculate error
    orientation_error = commanded_orientation - current_orientation
    
    return orientation_error


def base_lin_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vertical velocity of the base in world frame.
    
    This helps the policy understand height change dynamics.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Asset configuration.
    
    Returns:
        Vertical velocity tensor (num_envs, 1) in m/s.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, 2:3]


def base_ang_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Angular velocity around x and y axes (roll and pitch rates) in base frame.
    
    This helps the policy control orientation dynamics.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Asset configuration.
    
    Returns:
        Angular velocity tensor (num_envs, 2) in rad/s [omega_x, omega_y].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 0:2]


def feet_height_relative_to_base(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """Height of each foot relative to the base, useful for understanding leg extension.
    
    This helps the policy coordinate leg movements during height changes.
    
    Args:
        env: The environment instance.
        asset_cfg: Asset configuration.
        sensor_cfg: Contact sensor configuration for feet.
    
    Returns:
        Relative height tensor (num_envs, num_feet) in meters.
    """
    # Get base and feet positions
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    
    base_height = asset.data.root_pos_w[:, 2:3]  # (num_envs, 1)
    feet_positions = sensor.data.pos_w  # (num_envs, num_feet, 3)
    feet_heights = feet_positions[:, :, 2]  # (num_envs, num_feet)
    
    # Calculate relative heights
    relative_heights = feet_heights - base_height.squeeze(-1)  # (num_envs, num_feet)
    
    return relative_heights


def height_scanner_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Height of the base above ground (approximated by minimum foot height).
    
    This is more robust than absolute z-position on uneven terrain.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Asset configuration.
    
    Returns:
        Base height above ground tensor (num_envs, 1) in meters.
    """
    # Get asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get base z-position
    base_z = asset.data.root_pos_w[:, 2:3]
    
    # Try to get contact sensor for better ground reference
    try:
        # Get feet positions
        contact_sensor = env.scene["contact_forces"]
        feet_z = contact_sensor.data.pos_w[:, :, 2]  # (num_envs, num_feet)
        
        # Use minimum foot height as ground reference
        ground_z = torch.min(feet_z, dim=1, keepdim=True)[0]
        
        # Calculate height above ground
        height_above_ground = base_z - ground_z
    except (KeyError, AttributeError):
        # Fallback: use absolute z-position if contact sensor not available
        height_above_ground = base_z
    
    return height_above_ground


def last_action_with_height_pose(
    env: ManagerBasedRLEnv,
    action_name: str | None = None,
) -> torch.Tensor:
    """Last action tensor, useful for action smoothing.
    
    This is the same as the velocity task version, included for completeness.
    
    Args:
        env: The RL environment instance.
        action_name: Name of the action term (default: None for default action).
    
    Returns:
        Last action tensor (num_envs, action_dim).
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions


# ========================================
# ARX5 Arm Observations (Stage 1)
# ========================================

def arm_joint_pos_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
) -> torch.Tensor:
    """Arm joint positions relative to default position.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Asset configuration for arm joints.
    
    Returns:
        Relative joint positions (num_envs, 6) in radians.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get current and default positions
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    pos_rel = joint_pos - joint_pos_default
    
    # NUMERICAL STABILITY FIX: Clip extreme positions to prevent observation explosion
    # ARX5 joints have typical range ±π, extreme cases might reach ±2π
    # Clip at ±3π for safety without affecting normal motion
    pos_rel = torch.clamp(pos_rel, -3.0 * 3.14159, 3.0 * 3.14159)
    
    return pos_rel


def arm_joint_vel_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
) -> torch.Tensor:
    """Arm joint velocities.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Asset configuration for arm joints.
    
    Returns:
        Joint velocities (num_envs, 6) in rad/s.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    
    # NUMERICAL STABILITY FIX: Clip extreme velocities to prevent observation explosion
    # Without this, new arm motion modes (fishing/grasping/swinging/probing) can generate
    # extreme velocities that cause critic network to output huge values, leading to
    # value function loss explosion (e.g., 9.23×10²⁸ at iteration 26430)
    # Typical arm velocities: ±5 rad/s, extreme: ±20 rad/s, so clip at ±50 rad/s for safety
    joint_vel = torch.clamp(joint_vel, -50.0, 50.0)
    
    return joint_vel


def arm_end_effector_position_relative(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "link6",
) -> torch.Tensor:
    """End effector position relative to robot base.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Asset configuration.
        ee_body_name: Name of end effector body.
    
    Returns:
        Relative position (num_envs, 3) in meters.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get base position
    base_pos = asset.data.root_pos_w
    
    # Get end effector position
    ee_body_idx = asset.find_bodies(ee_body_name)[0][0]
    ee_pos = asset.data.body_pos_w[:, ee_body_idx, :]
    
    # Calculate relative position in base frame
    rel_pos = ee_pos - base_pos
    
    return rel_pos


def combined_center_of_mass_offset(
    env: ManagerBasedRLEnv,
    dog_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    dog_mass: float = 15.0,
    arm_mass: float = 3.0,
    arm_body_names: list[str] = ["link1", "link2", "link3", "link4", "link5", "link6"],
) -> torch.Tensor:
    """Combined center of mass offset from robot base.
    
    This computes the weighted center of mass of the entire system (dog + arm)
    relative to the dog's base position.
    
    Args:
        env: The RL environment instance.
        dog_cfg: Dog asset configuration.
        dog_mass: Total mass of the dog (kg).
        arm_mass: Total mass of the arm (kg).
        arm_body_names: Names of arm link bodies.
    
    Returns:
        CoM offset (num_envs, 3) in meters relative to base.
    """
    asset: Articulation = env.scene[dog_cfg.name]
    
    # Approximate dog CoM as base position
    dog_com = asset.data.root_pos_w
    
    # Calculate arm CoM as average of all link positions (simplified)
    arm_body_indices = [asset.find_bodies(name)[0][0] for name in arm_body_names]
    arm_link_positions = torch.stack([
        asset.data.body_pos_w[:, idx, :] for idx in arm_body_indices
    ], dim=1)  # (num_envs, num_links, 3)
    
    # Simplified: uniform weight for each link
    arm_com = arm_link_positions.mean(dim=1)  # (num_envs, 3)
    
    # Calculate combined CoM
    total_mass = dog_mass + arm_mass
    combined_com = (dog_mass * dog_com + arm_mass * arm_com) / total_mass
    
    # Return offset from base
    com_offset = combined_com - dog_com
    
    return com_offset


# ==============================================================================
# Placeholder observations for unified observation space
# These are used to align Low-Level and High-Level observation dimensions
# without affecting actual computation logic
# ==============================================================================

def placeholder_world_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Placeholder for world position (Low-Level only, zeros)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.zeros(env.num_envs, 3, device=asset.device)


def placeholder_world_yaw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Placeholder for world yaw angle (Low-Level only, zeros)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.zeros(env.num_envs, 1, device=asset.device)


def placeholder_ee_target_world(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Placeholder for EE target in world frame (Low-Level, zeros)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.zeros(env.num_envs, 3, device=asset.device)


def placeholder_ee_position_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Placeholder for EE position error (Low-Level only, zeros)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.zeros(env.num_envs, 3, device=asset.device)


def placeholder_trajectory_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Placeholder for trajectory progress (Low-Level, zeros)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.zeros(env.num_envs, 1, device=asset.device)


def placeholder_current_pose_commands(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Placeholder for pose commands (Low-Level only, zeros)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.zeros(env.num_envs, 3, device=asset.device)


def placeholder_velocity_pose_commands(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Placeholder for VelocityPose commands (High-Level, zeros)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.zeros(env.num_envs, 7, device=asset.device)


def placeholder_last_actions(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Placeholder for last actions (High-Level only, zeros)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.zeros(env.num_envs, 12, device=asset.device)


def placeholder_arm_details(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Placeholder for arm details (High-Level only, zeros)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.zeros(env.num_envs, 18, device=asset.device)


# Re-export common observations from velocity task
__all__ = [
    # New observations for VelocityPose
    "base_height_command",
    "base_orientation_command",
    "base_height_error",
    "base_height_normalized",
    "base_orientation_error",
    "base_lin_vel_z",
    "base_ang_vel_xy",
    "feet_height_relative_to_base",
    "height_scanner_base",
    "last_action_with_height_pose",
    # ARX5 Arm observations
    "arm_joint_pos_rel",
    "arm_joint_vel_rel",
    "arm_end_effector_position_relative",
    "combined_center_of_mass_offset",
    # Placeholder observations
    "placeholder_world_position",
    "placeholder_world_yaw",
    "placeholder_ee_target_world",
    "placeholder_ee_position_error",
    "placeholder_trajectory_progress",
    "placeholder_current_pose_commands",
    "placeholder_velocity_pose_commands",
    "placeholder_last_actions",
    "placeholder_arm_details",
    # Re-exported from velocity task
    "joint_pos_rel_without_wheel",
    "phase",
]
