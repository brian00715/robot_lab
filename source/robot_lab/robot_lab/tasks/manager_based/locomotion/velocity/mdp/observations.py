# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


##
# VelocityPose Observations - for height and pose control
##


def base_height_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Height command from the command manager.
    
    Returns:
        Height command tensor (num_envs, 1) in meters.
    """
    command = env.command_manager.get_command("base_velocity_pose")
    # Command format: [lin_vel_x, lin_vel_y, ang_vel_z, height, roll, pitch]
    return command[:, 3:4]  # Extract height (keep 2D shape)


def base_orientation_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Orientation command (roll, pitch) from the command manager.
    
    Returns:
        Orientation command tensor (num_envs, 2) in radians [roll, pitch].
    """
    command = env.command_manager.get_command("base_velocity_pose")
    # Command format: [lin_vel_x, lin_vel_y, ang_vel_z, height, roll, pitch]
    return command[:, 4:6]  # Extract roll and pitch


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
    # roll = atan2(gy, gz), pitch = atan2(-gx, gz)
    current_roll = torch.atan2(projected_gravity[:, 1], projected_gravity[:, 2])
    current_pitch = torch.atan2(-projected_gravity[:, 0], projected_gravity[:, 2])
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
