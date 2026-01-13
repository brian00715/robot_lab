# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reward functions for VelocityPose task with command-aware penalties."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
import isaaclab.envs.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Command-aware tracking rewards
##


def track_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking height command with exponential growth - more sensitive to small errors
    
    Uses exponential growth reward function: reward = exp(-|error|/std)
    Gives high reward when robot base height is close to commanded height.
    
    Args:
        env: Environment instance
        std: Tolerance parameter in meters (smaller = more sensitive)
              e.g., std=0.05m means ±5cm tolerance
        command_name: Command name in command manager (should be "base_velocity_pose")
        sensor_cfg: Height sensor config for getting terrain height. If None, uses world z coordinate
        asset_cfg: Robot asset configuration
        
    Returns:
        Reward values with shape (num_envs,)
        
    Design Intent:
        - Exponential growth: smaller errors get exponentially higher rewards
        - More sensitive than squared error version
        - Resolves learning conflicts caused by original lin_vel_z_l2 penalty
        
    Reward Characteristics:
        - Error = 0.00m: reward = 1.00 (perfect)
        - Error = 0.02m: reward ≈ 0.67 (good)
        - Error = 0.05m (=std): reward ≈ 0.37 (acceptable)
        - Error = 0.10m: reward ≈ 0.14 (poor)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get target height command (4th dimension of command)
    command = env.command_manager.get_command(command_name)
    target_height = command[:, 3]
    
    # Calculate current height (considering terrain)
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        # Check sensor data validity
        if not (torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any()):
            # Current height = base z coordinate - average terrain height
            current_height = asset.data.root_pos_w[:, 2] - torch.mean(ray_hits, dim=1)
        else:
            # Fall back to world coordinates when sensor is invalid
            current_height = asset.data.root_pos_w[:, 2]
    else:
        current_height = asset.data.root_pos_w[:, 2]
    
    # Calculate absolute height error (more sensitive to small errors)
    height_error_abs = torch.abs(target_height - current_height)
    
    # Use exponential growth reward: smaller error -> exponentially higher reward
    # reward = exp(-|error|/std) where std controls sensitivity
    # When error=0: reward=1.0, When error=std: reward≈0.37
    reward = torch.exp(-height_error_abs / std)
    
    # Only give reward when robot is upright (avoid rewarding when fallen)
    # projected_gravity_b[:, 2] is close to -1 when upright
    # Use clamp to limit to [0, 0.7] range, normalized to [0, 1]
    gz = env.scene["robot"].data.projected_gravity_b[:, 2]
    upright_factor = torch.clamp(-gz, 0, 0.7) / 0.7
    reward *= upright_factor
    
    # Debug: Print statistics every 100 steps to catch the issue early
    if not hasattr(env, "_height_debug_counter"):
        env._height_debug_counter = 0
    env._height_debug_counter += 1
    if env._height_debug_counter % 100 == 0:
        print(f"\n[DEBUG] Height Tracking Reward Statistics (Step {env._height_debug_counter}):")
        print(f"  Current height:               mean={current_height.mean().item():.4f}, min={current_height.min().item():.4f}, max={current_height.max().item():.4f}")
        print(f"  Target height:                mean={target_height.mean().item():.4f}, min={target_height.min().item():.4f}, max={target_height.max().item():.4f}")
        print(f"  Height error (abs):           mean={height_error_abs.mean().item():.4f}, max={height_error_abs.max().item():.4f}")
        print(f"  projected_gravity[:, 2] (gz): mean={gz.mean().item():.6f}, min={gz.min().item():.6f}, max={gz.max().item():.6f}")
        print(f"  Upright factor:               mean={upright_factor.mean().item():.6f}, min={upright_factor.min().item():.6f}, max={upright_factor.max().item():.6f}")
        print(f"  Raw reward (before upright):  mean={torch.exp(-height_error_abs / std).mean().item():.6f}")
        print(f"  Final reward (after upright): mean={reward.mean().item():.9f}, min={reward.min().item():.9f}, max={reward.max().item():.9f}")
    
    return reward


def track_orientation_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking orientation command with exponential growth - more sensitive to small errors
    
    Uses exponential growth reward function: reward = exp(-(|roll_error| + |pitch_error|)/std)
    Gives high reward when robot base orientation is close to commanded orientation.
    
    Args:
        env: Environment instance
        std: Tolerance parameter in radians (smaller = more sensitive)
              e.g., std=0.15rad ≈ 8.6° tolerance
        command_name: Command name in command manager
        asset_cfg: Robot asset configuration
        
    Returns:
        Reward values with shape (num_envs,)
        
    Design Intent:
        - Exponential growth: smaller errors get exponentially higher rewards
        - More sensitive than squared error version
        - Resolves learning conflicts caused by original ang_vel_xy_l2 penalty
        - Considers both roll and pitch dimensions
        
    Reward Characteristics (assuming single-axis error):
        - Error = 0.00rad (0°): reward = 1.00 (perfect)
        - Error = 0.05rad (2.9°): reward ≈ 0.72 (good)
        - Error = 0.15rad (8.6°, =std): reward ≈ 0.37 (acceptable)
        - Error = 0.30rad (17.2°): reward ≈ 0.14 (poor)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get target orientation command (5th and 6th dimensions of command)
    command = env.command_manager.get_command(command_name)
    target_roll = command[:, 4]
    target_pitch = command[:, 5]
    
    # Calculate current orientation from projected_gravity
    # projected_gravity_b is gravity projection in body frame [gx, gy, gz]
    # When robot is upright: gravity = [0, 0, -1], we want roll=0, pitch=0
    projected_gravity = asset.data.projected_gravity_b
    
    # FIXED: Correct formula to handle upright robot giving roll=0, pitch=0
    # Roll: rotation around x-axis, roll = atan2(gy, -gz)
    current_roll = torch.atan2(projected_gravity[:, 1], -projected_gravity[:, 2])
    
    # Pitch: rotation around y-axis, pitch = atan2(-gx, -gz)
    current_pitch = torch.atan2(-projected_gravity[:, 0], -projected_gravity[:, 2])
    
    # Calculate absolute errors for roll and pitch
    # Use modulo arithmetic to handle angle wrapping [-π, π]
    roll_error = target_roll - current_roll
    pitch_error = target_pitch - current_pitch
    
    # Normalize angles to [-π, π] range to avoid wrap-around issues
    roll_error = torch.atan2(torch.sin(roll_error), torch.cos(roll_error))
    pitch_error = torch.atan2(torch.sin(pitch_error), torch.cos(pitch_error))
    
    # Calculate absolute errors
    roll_error_abs = torch.abs(roll_error)
    pitch_error_abs = torch.abs(pitch_error)
    total_error_abs = roll_error_abs + pitch_error_abs
    
    # Use exponential growth reward: smaller error -> exponentially higher reward
    # reward = exp(-|error|/std) where std controls sensitivity
    # When error=0: reward=1.0, When error=std: reward≈0.37
    reward = torch.exp(-total_error_abs / std)
    
    # Only give reward when robot is upright
    upright_factor = torch.clamp(-projected_gravity[:, 2], 0, 0.7) / 0.7
    reward *= upright_factor
    
    # Debug: Print statistics every 100 steps to catch the issue early
    if not hasattr(env, "_orient_debug_counter"):
        env._orient_debug_counter = 0
    env._orient_debug_counter += 1
    if env._orient_debug_counter % 100 == 0:
        gz = projected_gravity[:, 2]
        print(f"\n[DEBUG] Orientation Tracking Reward Statistics (Step {env._orient_debug_counter}):")
        print(f"  projected_gravity[:, 2] (gz):  mean={gz.mean().item():.6f}, min={gz.min().item():.6f}, max={gz.max().item():.6f}")
        print(f"  Upright factor (-gz clamped):  mean={upright_factor.mean().item():.6f}, min={upright_factor.min().item():.6f}, max={upright_factor.max().item():.6f}")
        print(f"  Roll error (deg):              mean={torch.rad2deg(roll_error_abs).mean().item():.2f}, max={torch.rad2deg(roll_error_abs).max().item():.2f}")
        print(f"  Pitch error (deg):             mean={torch.rad2deg(pitch_error_abs).mean().item():.2f}, max={torch.rad2deg(pitch_error_abs).max().item():.2f}")
        print(f"  Total error (deg):             mean={torch.rad2deg(total_error_abs).mean().item():.2f}, max={torch.rad2deg(total_error_abs).max().item():.2f}")
        print(f"  Raw reward (before upright):   mean={torch.exp(-total_error_abs / std).mean().item():.6f}")
        print(f"  Final reward (after upright):  mean={reward.mean().item():.9f}, min={reward.min().item():.9f}, max={reward.max().item():.9f}")
    
    return reward


##
# Command-aware conditional penalties
##


def lin_vel_z_penalty_conditional(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Conditional penalty on vertical velocity - only penalize when height command is close to default
    
    This function replaces the original lin_vel_z_l2 by checking if the height command has changed
    to decide whether to penalize vertical velocity.
    
    Args:
        env: Environment instance
        command_name: Command name in command manager
        height_threshold: Height command change threshold (meters), below this value means "no height command change"
        asset_cfg: Robot asset configuration
        
    Returns:
        Penalty values with shape (num_envs,)
        
    Design Intent:
        - Allow vertical velocity during height adjustments (avoid conflicts)
        - Still penalize unnecessary vertical motion (e.g., jumping)
        - Implement intelligent switching through conditional logic
        
    Logic:
        IF |height_cmd - default_height| < threshold:
            Penalize vertical velocity (no height adjustment needed)
        ELSE:
            No penalty (height adjustment in progress)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get height command
    command = env.command_manager.get_command(command_name)
    height_cmd = command[:, 3]
    
    # Get default height (from command manager configuration)
    default_height = env.command_manager.get_term("base_velocity_pose").default_height
    
    # Calculate difference between height command and default value
    height_cmd_diff = torch.abs(height_cmd - default_height)
    
    # Only penalize vertical velocity when height command is close to default value
    should_penalize = height_cmd_diff < height_threshold
    
    # Calculate squared vertical velocity (z direction in body frame)
    penalty = torch.square(asset.data.root_lin_vel_b[:, 2])
    
    # Conditionally apply penalty: keep original value when should penalize, otherwise set to zero
    penalty = torch.where(should_penalize, penalty, torch.zeros_like(penalty))
    
    # Only penalize when robot is upright
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    return penalty


def ang_vel_xy_penalty_conditional(
    env: ManagerBasedRLEnv,
    command_name: str,
    angle_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Conditional penalty on roll/pitch angular velocity - only penalize when target orientation is close to zero
    
    This function replaces the original ang_vel_xy_l2 by checking if the target orientation (command) 
    is close to zero to decide whether to penalize angular velocity.
    
    Note: roll and pitch are orientation angles (not angular velocities). This function penalizes 
    the angular velocities (ω_x, ω_y) only when the commanded orientation angles are near zero.
    
    Args:
        env: Environment instance
        command_name: Command name in command manager
        angle_threshold: Target orientation threshold (radians), below this value means "target is flat"
                        e.g., 0.05 rad ≈ 2.86°
        asset_cfg: Robot asset configuration
        
    Returns:
        Penalty values with shape (num_envs,)
        
    Design Intent:
        - Allow angular velocity when target orientation requires tilting (avoid conflicts)
        - Penalize unnecessary angular velocity when target is to remain flat (maintain stability)
        - Consider both roll and pitch target angles
        
    Logic:
        IF sqrt(target_roll^2 + target_pitch^2) < threshold:
            Penalize angular velocities ω_x and ω_y (target is flat, shouldn't be rotating)
        ELSE:
            No penalty (target requires tilting, rotation is necessary)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get target roll and pitch angles (orientation commands, not angular velocities)
    command = env.command_manager.get_command(command_name)
    target_roll = command[:, 4]   # Target roll angle (rad)
    target_pitch = command[:, 5]  # Target pitch angle (rad)
    
    # Calculate L2 norm of target orientation
    target_orientation_norm = torch.sqrt(target_roll**2 + target_pitch**2)
    
    # Only penalize angular velocity when target orientation is close to zero (i.e., target is flat)
    should_penalize = target_orientation_norm < angle_threshold
    
    # Calculate sum of squared roll and pitch angular velocities (ω_x and ω_y in body frame)
    penalty = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    
    # Conditionally apply penalty: penalize only when target is flat
    penalty = torch.where(should_penalize, penalty, torch.zeros_like(penalty))
    
    # Only penalize when robot is upright
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    return penalty


def stand_still_full_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_threshold: float = 0.1,
    height_threshold: float = 0.02,
    angle_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stand still penalty considering full command (including height and orientation)
    
    This function replaces the original stand_still by checking all 6D commands to determine
    if the robot is truly "standing still".
    
    Args:
        env: Environment instance
        command_name: Command name in command manager
        velocity_threshold: Velocity command threshold (m/s or rad/s)
        height_threshold: Height command change threshold (m)
        angle_threshold: Orientation command threshold (rad)
        asset_cfg: Robot asset configuration
        
    Returns:
        Penalty values with shape (num_envs,)
        
    Design Intent:
        - Only penalize joint deviation when truly standing still (all velocity, height, orientation commands are zero)
        - Allow "standing still but adjusting posture" scenarios
        - Maintain original safety constraint strength
        
    Logic:
        Truly static = (velocity command small) AND (height close to default) AND (orientation close to zero)
        IF truly static:
            Penalize joint deviation from default position
        ELSE:
            No penalty (allow adjustments)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get full 6D command
    command = env.command_manager.get_command(command_name)
    velocity_cmd = command[:, :3]  # [vx, vy, ωz]
    height_cmd = command[:, 3]
    pose_cmd = command[:, 4:6]  # [roll, pitch]
    
    # Get default height
    default_height = env.command_manager.get_term("base_velocity_pose").default_height
    
    # Determine if each dimension is "still"
    velocity_small = torch.norm(velocity_cmd, dim=1) < velocity_threshold
    height_default = torch.abs(height_cmd - default_height) < height_threshold
    pose_zero = torch.norm(pose_cmd, dim=1) < angle_threshold
    
    # Only consider "truly static" when all dimensions are still
    is_truly_static = velocity_small & height_default & pose_zero
    
    # Calculate L1 penalty for joint deviation from default position
    penalty = mdp.joint_deviation_l1(env, asset_cfg)
    
    # Only apply penalty when truly static
    penalty = torch.where(is_truly_static, penalty, torch.zeros_like(penalty))
    
    # Only penalize when robot is upright
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    return penalty


def joint_pos_penalty_full_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float = 5.0,
    velocity_threshold: float = 0.5,
    velocity_cmd_threshold: float = 0.1,
    height_threshold: float = 0.02,
    angle_threshold: float = 0.05,
) -> torch.Tensor:
    """Joint position penalty considering full command
    
    This function replaces the original joint_pos_penalty, using full 6D command to determine motion state.
    
    Args:
        env: Environment instance
        command_name: Command name in command manager
        asset_cfg: Robot asset configuration (must include joint_ids)
        stand_still_scale: Penalty scale factor when standing still
        velocity_threshold: Actual velocity threshold (m/s)
        velocity_cmd_threshold: Velocity command threshold (m/s or rad/s)
        height_threshold: Height command change threshold (m)
        angle_threshold: Orientation command threshold (rad)
        
    Returns:
        Penalty values with shape (num_envs,)
        
    Design Intent:
        - Light penalty on joint deviation during motion (maintain flexibility)
        - Heavy penalty on joint deviation when standing still (return to default posture)
        - Consider all 6D commands to determine "motion" state
        
    Logic:
        Motion state = (velocity command large) OR (actual velocity large) OR (height change) OR (orientation change)
        IF moving:
            penalty = 1.0 × base_penalty
        ELSE:
            penalty = 5.0 × base_penalty
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get full 6D command
    command = env.command_manager.get_command(command_name)
    velocity_cmd = command[:, :3]
    height_cmd = command[:, 3]
    pose_cmd = command[:, 4:6]
    
    # Get default height
    default_height = env.command_manager.get_term("base_velocity_pose").default_height
    
    # Determine if in motion
    velocity_cmd_norm = torch.norm(velocity_cmd, dim=1)
    body_vel = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    height_cmd_diff = torch.abs(height_cmd - default_height)
    pose_cmd_norm = torch.norm(pose_cmd, dim=1)
    
    # Motion condition: velocity command large OR actual velocity large OR height change OR orientation change
    is_moving = (velocity_cmd_norm > velocity_cmd_threshold) | \
                (body_vel > velocity_threshold) | \
                (height_cmd_diff > height_threshold) | \
                (pose_cmd_norm > angle_threshold)
    
    # Calculate base penalty: L2 norm of joint position deviation from default
    base_penalty = torch.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
        dim=1
    )
    
    # Adjust penalty strength based on motion state
    # Moving: 1.0× penalty, Standing still: 5.0× penalty
    penalty = torch.where(is_moving, base_penalty, stand_still_scale * base_penalty)
    
    # Only penalize when robot is upright
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    return penalty


##
# Helper functions
##


def is_moving_full_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_threshold: float = 0.1,
    height_threshold: float = 0.02,
    angle_threshold: float = 0.05,
) -> torch.Tensor:
    """Determine if robot is moving (considering full 6D command)
    
    This is a helper function used in other reward functions to determine motion state.
    
    Args:
        env: Environment instance
        command_name: Command name in command manager
        velocity_threshold: Velocity command threshold
        height_threshold: Height command change threshold
        angle_threshold: Orientation command threshold
        
    Returns:
        Boolean tensor with shape (num_envs,), True indicates moving
        
    Usage:
        Can be used in other reward functions to determine whether to apply certain penalties:
        
        ```python
        is_moving = is_moving_full_cmd(env, "base_velocity_pose")
        penalty = torch.where(is_moving, small_penalty, large_penalty)
        ```
    """
    command = env.command_manager.get_command(command_name)
    
    # Calculate velocity command norm
    velocity_cmd_norm = torch.norm(command[:, :3], dim=1)
    
    # Calculate height command difference
    default_height = env.command_manager.get_term("base_velocity_pose").default_height
    height_cmd_diff = torch.abs(command[:, 3] - default_height)
    
    # Calculate orientation command norm
    pose_cmd_norm = torch.norm(command[:, 4:6], dim=1)
    
    # Consider moving if any dimension has command change
    return (velocity_cmd_norm > velocity_threshold) | \
           (height_cmd_diff > height_threshold) | \
           (pose_cmd_norm > angle_threshold)


##
# Updated existing reward functions to use full command awareness
##


def feet_contact_without_cmd_full(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    velocity_threshold: float = 0.1,
    height_threshold: float = 0.02,
    angle_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize feet not in contact with ground when standing still (using full command judgment)
    
    This is an improved version of the original feet_contact_without_cmd, considering height and orientation commands.
    
    Args:
        env: Environment instance
        command_name: Command name in command manager
        sensor_cfg: Contact sensor configuration
        velocity_threshold: Velocity command threshold
        height_threshold: Height command change threshold
        angle_threshold: Orientation command threshold
        
    Returns:
        Penalty values with shape (num_envs,)
    """
    # Use full command to determine if moving
    is_moving = is_moving_full_cmd(env, command_name, velocity_threshold, height_threshold, angle_threshold)
    
    # Get contact sensor data
    contact_sensor = env.scene[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    is_contact = torch.norm(contact_forces, dim=-1).max(dim=1)[0] > 1.0
    
    # Calculate number of feet not in contact
    num_feet = sensor_cfg.body_ids.__len__()
    not_contact_count = num_feet - is_contact.sum(dim=-1)
    
    # Only penalize when standing still
    penalty = torch.where(is_moving, torch.zeros_like(not_contact_count, dtype=torch.float), 
                         not_contact_count.float())
    
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    return penalty
