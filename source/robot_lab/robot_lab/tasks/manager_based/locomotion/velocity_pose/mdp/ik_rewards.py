# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
IK-based End-Effector Tracking Reward Functions

Adapted from Visual Wholebody B1Z1's maniploco_rewards.py for Isaac Lab.
These reward functions encourage the arm to track IK-generated target positions
while the quadruped learns optimal pose compensation.

Key Features:
- End-effector position tracking (world frame)
- End-effector orientation tracking
- Arm energy penalties
- Walking/standing-aware rewards
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import yaw_quat, quat_conjugate, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Helper Functions
##


def quat_apply_yaw_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply inverse yaw rotation from quaternion to vector.
    
    Extracts yaw from quaternion, creates inverse yaw quaternion, and rotates vector.
    
    Args:
        quat: Quaternion (N, 4) in [w, x, y, z] format
        vec: Vector (N, 3) to rotate
        
    Returns:
        Rotated vector (N, 3)
    """
    yaw_q = yaw_quat(quat)
    yaw_q_inv = quat_conjugate(yaw_q)
    return quat_apply(yaw_q_inv, vec)


##
# Coordinate Transformations
##


def cart2sphere(cart_coords: torch.Tensor) -> torch.Tensor:
    """Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        cart_coords: Cartesian coordinates (N, 3) in format [x, y, z]
        
    Returns:
        Spherical coordinates (N, 3) in format [l, pitch, yaw] where:
        - l: radial distance
        - pitch: elevation angle [-π/2, π/2]
        - yaw: azimuth angle [-π, π]
    """
    sphere_coords = torch.zeros_like(cart_coords)
    xy_len = torch.norm(cart_coords[:, :2], dim=1)
    sphere_coords[:, 0] = torch.norm(cart_coords, dim=1)  # radius
    sphere_coords[:, 1] = torch.atan2(cart_coords[:, 2], xy_len)  # pitch
    # yaw angle
    sphere_coords[:, 2] = torch.atan2(cart_coords[:, 1], cart_coords[:, 0])
    return sphere_coords


def sphere2cart(sphere_coords: torch.Tensor) -> torch.Tensor:
    """Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        sphere_coords: Spherical coordinates (N, 3) in format [l, pitch, yaw]
        
    Returns:
        Cartesian coordinates (N, 3) in format [x, y, z]
    """
    radius = sphere_coords[:, 0]
    pitch = sphere_coords[:, 1]
    yaw = sphere_coords[:, 2]
    cart_coords = torch.zeros_like(sphere_coords)
    cart_coords[:, 0] = radius * torch.cos(pitch) * torch.cos(yaw)
    cart_coords[:, 1] = radius * torch.cos(pitch) * torch.sin(yaw)
    cart_coords[:, 2] = radius * torch.sin(pitch)
    return cart_coords


def euler_from_quat(
    quat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quat: Quaternion (N, 4) in format [w, x, y, z]
        
    Returns:
        Tuple of (roll, pitch, yaw) tensors, each of shape (N,)
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)
    
    # Yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)
    
    return roll, pitch, yaw


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-π, π] range.
    
    Args:
        angles: Input angles in radians
        
    Returns:
        Wrapped angles in [-π, π]
    """
    return torch.atan2(torch.sin(angles), torch.cos(angles))


##
# End-Effector Position Tracking Rewards
##


def tracking_ee_world(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward end-effector position tracking in world frame (L1 distance).
    
    This is the primary IK tracking reward. Computes L1 distance between
    current end-effector position and IK target position in world coordinates.
    
    Uses exponential reward: exp(-error/sigma)
    
    Args:
        env: Environment instance
        std: Sigma parameter for exponential (larger = more tolerant)
        asset_cfg: Robot asset configuration
        
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Get IK controller if available
    if not hasattr(env, "_ik_controller"):
        return torch.zeros(env.num_envs, device=env.device)
    
    ik_controller = env._ik_controller
    
    # Get current end-effector position from IK controller state
    ee_pos = ik_controller.current_ee_pos  # (num_envs, 3)
    
    # Get target position (first 3 elements of current_targets)
    ee_target = ik_controller.current_targets[:, :3]  # (num_envs, 3)
    
    # Compute L1 error
    ee_pos_error = torch.sum(torch.abs(ee_pos - ee_target), dim=1)
    
    # Exponential reward (*2 matches B1Z1 implementation)
    reward = torch.exp(-ee_pos_error / (std * 2))
    
    return reward


def tracking_ee_sphere(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    sphere_error_scale: torch.Tensor | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward end-effector tracking in spherical coordinates.
    
    This tracks the EE position in spherical coordinates (l, pitch, yaw)
    relative to the robot's base frame. Useful for maintaining relative
    workspace positions during locomotion.
    
    Args:
        env: Environment instance
        std: Sigma parameter for exponential reward
        sphere_error_scale: Optional scaling for each spherical dimension
        asset_cfg: Robot asset configuration
        
    Returns:
        Reward tensor of shape (num_envs,)
    """
    if not hasattr(env, "_ik_controller"):
        return torch.zeros(env.num_envs, device=env.device)
    
    ik_controller = env._ik_controller
    asset = env.scene[asset_cfg.name]
    
    # Get robot base orientation (yaw only)
    base_quat_w = asset.data.root_quat_w  # (num_envs, 4) [w, x, y, z]
    
    # Get sphere center in world frame (robot base position + offset)
    base_pos = asset.data.root_pos_w[:, :3]  # (num_envs, 3)
    sphere_center = base_pos + ik_controller.sphere_center.unsqueeze(0)
    
    # Transform EE position to robot base frame
    ee_pos = ik_controller.current_ee_pos
    ee_pos_local = ee_pos - sphere_center
    
    # Rotate to base yaw frame (remove yaw rotation)
    ee_pos_local = quat_apply_yaw_inverse(base_quat_w, ee_pos_local)
    
    # Convert to spherical coordinates
    ee_pos_sphere = cart2sphere(ee_pos_local)  # (num_envs, 3) [l, pitch, yaw]
    
    # Get target in spherical coordinates from IK controller
    # NOTE: IK controller stores targets in Cartesian, need to convert
    ee_target = ik_controller.current_targets[:, :3]  # (num_envs, 3)
    ee_target_local = ee_target - sphere_center
    ee_target_local = quat_apply_yaw_inverse(base_quat_w, ee_target_local)
    ee_target_sphere = cart2sphere(ee_target_local)
    
    # Default error scale (equal weight to all dimensions)
    if sphere_error_scale is None:
        sphere_error_scale = torch.ones(3, device=env.device)
    
    # Compute weighted L1 error in spherical coordinates
    ee_pos_error = torch.sum(
        torch.abs(ee_pos_sphere - ee_target_sphere) * sphere_error_scale,
        dim=1
    )
    
    # Exponential reward
    reward = torch.exp(-ee_pos_error / std)
    
    return reward


def tracking_ee_cart(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward end-effector tracking in Cartesian coordinates.
    
    Similar to tracking_ee_sphere but uses Cartesian (x,y,z) error
    in robot frame.
    
    Args:
        env: Environment instance
        std: Sigma parameter for exponential reward
        asset_cfg: Robot asset configuration
        
    Returns:
        Reward tensor of shape (num_envs,)
    """
    if not hasattr(env, "_ik_controller"):
        return torch.zeros(env.num_envs, device=env.device)
    
    ik_controller = env._ik_controller
    asset = env.scene[asset_cfg.name]
    
    # Get robot base state
    base_quat_w = asset.data.root_quat_w
    base_pos = asset.data.root_pos_w[:, :3]
    sphere_center = base_pos + ik_controller.sphere_center.unsqueeze(0)
    
    # Transform EE and target to robot frame
    ee_pos = ik_controller.current_ee_pos
    ee_target = ik_controller.current_targets[:, :3]
    
    # Rotate to base yaw frame
    ee_pos_local = quat_apply_yaw_inverse(
        base_quat_w, ee_pos - sphere_center
    )
    ee_target_local = quat_apply_yaw_inverse(
        base_quat_w, ee_target - sphere_center
    )
    
    # Compute L1 error
    ee_pos_error = torch.sum(torch.abs(ee_pos_local - ee_target_local), dim=1)
    
    # Exponential reward
    reward = torch.exp(-ee_pos_error / std)
    
    return reward


##
# End-Effector Orientation Tracking Rewards
##


def tracking_ee_orn(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    orn_error_scale: torch.Tensor | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward end-effector orientation tracking (full RPY).
    
    Tracks all three Euler angles (roll, pitch, yaw) of the
    end-effector orientation.
    
    Args:
        env: Environment instance
        std: Sigma parameter for exponential reward
        orn_error_scale: Optional scaling for each orientation dimension
        asset_cfg: Robot asset configuration
        
    Returns:
        Reward tensor of shape (num_envs,)
    """
    if not hasattr(env, "_ik_controller"):
        return torch.zeros(env.num_envs, device=env.device)
    
    ik_controller = env._ik_controller
    
    # Get current EE orientation
    ee_orn_quat = ik_controller.current_ee_orn  # (num_envs, 4) [w, x, y, z]
    roll, pitch, yaw = euler_from_quat(ee_orn_quat)
    ee_orn_euler = torch.stack([roll, pitch, yaw], dim=-1)  # (num_envs, 3)
    
    # Get target orientation (convert RPY to compare)
    ee_target_orn_rpy = ik_controller.current_targets[:, 3:6]  # (num_envs, 3)
    
    # Compute angular error with wrapping
    orn_error = wrap_to_pi(ee_target_orn_rpy - ee_orn_euler)
    
    # Default error scale
    if orn_error_scale is None:
        orn_error_scale = torch.ones(3, device=env.device)
    
    # Weighted L1 error
    orn_err = torch.sum(torch.abs(orn_error) * orn_error_scale, dim=1)
    
    # Exponential reward
    reward = torch.exp(-orn_err / std)
    
    return reward


def tracking_ee_orn_ry(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    orn_error_scale: torch.Tensor | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward end-effector orientation tracking (roll and yaw only).
    
    Tracks only roll and yaw angles, ignoring pitch. Useful when pitch
    is less important or controlled separately.
    
    Args:
        env: Environment instance
        std: Sigma parameter for exponential reward
        orn_error_scale: Optional scaling for orientation dimensions
        asset_cfg: Robot asset configuration
        
    Returns:
        Reward tensor of shape (num_envs,)
    """
    if not hasattr(env, "_ik_controller"):
        return torch.zeros(env.num_envs, device=env.device)
    
    ik_controller = env._ik_controller
    
    # Get current EE orientation
    ee_orn_quat = ik_controller.current_ee_orn
    roll, pitch, yaw = euler_from_quat(ee_orn_quat)
    ee_orn_euler = torch.stack([roll, pitch, yaw], dim=-1)
    
    # Get target orientation
    ee_target_orn_rpy = ik_controller.current_targets[:, 3:6]
    
    # Compute angular error
    orn_error = wrap_to_pi(ee_target_orn_rpy - ee_orn_euler)
    
    # Default error scale
    if orn_error_scale is None:
        orn_error_scale = torch.ones(3, device=env.device)
    
    # Apply scaling and select only roll (0) and yaw (2)
    orn_error_scaled = orn_error * orn_error_scale
    orn_err = torch.sum(torch.abs(orn_error_scaled[:, [0, 2]]), dim=1)
    
    # Exponential reward
    reward = torch.exp(-orn_err / std)
    
    return reward


##
# Arm Energy Penalties
##


def arm_energy_abs_sum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize absolute sum of arm joint power consumption.
    
    Computes sum of |torque * velocity| for all arm joints.
    Encourages energy-efficient arm motion.
    
    Args:
        env: Environment instance
        asset_cfg: Robot asset configuration
        
    Returns:
        Energy penalty tensor of shape (num_envs,)
    """
    asset = env.scene[asset_cfg.name]
    
    # Get arm joint indices (assumes last 6 DOFs are arm joints for Go2X5)
    # Go2X5: 12 leg joints + 6 arm joints = 18 total
    num_arm_joints = 6
    arm_start_idx = asset.data.joint_pos.shape[1] - num_arm_joints
    
    # Get arm torques and velocities
    arm_torques = asset.data.applied_torque[:, arm_start_idx:]  # (num_envs, 6)
    arm_vel = asset.data.joint_vel[:, arm_start_idx:]  # (num_envs, 6)
    
    # Compute absolute power
    energy = torch.sum(torch.abs(arm_torques * arm_vel), dim=1)
    
    return energy


def arm_action_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize L2 norm of arm actions.
    
    Encourages smaller, smoother arm motions.
    
    Args:
        env: Environment instance
        asset_cfg: Robot asset configuration
        
    Returns:
        Action penalty tensor of shape (num_envs,)
    """
    # NOTE: In IK mode, arm actions come from IK controller, not policy
    # This reward may not be applicable for Stage 2 IK training
    # Keep for compatibility but will return zeros
    return torch.zeros(env.num_envs, device=env.device)


##
# Conditional Tracking Rewards (Walking/Standing)
##


def tracking_ee_sphere_walking(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    speed_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward EE tracking in spherical coordinates, only when robot is walking.
    
    Args:
        env: Environment instance
        std: Sigma parameter for exponential reward
        speed_threshold: Linear velocity threshold to detect walking (m/s)
        asset_cfg: Robot asset configuration
        
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Get base tracking reward
    reward = tracking_ee_sphere(env, std, asset_cfg=asset_cfg)
    
    # Get walking mask (robot is moving)
    asset = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b[:, :2]  # (num_envs, 2) [vx, vy]
    speed = torch.norm(lin_vel, dim=1)
    walking_mask = speed > speed_threshold
    
    # Zero out reward for standing robots
    reward[~walking_mask] = 0.0
    
    return reward


def tracking_ee_sphere_standing(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    speed_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward EE tracking in spherical coordinates, only when robot is standing still.
    
    Args:
        env: Environment instance
        std: Sigma parameter for exponential reward
        speed_threshold: Linear velocity threshold to detect walking (m/s)
        asset_cfg: Robot asset configuration
        
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Get base tracking reward
    reward = tracking_ee_sphere(env, std, asset_cfg=asset_cfg)
    
    # Get standing mask (robot is stationary)
    asset = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b[:, :2]
    speed = torch.norm(lin_vel, dim=1)
    standing_mask = speed <= speed_threshold
    
    # Zero out reward for walking robots
    reward[~standing_mask] = 0.0
    
    return reward
