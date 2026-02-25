# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Domain randomization events for VelocityPose task.

This module extends velocity task events with additional randomizations
specific to height and pose control, such as:
- Height sensor noise
- IMU orientation noise
- Center of mass variations
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

# Import utility functions from velocity task
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.utils import is_env_assigned_to_terrain
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.events import (
    randomize_rigid_body_inertia,
    randomize_com_positions,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float],
    damping_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the actuator gains (stiffness and damping) for the asset.
    
    This is important for height and pose control as different gains affect
    the robot's ability to maintain target heights and orientations.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Asset configuration.
        stiffness_distribution_params: Distribution parameters for stiffness (min, max).
        damping_distribution_params: Distribution parameters for damping (min, max).
        operation: Operation to apply - "add", "scale", or "abs".
        distribution: Distribution type - "uniform", "log_uniform", or "gaussian".
    """
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Resolve environment and joint indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    
    if asset_cfg.joint_ids == slice(None):
        joint_ids = torch.arange(asset.num_joints, dtype=torch.int, device=asset.device)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)
    
    # Get default actuator parameters
    default_stiffness = asset.data.default_joint_stiffness[env_ids][:, joint_ids]
    default_damping = asset.data.default_joint_damping[env_ids][:, joint_ids]
    
    # Randomize stiffness
    if distribution == "uniform":
        stiffness_scale = torch.rand(len(env_ids), len(joint_ids), device=asset.device)
        stiffness_scale = (
            stiffness_scale * (stiffness_distribution_params[1] - stiffness_distribution_params[0])
            + stiffness_distribution_params[0]
        )
    elif distribution == "gaussian":
        stiffness_scale = torch.randn(len(env_ids), len(joint_ids), device=asset.device)
        stiffness_scale = (
            stiffness_scale * stiffness_distribution_params[1] + stiffness_distribution_params[0]
        )
    else:  # log_uniform
        log_min = torch.log(torch.tensor(stiffness_distribution_params[0]))
        log_max = torch.log(torch.tensor(stiffness_distribution_params[1]))
        stiffness_scale = torch.exp(
            torch.rand(len(env_ids), len(joint_ids), device=asset.device) * (log_max - log_min) + log_min
        )
    
    # Randomize damping
    if distribution == "uniform":
        damping_scale = torch.rand(len(env_ids), len(joint_ids), device=asset.device)
        damping_scale = (
            damping_scale * (damping_distribution_params[1] - damping_distribution_params[0])
            + damping_distribution_params[0]
        )
    elif distribution == "gaussian":
        damping_scale = torch.randn(len(env_ids), len(joint_ids), device=asset.device)
        damping_scale = damping_scale * damping_distribution_params[1] + damping_distribution_params[0]
    else:  # log_uniform
        log_min = torch.log(torch.tensor(damping_distribution_params[0]))
        log_max = torch.log(torch.tensor(damping_distribution_params[1]))
        damping_scale = torch.exp(
            torch.rand(len(env_ids), len(joint_ids), device=asset.device) * (log_max - log_min) + log_min
        )
    
    # Apply operation
    if operation == "scale":
        new_stiffness = default_stiffness * stiffness_scale
        new_damping = default_damping * damping_scale
    elif operation == "add":
        new_stiffness = default_stiffness + stiffness_scale
        new_damping = default_damping + damping_scale
    else:  # abs
        new_stiffness = stiffness_scale
        new_damping = damping_scale
    
    # Set new gains
    asset.write_joint_stiffness_to_sim(new_stiffness, env_ids=env_ids, joint_ids=joint_ids)
    asset.write_joint_damping_to_sim(new_damping, env_ids=env_ids, joint_ids=joint_ids)


def randomize_base_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the base link mass of the robot.
    
    Mass variations significantly affect height control dynamics and
    orientation stability during locomotion.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Asset configuration (should specify base body).
        mass_distribution_params: Distribution parameters (min, max).
        operation: Operation to apply - "add", "scale", or "abs".
        distribution: Distribution type.
    """
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    
    # Get base body index (typically 0 or specified in asset_cfg)
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.tensor([0], dtype=torch.int, device="cpu")  # Base link
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    
    # Get current masses
    masses = asset.root_physx_view.get_masses()
    default_masses = asset.data.default_mass[env_ids][:, body_ids].clone()
    
    # Generate random values
    if distribution == "uniform":
        random_values = torch.rand(len(env_ids), len(body_ids), device="cpu")
        random_values = (
            random_values * (mass_distribution_params[1] - mass_distribution_params[0])
            + mass_distribution_params[0]
        )
    elif distribution == "gaussian":
        random_values = torch.randn(len(env_ids), len(body_ids), device="cpu")
        random_values = random_values * mass_distribution_params[1] + mass_distribution_params[0]
    else:  # log_uniform
        log_min = torch.log(torch.tensor(mass_distribution_params[0]))
        log_max = torch.log(torch.tensor(mass_distribution_params[1]))
        random_values = torch.exp(
            torch.rand(len(env_ids), len(body_ids), device="cpu") * (log_max - log_min) + log_min
        )
    
    # Apply operation
    if operation == "scale":
        masses[env_ids[:, None], body_ids] = default_masses * random_values
    elif operation == "add":
        masses[env_ids[:, None], body_ids] = default_masses + random_values
    else:  # abs
        masses[env_ids[:, None], body_ids] = random_values
    
    # Clamp to positive values
    masses[env_ids[:, None], body_ids] = torch.clamp(masses[env_ids[:, None], body_ids], min=0.1)
    
    # Set masses
    asset.root_physx_view.set_masses(masses, env_ids)


def add_base_mass_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    mass_offset_range: tuple[float, float] = (-2.0, 2.0),
):
    """Add a random mass offset to the base link.
    
    This simulates carrying payloads or varying configurations,
    which affects height control and balance.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Asset configuration.
        mass_offset_range: Range of mass to add (kg), can be negative.
    """
    randomize_base_mass(
        env,
        env_ids,
        asset_cfg,
        mass_offset_range,
        operation="add",
        distribution="uniform",
    )


def randomize_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float] = (0.4, 1.2),
    dynamic_friction_range: tuple[float, float] = (0.4, 1.2),
):
    """Randomize ground friction coefficients for the feet.
    
    Friction affects stability during height changes and pose adjustments.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Asset configuration (should specify foot bodies).
        static_friction_range: Range for static friction coefficient.
        dynamic_friction_range: Range for dynamic friction coefficient.
    """
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Resolve body indices (foot bodies)
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device=asset.device)
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device=asset.device)
    
    # Get material properties view
    material_props = asset.root_physx_view.get_material_properties()
    
    # Randomize static friction
    static_friction = torch.rand(len(env_ids), len(body_ids), device=asset.device)
    static_friction = (
        static_friction * (static_friction_range[1] - static_friction_range[0])
        + static_friction_range[0]
    )
    
    # Randomize dynamic friction
    dynamic_friction = torch.rand(len(env_ids), len(body_ids), device=asset.device)
    dynamic_friction = (
        dynamic_friction * (dynamic_friction_range[1] - dynamic_friction_range[0])
        + dynamic_friction_range[0]
    )
    
    # Update material properties
    # Note: Material properties format is [static_friction, dynamic_friction, restitution]
    for i, env_id in enumerate(env_ids):
        for j, body_id in enumerate(body_ids):
            material_props[env_id, body_id, 0] = static_friction[i, j]  # Static friction
            material_props[env_id, body_id, 1] = dynamic_friction[i, j]  # Dynamic friction
    
    # Set updated material properties
    asset.root_physx_view.set_material_properties(material_props, env_ids)


def reset_base_to_default_height(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    """Reset the robot base to its default height on episode reset.
    
    This ensures consistent starting conditions for height control learning.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        asset_cfg: Asset configuration.
    """
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get default root state
    default_root_state = asset.data.default_root_state[env_ids].clone()
    
    # Set the root state (position, orientation, velocities)
    asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)


# Re-export common events from velocity task for convenience
__all__ = [
    "randomize_actuator_gains",
    "randomize_base_mass",
    "add_base_mass_offset",
    "randomize_friction",
    "reset_base_to_default_height",
    # Re-export from velocity task
    "randomize_rigid_body_inertia",
    "randomize_com_positions",
    "is_env_assigned_to_terrain",
]
