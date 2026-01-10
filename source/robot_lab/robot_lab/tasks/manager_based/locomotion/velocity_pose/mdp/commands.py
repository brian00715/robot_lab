# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as velocity_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformVelocityPoseCommand(velocity_mdp.UniformThresholdVelocityCommand):
    """Command generator that extends velocity commands with height and pose (roll, pitch) control.
    
    This command generator adds:
    - Base height command: target height for robot base CoM
    - Base roll command: target roll angle for robot base
    - Base pitch command: target pitch angle for robot base
    
    For curriculum learning, these new commands initially keep default values.
    """

    cfg: UniformVelocityPoseCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        
        # Create buffers for height and pose commands
        # Height command: (num_envs, 1) - target height for base CoM
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        # Pose command: (num_envs, 2) - [roll, pitch]
        self.pose_command = torch.zeros(self.num_envs, 2, device=self.device)
        
        # Store default height (will be set in _resample_command based on robot)
        self.default_height = self.cfg.default_height

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity and pose command in base frame. Shape is (num_envs, 6).
        
        The command is structured as: [lin_vel_x, lin_vel_y, ang_vel_z, height, roll, pitch]
        """
        return torch.cat([self.vel_command_b, self.height_command, self.pose_command], dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity and pose commands.
        
        For curriculum learning, height and pose commands are set to default values initially.
        """
        # First resample velocity commands using parent class
        super()._resample_command(env_ids)
        
        # Sample height command (initially using default height)
        # For curriculum learning, we keep it at default
        height_range = self.cfg.ranges.height
        if height_range[0] == height_range[1]:
            # Use default height when range is zero
            self.height_command[env_ids] = self.default_height
        else:
            # Sample from range
            self.height_command[env_ids] = torch.rand(
                len(env_ids), 1, device=self.device
            ) * (height_range[1] - height_range[0]) + height_range[0]
        
        # Sample roll and pitch commands (initially set to zero for curriculum learning)
        roll_range = self.cfg.ranges.roll
        pitch_range = self.cfg.ranges.pitch
        
        if roll_range[0] == roll_range[1]:
            self.pose_command[env_ids, 0] = 0.0  # Default roll = 0
        else:
            self.pose_command[env_ids, 0] = torch.rand(
                len(env_ids), device=self.device
            ) * (roll_range[1] - roll_range[0]) + roll_range[0]
        
        if pitch_range[0] == pitch_range[1]:
            self.pose_command[env_ids, 1] = 0.0  # Default pitch = 0
        else:
            self.pose_command[env_ids, 1] = torch.rand(
                len(env_ids), device=self.device
            ) * (pitch_range[1] - pitch_range[0]) + pitch_range[0]

    def _update_metrics(self):
        """Update metrics for the command generator."""
        # Call parent's update metrics for velocity tracking
        super()._update_metrics()
        
        # Add height and pose tracking metrics (can be extended later)
        pass


@configclass
class UniformVelocityPoseCommandCfg(velocity_mdp.UniformThresholdVelocityCommandCfg):
    """Configuration for the uniform velocity and pose command generator."""

    class_type: type = UniformVelocityPoseCommand
    
    # Default height for the robot base (will be overridden per robot)
    default_height: float = 0.35
    """Default height for robot base CoM in meters."""
    
    @configclass
    class Ranges(velocity_mdp.UniformThresholdVelocityCommandCfg.Ranges):
        """Ranges for the velocity and pose commands."""
        
        # Height command range (relative to default height)
        height: tuple[float, float] = (0.0, 0.0)
        """Range for height command in meters. Set to (0.0, 0.0) for curriculum learning."""
        
        # Roll command range
        roll: tuple[float, float] = (0.0, 0.0)
        """Range for roll command in radians. Set to (0.0, 0.0) for curriculum learning."""
        
        # Pitch command range  
        pitch: tuple[float, float] = (0.0, 0.0)
        """Range for pitch command in radians. Set to (0.0, 0.0) for curriculum learning."""
    
    # Override ranges with extended Ranges class
    ranges: Ranges = Ranges()
