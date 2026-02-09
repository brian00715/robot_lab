# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Velocity and Pose Tracking Environment for Quadruped Locomotion

This environment extends velocity tracking with height and orientation control.
For GO2+ARX5, it integrates the arm trajectory controller to handle composite actions.
"""

from __future__ import annotations

import torch
from typing import Any

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg


class VelocityPoseEnv(ManagerBasedRLEnv):
    """Environment for quadruped locomotion with velocity and pose tracking.
    
    For robots with manipulators (e.g., GO2+ARX5), this environment automatically
    handles the composite action space:
    - Policy outputs action for dog joints only (12D for GO2)
    - Arm trajectory controller generates arm actions (6D for ARX5)
    - Combined actions are applied to the scene (18D total)
    """

    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the velocity pose environment.
        
        Args:
            cfg: Environment configuration.
            render_mode: Render mode for the environment.
            **kwargs: Additional arguments.
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # Check if this is a GO2+X5 configuration (has arm)
        self._has_arm = self._detect_arm_configuration()
        
        if self._has_arm:
            print("[VelocityPoseEnv] Detected GO2+ARX5 configuration, initializing arm controller...")
            self._initialize_arm_controller()
        else:
            print("[VelocityPoseEnv] Standard quadruped configuration (no arm)")
            self._arm_controller = None

    def _detect_arm_configuration(self) -> bool:
        """Detect if the robot has an arm based on configuration.
        
        Returns:
            True if arm is present, False otherwise.
        """
        # Check if the task name contains "X5" or "ARX5"
        if hasattr(self.cfg, '__class__'):
            class_name = self.cfg.__class__.__name__
            if "X5" in class_name or "ARX5" in class_name or "x5" in class_name:
                return True
        
        # Check if arm joint names are defined in the config
        if hasattr(self.cfg, 'arm_joint_names'):
            return len(self.cfg.arm_joint_names) > 0
        
        # Check if robot has joints with "joint" prefix (ARX5 naming convention)
        robot = self.scene["robot"]
        joint_names = robot.data.joint_names
        arm_joints = [name for name in joint_names if name.startswith("joint")]
        
        return len(arm_joints) > 0

    def _initialize_arm_controller(self):
        """Initialize the arm trajectory controller."""
        try:
            from robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp.arm_controller import (
                create_arm_controller,
            )
            
            # Determine curriculum stage (default to Stage 1)
            stage = getattr(self.cfg, 'arm_curriculum_stage', 1)
            
            # Create controller
            self._arm_controller = create_arm_controller(
                num_envs=self.num_envs,
                device=self.device,
                stage=stage,
            )
            
            # Get arm joint indices
            robot = self.scene["robot"]
            if hasattr(self.cfg, 'arm_joint_names'):
                arm_joint_names = self.cfg.arm_joint_names
            else:
                # Fallback: find joints with "joint" prefix
                arm_joint_names = [name for name in robot.data.joint_names if name.startswith("joint")]
            
            self._arm_joint_ids = [robot.find_joints(name)[0][0] for name in arm_joint_names]
            self._num_arm_joints = len(self._arm_joint_ids)
            
            print(f"[VelocityPoseEnv] Arm controller initialized with {self._num_arm_joints} joints")
            print(f"[VelocityPoseEnv] Arm joint IDs: {self._arm_joint_ids}")
            
        except Exception as e:
            print(f"[VelocityPoseEnv] ERROR: Failed to initialize arm controller: {e}")
            print(f"[VelocityPoseEnv] Falling back to no arm mode")
            self._has_arm = False
            self._arm_controller = None

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Pre-process actions before stepping through the physics.
        
        For GO2+ARX5:
        - Input actions: (num_envs, 12) - Dog joints from policy
        - Generate arm actions: (num_envs, 6) - From trajectory controller
        - Combined actions: (num_envs, 18) - Sent to physics
        
        Args:
            actions: Actions from the policy (num_envs, action_dim).
        """
        if self._has_arm and self._arm_controller is not None:
            # Policy outputs dog joint actions only (12D for GO2)
            dog_actions = actions  # (num_envs, 12)
            
            # Generate arm actions from trajectory controller (6D for ARX5)
            arm_actions = self._arm_controller.generate_arm_action(self)
            
            # Combine dog and arm actions
            combined_actions = torch.cat([dog_actions, arm_actions], dim=1)  # (num_envs, 18)
            
            # Pass combined actions to parent class
            super()._pre_physics_step(combined_actions)
        else:
            # Standard quadruped (no arm)
            super()._pre_physics_step(actions)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments at the given indices.
        
        Args:
            env_ids: Indices of environments to reset.
        """
        super()._reset_idx(env_ids)
        
        # Reset arm controller state for these environments
        if self._has_arm and self._arm_controller is not None and env_ids is not None:
            self._arm_controller.reset_idx(env_ids)

    def update_arm_curriculum_stage(self, stage: int):
        """Update the arm motion curriculum stage.
        
        Args:
            stage: New curriculum stage (1-4).
        """
        if self._has_arm and self._arm_controller is not None:
            self._arm_controller.update_curriculum(stage)
            print(f"[VelocityPoseEnv] Updated arm curriculum to stage {stage}")
        else:
            print("[VelocityPoseEnv] WARNING: No arm controller to update")

    def get_arm_motion_info(self) -> dict | None:
        """Get information about the current arm motion state.
        
        Returns:
            Dictionary with motion statistics or None if no arm.
        """
        if self._has_arm and self._arm_controller is not None:
            return self._arm_controller.get_motion_info()
        return None
