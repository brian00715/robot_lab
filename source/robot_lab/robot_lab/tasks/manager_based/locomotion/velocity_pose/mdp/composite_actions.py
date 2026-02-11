# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Composite action term for combining policy actions with trajectory generation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .arm_controller import create_arm_controller


class DogArmCompositeAction(ActionTerm):
    """Composite action term that combines policy actions for dog with trajectory actions for arm.
    
    This action term handles the composite control of GO2+ARM robots:
    - Policy outputs 12D actions for dog locomotion
    - Arm controller generates 6D trajectory actions
    - Combined 18D actions are applied to all joints
    
    This follows the standard ActionTerm interface and works with Isaac Lab's ActionManager.
    """
    
    cfg: ActionTermCfg
    
    def __init__(self, cfg: ActionTermCfg, env):
        """Initialize the composite action term.
        
        Args:
            cfg: Configuration for the action term.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        
        # Get the articulation asset
        self._asset = env.scene[cfg.asset_name]
        
        # Detect if this is a dog+arm configuration
        self._has_arm = self._detect_arm_configuration()
        
        # Initialize arm controller if arm is detected
        self._arm_controller = None
        if self._has_arm:
            try:
                self._arm_controller = create_arm_controller(
                    num_envs=env.num_envs,
                    device=env.device,
                    stage=1  # Start with stage 1 (static arm)
                )
                print(f"[DogArmCompositeAction] Initialized arm controller for {env.num_envs} envs")
            except Exception as e:
                print(f"[DogArmCompositeAction] ERROR: Failed to initialize arm controller: {e}")
                self._has_arm = False
        
        # Store joint indices
        if self._has_arm:
            # Assume first 12 joints are dog, next 6 are arm
            self._dog_joint_ids = list(range(12))
            self._arm_joint_ids = list(range(12, 18))
            print(f"[DogArmCompositeAction] Dog joint IDs: {self._dog_joint_ids}")
            print(f"[DogArmCompositeAction] Arm joint IDs: {self._arm_joint_ids}")
        
        # Use default scale and offset (will be applied through PD controller)
        self._scale = 0.25
        self._offset = 0.0
        
        # Debug counter
        self._step_counter = 0
    
    def _detect_arm_configuration(self):
        """Detect if the robot has an arm based on configuration.
        
        Returns:
            True if arm is detected, False otherwise.
        """
        # Check joint count - GO2 has 12 DOF, GO2+ARX5 has 18+ DOF (may include gripper joints)
        num_joints = self._asset.num_joints
        if num_joints >= 18:  # 18 or more joints means it has an arm
            print(f"[DogArmCompositeAction] Detected {num_joints} DOF robot - GO2+ARM configuration")
            return True
        elif num_joints == 12:
            print("[DogArmCompositeAction] Detected 12 DOF robot - Dog only configuration")
            return False
        else:
            print(f"[DogArmCompositeAction] WARNING: Unexpected joint count: {num_joints}")
            return False
    
    @property
    def action_dim(self):
        """Dimension of the action space.
        
        For composite control, the policy only outputs dog actions (12D).
        The arm actions (6D) are generated internally.
        """
        if self._has_arm:
            return 12  # Policy only controls dog
        else:
            return self._asset.num_joints  # Standard control
    
    @property
    def raw_actions(self):
        """The input/raw actions sent to the term."""
        return self._raw_actions
    
    @property
    def processed_actions(self):
        """The actions computed by the term after applying any processing."""
        return self._processed_actions
    
    def process_actions(self, actions):
        """Process the actions and combine with arm trajectory.
        
        Args:
            actions: Policy actions (num_envs, 12) for dog joints.
        """
        self._step_counter += 1
        
        # Store raw actions
        self._raw_actions = actions.clone()
        
        if self._has_arm and self._arm_controller is not None:
            # Debug: First step
            if self._step_counter == 1:
                print(f"\n{'='*80}")
                print("[DogArmCompositeAction] First step:")
                print(f"  Policy actions shape: {actions.shape}")
                print(f"  Policy actions[0]: {actions[0]}")
                print(f"{'='*80}\n")
            
            # Policy outputs dog actions (12D)
            dog_actions = actions
            
            # Generate arm trajectory actions (6D)
            arm_actions = self._arm_controller.generate_arm_action(self._env)
            
            # Debug: Every 100 steps
            if self._step_counter % 100 == 0:
                print(f"[DogArmCompositeAction] Step {self._step_counter}:")
                print(f"  Arm actions mean: {arm_actions.mean(dim=0).cpu().numpy()}")
                print(f"  Arm actions std: {arm_actions.std(dim=0).cpu().numpy()}")
            
            # Apply scale to dog actions only (arm uses full trajectory output)
            scaled_dog_actions = self._offset + self._scale * dog_actions
            
            # Check total joint count to handle gripper joints
            num_joints = self._asset.num_joints
            
            if num_joints == 20:
                # 12 dog + 6 arm + 2 gripper = 20 joints
                # Add zero actions for gripper (keep them fixed)
                gripper_actions = torch.zeros(dog_actions.shape[0], 2, device=dog_actions.device)
                # Combine: scaled dog actions + full arm actions + zero gripper actions
                self._processed_actions = torch.cat([scaled_dog_actions, arm_actions, gripper_actions], dim=1)
            else:
                # 12 dog + 6 arm = 18 joints (no gripper)
                # Combine: scaled dog actions + full arm actions
                self._processed_actions = torch.cat([scaled_dog_actions, arm_actions], dim=1)
            
        else:
            # Standard dog-only control
            self._processed_actions = self._offset + self._scale * actions
    
    def apply_actions(self):
        """Apply the processed actions to the articulation."""
        # Set joint position targets
        self._asset.set_joint_position_target(self._processed_actions)
    
    def reset(self, env_ids=None):
        """Reset the action term.
        
        Args:
            env_ids: The environment indices to reset. If None, reset all environments.
        """
        # Reset arm controller state
        if self._has_arm and self._arm_controller is not None and env_ids is not None:
            self._arm_controller.reset_idx(env_ids)
