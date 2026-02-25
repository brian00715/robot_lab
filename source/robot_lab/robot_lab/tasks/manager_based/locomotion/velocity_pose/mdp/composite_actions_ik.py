# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Composite action term with IK-based arm controller."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .arm_controller_ik import create_ik_arm_controller


class DogArmIKCompositeAction(ActionTerm):
    """Composite action with IK-based arm control for Stage 2 training.
    
    This action term uses inverse kinematics to control the arm:
    - Policy outputs 12D actions for dog locomotion  
    - IK controller generates 6D arm actions to reach workspace targets
    - Combined 18D actions are applied to all joints
    """
    
    cfg: ActionTermCfg
    
    def __init__(self, cfg: ActionTermCfg, env):
        """Initialize the composite action term with IK controller.
        
        Args:
            cfg: Configuration for the action term.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        
        # Get the articulation asset
        self._asset = env.scene[cfg.asset_name]
        
        # Detect if this is a dog+arm configuration
        self._has_arm = self._detect_arm_configuration()
        
        # Initialize IK arm controller
        self._arm_controller = None
        self._env = env
        if self._has_arm:
            try:
                initial_stage = 1
                
                if hasattr(env.cfg, 'inference_stage') and env.cfg.inference_stage is not None:
                    initial_stage = env.cfg.inference_stage
                    print(f"[DogArmIKCompositeAction] Detected inference stage from env_cfg: {initial_stage}")
                elif hasattr(env.unwrapped, '_inference_curriculum_stage'):
                    initial_stage = env.unwrapped._inference_curriculum_stage
                    print(f"[DogArmIKCompositeAction] Detected inference stage from unwrapped: {initial_stage}")
                
                self._arm_controller = create_ik_arm_controller(
                    num_envs=env.num_envs,
                    device=env.device,
                    stage=initial_stage,
                )
                print(f"[DogArmIKCompositeAction] Initialized IK arm controller for {env.num_envs} envs at stage {initial_stage}")
            except Exception as e:
                print(f"[DogArmIKCompositeAction] ERROR: Failed to initialize IK arm controller: {e}")
                import traceback
                traceback.print_exc()
                self._has_arm = False
        
        # Store joint indices (same logic as trajectory-based composite action)
        if self._has_arm:
            all_joint_names = self._asset.joint_names
            
            policy_joint_order = [
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            ]
            
            dog_joint_names_urdf = [name for name in all_joint_names 
                                    if any(leg in name for leg in ['FL_', 'FR_', 'RL_', 'RR_'])]
            
            self._policy_to_urdf_mapping = []
            self._dog_joint_ids = []
            
            for policy_joint_name in policy_joint_order:
                try:
                    urdf_global_index = all_joint_names.index(policy_joint_name)
                    self._dog_joint_ids.append(urdf_global_index)
                    urdf_dog_index = dog_joint_names_urdf.index(policy_joint_name)
                    self._policy_to_urdf_mapping.append(urdf_dog_index)
                except ValueError:
                    raise ValueError(f"Joint {policy_joint_name} not found in URDF!")
            
            # Arm joint IDs
            self._arm_joint_ids = [i for i, name in enumerate(all_joint_names) 
                                   if i not in self._dog_joint_ids]
            print(f"[DogArmIKCompositeAction] Joint mapping: {len(self._dog_joint_ids)} dog joints, {len(self._arm_joint_ids)} arm joints")
            
            # Process dog joint scale configuration
            dog_joint_names = [all_joint_names[i] for i in self._dog_joint_ids]
            
            if hasattr(cfg, 'scale') and isinstance(cfg.scale, dict):
                scale_values = []
                for joint_name in dog_joint_names:
                    scale_val = 0.25  # default
                    for pattern, val in cfg.scale.items():
                        import re
                        if re.match(pattern, joint_name):
                            scale_val = val
                            break
                    scale_values.append(scale_val)
                
                self._dog_scale = torch.tensor(scale_values, device=env.device)
                print(f"[DogArmIKCompositeAction] Processed dog joint scales: {scale_values}")
            else:
                scale_val = getattr(cfg, 'scale', 0.25) if not isinstance(getattr(cfg, 'scale', 0.25), dict) else 0.25
                self._dog_scale = torch.full((12,), scale_val, device=env.device)
                print(f"[DogArmIKCompositeAction] Using uniform scale: {scale_val}")
            
            # Dog joint offsets from default positions
            default_positions = self._asset.data.default_joint_pos[0, self._dog_joint_ids]
            self._dog_offset = default_positions.clone()
            print(f"[DogArmIKCompositeAction] Dog joint offsets: {self._dog_offset.cpu().numpy()}")
        
        self._step_counter = 0
    
    def _detect_arm_configuration(self):
        """Detect if the robot has an arm."""
        num_joints = self._asset.num_joints
        if num_joints >= 18:
            print(f"[DogArmIKCompositeAction] Detected {num_joints} DOF robot - GO2+ARM configuration")
            return True
        elif num_joints == 12:
            print("[DogArmIKCompositeAction] Detected 12 DOF robot - Dog only configuration")
            return False
        else:
            print(f"[DogArmIKCompositeAction] WARNING: Unexpected joint count: {num_joints}")
            return False
    
    @property
    def action_dim(self):
        """Dimension of the action space."""
        if self._has_arm:
            return 12  # Policy only controls dog
        else:
            return self._asset.num_joints
    
    @property
    def raw_actions(self):
        """The input/raw actions sent to the term."""
        return self._raw_actions
    
    @property
    def processed_actions(self):
        """The actions computed by the term after applying any processing."""
        return self._processed_actions
    
    def _update_arm_state(self):
        """Update arm state in the IK controller from environment."""
        if not self._has_arm or self._arm_controller is None:
            return
        
        # Get arm rigid body indices
        # Find end-effector and extract its state
        # Note: This assumes specific link names - adjust if needed
        try:
            # Get rigid body states
            rigid_body_state = self._asset.data.body_state_w  # World frame positions
            
            # Find end-effector index
            # For ARX5, the end-effector link might be named differently
            # Common names: "ee_link", "gripper_link", "ee_gripper_link", "link6"
            body_names = self._asset.body_names
            ee_link_candidates = ["ee_gripper_link", "ee_link", "gripper_link", "link6", "link06"]
            
            ee_idx = None
            for candidate in ee_link_candidates:
                if candidate in body_names:
                    ee_idx = body_names.index(candidate)
                    break
            
            if ee_idx is None:
                print(f"[WARNING] Could not find end-effector link in: {body_names}")
                return
            
            # Extract end-effector pose
            ee_pos = rigid_body_state[:, ee_idx, :3]  # Position
            ee_quat = rigid_body_state[:, ee_idx, 3:7]  # Orientation (quaternion)
            
            # Get current arm joint positions
            arm_dof_pos = self._asset.data.joint_pos[:, self._arm_joint_ids[:6]]  # Only first 6 arm joints
            
            # Get Jacobian for arm
            # Note: Isaac Lab may not expose Jacobian directly
            # For now, we'll use a simplified approach or compute numerically
            # TODO: Implement proper Jacobian extraction or computation
            
            # Placeholder: Identity-based pseudo-Jacobian (will be replaced)
            jacobian = torch.eye(
                6, device=self._env.device
            ).unsqueeze(0).repeat(self._env.num_envs, 1, 1)
            
            # Get base state for yaw-aligned coordinate system
            base_pos = self._asset.data.root_pos_w  # (num_envs, 3)
            base_quat = self._asset.data.root_quat_w  # (num_envs, 4) in [w,x,y,z]
            
            # Update controller state with base state for dynamic sphere center
            self._arm_controller.update_arm_state(
                ee_pos=ee_pos,
                ee_orn=ee_quat,
                arm_dof_pos=arm_dof_pos,
                jacobian=jacobian,
                base_pos=base_pos,
                base_quat=base_quat,
            )
        except Exception as e:
            print(f"[WARNING] Failed to update arm state: {e}")
    
    def process_actions(self, actions):
        """Process actions and combine with IK arm control.
        
        Args:
            actions: Policy actions (num_envs, 12) for dog joints.
        """
        self._step_counter += 1
        
        # Store raw actions
        self._raw_actions = actions.clone()
        
        if self._has_arm and self._arm_controller is not None:
            # Update arm state before computing IK
            self._update_arm_state()
            
            # Policy outputs dog actions (12D)
            dog_actions = actions
            
            # Generate arm IK actions (6D)
            arm_actions = self._arm_controller.generate_arm_action(self._env)
            
            # Apply scale to dog actions
            scaled_dog_actions = self._dog_offset + self._dog_scale * dog_actions
            
            # Create full action vector
            num_joints = self._asset.num_joints
            full_actions = torch.zeros(dog_actions.shape[0], num_joints, device=dog_actions.device)
            
            # Place dog actions at correct global indices
            for policy_idx in range(len(self._dog_joint_ids)):
                global_idx = self._dog_joint_ids[policy_idx]
                full_actions[:, global_idx] = scaled_dog_actions[:, policy_idx]
            
            # Place arm actions at correct global indices
            for arm_action_idx, global_idx in enumerate(self._arm_joint_ids):
                if arm_action_idx < arm_actions.shape[1]:
                    full_actions[:, global_idx] = arm_actions[:, arm_action_idx]
            
            self._processed_actions = full_actions
        else:
            # Standard dog-only control
            self._processed_actions = self._offset + self._scale * actions
    
    def apply_actions(self):
        """Apply the processed actions to the articulation."""
        self._asset.set_joint_position_target(self._processed_actions)
    
    def reset(self, env_ids=None):
        """Reset the action term.
        
        Args:
            env_ids: The environment indices to reset.
        """
        if self._has_arm and self._arm_controller is not None and env_ids is not None:
            self._arm_controller.reset_idx(env_ids)
