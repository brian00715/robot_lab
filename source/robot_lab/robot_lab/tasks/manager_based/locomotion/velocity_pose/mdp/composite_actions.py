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
            # CRITICAL FIX: Map DOG_JOINT_NAMES order to URDF order
            # DOG_JOINT_NAMES (policy output order): FR, FL, RR, RL
            # URDF order: FL, FR, RL, RR
            # We need to reorder policy actions to match URDF
            
            # Get all joint names from asset
            all_joint_names = self._asset.joint_names
            # print(f"[DogArmCompositeAction] All URDF joints: {all_joint_names}")
            
            # Define expected policy output order (from DOG_JOINT_NAMES in rough_env_cfg.py)
            policy_joint_order = [
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            ]
            
            # Filter to get only dog joint names (exclude arm joints)
            dog_joint_names_urdf = [name for name in all_joint_names 
                                    if any(leg in name for leg in ['FL_', 'FR_', 'RL_', 'RR_'])]
            # print(f"[DogArmCompositeAction] Dog joints in URDF: {dog_joint_names_urdf}")
            
            # Create mapping: policy_index -> urdf_dog_index
            # We need to map policy order to the actual indices in all_joint_names
            self._policy_to_urdf_mapping = []
            self._dog_joint_ids = []  # Store actual indices in all_joint_names
            
            for policy_joint_name in policy_joint_order:
                try:
                    # Find the actual index in all_joint_names
                    urdf_global_index = all_joint_names.index(policy_joint_name)
                    self._dog_joint_ids.append(urdf_global_index)
                    # Find position within dog_joint_names_urdf for reordering
                    urdf_dog_index = dog_joint_names_urdf.index(policy_joint_name)
                    self._policy_to_urdf_mapping.append(urdf_dog_index)
                except ValueError:
                    raise ValueError(f"Joint {policy_joint_name} not found in URDF!")
            
            # print(f"\n{'='*80}")
            # print("[DogArmCompositeAction] JOINT MAPPING INITIALIZATION")
            # print(f"{'='*80}")
            # print(f"Policy-to-URDF mapping: {self._policy_to_urdf_mapping}")
            # print("\nDetailed Mapping:")
            # policy_names = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
            #                 "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
            # for policy_idx, (urdf_idx, name) in enumerate(zip(self._policy_to_urdf_mapping, policy_names)):
            #     urdf_name = dog_joint_names_urdf[urdf_idx]
            #     print(f"  Policy[{policy_idx:2d}] {name:12s} -> URDF[{urdf_idx:2d}] {urdf_name}")
            
            # Verify critical mappings
            # print("\nCritical Verification:")
            # print(f"  Policy FR_hip (idx 0) -> Dog[{self._policy_to_urdf_mapping[0]}] Global[{self._dog_joint_ids[0]}]")
            # print(f"  Policy FL_hip (idx 3) -> Dog[{self._policy_to_urdf_mapping[3]}] Global[{self._dog_joint_ids[3]}]")
            # print(f"  Policy RR_hip (idx 6) -> Dog[{self._policy_to_urdf_mapping[6]}] Global[{self._dog_joint_ids[6]}]")
            # print(f"  Policy RL_hip (idx 9) -> Dog[{self._policy_to_urdf_mapping[9]}] Global[{self._dog_joint_ids[9]}]")
            # print(f"{'='*80}\n")
            
            # Arm joint IDs - find joints that are not dog joints
            self._arm_joint_ids = [i for i, name in enumerate(all_joint_names) 
                                   if i not in self._dog_joint_ids]
            print(f"[DogArmCompositeAction] Joint mapping initialized: {len(self._dog_joint_ids)} dog joints, {len(self._arm_joint_ids)} arm joints")
            
            # CRITICAL: Process scale configuration for dog joints
            # Isaac Lab's ActionTerm doesn't handle regex dict scale for composite actions
            # We need to manually process it
            dog_joint_names = [all_joint_names[i] for i in self._dog_joint_ids]
            
            if hasattr(cfg, 'scale') and isinstance(cfg.scale, dict):
                # Process scale dict
                scale_values = []
                for joint_name in dog_joint_names:
                    # Check each pattern in scale dict
                    scale_val = 0.25  # default
                    for pattern, val in cfg.scale.items():
                        import re
                        if re.match(pattern, joint_name):
                            scale_val = val
                            break
                    scale_values.append(scale_val)
                
                self._dog_scale = torch.tensor(scale_values, device=env.device)
                print(f"[DogArmCompositeAction] Processed dog joint scales: {scale_values}")
            else:
                # Use uniform scale
                scale_val = getattr(cfg, 'scale', 0.25) if not isinstance(getattr(cfg, 'scale', 0.25), dict) else 0.25
                self._dog_scale = torch.full((12,), scale_val, device=env.device)
                print(f"[DogArmCompositeAction] Using uniform scale: {scale_val}")
            
            # CRITICAL FIX: Use default joint positions as offset (in policy order!)
            # Extract default positions for dog joints in the policy output order
            default_positions = self._asset.data.default_joint_pos[0, self._dog_joint_ids]
            self._dog_offset = default_positions.clone()
            print(f"[DogArmCompositeAction] Dog joint offsets (policy order): {self._dog_offset.cpu().numpy()}")
        else:
            # For dog-only mode, use parent class processing
            # Parent ActionTerm will handle _scale and _offset
            pass
        
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
            # Debug: First step (DISABLED for performance)
            # if self._step_counter == 1:
            #     print(f"\n{'='*80}")
            #     print("[DogArmCompositeAction] First step:")
            #     print(f"  Policy actions shape: {actions.shape}")
            #     print(f"  Policy actions[0]: {actions[0]}")
            #     print(f"{'='*80}\n")
            
            # Policy outputs dog actions (12D)
            dog_actions = actions
            
            # Generate arm trajectory actions (6D)
            arm_actions = self._arm_controller.generate_arm_action(self._env)
            
            # Debug: Every 100 steps (DISABLED for performance)
            # if self._step_counter % 100 == 0:
            #     print(f"[DogArmCompositeAction] Step {self._step_counter}:")
            #     print(f"  Arm actions mean: {arm_actions.mean(dim=0).cpu().numpy()}")
            #     print(f"  Arm actions std: {arm_actions.std(dim=0).cpu().numpy()}")
            
            # Apply scale to dog actions only (arm uses full trajectory output)
            # Use per-joint scale tensor (handles different scales for hip vs other joints)
            scaled_dog_actions = self._dog_offset + self._dog_scale * dog_actions
            
            # CRITICAL FIX: Create full action vector with correct indices
            # Dog joints may not be at indices 0-11 in URDF
            # We need to place each policy action at its correct global index
            
            num_joints = self._asset.num_joints
            full_actions = torch.zeros(dog_actions.shape[0], num_joints, device=dog_actions.device)
            
            # Place dog actions at their correct global indices
            # policy_idx corresponds to DOG_JOINT_NAMES order
            # self._dog_joint_ids[policy_idx] is the global URDF index
            for policy_idx in range(len(self._dog_joint_ids)):
                global_idx = self._dog_joint_ids[policy_idx]
                full_actions[:, global_idx] = scaled_dog_actions[:, policy_idx]
            
            # Place arm actions at their correct global indices
            for arm_action_idx, global_idx in enumerate(self._arm_joint_ids):
                if arm_action_idx < arm_actions.shape[1]:  # Within arm action dimensions
                    full_actions[:, global_idx] = arm_actions[:, arm_action_idx]
            
            # Debug first 10 steps to verify mapping works correctly (VERIFIED - DISABLED)
            # if self._step_counter <= 10:
            #     env_idx = 0  # Monitor first environment
            #     print(f"\n[Step {self._step_counter}] Action Mapping Verification (env {env_idx}):")
            #     print(f"  Policy outputs (12D dog actions):")
            #     print(f"    FR_hip={scaled_dog_actions[env_idx, 0]:.4f}, FR_thigh={scaled_dog_actions[env_idx, 1]:.4f}, FR_calf={scaled_dog_actions[env_idx, 2]:.4f}")
            #     print(f"    FL_hip={scaled_dog_actions[env_idx, 3]:.4f}, FL_thigh={scaled_dog_actions[env_idx, 4]:.4f}, FL_calf={scaled_dog_actions[env_idx, 5]:.4f}")
            #     print(f"    RR_hip={scaled_dog_actions[env_idx, 6]:.4f}, RR_thigh={scaled_dog_actions[env_idx, 7]:.4f}, RR_calf={scaled_dog_actions[env_idx, 8]:.4f}")
            #     print(f"    RL_hip={scaled_dog_actions[env_idx, 9]:.4f}, RL_thigh={scaled_dog_actions[env_idx, 10]:.4f}, RL_calf={scaled_dog_actions[env_idx, 11]:.4f}")
            #     
            #     all_joint_names = self._asset.joint_names
            #     print(f"  Mapped to global URDF indices:")
            #     for policy_idx in range(min(12, len(self._dog_joint_ids))):
            #         global_idx = self._dog_joint_ids[policy_idx]
            #         joint_name = all_joint_names[global_idx]
            #         value = full_actions[env_idx, global_idx]
            #         print(f"    Policy[{policy_idx:2d}] -> Global[{global_idx:2d}] {joint_name:20s} = {value:.4f}")

            self._processed_actions = full_actions
            
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
