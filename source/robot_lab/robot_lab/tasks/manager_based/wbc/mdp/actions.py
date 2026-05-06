# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Custom action terms for velocity_pose locomotion tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CompositeJointAction(ActionTerm):
    """Composite action term that combines policy actions with trajectory controller actions.
    
    This action term is designed for robots with manipulators (e.g., GO2+ARX5):
    - Policy outputs actions for base joints (e.g., 12D for dog legs)
    - Arm trajectory controller generates actions for manipulator joints (e.g., 6D for arm)
    - Both are combined and applied to the robot
    
    The arm controller can be disabled during curriculum Stage 1 to allow the policy
    to learn basic locomotion before dealing with arm disturbances.
    """

    cfg: CompositeJointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""

    def __init__(self, cfg: CompositeJointActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._base_joint_ids, self._base_joint_names = self._asset.find_joints(cfg.base_joint_names)
        self._arm_joint_ids, self._arm_joint_names = self._asset.find_joints(cfg.arm_joint_names)
        self._num_base_joints = len(self._base_joint_ids)
        self._num_arm_joints = len(self._arm_joint_ids)
        
        # log the joint names
        print(f"[CompositeJointAction] Base joints ({self._num_base_joints}): {self._base_joint_names}")
        print(f"[CompositeJointAction] Arm joints ({self._num_arm_joints}): {self._arm_joint_names}")

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # save the scale and offset as tensors
        self._scale = torch.zeros_like(self._raw_actions)
        self._offset = torch.zeros_like(self._raw_actions)
        
        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale[:] = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            # resolve the scale for each joint
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                cfg.scale, self._base_joint_names
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}")
        
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset[:] = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            # resolve the offset for each joint
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                cfg.offset, self._base_joint_names
            )
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        elif cfg.offset is None and cfg.use_default_offset:
            # use the default joint positions as offset
            self._offset[:] = self._asset.data.default_joint_pos[:, self._base_joint_ids]
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}")

        # Initialize arm controller flag (will be set by environment)
        self._arm_enabled = False
        
        # Arm controller will be set by the environment
        self._arm_controller = None

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action term (base joints only - policy action space)."""
        return self._num_base_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the action term (base joints only)."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the action term (base joints only, after scale/offset)."""
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Process the policy actions (base joints only).
        
        Args:
            actions: The input actions from the policy (num_envs, num_base_joints).
        """
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions[:] = self._raw_actions * self._scale + self._offset

    def apply_actions(self):
        """Apply the processed actions to the robot (combines base + arm actions)."""
        # Get processed base joint actions
        base_actions = self._processed_actions  # (num_envs, num_base_joints)
        
        # Generate arm actions (zeros if disabled)
        if self._arm_enabled and self._arm_controller is not None:
            # Check if curriculum allows arm randomization
            if hasattr(self._env, '_arm_randomization_enabled') and self._env._arm_randomization_enabled:
                arm_actions = self._arm_controller.generate_arm_action(self._env)
            else:
                # Stage 1: Arm stays at default position (zero actions)
                arm_actions = torch.zeros(self.num_envs, self._num_arm_joints, device=self.device)
        else:
            # No arm controller or disabled
            arm_actions = torch.zeros(self.num_envs, self._num_arm_joints, device=self.device)
        
        # Combine base and arm actions
        combined_actions = torch.zeros(
            self.num_envs,
            self._num_base_joints + self._num_arm_joints,
            device=self.device
        )
        
        # Fill in base joint actions
        combined_actions[:, self._base_joint_ids] = base_actions
        
        # Fill in arm joint actions
        combined_actions[:, self._arm_joint_ids] = arm_actions
        
        # Apply to robot
        self._asset.set_joint_position_target(combined_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term.
        
        Args:
            env_ids: The environment ids to reset. Defaults to None (all environments).
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        
        # reset raw and processed actions
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        
        # reset arm controller if it exists
        if self._arm_controller is not None and hasattr(self._arm_controller, 'reset_idx'):
            if isinstance(env_ids, slice):
                env_ids_tensor = torch.arange(self.num_envs, device=self.device)
            else:
                env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            self._arm_controller.reset_idx(env_ids_tensor)
    
    def set_arm_controller(self, controller):
        """Set the arm trajectory controller.

        Args:
            controller: ARX5TrajectoryController instance.
        """
        self._arm_controller = controller
        self._arm_enabled = controller is not None
        if controller is not None:
            print("[CompositeJointAction] Arm controller enabled")
        else:
            print("[CompositeJointAction] Arm controller disabled")


class CompositeJointActionCfg(ActionTermCfg):
    """Configuration for composite joint action term.

    This configuration separates base joints (controlled by policy) from
    arm joints (controlled by trajectory controller).
    """

    class_type: type[ActionTerm] = CompositeJointAction
    """The class corresponding to the action term."""

    base_joint_names: list[str] | str = []
    """List of base joint names or regex patterns (e.g., dog leg joints)."""

    arm_joint_names: list[str] | str = []
    """List of arm joint names or regex patterns (e.g., manipulator joints)."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the base joint actions. Defaults to 1.0."""

    offset: float | dict[str, float] | None = None
    """Offset applied to the base joint actions. Defaults to None."""

    use_default_offset: bool = True
    """Whether to use default joint positions as offset when offset is None. Defaults to True."""

    preserve_order: bool = False
    """Whether to preserve the order of the joint names. Defaults to False."""
