"""IK-based arm controller for ARX5 manipulator.

Based on visual_wholebody/low-level implementation.
Samples target points in workspace and uses IK to generate joint commands.
The target end-effector position is included in observations for the policy to learn
optimal body pose adjustments.
"""

import torch
from torch import Tensor
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class ArmIKController:
    """IK-based controller for ARX5 arm.
    
    Samples target end-effector positions in workspace and uses damped least squares
    IK to compute joint commands. The target position is added to observations so the
    policy can learn to adjust the robot's body pose for optimal energy and robustness.
    
    Based on visual_wholebody/low-level implementation.
    """
    
    def __init__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg,
    ):
        """Initialize IK controller.
        
        Args:
            env: The environment instance
            asset_cfg: Configuration for the articulation asset
        """
        self.env = env
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._device = self._asset.device
        self._num_envs = self._asset.num_instances
        
        # ARX5 arm joint indices (assuming go2x5 robot structure)
        # Typically: [shoulder_roll, shoulder_pitch, elbow_pitch, elbow_roll, wrist_pitch, wrist_roll]
        self._arm_joint_ids = list(range(12, 18))  # joints 12-17 for ARX5
        self._num_arm_joints = 6
        
        # Arm base offset in robot base frame (adjust based on actual URDF)
        # This should match your robot's configuration
        self.arm_base_offset = torch.tensor([0.0, 0.0, 0.3], device=self._device).repeat(self._num_envs, 1)
        
        # Workspace boundaries for target sampling (in arm base frame)
        # These define a reachable workspace for ARX5
        self.workspace_limits = {
            "x": [0.15, 0.45],  # Forward reach
            "y": [-0.25, 0.25],  # Side reach  
            "z": [-0.15, 0.30],  # Height range
        }
        
        # IK parameters
        self.damping = 0.05  # Damping factor for damped least squares
        
        # Goal tracking in world frame
        self.ee_goal_cart_world = torch.zeros(self._num_envs, 3, device=self._device)
        # Goal in robot base frame (for observation)
        self.ee_goal_cart_base = torch.zeros(self._num_envs, 3, device=self._device)
        
        # Trajectory parameters
        self.traj_timesteps = 100  # Steps to reach goal (2 seconds at 50Hz)
        self.traj_total_timesteps = 150  # Total steps before resampling (3 seconds)
        self.goal_timer = torch.zeros(self._num_envs, device=self._device)
        
        # Trajectory endpoints in arm base frame
        self.ee_start_cart = torch.zeros(self._num_envs, 3, device=self._device)
        self.ee_goal_cart = torch.zeros(self._num_envs, 3, device=self._device)
        
        # Current interpolated goal
        self.curr_ee_goal_cart_local = torch.zeros(self._num_envs, 3, device=self._device)
        
        # Initialize first goals
        self._resample_ee_goals(torch.arange(self._num_envs, device=self._device))
        
    def get_ee_goal_observation(self) -> Tensor:
        """Get end-effector goal position in robot base frame for observation.
        
        This is the key for curriculum stage 2: the policy observes the target position
        and learns to adjust the robot's body pose (via 7D command: vx, vy, vz, roll, pitch, height)
        to achieve optimal energy consumption and robustness.
        
        Returns:
            Tensor: Goal position in robot base frame (num_envs, 3)
        """
        return self.ee_goal_cart_base.clone()
        
    def compute_joint_commands(self) -> Tensor:
        """Compute arm joint commands using IK.
        
        This replaces the random arm actions from curriculum stage 1.
        Uses damped least squares IK to track the sampled target position.
        
        Returns:
            Tensor: Joint position commands for arm (num_envs, 6)
        """
        # Update trajectory interpolation
        self._update_curr_ee_goal()
        
        # Get current end-effector position
        ee_pos_world = self._get_ee_position()
        
        # Get arm base position in world frame
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        arm_base_pos_w = root_pos_w + math_utils.quat_rotate(root_quat_w, self.arm_base_offset)
        
        # Compute error in world frame
        dpos = self.ee_goal_cart_world - ee_pos_world
        
        # For full 6D control, we'd need orientation error too
        # For now, just position control
        dpose = torch.cat([dpos, torch.zeros(self._num_envs, 3, device=self._device)], dim=-1)
        
        # Get Jacobian for arm joints
        jacobian = self._get_arm_jacobian()
        
        # Solve IK using damped least squares
        arm_joint_delta = self._control_ik(jacobian, dpose)
        
        # Get current arm joint positions
        current_arm_pos = self._asset.data.joint_pos[:, self._arm_joint_ids]
        
        # Compute target positions
        arm_pos_targets = current_arm_pos + arm_joint_delta
        
        return arm_pos_targets
        
    def reset_idx(self, env_ids: Tensor):
        """Reset goals for specified environments.
        
        Args:
            env_ids: Environment indices to reset
        """
        if len(env_ids) > 0:
            self._resample_ee_goals(env_ids)
            self.goal_timer[env_ids] = 0
            
    def _update_curr_ee_goal(self):
        """Update current interpolated goal and resample if needed."""
        # Linear interpolation between start and goal
        t = torch.clamp(self.goal_timer / self.traj_timesteps, 0.0, 1.0)
        self.curr_ee_goal_cart_local = torch.lerp(self.ee_start_cart, self.ee_goal_cart, t.unsqueeze(-1))
        
        # Transform to world frame
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        
        # Get arm base in world frame
        arm_base_pos_w = root_pos_w + math_utils.quat_rotate(root_quat_w, self.arm_base_offset)
        
        # Transform goal from arm base frame to world frame
        self.ee_goal_cart_world = arm_base_pos_w + math_utils.quat_rotate(root_quat_w, self.curr_ee_goal_cart_local)
        
        # Store in base frame for observation
        self.ee_goal_cart_base = self.curr_ee_goal_cart_local.clone()
        
        # Increment timer
        self.goal_timer += 1
        
        # Resample when trajectory is complete
        resample_ids = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_ee_goals(resample_ids)
            
    def _resample_ee_goals(self, env_ids: Tensor):
        """Sample new trajectory endpoints for specified environments.
        
        Args:
            env_ids: Environment indices to resample
        """
        num_resets = len(env_ids)
        
        # Set start position as current goal (for smooth transitions)
        self.ee_start_cart[env_ids] = self.curr_ee_goal_cart_local[env_ids]
        
        # Sample new goal in arm base frame
        self.ee_goal_cart[env_ids, 0] = torch.rand(num_resets, device=self._device) * \
            (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) + self.workspace_limits["x"][0]
        self.ee_goal_cart[env_ids, 1] = torch.rand(num_resets, device=self._device) * \
            (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) + self.workspace_limits["y"][0]
        self.ee_goal_cart[env_ids, 2] = torch.rand(num_resets, device=self._device) * \
            (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) + self.workspace_limits["z"][0]
        
        # Reset timer
        self.goal_timer[env_ids] = 0
        
    def _get_ee_position(self) -> Tensor:
        """Get current end-effector position in world frame.
        
        Returns:
            Tensor: EE position in world frame (num_envs, 3)
        """
        # This assumes the last body in the arm chain is the end-effector
        # Adjust body_ids based on your robot's configuration
        # For ARX5, the gripper link is typically the last link
        # You may need to check your URDF for the exact link name/index
        
        # Get all body positions
        body_pos_w = self._asset.data.body_pos_w
        
        # Assuming EE is at body index for gripper (adjust as needed)
        # This is a placeholder - you need to find the correct body index
        ee_body_idx = -1  # Last body, or find specific index
        ee_pos = body_pos_w[:, ee_body_idx, :]
        
        return ee_pos
        
    def _get_arm_jacobian(self) -> Tensor:
        """Get Jacobian matrix for arm joints.
        
        Returns:
            Tensor: Jacobian matrix (num_envs, 6, num_arm_joints)
        """
        # Get full Jacobian from Isaac Sim
        # This returns Jacobian for all bodies
        full_jacobian = self._asset.root_physx_view.get_jacobians()
        
        # Extract Jacobian for arm joints only
        # Shape: (num_envs, num_bodies, 6, num_dofs)
        # We need the EE body and arm joints
        
        # Assuming EE is the last body
        ee_body_idx = -1
        
        # Extract arm joint columns
        jacobian = full_jacobian[:, ee_body_idx, :, self._arm_joint_ids]
        
        return jacobian
        
    def _control_ik(self, jacobian: Tensor, dpose: Tensor) -> Tensor:
        """Solve damped least squares IK.
        
        Args:
            jacobian: Jacobian matrix (num_envs, 6, num_arm_joints)
            dpose: Desired pose delta (num_envs, 6) [dx, dy, dz, dr, dp, dy]
            
        Returns:
            Tensor: Joint velocity commands (num_envs, num_arm_joints)
        """
        # Transpose Jacobian
        j_T = torch.transpose(jacobian, 1, 2)  # (num_envs, num_arm_joints, 6)
        
        # Damping matrix
        lmbda = torch.eye(6, device=self._device) * (self.damping ** 2)
        lmbda = lmbda.unsqueeze(0).repeat(self._num_envs, 1, 1)
        
        # Solve: (J @ J^T + λI)^-1 @ J^T @ dpose
        A = torch.bmm(jacobian, j_T) + lmbda  # (num_envs, 6, 6)
        
        # Solve A @ x = dpose for x
        dpose_expanded = dpose.unsqueeze(-1)  # (num_envs, 6, 1)
        x = torch.linalg.solve(A, dpose_expanded)  # (num_envs, 6, 1)
        
        # Joint velocity: u = J^T @ x
        u = torch.bmm(j_T, x).squeeze(-1)  # (num_envs, num_arm_joints)
        
        return u
