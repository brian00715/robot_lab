# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
ARX5 Arm IK Controller for Curriculum Stage 2 Training

This controller uses Inverse Kinematics to compute arm joint positions
based on target end-effector positions sampled from workspace.

The arm moves to reach random workspace targets, and the dog learns to
maintain optimal pose (energy-efficient and robust) during these motions.

Key Features:
- Damped Least Squares IK solver (matches B1Z1 implementation)
- Workspace sampling for target generation
- Smooth interpolation between targets
- Curriculum-aware motion scaling
"""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def extract_yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only quaternion from full quaternion.
    
    This creates a quaternion representing only the yaw (z-axis) rotation,
    discarding roll and pitch components.
    
    Args:
        quat: Full quaternion (N, 4) in [w, x, y, z] format.
        
    Returns:
        Yaw-only quaternion (N, 4).
    """
    # Convert quaternion to Euler angles
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Extract yaw angle
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)
    
    # Create yaw-only quaternion (roll=0, pitch=0, yaw=extracted)
    zeros = torch.zeros_like(yaw)
    yaw_quat = quat_from_euler_xyz(zeros, zeros, yaw)
    
    return yaw_quat


class ARX5IKController:
    """IK-based controller for ARX5 arm with workspace target reaching.
    
    Features:
    - Damped Least Squares IK solver
    - Spherical coordinate workspace sampling (matching Visual Wholebody B1Z1)
    - Smooth target transitions
    - Curriculum-aware scaling
    
    Workspace Configuration (Go2X5):
    - sphere_center: [x=0.1, y=0.0, z=0.53] (from URDF analysis)
    - Spherical range: l∈[0.30,0.60]m, pitch∈[-72°,50°], yaw∈[-69°,69°]
    """
    
    def __init__(
        self,
        num_envs: int,
        device: str,
        # Sphere center configuration (Go2X5 specific)
        sphere_center_x: float = 0.1,    # From URDF: arm_mount xyz[0]
        sphere_center_y: float = 0.0,    # From URDF: arm_mount xyz[1]
        # Calculated: 0.33(standing) + 0.20(base)
        sphere_center_z: float = 0.53,
        # Spherical coordinate ranges (adapted for ARX5 0.75m reach)
        l_range: tuple = (0.30, 0.60),   # Radius [m]: 40-80% of arm reach
        pitch_range: tuple = (-1.256, 1.047),  # -72° to 60° [rad]
        yaw_range: tuple = (-1.2, 1.2),    # -69° to 69° [rad]
        target_hold_steps: int = 100,  # Steps to hold each target
        motion_scale: float = 1.0,
        dt: float = 0.02,
        seed: int = 42,
    ):
        """Initialize the IK controller with spherical workspace sampling.
        
        Args:
            num_envs: Number of parallel environments.
            device: Device to run on ('cuda' or 'cpu').
            sphere_center_x: X offset of workspace sphere center [m].
            sphere_center_y: Y offset of workspace sphere center [m].
            sphere_center_z: Z offset of workspace sphere center [m].
            l_range: Radial distance range (min, max) [m].
            pitch_range: Pitch angle range (min, max) [rad].
            yaw_range: Yaw angle range (min, max) [rad].
            target_hold_steps: Steps to hold each target before switching.
            motion_scale: Overall motion scale factor.
            dt: Time step (seconds).
            seed: Random seed for reproducibility.
        """
        self.num_envs = num_envs
        self.device = device
        
        # Spherical workspace configuration
        self.sphere_center = torch.tensor(
            [sphere_center_x, sphere_center_y, sphere_center_z],
            device=device,
            dtype=torch.float32
        )
        self.l_range = l_range
        self.pitch_range = pitch_range
        self.yaw_range = yaw_range
        
        self.target_hold_steps = target_hold_steps
        self.motion_scale = motion_scale
        self.dt = dt
        
        # Random generator
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        
        # Collision avoidance configuration (matching B1Z1 strategy)
        # Defined in LOCAL (yaw-aligned) frame relative to sphere_center
        self.collision_upper_limits = torch.tensor(
            [0.08, 0.15, -0.03],  # [x_max, y_max, z_max] in LOCAL frame
            device=device,
            dtype=torch.float32
        )
        self.collision_lower_limits = torch.tensor(
            [-0.65, -0.15, -0.55],  # [x_min, y_min, z_min] in LOCAL frame
            device=device,
            dtype=torch.float32
        )
        self.underground_limit = -0.55  # Absolute Z limit (LOCAL frame)
        self.num_collision_check_samples = 10  # Check 10 points along trajectory
        self.collision_check_t = torch.linspace(
            0, 1, self.num_collision_check_samples, device=device
        )
        
        print("[ARX5IKController] Workspace Configuration:")
        print(f"  Sphere center: [{sphere_center_x:.2f}, "
              f"{sphere_center_y:.2f}, {sphere_center_z:.2f}]m")
        print(f"  Radial range: [{l_range[0]:.2f}, {l_range[1]:.2f}]m")
        pitch_min_deg = math.degrees(pitch_range[0])
        pitch_max_deg = math.degrees(pitch_range[1])
        print(f"  Pitch: [{pitch_min_deg:.1f}°, {pitch_max_deg:.1f}°]")
        yaw_min_deg = math.degrees(yaw_range[0])
        yaw_max_deg = math.degrees(yaw_range[1])
        print(f"  Yaw: [{yaw_min_deg:.1f}°, {yaw_max_deg:.1f}°]")
        print("[ARX5IKController] Collision Avoidance:")
        print(f"  AABB limits (LOCAL): [{self.collision_lower_limits.tolist()}, "
              f"{self.collision_upper_limits.tolist()}]")
        print(f"  Underground limit: {self.underground_limit}m")
        
        # IK parameters (matching B1Z1 implementation)
        self.ik_damping = 0.05
        
        # Target tracking (6D: position + orientation)
        self.current_targets = torch.zeros(
            num_envs, 6, device=device
        )
        self.target_steps = torch.zeros(
            num_envs, device=device, dtype=torch.long
        )
        
        # Arm state (will be set by environment)
        self.current_ee_pos = torch.zeros(num_envs, 3, device=device)
        # Quaternion format
        self.current_ee_orn = torch.zeros(num_envs, 4, device=device)
        self.current_arm_dof_pos = torch.zeros(
            num_envs, 6, device=device
        )
        self.arm_jacobian = torch.zeros(num_envs, 6, 6, device=device)
        
        print(f"[ARX5IKController] Initialized with {num_envs} envs")
        print(f"  Target hold steps: {target_hold_steps}")
    
    def update_arm_state(
        self,
        ee_pos: torch.Tensor,
        ee_orn: torch.Tensor,
        arm_dof_pos: torch.Tensor,
        jacobian: torch.Tensor,
        base_pos: torch.Tensor = None,
        base_quat: torch.Tensor = None,
    ):
        """Update current arm state from environment.
        
        Args:
            ee_pos: End-effector position (num_envs, 3).
            ee_orn: End-effector orientation as quaternion (num_envs, 4).
            arm_dof_pos: Current arm joint positions (num_envs, 6).
            jacobian: Arm Jacobian matrix (num_envs, 6, 6).
            base_pos: Robot base position (num_envs, 3). Optional.
            base_quat: Robot base quaternion (num_envs, 4). Optional.
        """
        self.current_ee_pos = ee_pos
        self.current_ee_orn = ee_orn
        self.current_arm_dof_pos = arm_dof_pos
        self.arm_jacobian = jacobian
        
        # Store base state for dynamic sphere center calculation
        if base_pos is not None:
            self.current_base_pos = base_pos
        if base_quat is not None:
            self.current_base_quat = base_quat
    
    def reset_idx(self, env_ids: torch.Tensor):
        """Reset controller state for specific environments.
        
        Args:
            env_ids: Indices of environments to reset.
        """
        self.target_steps[env_ids] = 0
        # Force new target generation on next step
        self._generate_new_targets(env_ids)
    
    def _collision_check_trajectory(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Check if trajectory from start to goal has collision.
        
        Implements B1Z1's collision detection strategy:
        - Interpolates N points along trajectory
        - Checks AABB collision in LOCAL (yaw-aligned) frame
        - Checks underground collision
        
        Args:
            start_pos: Starting position (N, 3) in LOCAL frame.
            goal_pos: Goal position (N, 3) in LOCAL frame.
            
        Returns:
            Boolean mask (N,): True if collision detected, False if safe.
        """
        # Interpolate trajectory points
        # start_pos: (N, 3), goal_pos: (N, 3)
        # collision_check_t: (num_samples,)
        
        # Expand for broadcasting: (N, 3, num_samples)
        start_expanded = start_pos.unsqueeze(-1)  # (N, 3, 1)
        goal_expanded = goal_pos.unsqueeze(-1)    # (N, 3, 1)
        t_expanded = self.collision_check_t[None, None, :]  # (1, 1, num_samples)
        
        # Interpolate: (N, 3, num_samples)
        trajectory = start_expanded + (goal_expanded - start_expanded) * t_expanded
        
        # Check AABB collision for each sample point
        # trajectory: (N, 3, num_samples)
        # collision_limits: (3,)
        
        # Broadcasting: (N, 3, num_samples) vs (3, 1)
        upper_check = trajectory < self.collision_upper_limits[:, None]
        lower_check = trajectory > self.collision_lower_limits[:, None]
        
        # All 3 dimensions must be within bounds: (N, num_samples)
        within_bbox = torch.all(upper_check, dim=1) & torch.all(lower_check, dim=1)
        
        # Any sample point in collision bbox means collision
        collision_mask = torch.any(within_bbox, dim=1)  # (N,)
        
        # Check underground collision (Z < underground_limit)
        # trajectory[:, 2, :]: (N, num_samples) - Z coordinates
        underground_mask = torch.any(
            trajectory[:, 2, :] < self.underground_limit, dim=1
        )  # (N,)
        
        return collision_mask | underground_mask
    
    def _generate_new_targets(self, env_ids: torch.Tensor):
        """Generate collision-free targets using rejection sampling.
        
        Matches B1Z1's implementation:
        1. Sample targets in spherical coordinates (LOCAL frame)
        2. Check trajectory for collisions (AABB + ground)
        3. Re-sample if collision detected (max 10 attempts)
        4. Transform to world frame using yaw-aligned coordinates
        
        Args:
            env_ids: Indices of environments to generate targets for.
        """
        N = len(env_ids)
        
        # Get current EE position in LOCAL frame for collision checking
        from isaaclab.utils.math import quat_conjugate
        
        if hasattr(self, 'current_base_pos') and hasattr(self, 'current_base_quat'):
            # Transform current EE to LOCAL frame
            base_yaw_quat = extract_yaw_quat(self.current_base_quat[env_ids])
            base_yaw_quat_inv = quat_conjugate(base_yaw_quat)
            
            # Sphere center in world frame
            base_xy = self.current_base_pos[env_ids, :2]
            zeros = torch.zeros(N, 1, device=self.device)
            sphere_center_base = torch.cat([base_xy, zeros], dim=1)
            sphere_offset = self.sphere_center.unsqueeze(0).expand(N, -1)
            sphere_center_rotated = quat_apply(base_yaw_quat, sphere_offset)
            sphere_center_world = sphere_center_base + sphere_center_rotated
            
            # Current EE relative to sphere center (world frame)
            ee_relative_world = self.current_ee_pos[env_ids] - sphere_center_world
            
            # Transform to LOCAL frame
            ee_pos_local = quat_apply(base_yaw_quat_inv, ee_relative_world)
        else:
            # Fallback
            ee_pos_local = (
                self.current_ee_pos[env_ids] - self.sphere_center.unsqueeze(0)
            )
            base_yaw_quat = None
            sphere_center_world = self.sphere_center.unsqueeze(0).expand(N, -1)
        
        # Rejection sampling (matching B1Z1's strategy)
        remaining_ids = torch.arange(N, device=self.device)
        stored_goals_local = torch.zeros(N, 3, device=self.device)
        
        for attempt in range(10):  # Max 10 attempts like B1Z1
            if len(remaining_ids) == 0:
                break  # All envs found valid targets
            
            N_rem = len(remaining_ids)
            
            # Sample in spherical coordinates
            radial_dist = (
                torch.rand(N_rem, device=self.device, generator=self.rng)
                * (self.l_range[1] - self.l_range[0]) + self.l_range[0]
            )
            pitch = (
                torch.rand(N_rem, device=self.device, generator=self.rng)
                * (self.pitch_range[1] - self.pitch_range[0])
                + self.pitch_range[0]
            )
            yaw = (
                torch.rand(N_rem, device=self.device, generator=self.rng)
                * (self.yaw_range[1] - self.yaw_range[0]) + self.yaw_range[0]
            )
            
            # Convert to Cartesian (LOCAL frame)
            x_l = radial_dist * torch.cos(pitch) * torch.cos(yaw)
            y_l = radial_dist * torch.cos(pitch) * torch.sin(yaw)
            z_l = radial_dist * torch.sin(pitch)
            goal_pos_local = torch.stack([x_l, y_l, z_l], dim=1)
            
            # Collision check (LOCAL frame)
            start_local = ee_pos_local[remaining_ids]
            has_collision = self._collision_check_trajectory(
                start_local, goal_pos_local
            )
            
            # Store valid goals
            valid_mask = ~has_collision
            if valid_mask.any():
                valid_indices = remaining_ids[valid_mask]
                stored_goals_local[valid_indices] = goal_pos_local[valid_mask]
            
            # Update remaining (only retry collisions)
            remaining_ids = remaining_ids[has_collision]
        
        # Warning for failed samples
        if len(remaining_ids) > 0:
            print(f"[WARNING] {len(remaining_ids)}/{N} envs failed collision check")
            # Use last sampled goal anyway (better than no goal)
            stored_goals_local[remaining_ids] = goal_pos_local[has_collision]
        
        # Transform to world frame
        if base_yaw_quat is not None:
            # Re-extract for all envs
            base_yaw_quat_all = extract_yaw_quat(self.current_base_quat[env_ids])
            cart_yaw_aligned = quat_apply(base_yaw_quat_all, stored_goals_local)
            target_pos = sphere_center_world + cart_yaw_aligned
        else:
            target_pos = stored_goals_local + self.sphere_center.unsqueeze(0)
        
        # Fixed orientation (gripper pointing down)
        target_orn = torch.zeros(N, 3, device=self.device)
        target_orn[:, 0] = math.pi / 2
        
        # Store targets
        self.current_targets[env_ids] = torch.cat([target_pos, target_orn], dim=1)
    
    def _control_ik(self, dpose: torch.Tensor) -> torch.Tensor:
        """Compute joint velocity using Damped Least Squares IK.
        
        This matches the B1Z1 implementation exactly.
        
        Args:
            dpose: Desired pose change (num_envs, 6, 1).
                Format: [dx, dy, dz, droll, dpitch, dyaw].
        
        Returns:
            Joint velocity (num_envs, 6).
        """
        # Transpose Jacobian: (num_envs, 6, 6)
        j_eef_T = torch.transpose(self.arm_jacobian, 1, 2)
        
        # Damping matrix: lambda * I
        lmbda = torch.eye(6, device=self.device) * (self.ik_damping ** 2)
        
        # A = J * J^T + lambda * I
        A = torch.bmm(self.arm_jacobian, j_eef_T) + lmbda[None, ...]
        
        # u = J^T * (A^-1 * dpose)
        u = torch.bmm(j_eef_T, torch.linalg.solve(A, dpose))
        
        return u.squeeze(-1)
    
    def _orientation_error(
        self,
        desired_quat: torch.Tensor,
        current_quat: torch.Tensor
    ) -> torch.Tensor:
        """Compute orientation error in axis-angle form.
        
        Args:
            desired_quat: Desired orientation (num_envs, 4).
            current_quat: Current orientation (num_envs, 4).
        
        Returns:
            Orientation error (num_envs, 3).
        """
        # Compute quaternion difference: q_error = q_desired * q_current^(-1)
        # For unit quaternions, q^(-1) = q_conjugate = [w, -x, -y, -z]
        
        current_quat_conj = current_quat.clone()
        current_quat_conj[:, 1:] *= -1  # Conjugate
        
        # Quaternion multiplication
        q_error = self._quat_mul(desired_quat, current_quat_conj)
        
        # Extract axis-angle (using small angle approximation for speed)
        # For small rotations: axis_angle ≈ 2 * [x, y, z]
        axis_angle = 2.0 * q_error[:, 1:]
        
        return axis_angle
    
    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions.
        
        Args:
            q1: First quaternion (num_envs, 4) - [w, x, y, z].
            q2: Second quaternion (num_envs, 4) - [w, x, y, z].
        
        Returns:
            Product quaternion (num_envs, 4).
        """
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=1)
    
    def _rpy_to_quat(self, rpy: torch.Tensor) -> torch.Tensor:
        """Convert roll-pitch-yaw to quaternion.
        
        Args:
            rpy: Roll-pitch-yaw angles (num_envs, 3).
        
        Returns:
            Quaternion (num_envs, 4) - [w, x, y, z].
        """
        roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return torch.stack([w, x, y, z], dim=1)
    
    def generate_arm_action(
        self, env: ManagerBasedRLEnv | None = None
    ) -> torch.Tensor:
        """Generate arm joint target positions using IK.
        
        Args:
            env: Optional environment instance (for curriculum control).
        
        Returns:
            Arm actions (num_envs, 6) - target joint positions in radians.
        """
        # Get motion_scale from curriculum if available
        if env is not None and hasattr(env, "_arm_motion_scale"):
            current_motion_scale = env._arm_motion_scale
            
            # Stage 1: arm fixed at zero
            if current_motion_scale == 0.0:
                return torch.zeros((self.num_envs, 6), device=self.device)
            
            # Update local scale
            if self.motion_scale != current_motion_scale:
                self.motion_scale = current_motion_scale
        else:
            if self.motion_scale == 0.0:
                return torch.zeros((self.num_envs, 6), device=self.device)
        
        # Check which environments need new targets
        needs_new_target = (
            (self.target_steps == 0)
            | (self.target_steps >= self.target_hold_steps)
        )
        
        if needs_new_target.any():
            env_ids = torch.where(needs_new_target)[0]
            self._generate_new_targets(env_ids)
            self.target_steps[env_ids] = 0
        
        # Compute IK to reach current targets
        # Extract target position and orientation
        target_pos = self.current_targets[:, :3]
        target_orn_rpy = self.current_targets[:, 3:]
        target_orn_quat = self._rpy_to_quat(target_orn_rpy)
        
        # Compute pose error
        dpos = target_pos - self.current_ee_pos
        drot = self._orientation_error(target_orn_quat, self.current_ee_orn)
        
        # Combine into 6D pose error
        # Shape: (num_envs, 6, 1)
        dpose = torch.cat([dpos, drot], dim=-1).unsqueeze(-1)
        
        # Apply motion scaling to pose error (slower/faster approach)
        dpose = dpose * self.motion_scale
        
        # Compute joint velocities using IK
        joint_vel = self._control_ik(dpose)
        
        # Integrate to get target positions: q_target = q_current + dq
        arm_target_pos = self.current_arm_dof_pos + joint_vel
        
        # Increment step counter
        self.target_steps += 1
        
        return arm_target_pos
    
    def update_curriculum(self, stage: int):
        """[DEPRECATED] Update parameters based on curriculum stage.
        
        NOTE: This method is now deprecated. Curriculum parameters are managed
        centrally in curriculums.py and accessed via env._arm_motion_scale.
        """
        pass
    
    def get_motion_info(self) -> dict:
        """Get information about current motion state.
        
        Returns:
            Dictionary with motion statistics.
        """
        return {
            "motion_scale": self.motion_scale,
            "workspace_center": self.workspace_center.cpu().numpy().tolist(),
            "workspace_radius": self.workspace_radius,
            "mean_target_step": self.target_steps.float().mean().item(),
        }


def create_ik_arm_controller(
    num_envs: int,
    device: str,
    stage: int = 1,
) -> ARX5IKController:
    """Factory function to create IK controller for Go2+ARX5.
    
    Workspace parameters are based on URDF analysis:
    - sphere_center: [0.1, 0.0, 0.53]m (from Go2X5 URDF)
    - l_range: [0.30, 0.60]m (ARX5 arm reach)
    - pitch: [-72°, 50°], yaw: [-69°, 69°]
    
    Args:
        num_envs: Number of parallel environments.
        device: Device to run on.
        stage: Initial curriculum stage (1-5).
    
    Returns:
        Initialized ARX5IKController with Go2X5 workspace.
    """
    # Go2X5 workspace configuration (from URDF analysis)
    # See GO2X5_SPHERE_CENTER_CALCULATION.md for derivation
    sphere_center_cfg = {
        'sphere_center_x': 0.1,   # From arm_mount_joint xyz[0]
        'sphere_center_y': 0.0,   # From arm_mount_joint xyz[1]
        'sphere_center_z': 0.53,  # 0.33 (standing) + 0.20 (base)
    }
    
    # Spherical coordinate ranges (matching B1Z1 style)
    workspace_cfg = {
        'l_range': (0.30, 0.60),        # Radius: 30-60cm
        'pitch_range': (-1.256, 1.047),  # -72° to 60°
        'yaw_range': (-1.2, 1.2),        # -69° to 69°
    }
    
    # Stage-specific motion parameters
    if stage == 1:
        motion_scale = 0.0  # Arm fixed
        target_hold_steps = 100
    elif stage == 2:
        motion_scale = 1.0  # Slow motion
        target_hold_steps = 150
    elif stage == 3:
        motion_scale = 1.5
        target_hold_steps = 100
    elif stage == 4:
        motion_scale = 2.0
        target_hold_steps = 80
    else:  # stage >= 5
        motion_scale = 2.5  # Full speed
        target_hold_steps = 60
    
    controller = ARX5IKController(
        num_envs=num_envs,
        device=device,
        **sphere_center_cfg,
        **workspace_cfg,
        target_hold_steps=target_hold_steps,
        motion_scale=motion_scale,
    )
    
    print(f"[create_ik_arm_controller] Created for stage {stage}")
    print(f"  Motion scale: {motion_scale}")
    
    return controller
