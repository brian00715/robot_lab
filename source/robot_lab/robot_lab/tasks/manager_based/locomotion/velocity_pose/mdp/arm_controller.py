# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
ARX5 Arm Trajectory Controller for Stage 1 Training

This controller generates predefined trajectories for the ARX5 arm to create
diverse disturbances for the dog to learn robust locomotion control.

The arm acts as a "disturbance source" - Policy does not control it, but must
learn to compensate for the arm's motion to maintain balance and tracking.
"""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ARX5TrajectoryController:
    """Predefined trajectory controller for ARX5 arm (Stage 1).
    
    Generates diverse arm motions to create controllable disturbances for
    the dog policy to learn robust compensation strategies.
    
    Features:
    - 5 motion modes: circular, figure_eight, sinusoidal, random_walk, reach_points
    - Per-environment random mode assignment
    - Configurable frequency and amplitude
    - Smooth transitions to avoid sudden jerks
    """
    
    def __init__(
        self,
        num_envs: int,
        device: str,
        motion_scale: float = 0.3,
        frequency_range: tuple = (0.2, 0.5),
        amplitude_range: tuple = (0.1, 0.3),
        dt: float = 0.02,  # 50Hz control frequency
        seed: int = 42,
    ):
        """Initialize the trajectory controller.
        
        Args:
            num_envs: Number of parallel environments.
            device: Device to run on ('cuda' or 'cpu').
            motion_scale: Overall motion scale factor (0.3 = 30% speed).
            frequency_range: Range of motion frequencies (Hz).
            amplitude_range: Range of motion amplitudes (radians).
            dt: Time step (seconds).
            seed: Random seed for reproducibility.
        """
        self.num_envs = num_envs
        self.device = device
        self.motion_scale = motion_scale
        self.dt = dt
        
        # Random generator for reproducibility
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        
        # Motion modes (one per environment)
        self.motion_modes = self._assign_motion_modes()
        
        # Per-environment parameters
        self.frequencies = torch.rand(num_envs, device=device, generator=self.rng) * \
                          (frequency_range[1] - frequency_range[0]) + frequency_range[0]
        self.amplitudes = torch.rand(num_envs, device=device, generator=self.rng) * \
                         (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
        
        # Phase offsets for variety
        self.phase_offsets = torch.rand(num_envs, device=device, generator=self.rng) * 2 * math.pi
        
        # Current timestep counter
        self.timesteps = torch.zeros(num_envs, device=device, dtype=torch.long)
        
        # Random walk state (for random_walk mode)
        self.random_walk_targets = torch.zeros(num_envs, 6, device=device)
        self.random_walk_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        
        # Reach points state (for reach_points mode)
        self.reach_points_targets = torch.zeros(num_envs, 6, device=device)
        self.reach_points_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        
        # Debug info (DISABLED for performance)
        # print(f"[ARX5Controller] Initialized with {num_envs} envs, scale={motion_scale}")
        # print(f"[ARX5Controller] Motion modes: {dict(zip(*torch.unique(torch.tensor([self.motion_modes.index(m) for m in self.motion_modes]), return_counts=True)))}")
    
    def _assign_motion_modes(self) -> list[str]:
        """Assign random motion modes to each environment.
        
        Returns:
            List of motion mode strings (one per environment).
        """
        modes = ["circular", "figure_eight", "sinusoidal", "random_walk", "reach_points"]
        # Cyclically assign modes to ensure diversity
        return [modes[i % len(modes)] for i in range(self.num_envs)]
    
    def reset_idx(self, env_ids: torch.Tensor):
        """Reset controller state for specific environments.
        
        Args:
            env_ids: Indices of environments to reset.
        """
        self.timesteps[env_ids] = 0
        self.phase_offsets[env_ids] = torch.rand(
            len(env_ids), device=self.device, generator=self.rng
        ) * 2 * math.pi
        self.random_walk_targets[env_ids] = 0
        self.random_walk_steps[env_ids] = 0
        self.reach_points_targets[env_ids] = 0
        self.reach_points_steps[env_ids] = 0
    
    def generate_arm_action(self, env: ManagerBasedRLEnv | None = None) -> torch.Tensor:
        """Generate arm joint target positions for current timestep.
        
        Args:
            env: Optional environment instance (for future use).
        
        Returns:
            Arm actions (num_envs, 6) - target joint positions in radians.
        """
        # Debug: Log first call (DISABLED for performance)
        # if not hasattr(self, '_generate_debug_logged'):
        #     self._generate_debug_logged = True
        #     print(f"\n{'='*80}")
        #     print("[DEBUG] ARX5Controller.generate_arm_action() first call:")
        #     print(f"  num_envs: {self.num_envs}")
        #     print(f"  motion_scale: {self.motion_scale}")
        #     print(f"  dt: {self.dt}")
        #     print(f"  timesteps: {self.timesteps[:5]}")  # First 5 envs
        #     print(f"{'='*80}\n")
        
        # CRITICAL: In Stage 1, keep arm fixed at zero position (no motion)
        # This prevents arm swinging due to base motion and helps dog learn stable locomotion
        if env is not None and hasattr(env, "_curriculum_stage"):
            current_stage = env._curriculum_stage
            if current_stage == 1:
                # Return zero actions - arm stays at initial position (all joints at 0)
                return torch.zeros((self.num_envs, 6), device=self.device)
        
        arm_actions = torch.zeros((self.num_envs, 6), device=self.device)
        
        # Compute time in seconds
        time = self.timesteps.float() * self.dt
        
        # Generate actions based on mode assignment (VECTORIZED for performance)
        # Create mode masks for batch processing
        mode_indices = {
            "circular": 0,
            "figure_eight": 1,
            "sinusoidal": 2,
            "random_walk": 3,
            "reach_points": 4
        }
        
        # Convert motion_modes list to tensor indices
        mode_tensor = torch.tensor([mode_indices[m] for m in self.motion_modes], 
                                   device=self.device, dtype=torch.long)
        
        # Process each mode in batch
        for mode_name, mode_idx in mode_indices.items():
            mask = (mode_tensor == mode_idx)
            if not mask.any():
                continue
                
            env_indices = torch.where(mask)[0]
            
            if mode_name == "circular":
                arm_actions[mask] = self._circular_motion_batch(
                    time[mask], 
                    self.frequencies[mask], 
                    self.amplitudes[mask], 
                    self.phase_offsets[mask]
                )
            elif mode_name == "figure_eight":
                arm_actions[mask] = self._figure_eight_motion_batch(
                    time[mask], 
                    self.frequencies[mask], 
                    self.amplitudes[mask], 
                    self.phase_offsets[mask]
                )
            elif mode_name == "sinusoidal":
                arm_actions[mask] = self._sinusoidal_motion_batch(
                    time[mask], 
                    self.frequencies[mask], 
                    self.amplitudes[mask], 
                    self.phase_offsets[mask]
                )
            elif mode_name == "random_walk":
                arm_actions[mask] = self._random_walk_motion_batch(env_indices)
            elif mode_name == "reach_points":
                arm_actions[mask] = self._reach_points_motion_batch(env_indices)
        
        # Apply overall motion scale
        arm_actions = arm_actions * self.motion_scale
        
        # Increment timesteps
        self.timesteps += 1
        
        return arm_actions
    
    def _circular_motion_batch(self, t: torch.Tensor, freq: torch.Tensor,
                               amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Generate circular motion trajectory (BATCH VERSION).
        
        Args:
            t: Current time for each env (N,).
            freq: Motion frequency for each env (N,).
            amp: Motion amplitude for each env (N,).
            phase: Phase offset for each env (N,).
        
        Returns:
            Joint positions (N, 6).
        """
        N = t.shape[0]
        angle = 2 * math.pi * freq * t + phase  # (N,)
        
        # Circular motion in joints 1-3
        joint1 = amp * torch.sin(angle)           # Base rotation
        joint2 = amp * torch.cos(angle)           # Shoulder
        joint3 = amp * torch.sin(angle * 2)       # Elbow (double frequency)
        joint4 = torch.zeros(N, device=self.device)  # Wrist 1
        joint5 = amp * 0.5 * torch.cos(angle)     # Wrist 2
        joint6 = torch.zeros(N, device=self.device)  # Wrist 3
        
        return torch.stack([joint1, joint2, joint3, joint4, joint5, joint6], dim=1)
    
    def _figure_eight_motion_batch(self, t: torch.Tensor, freq: torch.Tensor,
                                   amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Generate figure-eight motion trajectory (BATCH VERSION).
        
        Args:
            t: Current time for each env (N,).
            freq: Motion frequency for each env (N,).
            amp: Motion amplitude for each env (N,).
            phase: Phase offset for each env (N,).
        
        Returns:
            Joint positions (N, 6).
        """
        N = t.shape[0]
        angle = 2 * math.pi * freq * t + phase  # (N,)
        
        # Figure-eight using Lissajous curves
        joint1 = amp * torch.sin(angle)
        joint2 = amp * torch.sin(2 * angle)
        joint3 = amp * torch.cos(angle)
        joint4 = amp * 0.3 * torch.sin(angle + math.pi/2)
        joint5 = amp * 0.5 * torch.cos(2 * angle)
        joint6 = torch.zeros(N, device=self.device)
        
        return torch.stack([joint1, joint2, joint3, joint4, joint5, joint6], dim=1)
    
    def _sinusoidal_motion_batch(self, t: torch.Tensor, freq: torch.Tensor,
                                 amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal motion trajectory (BATCH VERSION).
        
        Args:
            t: Current time for each env (N,).
            freq: Motion frequency for each env (N,).
            amp: Motion amplitude for each env (N,).
            phase: Phase offset for each env (N,).
        
        Returns:
            Joint positions (N, 6).
        """
        N = t.shape[0]
        phase_offsets = torch.tensor([0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3],
                                    device=self.device).unsqueeze(0)  # (1, 6)
        
        # Broadcast angle calculation: (N, 1) + (1, 6) -> (N, 6)
        angle = (2 * math.pi * freq * t + phase).unsqueeze(1) + phase_offsets
        
        # Amplitude reduction per joint: (1, 6)
        amp_factors = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5], device=self.device).unsqueeze(0)
        
        # (N, 1) * (1, 6) -> (N, 6)
        joint_amp = amp.unsqueeze(1) * amp_factors
        
        return joint_amp * torch.sin(angle)
    
    def _random_walk_motion_batch(self, env_indices: torch.Tensor) -> torch.Tensor:
        """Generate random walk trajectory (BATCH VERSION).
        
        Args:
            env_indices: Environment indices with this mode (N,).
        
        Returns:
            Joint positions (N, 6).
        """
        N = len(env_indices)
        result = torch.zeros(N, 6, device=self.device)
        
        for idx_in_batch, env_id in enumerate(env_indices):
            env_id_int = env_id.item()  # Only one .item() per env with this mode
            
            # Check if we need a new target
            if self.random_walk_steps[env_id_int] == 0 or self.random_walk_steps[env_id_int] >= 100:
                self.random_walk_targets[env_id_int] = torch.randn(6, device=self.device) * 0.5
                self.random_walk_steps[env_id_int] = 0
            
            # Interpolate towards target (vectorized per env)
            alpha = min(self.random_walk_steps[env_id_int].item() / 100.0, 1.0)
            result[idx_in_batch] = self.random_walk_targets[env_id_int] * alpha
            
            self.random_walk_steps[env_id_int] += 1
        
        return result
    
    def _reach_points_motion_batch(self, env_indices: torch.Tensor) -> torch.Tensor:
        """Generate reach-to-points trajectory (BATCH VERSION).
        
        Args:
            env_indices: Environment indices with this mode (N,).
        
        Returns:
            Joint positions (N, 6).
        """
        N = len(env_indices)
        result = torch.zeros(N, 6, device=self.device)
        
        # Define target configurations (shared across all envs)
        target_configs = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.3, 0.4, 0.0, 0.2, 0.0],
            [-0.5, 0.3, 0.4, 0.0, 0.2, 0.0],
            [0.0, 0.5, 0.6, 0.2, 0.3, 0.0],
            [0.3, -0.2, -0.3, 0.0, -0.2, 0.0],
            [-0.3, -0.2, -0.3, 0.0, -0.2, 0.0],
        ], device=self.device)
        
        for idx_in_batch, env_id in enumerate(env_indices):
            env_id_int = env_id.item()  # Only one .item() per env with this mode
            
            # Check if we need a new target
            if self.reach_points_steps[env_id_int] == 0 or self.reach_points_steps[env_id_int] >= 150:
                target_idx = int((self.reach_points_steps[env_id_int] // 150) % len(target_configs))
                self.reach_points_targets[env_id_int] = target_configs[target_idx]
                self.reach_points_steps[env_id_int] = 0
            
            # Smooth interpolation
            alpha = min(self.reach_points_steps[env_id_int].item() / 150.0, 1.0)
            alpha_smooth = 3 * alpha**2 - 2 * alpha**3  # Smoothstep
            result[idx_in_batch] = self.reach_points_targets[env_id_int] * alpha_smooth
            
            self.reach_points_steps[env_id_int] += 1
        
        return result
    
    def update_curriculum(self, stage: int):
        """Update motion parameters based on curriculum stage.
        
        Args:
            stage: Current training stage (1-4).
        """
        if stage == 1:
            # Stage 1: Conservative motion (30% speed)
            self.motion_scale = 0.3
            print(f"[ARX5Controller] Stage 1: motion_scale = {self.motion_scale}")
        elif stage == 2:
            # Stage 2: Moderate motion (60% speed)
            self.motion_scale = 0.6
            print(f"[ARX5Controller] Stage 2: motion_scale = {self.motion_scale}")
        elif stage == 3:
            # Stage 3: Active motion (90% speed)
            self.motion_scale = 0.9
            print(f"[ARX5Controller] Stage 3: motion_scale = {self.motion_scale}")
        else:
            # Stage 4+: Full speed
            self.motion_scale = 1.0
            print(f"[ARX5Controller] Stage 4: motion_scale = {self.motion_scale}")
    
    def get_motion_info(self) -> dict:
        """Get information about current motion state.
        
        Returns:
            Dictionary with motion statistics.
        """
        return {
            "motion_scale": self.motion_scale,
            "mean_frequency": self.frequencies.mean().item(),
            "mean_amplitude": self.amplitudes.mean().item(),
            "timestep": self.timesteps.float().mean().item(),
        }


def create_arm_controller(
    num_envs: int,
    device: str,
    stage: int = 1,
) -> ARX5TrajectoryController:
    """Factory function to create an arm trajectory controller.
    
    Args:
        num_envs: Number of parallel environments.
        device: Device to run on.
        stage: Initial curriculum stage (1-4).
    
    Returns:
        Initialized ARX5TrajectoryController.
    """
    # Stage-specific parameters - INCREASED RANGE for more visible arm motion
    if stage == 1:
        motion_scale = 0 
        frequency_range = (0.3, 0.8)  
        amplitude_range = (0.3, 0.6)  
    elif stage == 2:
        motion_scale = 1.2  
        frequency_range = (0.5, 1.2)  
        amplitude_range = (0.4, 0.8) 
    elif stage == 3:
        motion_scale = 1.8  
        frequency_range = (0.8, 1.8)  
        amplitude_range = (0.5, 1.0)  
    else:  # stage >= 4
        motion_scale = 2.5 
        frequency_range = (1.0, 2.5) 
        amplitude_range = (0.6, 1.2)
    
    controller = ARX5TrajectoryController(
        num_envs=num_envs,
        device=device,
        motion_scale=motion_scale,
        frequency_range=frequency_range,
        amplitude_range=amplitude_range,
    )
    
    return controller
