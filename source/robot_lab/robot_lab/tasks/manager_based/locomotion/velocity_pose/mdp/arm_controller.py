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
        
        print(f"[ARX5Controller] Initialized with {num_envs} envs, scale={motion_scale}")
        print(f"[ARX5Controller] Motion modes: {dict(zip(*torch.unique(torch.tensor([self.motion_modes.index(m) for m in self.motion_modes]), return_counts=True)))}")
    
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
        arm_actions = torch.zeros((self.num_envs, 6), device=self.device)
        
        # Compute time in seconds
        time = self.timesteps.float() * self.dt
        
        # Generate actions for each environment based on its mode
        for i in range(self.num_envs):
            mode = self.motion_modes[i]
            freq = self.frequencies[i].item()
            amp = self.amplitudes[i].item()
            t = time[i].item()
            phase = self.phase_offsets[i].item()
            
            if mode == "circular":
                arm_actions[i] = self._circular_motion(t, freq, amp, phase)
            elif mode == "figure_eight":
                arm_actions[i] = self._figure_eight_motion(t, freq, amp, phase)
            elif mode == "sinusoidal":
                arm_actions[i] = self._sinusoidal_motion(t, freq, amp, phase)
            elif mode == "random_walk":
                arm_actions[i] = self._random_walk_motion(i, t)
            elif mode == "reach_points":
                arm_actions[i] = self._reach_points_motion(i, t)
        
        # Apply overall motion scale
        arm_actions = arm_actions * self.motion_scale
        
        # Increment timesteps
        self.timesteps += 1
        
        return arm_actions
    
    def _circular_motion(self, t: float, freq: float, amp: float, phase: float) -> torch.Tensor:
        """Generate circular motion trajectory.
        
        The arm moves in a circular pattern, creating lateral and forward/backward
        shifts in the center of mass.
        
        Args:
            t: Current time (seconds).
            freq: Motion frequency (Hz).
            amp: Motion amplitude (radians).
            phase: Phase offset (radians).
        
        Returns:
            Joint positions (6,).
        """
        angle = 2 * math.pi * freq * t + phase
        
        # Circular motion in joints 1-3 (base, shoulder, elbow)
        joint1 = amp * math.sin(angle)           # Base rotation
        joint2 = amp * math.cos(angle)           # Shoulder
        joint3 = amp * math.sin(angle * 2)       # Elbow (double frequency)
        joint4 = 0.0                              # Wrist 1 (minimal motion)
        joint5 = amp * 0.5 * math.cos(angle)     # Wrist 2 (half amplitude)
        joint6 = 0.0                              # Wrist 3 (minimal motion)
        
        return torch.tensor([joint1, joint2, joint3, joint4, joint5, joint6], 
                           device=self.device, dtype=torch.float32)
    
    def _figure_eight_motion(self, t: float, freq: float, amp: float, phase: float) -> torch.Tensor:
        """Generate figure-eight (∞) motion trajectory.
        
        Creates a complex 3D motion pattern that significantly shifts the CoM.
        
        Args:
            t: Current time (seconds).
            freq: Motion frequency (Hz).
            amp: Motion amplitude (radians).
            phase: Phase offset (radians).
        
        Returns:
            Joint positions (6,).
        """
        angle = 2 * math.pi * freq * t + phase
        
        # Figure-eight using Lissajous curves
        joint1 = amp * math.sin(angle)                    # Base: sin(θ)
        joint2 = amp * math.sin(2 * angle)                # Shoulder: sin(2θ)
        joint3 = amp * math.cos(angle)                    # Elbow: cos(θ)
        joint4 = amp * 0.3 * math.sin(angle + math.pi/2)  # Wrist 1: 90° phase
        joint5 = amp * 0.5 * math.cos(2 * angle)          # Wrist 2: cos(2θ)
        joint6 = 0.0                                       # Wrist 3: stable
        
        return torch.tensor([joint1, joint2, joint3, joint4, joint5, joint6], 
                           device=self.device, dtype=torch.float32)
    
    def _sinusoidal_motion(self, t: float, freq: float, amp: float, phase: float) -> torch.Tensor:
        """Generate sinusoidal motion trajectory.
        
        Simple periodic motion with different phases for each joint.
        
        Args:
            t: Current time (seconds).
            freq: Motion frequency (Hz).
            amp: Motion amplitude (radians).
            phase: Phase offset (radians).
        
        Returns:
            Joint positions (6,).
        """
        # Each joint has a different phase offset
        phase_offsets = [0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3]
        
        joints = []
        for i, p in enumerate(phase_offsets):
            angle = 2 * math.pi * freq * t + phase + p
            # Vary amplitude by joint
            joint_amp = amp * (1.0 - i * 0.1)  # Gradually reduce amplitude
            joints.append(joint_amp * math.sin(angle))
        
        return torch.tensor(joints, device=self.device, dtype=torch.float32)
    
    def _random_walk_motion(self, env_id: int, t: float) -> torch.Tensor:
        """Generate random walk trajectory.
        
        The arm randomly selects target positions and smoothly moves towards them.
        
        Args:
            env_id: Environment index.
            t: Current time (seconds).
        
        Returns:
            Joint positions (6,).
        """
        # Check if we need a new target (every 100 steps ≈ 2 seconds)
        if self.random_walk_steps[env_id] == 0 or self.random_walk_steps[env_id] >= 100:
            # Generate new random target
            self.random_walk_targets[env_id] = torch.randn(6, device=self.device) * 0.5
            self.random_walk_steps[env_id] = 0
        
        # Interpolate towards target
        alpha = min(self.random_walk_steps[env_id].item() / 100.0, 1.0)
        current_target = self.random_walk_targets[env_id] * alpha
        
        self.random_walk_steps[env_id] += 1
        
        return current_target
    
    def _reach_points_motion(self, env_id: int, t: float) -> torch.Tensor:
        """Generate reach-to-points trajectory.
        
        The arm reaches to predefined spatial points in sequence.
        
        Args:
            env_id: Environment index.
            t: Current time (seconds).
        
        Returns:
            Joint positions (6,).
        """
        # Define a set of "interesting" joint configurations
        # These are roughly IK solutions for different end-effector positions
        target_configs = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # Home position
            [0.5, 0.3, 0.4, 0.0, 0.2, 0.0],      # Reach forward-right
            [-0.5, 0.3, 0.4, 0.0, 0.2, 0.0],     # Reach forward-left
            [0.0, 0.5, 0.6, 0.2, 0.3, 0.0],      # Reach up
            [0.3, -0.2, -0.3, 0.0, -0.2, 0.0],   # Reach down-right
            [-0.3, -0.2, -0.3, 0.0, -0.2, 0.0],  # Reach down-left
        ], device=self.device)
        
        # Check if we need a new target (every 150 steps ≈ 3 seconds)
        if self.reach_points_steps[env_id] == 0 or self.reach_points_steps[env_id] >= 150:
            # Cycle through target configurations
            target_idx = (self.reach_points_steps[env_id] // 150) % len(target_configs)
            self.reach_points_targets[env_id] = target_configs[target_idx]
            self.reach_points_steps[env_id] = 0
        
        # Smooth interpolation towards target
        alpha = min(self.reach_points_steps[env_id].item() / 150.0, 1.0)
        # Use ease-in-out for smoother motion
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3  # Smoothstep function
        current_target = self.reach_points_targets[env_id] * alpha_smooth
        
        self.reach_points_steps[env_id] += 1
        
        return current_target
    
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
    # Stage-specific parameters
    if stage == 1:
        motion_scale = 0.3
        frequency_range = (0.2, 0.5)
        amplitude_range = (0.1, 0.3)
    elif stage == 2:
        motion_scale = 0.6
        frequency_range = (0.5, 1.0)
        amplitude_range = (0.2, 0.4)
    elif stage == 3:
        motion_scale = 0.9
        frequency_range = (0.5, 1.5)
        amplitude_range = (0.3, 0.5)
    else:  # stage >= 4
        motion_scale = 1.0
        frequency_range = (0.5, 2.0)
        amplitude_range = (0.3, 0.6)
    
    controller = ARX5TrajectoryController(
        num_envs=num_envs,
        device=device,
        motion_scale=motion_scale,
        frequency_range=frequency_range,
        amplitude_range=amplitude_range,
    )
    
    return controller
