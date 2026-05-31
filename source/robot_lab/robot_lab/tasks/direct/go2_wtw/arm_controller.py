# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
ARX5 Arm Trajectory Controller for Curriculum Training

This controller generates predefined trajectories for the ARX5 arm to create
diverse disturbances for the dog to learn robust locomotion control.

The arm acts as a "disturbance source" - Policy does not control it, but must
learn to compensate for the arm's motion to maintain balance and tracking.

Motion Modes (9 total):
1. Circular - Smooth circular trajectories
2. Figure-Eight - Lissajous curves forming figure-8 patterns
3. Sinusoidal - Multi-joint sinusoidal waves with phase offsets
4. Random Walk - Brownian motion with smooth interpolation
5. Reach Points - Sequential reaching to 6 predefined target poses
6. Fishing - Extend→hold→retract (simulates fishing/casting, 6s cycles)
7. Grasping - Quick reach→grasp→retract (simulates pick-and-place, 3s cycles)
8. Swinging - Pendulum-like motion (simulates swinging loads, continuous)
9. Probing - Quick poke in random directions (simulates exploration, 1.2s cycles)

Modes 6-9 specifically simulate realistic manipulation tasks with fishing-like
motions, providing diverse and task-relevant disturbances for robust training.
"""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


class ARX5TrajectoryController:
    """Predefined trajectory controller for ARX5 arm with 9 diverse motion modes.
    
    Generates diverse arm motions to create controllable disturbances for
    the dog policy to learn robust compensation strategies.
    
    Features:
    - 9 motion modes including fishing/grasping simulations
    - Per-environment random mode assignment (cyclic distribution)
    - Configurable frequency and amplitude per environment
    - Smooth transitions to avoid sudden jerks
    - Curriculum-aware motion scaling (0.0 → 1.2 → 1.8 → 2.5 → 3.75)
    
    Motion Modes:
    1-5: Original modes (circular, figure_eight, sinusoidal, random_walk, reach_points)
    6-9: NEW fishing-inspired modes (fishing, grasping, swinging, probing)
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
        fixed_mode_idx: int | None = None,  # NEW: Force all envs to use specific mode
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
            fixed_mode_idx: If set, forces all environments to use this mode index
                           (0=circular, 1=figure_eight, 2=sinusoidal, 3=random_walk,
                            4=reach_points, 5=fishing, 6=grasping, 7=swinging, 8=probing).
        """
        self.num_envs = num_envs
        self.device = device
        self.motion_scale = motion_scale
        self.dt = dt
        self.fixed_mode_idx = fixed_mode_idx  # Store fixed mode index
        
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
        
        # Fishing state (for fishing mode - extend→hold→retract)
        self.fishing_targets = torch.zeros(num_envs, 6, device=device)
        self.fishing_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.fishing_phase = torch.zeros(num_envs, device=device, dtype=torch.long)  # 0=extend, 1=hold, 2=retract
        
        # Grasping state (for grasping mode - quick reach→grasp→quick retract)
        self.grasping_targets = torch.zeros(num_envs, 6, device=device)
        self.grasping_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.grasping_phase = torch.zeros(num_envs, device=device, dtype=torch.long)  # 0=reach, 1=grasp, 2=retract
        
        # Swinging state (for swinging mode - pendulum-like motion)
        self.swinging_angle = torch.zeros(num_envs, device=device)
        self.swinging_direction = torch.ones(num_envs, device=device)  # 1 or -1
        
        # Probing state (for probing mode - quick poke in random directions)
        self.probing_targets = torch.zeros(num_envs, 6, device=device)
        self.probing_steps = torch.zeros(num_envs, device=device, dtype=torch.long)

        # Home pose: all joints at zero (arm extends backward from mount)
        # Matches original Manager-Based implementation which trained successfully.
        self.home_pose = torch.zeros(6, dtype=torch.float32, device=device)  # (6,)
        
        # Debug info (DISABLED for performance)
        # print(f"[ARX5Controller] Initialized with {num_envs} envs, scale={motion_scale}")
        # print(f"[ARX5Controller] Motion modes: {dict(zip(*torch.unique(torch.tensor([self.motion_modes.index(m) for m in self.motion_modes]), return_counts=True)))}")
    
    def _assign_motion_modes(self) -> list[str]:
        """Assign random motion modes to each environment.
        
        Returns:
            List of motion mode strings (one per environment).
        """
        modes = [
            "circular", "figure_eight", "sinusoidal", "random_walk", "reach_points",
            "fishing", "grasping", "swinging", "probing"
        ]
        
        # If fixed mode index is specified, assign that mode to all environments
        if self.fixed_mode_idx is not None:
            if 0 <= self.fixed_mode_idx < len(modes):
                return [modes[self.fixed_mode_idx]] * self.num_envs
            else:
                print(f"[WARNING] Invalid fixed_mode_idx={self.fixed_mode_idx}, using cyclic assignment")
        
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
        
        # Reset new motion mode states
        self.fishing_targets[env_ids] = 0
        self.fishing_steps[env_ids] = 0
        self.fishing_phase[env_ids] = 0
        self.grasping_targets[env_ids] = 0
        self.grasping_steps[env_ids] = 0
        self.grasping_phase[env_ids] = 0
        self.swinging_angle[env_ids] = 0
        self.swinging_direction[env_ids] = 1
        self.probing_targets[env_ids] = 0
        self.probing_steps[env_ids] = 0
    
    def generate_arm_action(self, env: "DirectRLEnv | None" = None) -> torch.Tensor:
        """Generate arm joint target positions for current timestep.

        Args:
            env: Optional environment instance (for curriculum control).

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

        # CRITICAL: Get motion_scale from curriculum (managed in curriculums.py)
        # This allows arm motion to increase with training difficulty
        if env is not None and hasattr(env, "_arm_motion_scale"):
            # Use curriculum-managed motion scale
            current_motion_scale = env._arm_motion_scale

            # In Stage 1 (motion_scale=0), keep arm fixed at safe home pose
            if current_motion_scale == 0.0:
                # Stage 1: keep arm locked at zero position (matches original curriculum)
                return self.home_pose.unsqueeze(0).expand(self.num_envs, -1).clone()

            # Update local motion_scale if changed
            if self.motion_scale != current_motion_scale:
                self.motion_scale = current_motion_scale
        else:
            # Fallback: No curriculum control, use initialization value
            if self.motion_scale == 0.0:
                return self.home_pose.unsqueeze(0).expand(self.num_envs, -1).clone()

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
            "reach_points": 4,
            "fishing": 5,
            "grasping": 6,
            "swinging": 7,
            "probing": 8
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
            elif mode_name == "fishing":
                arm_actions[mask] = self._fishing_motion_batch(env_indices)
            elif mode_name == "grasping":
                arm_actions[mask] = self._grasping_motion_batch(env_indices)
            elif mode_name == "swinging":
                arm_actions[mask] = self._swinging_motion_batch(env_indices, time[mask])
            elif mode_name == "probing":
                arm_actions[mask] = self._probing_motion_batch(env_indices)
        
        # Apply overall motion scale and add home pose as base
        # Trajectories are offsets around home pose (arm pointing up, elbow folded)
        arm_actions = self.home_pose + arm_actions * self.motion_scale
        
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
    
    def _fishing_motion_batch(self, env_indices: torch.Tensor) -> torch.Tensor:
        """Generate fishing motion trajectory (FULLY VECTORIZED).
        
        Simulates fishing rod motion: extend in random direction → hold → retract.
        
        Args:
            env_indices: Environment indices with this mode (N,).
        
        Returns:
            Joint positions (N, 6).
        """
        N = len(env_indices)
        
        # Check which envs need new fishing cycles (vectorized)
        needs_new_cycle = self.fishing_steps[env_indices] == 0
        
        if needs_new_cycle.any():
            # Generate random targets for envs needing new cycles (vectorized)
            num_new = needs_new_cycle.sum().item()
            azimuths = torch.rand(num_new, device=self.device, generator=self.rng) * 2 * math.pi
            elevations = (torch.rand(num_new, device=self.device, generator=self.rng) - 0.3) * 0.8
            reach_distances = 0.5 + torch.rand(num_new, device=self.device, generator=self.rng) * 0.7
            
            # Compute targets (vectorized)
            new_targets = torch.stack([
                azimuths * 0.8,
                elevations + reach_distances * 0.6,
                reach_distances * 0.8,
                torch.zeros_like(azimuths),
                elevations * 0.5,
                azimuths * 0.3,
            ], dim=1)  # (num_new, 6)
            
            # Update targets for envs needing new cycles
            self.fishing_targets[env_indices[needs_new_cycle]] = new_targets
            self.fishing_phase[env_indices[needs_new_cycle]] = 0
        
        # Get current steps (vectorized)
        steps = self.fishing_steps[env_indices].float()  # (N,)
        targets = self.fishing_targets[env_indices]  # (N, 6)
        
        # Phase 0: Extend (0-100 steps) - vectorized
        phase0_mask = steps < 100
        alpha_0 = (steps / 100.0).clamp(0, 1)  # (N,)
        alpha_smooth_0 = 3 * alpha_0**2 - 2 * alpha_0**3
        
        # Phase 1: Hold (100-200 steps) - vectorized
        phase1_mask = (steps >= 100) & (steps < 200)
        wiggle = torch.sin(steps * 0.3) * 0.05  # (N,)
        
        # Phase 2: Retract (200-300 steps) - vectorized
        phase2_mask = steps >= 200
        alpha_2 = ((steps - 200) / 100.0).clamp(0, 1)
        alpha_smooth_2 = 3 * alpha_2**2 - 2 * alpha_2**3
        
        # Compute result (vectorized)
        result = torch.zeros(N, 6, device=self.device)
        result[phase0_mask] = targets[phase0_mask] * alpha_smooth_0[phase0_mask].unsqueeze(1)
        result[phase1_mask] = targets[phase1_mask] * (1.0 + wiggle[phase1_mask].unsqueeze(1))
        result[phase2_mask] = targets[phase2_mask] * (1.0 - alpha_smooth_2[phase2_mask].unsqueeze(1))
        
        # Update steps and phases (vectorized)
        self.fishing_steps[env_indices] += 1
        self.fishing_phase[env_indices[phase0_mask]] = 0
        self.fishing_phase[env_indices[phase1_mask]] = 1
        self.fishing_phase[env_indices[phase2_mask]] = 2
        
        # Reset after full cycle (vectorized)
        cycle_complete = self.fishing_steps[env_indices] >= 300
        self.fishing_steps[env_indices[cycle_complete]] = 0
        
        return result
    
    def _grasping_motion_batch(self, env_indices: torch.Tensor) -> torch.Tensor:
        """Generate grasping motion trajectory (FULLY VECTORIZED).
        
        Simulates object grasping: quick reach → grasp pause → quick retract.
        
        Args:
            env_indices: Environment indices with this mode (N,).
        
        Returns:
            Joint positions (N, 6).
        """
        N = len(env_indices)
        
        # Check which envs need new grasping cycles (vectorized)
        needs_new_cycle = self.grasping_steps[env_indices] == 0
        
        if needs_new_cycle.any():
            # Generate random targets for envs needing new cycles (vectorized)
            num_new = needs_new_cycle.sum().item()
            azimuths = (torch.rand(num_new, device=self.device, generator=self.rng) - 0.5) * 1.5
            reaches = 0.6 + torch.rand(num_new, device=self.device, generator=self.rng) * 0.6
            heights = (torch.rand(num_new, device=self.device, generator=self.rng) - 0.5) * 0.8
            
            # Compute targets (vectorized)
            new_targets = torch.stack([
                azimuths,
                heights + reaches * 0.5,
                reaches * 0.9,
                azimuths * 0.4,
                heights * 0.6,
                -azimuths * 0.5,
            ], dim=1)  # (num_new, 6)
            
            # Update targets for envs needing new cycles
            self.grasping_targets[env_indices[needs_new_cycle]] = new_targets
            self.grasping_phase[env_indices[needs_new_cycle]] = 0
        
        # Get current steps (vectorized)
        steps = self.grasping_steps[env_indices].float()  # (N,)
        targets = self.grasping_targets[env_indices]  # (N, 6)
        
        # Phase 0: Quick reach (0-40 steps) - vectorized
        phase0_mask = steps < 40
        alpha_0 = (steps / 40.0).clamp(0, 1)
        alpha_fast_0 = alpha_0 * alpha_0 * (3.0 - 2.0 * alpha_0)  # Cubic easing
        
        # Phase 1: Grasp hold (40-80 steps) - vectorized
        phase1_mask = (steps >= 40) & (steps < 80)
        tremor = torch.sin(steps * 0.5) * 0.02
        
        # Phase 2: Quick retract (80-150 steps) - vectorized
        phase2_mask = steps >= 80
        alpha_2 = ((steps - 80) / 70.0).clamp(0, 1)
        alpha_fast_2 = alpha_2 * alpha_2 * (3.0 - 2.0 * alpha_2)
        
        # Compute result (vectorized)
        result = torch.zeros(N, 6, device=self.device)
        result[phase0_mask] = targets[phase0_mask] * alpha_fast_0[phase0_mask].unsqueeze(1)
        result[phase1_mask] = targets[phase1_mask] * (1.0 + tremor[phase1_mask].unsqueeze(1))
        result[phase2_mask] = targets[phase2_mask] * (1.0 - alpha_fast_2[phase2_mask].unsqueeze(1))
        
        # Update steps and phases (vectorized)
        self.grasping_steps[env_indices] += 1
        self.grasping_phase[env_indices[phase0_mask]] = 0
        self.grasping_phase[env_indices[phase1_mask]] = 1
        self.grasping_phase[env_indices[phase2_mask]] = 2
        
        # Reset after full cycle (vectorized)
        cycle_complete = self.grasping_steps[env_indices] >= 150
        self.grasping_steps[env_indices[cycle_complete]] = 0
        
        return result
    
    def _swinging_motion_batch(self, env_indices: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Generate swinging motion trajectory (FULLY VECTORIZED).
        
        Simulates pendulum-like swinging motion, like a fishing rod casting.
        
        Args:
            env_indices: Environment indices with this mode (N,).
            t: Current time for each env (N,).
        
        Returns:
            Joint positions (N, 6).
        """
        # Update swing angles (vectorized pendulum motion)
        swing_speeds = 0.05 * self.amplitudes[env_indices]
        self.swinging_angle[env_indices] += swing_speeds * self.swinging_direction[env_indices]
        
        # Reverse direction at limits (vectorized)
        max_angle = 1.2
        exceed_limit = torch.abs(self.swinging_angle[env_indices]) > max_angle
        self.swinging_direction[env_indices[exceed_limit]] *= -1
        self.swinging_angle[env_indices] = torch.clamp(self.swinging_angle[env_indices], -max_angle, max_angle)
        
        # Get current angles (vectorized)
        angles = self.swinging_angle[env_indices]  # (N,)
        abs_angles = torch.abs(angles)
        
        # Convert swing angles to joint positions (vectorized)
        result = torch.stack([
            angles * 0.7,
            abs_angles * 0.5,
            abs_angles * 0.6 + 0.3,
            torch.sin(angles * 2.0) * 0.3,
            angles * 0.4,
            -angles * 0.5,
        ], dim=1)  # (N, 6)
        
        return result
    
    def _probing_motion_batch(self, env_indices: torch.Tensor) -> torch.Tensor:
        """Generate probing motion trajectory (FULLY VECTORIZED).
        
        Simulates quick poke/probe motions in random directions.
        
        Args:
            env_indices: Environment indices with this mode (N,).
        
        Returns:
            Joint positions (N, 6).
        """
        # Check which envs need new probe targets (vectorized)
        needs_new_target = (self.probing_steps[env_indices] == 0) | (self.probing_steps[env_indices] >= 60)
        
        if needs_new_target.any():
            # Generate random directions and reaches (vectorized)
            num_new = needs_new_target.sum().item()
            directions = torch.randn(num_new, 3, device=self.device, generator=self.rng) * 0.5
            reaches = 0.3 + torch.rand(num_new, device=self.device, generator=self.rng) * 0.5
            
            # Compute targets (vectorized)
            new_targets = torch.stack([
                directions[:, 0],
                directions[:, 1] + reaches * 0.4,
                reaches * 0.7,
                directions[:, 2] * 0.6,
                directions[:, 1] * 0.5,
                -directions[:, 0] * 0.4,
            ], dim=1)  # (num_new, 6)
            
            # Update targets for envs needing new probes
            self.probing_targets[env_indices[needs_new_target]] = new_targets
            self.probing_steps[env_indices[needs_new_target]] = 0
        
        # Get current steps (vectorized)
        steps = self.probing_steps[env_indices].float()  # (N,)
        targets = self.probing_targets[env_indices]  # (N, 6)
        
        # Triangular wave: extend (0-30) → retract (30-60) - vectorized
        extend_mask = steps < 30
        alpha = torch.where(
            extend_mask,
            steps / 30.0,  # Extend phase
            (60 - steps) / 30.0  # Retract phase
        ).clamp(0, 1)
        
        result = targets * alpha.unsqueeze(1)
        
        # Update steps (vectorized)
        self.probing_steps[env_indices] += 1

        return result

    def update_curriculum(self, stage: int):
        """[DEPRECATED] Update motion parameters based on curriculum stage.

        NOTE: This method is now deprecated. Curriculum parameters are managed
        centrally in curriculums.py and accessed via env._arm_motion_scale.

        Args:
            stage: Current training stage (1-5).
        """
        # DEPRECATED: Motion scale is now managed by curriculums.py
        # This method is kept for backward compatibility but does nothing
        # The generate_arm_action() method reads from env._arm_motion_scale instead
        pass

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
    fixed_mode_idx: int | None = None,
) -> ARX5TrajectoryController:
    """Factory function to create an arm trajectory controller.

    NOTE: Stage-specific parameters (motion_scale, frequency_range, amplitude_range)
    are now managed centrally in curriculums.py and accessed via env attributes:
    - env._arm_motion_scale
    - env._arm_frequency_range
    - env._arm_amplitude_range

    This function still provides initial values for backward compatibility,
    but during training these will be overridden by curriculum system.

    Args:
        num_envs: Number of parallel environments.
        device: Device to run on.
        stage: Initial curriculum stage (1-5) - used for initialization only.
        fixed_mode_idx: If set, forces all environments to use this mode index.

    Returns:
        Initialized ARX5TrajectoryController.
    """
    # Stage-specific parameters - FOR INITIALIZATION ONLY
    # During training, these are managed by curriculums.py
    if stage == 1:
        motion_scale = 0.0
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
    elif stage == 4:
        motion_scale = 2.5
        frequency_range = (1.0, 2.5)
        amplitude_range = (0.6, 1.2)
    else:  # stage >= 5
        motion_scale = 3.75  # 1.5x Stage 4
        frequency_range = (1.0, 2.5)
        amplitude_range = (0.9, 1.8)  # 1.5x Stage 4

    controller = ARX5TrajectoryController(
        num_envs=num_envs,
        device=device,
        motion_scale=motion_scale,
        frequency_range=frequency_range,
        amplitude_range=amplitude_range,
        fixed_mode_idx=fixed_mode_idx,
    )

    return controller
