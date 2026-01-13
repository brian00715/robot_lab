# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Curriculum learning functions for VelocityPose task with stage-based progression.

This module implements a fixed-iteration stage-based curriculum:
- Stage 1 (0-20k iterations): Height and pose commands fixed at default (ranges = 0)
- Stage 2 (20k-40k iterations): Small range for height and pose commands
- Stage 3 (40k-60k+ iterations): Large range for height and pose commands

The curriculum automatically tracks total iterations across training sessions,
so --resume will correctly continue from the accumulated iteration count.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_velocity_pose(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term_name: str
) -> torch.Tensor:
    """Curriculum that updates the terrain levels based on velocity and pose tracking performance.
    
    This adapts the original terrain curriculum to also consider height and orientation tracking.
    """
    # Get the current terrain levels
    terrain_levels = env.terrain_levels.float()
    
    # Get tracking rewards
    vel_tracking_reward = env.reward_manager._episode_sums.get("track_lin_vel_xy_exp", 0.0)
    ang_tracking_reward = env.reward_manager._episode_sums.get("track_ang_vel_z_exp", 0.0)
    height_tracking_reward = env.reward_manager._episode_sums.get("track_height_exp", 0.0)
    pose_tracking_reward = env.reward_manager._episode_sums.get("track_orientation_exp", 0.0)
    
    # Combine all tracking rewards (weighted average)
    total_tracking = (
        vel_tracking_reward * 0.4 + 
        ang_tracking_reward * 0.2 + 
        height_tracking_reward * 0.2 + 
        pose_tracking_reward * 0.2
    )
    
    # Normalize by episode length and get weight
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    normalized_reward = total_tracking[env_ids] / env.max_episode_length_s
    
    # Increase terrain level if tracking is good (>80% of max reward)
    increase_level = normalized_reward > 0.8 * (
        reward_term_cfg.weight * 0.4 +  # vel weight portion
        2.0 * 0.2 +  # height weight (from config)
        1.0 * 0.2    # pose weight (from config)
    )
    
    # Decrease terrain level if robot dies (episode ended but not timeout)
    decrease_level = ~env.termination_manager.get_term("time_out").time_out[env_ids]
    
    # Update terrain levels
    terrain_levels[env_ids] += increase_level.float()
    terrain_levels[env_ids] -= decrease_level.float()
    
    # Clamp terrain levels to valid range
    terrain_levels[env_ids] = torch.clamp(terrain_levels[env_ids], min=0)
    
    return terrain_levels[env_ids]


def command_curriculum_height_pose(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity_pose",
) -> torch.Tensor:
    """Stage-based curriculum for height and pose commands with fixed iteration thresholds.
    
    This curriculum implements a 3-stage progression based on total training iterations:
    
    Stage 1 (0-14,500 iterations):
        - Height range: 0.33m (fixed, ±0cm)
        - Roll range: 0° (fixed)
        - Pitch range: 0° (fixed)
        - Focus: Learn basic locomotion on flat/rough terrain
    
    Stage 2 (14,500-25,000 iterations):
        - Height range: [0.30m, 0.36m] (±3cm from default)
        - Roll range: [-8°, +8°] (±0.14 rad)
        - Pitch range: 0° (keep fixed, introduce roll first)
        - Focus: Learn height adjustment and lateral balance
    
    Stage 3 (25,000-40,000+ iterations):
        - Height range: [0.26m, 0.40m] (±7cm, near physical limits)
        - Roll range: [-10°, +10°] (±0.17 rad)
        - Pitch range: [-8°, +8°] (±0.14 rad)
        - Focus: Master full pose control within safe limits
    
    The curriculum automatically handles --resume by tracking env.common_step_counter,
    which persists across training sessions.
    
    Args:
        env: The RL environment instance
        env_ids: Environment IDs to update (not used, curriculum is global)
        command_name: Name of the command term (default: "base_velocity_pose")
    
    Returns:
        Current stage number as a tensor for logging (1, 2, or 3)
    """
    # Get command manager ranges
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges
    default_height = command_term.default_height
    
    # Calculate total iterations from common_step_counter
    # common_step_counter tracks total steps across all training sessions
    steps_per_iteration = env.num_envs * env.max_episode_length
    total_iterations = env.common_step_counter // steps_per_iteration
    
    # Initialize curriculum state on first call
    if not hasattr(env, "_curriculum_stage"):
        env._curriculum_stage = 0
        env._curriculum_last_update = 0
        print(f"\n{'='*80}")
        print(f"[Curriculum] Initialized at iteration {total_iterations}")
        print(f"{'='*80}\n")
    
    # Determine current stage based on total iterations
    # TEMPORARY: Skipping Stage 1 as we're resuming from 14.5k checkpoint
    if total_iterations < 0:  # Changed from 14500 to 0, effectively skip Stage 1
        target_stage = 1
        height_range = (default_height, default_height)  # Fixed at 0.33m
        roll_range = (0.0, 0.0)  # Fixed at 0°
        pitch_range = (0.0, 0.0)  # Fixed at 0°
    elif total_iterations < 10500:  # Stage 2 duration: 10,500 iterations (was 14.5k-25k)
        target_stage = 2
        height_range = (0.30, 0.36)  # ±3cm
        roll_range = (-0.14, 0.14)  # ±8°
        pitch_range = (0.0, 0.0)  # Keep pitch fixed
    else:  # >= 10500, Stage 3
        target_stage = 3
        height_range = (0.26, 0.40)  # ±7cm (near limits)
        roll_range = (-0.17, 0.17)  # ±10°
        pitch_range = (-0.14, 0.14)  # ±8°
    
    # Update ranges if stage changed
    if target_stage != env._curriculum_stage:
        env._curriculum_stage = target_stage
        env._curriculum_last_update = total_iterations
        
        # Update command ranges
        ranges.height = height_range
        ranges.roll = roll_range
        ranges.pitch = pitch_range
        
        # Print stage transition message
        print(f"\n{'='*80}")
        print(f"[Curriculum] Stage Transition at Iteration {total_iterations}")
        print(f"{'='*80}")
        print(f"  New Stage: {target_stage}")
        print(f"  Height Range: [{height_range[0]:.3f}, {height_range[1]:.3f}] m")
        print(f"  Roll Range:   [{roll_range[0]:.3f}, {roll_range[1]:.3f}] rad = [{math.degrees(roll_range[0]):.1f}, {math.degrees(roll_range[1]):.1f}]°")
        print(f"  Pitch Range:  [{pitch_range[0]:.3f}, {pitch_range[1]:.3f}] rad = [{math.degrees(pitch_range[0]):.1f}, {math.degrees(pitch_range[1]):.1f}]°")
        print(f"{'='*80}\n")
    
    # Log current stage every 1000 iterations
    if total_iterations > 0 and total_iterations % 1000 == 0 and total_iterations != env._curriculum_last_update:
        iterations_in_stage = total_iterations - (
            0 if target_stage == 1 else
            14500 if target_stage == 2 else
            25000
        )
        stage_total = (
            14500 if target_stage == 1 else
            10500 if target_stage == 2 else
            15000  # Stage 3 is open-ended but set 15000 as reference
        )
        progress = min(100.0, (iterations_in_stage / stage_total) * 100)
        
        print(f"[Curriculum] Stage {target_stage}: Iteration {total_iterations} "
              f"(Stage progress: {iterations_in_stage}/{stage_total} = {progress:.1f}%)")
    
    # Return current stage as scalar for logging (take mean of all envs, which are all the same)
    return torch.tensor(float(target_stage), device=env.device)


# Legacy function names for backward compatibility (if needed)
def command_levels_height(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity_pose",
) -> torch.Tensor:
    """Legacy wrapper for height curriculum. Use command_curriculum_height_pose instead."""
    return command_curriculum_height_pose(env, env_ids, command_name)


def command_levels_orientation(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity_pose",
) -> torch.Tensor:
    """Legacy wrapper for orientation curriculum. Use command_curriculum_height_pose instead."""
    return command_curriculum_height_pose(env, env_ids, command_name)


# Import math for degree conversion in print statements
import math
