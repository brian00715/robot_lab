# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Curriculum learning functions for VelocityPose task with stage-based progression.

This module implements a fixed-iteration stage-based curriculum:
- Stage 1 (< 0 iterations): Skipped - Height and pose commands fixed at default
- Stage 2 (< 0 iterations): Skipped - Small range for height and pose commands
- Stage 3 (0-15,000 iterations): Medium range for height and pose commands (±10cm height, ±20° roll, ±12° pitch)
- Stage 4 (15,000-25,000+ iterations): Large range for height and pose commands (±15cm height, ±30° roll, ±15° pitch)

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
    
    This curriculum implements a 2-stage progression based on total training iterations:
    
    Stage 1 (< 0 iterations):
        - SKIPPED
        - Height range: 0.33m (fixed, ±0cm)
        - Roll range: 0° (fixed)
        - Pitch range: 0° (fixed)
    
    Stage 2 (< 0 iterations):
        - SKIPPED
        - Height range: [0.30m, 0.36m] (±3cm from default)
        - Roll range: [-8°, +8°] (±0.14 rad)
        - Pitch range: 0° (keep fixed)
    
    Stage 3 (0-15,000 iterations):
        - Height range: [0.23m, 0.43m] (±10cm)
        - Roll range: [-20°, +20°] (±0.349 rad)
        - Pitch range: [-12°, +12°] (±0.21 rad)
        - Focus: Learn medium-range pose control
        - Duration: 15,000 iterations
    
    Stage 4 (15,000-25,000+ iterations):
        - Height range: [0.18m, 0.48m] (±15cm, maximum range)
        - Roll range: [-30°, +30°] (±0.524 rad, π/6)
        - Pitch range: [-15°, +15°] (±0.262 rad)
        - Focus: Master full pose control at maximum safe limits
        - Duration: 10,000 iterations
    
    The curriculum automatically handles --resume by tracking env.common_step_counter,
    which persists across training sessions.
    
    Args:
        env: The RL environment instance
        env_ids: Environment IDs to update (not used, curriculum is global)
        command_name: Name of the command term (default: "base_velocity_pose")
    
    Returns:
        Current stage number as a tensor for logging (1, 2, 3, or 4)
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
    # Starting directly from Stage 3 (for resuming from model_14500.pt)
    if total_iterations < 0:  # Stage 1: Skipped
        target_stage = 1
        height_range = (default_height, default_height)  # Fixed at 0.33m
        roll_range = (0.0, 0.0)  # Fixed at 0°
        pitch_range = (0.0, 0.0)  # Fixed at 0°
    elif total_iterations < 0:  # Stage 2: Skipped (changed from 10500 to 0)
        target_stage = 2
        height_range = (0.30, 0.36)  # ±3cm
        roll_range = (-0.14, 0.14)  # ±8°
        pitch_range = (0.0, 0.0)  # Keep pitch fixed
    elif total_iterations < 15000:  # Stage 3 duration: 15,000 iterations (0-15k)
        target_stage = 3
        height_range = (0.23, 0.43)  # ±10cm (near limits)
        roll_range = (-0.349, 0.349)  # ±20°
        pitch_range = (-0.21, 0.21)  # ±12°
    else:  # >= 15000, Stage 4
        target_stage = 4
        height_range = (0.18, 0.48)  # ±15cm (maximum range)
        roll_range = (-0.524, 0.524)  # ±30° (π/6 rad)
        pitch_range = (-0.262, 0.262)  # ±15°
    
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
            0 if target_stage == 2 else
            0 if target_stage == 3 else  # Stage 3 starts from 0
            15000  # Stage 4 starts from 15000
        )
        stage_total = (
            0 if target_stage == 1 else  # Stage 1 skipped
            0 if target_stage == 2 else  # Stage 2 skipped
            15000 if target_stage == 3 else  # Stage 3: 0-15000
            10000  # Stage 4: 15000-25000
        )
        progress = min(100.0, (iterations_in_stage / stage_total) * 100) if stage_total > 0 else 0.0
        
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
