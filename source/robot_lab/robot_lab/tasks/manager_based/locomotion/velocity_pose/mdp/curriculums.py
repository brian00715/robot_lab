# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Curriculum learning functions for VelocityPose task with stage-based progression.

This module implements a fixed-iteration stage-based curriculum:
- Stage 1 (0-20,000 iterations): Base training - Height and pose commands fixed at default (roll=0°, pitch=0°, yaw=0°, height=0.33m)
- Stage 2 (20,000-30,000 iterations): Small range for height and pose commands (±3cm height, ±8° roll, pitch=0°, yaw=0°)
- Stage 3 (30,000-45,000 iterations): Medium range for height and pose commands (±10cm height, ±20° roll, ±12° pitch, ±12° yaw)
- Stage 4 (45,000+ iterations): Large range for height and pose commands (±15cm height, ±30° roll, ±15° pitch, ±15° yaw)

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
    
    This curriculum implements a 4-stage progression based on total training iterations:
    
    Stage 1 (0-20,000 iterations):
        - Base training phase
        - Height: 0.33m (fixed at default, ±0cm)
        - Roll: 0° (fixed)
        - Pitch: 0° (fixed)
        - Yaw: 0° (fixed)
        - Focus: Learn basic locomotion without pose control
        - Duration: 20,000 iterations
    
    Stage 2 (20,000-30,000 iterations):
        - Small range introduction
        - Height range: [0.30m, 0.36m] (±3cm from default)
        - Roll range: [-8°, +8°] (±0.14 rad)
        - Pitch: 0° (keep fixed)
        - Yaw: 0° (keep fixed)
        - Focus: Start learning height and roll control
        - Duration: 10,000 iterations
    
    Stage 3 (30,000-45,000 iterations):
        - Medium range expansion
        - Height range: [0.23m, 0.43m] (±10cm)
        - Roll range: [-20°, +20°] (±0.349 rad)
        - Pitch range: [-12°, +12°] (±0.21 rad)
        - Yaw range: [-12°, +12°] (±0.21 rad)
        - Focus: Learn medium-range pose control with pitch and yaw
        - Duration: 15,000 iterations
    
    Stage 4 (45,000+ iterations):
        - Maximum range mastery
        - Height range: [0.18m, 0.48m] (±15cm, maximum range)
        - Roll range: [-30°, +30°] (±0.524 rad, π/6)
        - Pitch range: [-15°, +15°] (±0.262 rad)
        - Yaw range: [-15°, +15°] (±0.262 rad)
        - Focus: Master full pose control at maximum safe limits
        - Duration: Open-ended (continue until performance plateaus)
    
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
    
    # Determine current stage based on total iterations
    # Stage 1: 0-20,000 (Base training with fixed pose)
    # Stage 2: 20,000-30,000 (Small range)
    # Stage 3: 30,000-45,000 (Medium range)
    # Stage 4: 45,000+ (Maximum range)
    if total_iterations < 20000:  # Stage 1: Base training
        target_stage = 1
        height_range = (default_height, default_height)  # Fixed at 0.33m
        roll_range = (0.0, 0.0)  # Fixed at 0°
        pitch_range = (0.0, 0.0)  # Fixed at 0°
        yaw_range = (0.0, 0.0)  # Fixed at 0°
    elif total_iterations < 30000:  # Stage 2: Small range
        target_stage = 2
        height_range = (0.30, 0.36)  # ±3cm
        roll_range = (-0.14, 0.14)  # ±8°
        pitch_range = (0.0, 0.0)  # Keep pitch fixed
        yaw_range = (0.0, 0.0)  # Keep yaw fixed
    elif total_iterations < 45000:  # Stage 3: Medium range (duration: 15,000 iterations)
        target_stage = 3
        height_range = (0.23, 0.43)  # ±10cm
        roll_range = (-0.349, 0.349)  # ±20°
        pitch_range = (-0.21, 0.21)  # ±12°
        yaw_range = (-0.21, 0.21)  # ±12°
    else:  # >= 45000, Stage 4: Maximum range
        target_stage = 4
        height_range = (0.18, 0.48)  # ±15cm (maximum range)
        roll_range = (-0.524, 0.524)  # ±30° (π/6 rad)
        pitch_range = (-0.262, 0.262)  # ±15°
        yaw_range = (-0.262, 0.262)  # ±15°
    
    # Initialize curriculum state on first call, using the target_stage we just determined
    if not hasattr(env, "_curriculum_stage"):
        env._curriculum_stage = target_stage  # Initialize with correct stage instead of 0
        env._curriculum_last_update = 0
        print(f"\n{'='*80}")
        print(f"[Curriculum] Initialized at iteration {total_iterations}")
        print(f"  Starting Stage: {target_stage}")
        print(f"{'='*80}\n")
    
    # Update ranges if stage changed
    if target_stage != env._curriculum_stage:
        env._curriculum_stage = target_stage
        env._curriculum_last_update = total_iterations
        
        # Update command ranges
        ranges.height = height_range
        ranges.roll = roll_range
        ranges.pitch = pitch_range
        ranges.yaw = yaw_range
        
        # Print stage transition message
        print(f"\n{'='*80}")
        print(f"[Curriculum] Stage Transition at Iteration {total_iterations}")
        print(f"{'='*80}")
        print(f"  New Stage: {target_stage}")
        print(f"  Height Range: [{height_range[0]:.3f}, {height_range[1]:.3f}] m")
        print(f"  Roll Range:   [{roll_range[0]:.3f}, {roll_range[1]:.3f}] rad = [{math.degrees(roll_range[0]):.1f}, {math.degrees(roll_range[1]):.1f}]°")
        print(f"  Pitch Range:  [{pitch_range[0]:.3f}, {pitch_range[1]:.3f}] rad = [{math.degrees(pitch_range[0]):.1f}, {math.degrees(pitch_range[1]):.1f}]°")
        print(f"  Yaw Range:    [{yaw_range[0]:.3f}, {yaw_range[1]:.3f}] rad = [{math.degrees(yaw_range[0]):.1f}, {math.degrees(yaw_range[1]):.1f}]°")
        print(f"{'='*80}\n")
    
    # Log current stage every 1000 iterations
    if total_iterations > 0 and total_iterations % 1000 == 0 and total_iterations != env._curriculum_last_update:
        iterations_in_stage = total_iterations - (
            0 if target_stage == 1 else        # Stage 1 starts from 0
            20000 if target_stage == 2 else    # Stage 2 starts from 20000
            30000 if target_stage == 3 else    # Stage 3 starts from 30000
            45000                              # Stage 4 starts from 45000
        )
        stage_total = (
            20000 if target_stage == 1 else    # Stage 1: 0-20000 (20k iterations)
            10000 if target_stage == 2 else    # Stage 2: 20000-30000 (10k iterations)
            15000 if target_stage == 3 else    # Stage 3: 30000-45000 (15k iterations)
            10000                              # Stage 4: 45000+ (estimate 10k iterations)
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
