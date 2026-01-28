# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Curriculum learning functions for VelocityPose task with stage-based progression.

This module implements a fixed-iteration stage-based curriculum:

- Stage 1 (0-20,000 iterations): Base training - Height and pose commands fixed at default (roll=0°, pitch=0°, yaw=0°, height=0.33m)
- Stage 2 (20,000-30,000 iterations): Small range - Height and pose commands with limited variation (±3cm height, ±8° roll, pitch=0°, yaw=0°)
- Stage 3 (30,000-45,000 iterations): Medium range for height and pose commands (±10cm height, ±20° roll, ±12° pitch, yaw=0°)
- Stage 4 (45,000+ iterations): Large range for height and pose commands (±15cm height, ±30° roll, ±15° pitch, yaw=0°)

NOTE: Yaw is ALWAYS fixed at 0° to decouple from localization systems. Only roll and pitch are tracked,
which can be determined from IMU gravity projection alone without external localization.

The curriculum automatically tracks total iterations across training sessions,
so --resume will correctly continue from the accumulated iteration count.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _update_reward_parameters(env: ManagerBasedRLEnv, stage: int):
    """Update reward parameters (std and weight) based on curriculum stage.
    
    Stage-based reward parameter progression:
    - Stage 1: Pose tracking disabled, upward reward enabled to maintain stability
    - Stage 2: Pose tracking with relaxed tolerance, upward disabled
    - Stage 3: Strict parameters + increased weight
      * Height: std=0.22m, weight=6.0
      * Orientation: std=0.316rad (18°), weight=6.0
    - Stage 4: Very strict parameters + high weight
      * Height: std=0.22m, weight=8.0
      * Orientation: std=0.316rad (18°), weight=8.0
    """
    import math
    
    # Try to get pose tracking reward terms (may not exist in all environments)
    try:
        height_reward_cfg = env.reward_manager.get_term_cfg("track_height_exp")
        orient_reward_cfg = env.reward_manager.get_term_cfg("track_orientation_exp")
    except (AttributeError, KeyError):
        height_reward_cfg = None
        orient_reward_cfg = None
    
    # Try to get locomotion penalty terms
    try:
        lin_vel_z_l2_cfg = env.reward_manager.get_term_cfg("lin_vel_z_l2")
    except (AttributeError, KeyError):
        lin_vel_z_l2_cfg = None
    
    try:
        ang_vel_xy_l2_cfg = env.reward_manager.get_term_cfg("ang_vel_xy_l2")
    except (AttributeError, KeyError):
        ang_vel_xy_l2_cfg = None
    
    # Try to get upward reward (encourages keeping base upright)
    try:
        upward_cfg = env.reward_manager.get_term_cfg("upward")
    except (AttributeError, KeyError):
        upward_cfg = None
    
    # Stage 1: Disable pose tracking rewards, enable locomotion penalties and upward reward
    if stage == 1:
        if height_reward_cfg:
            height_reward_cfg.weight = 0.0
        if orient_reward_cfg:
            orient_reward_cfg.weight = 0.0
        
        # Enable locomotion penalties in Stage 1 only
        if lin_vel_z_l2_cfg:
            lin_vel_z_l2_cfg.weight = -2.0  # Penalize vertical velocity
        if ang_vel_xy_l2_cfg:
            ang_vel_xy_l2_cfg.weight = 0.0  # Don't penalize roll/pitch in Stage 1
        
        # Enable upward reward in Stage 1 to maintain stability during basic locomotion learning
        if upward_cfg:
            upward_cfg.weight = 1.0
    
    # Stage 2: Enable pose tracking with relaxed tolerance, disable locomotion penalties and upward
    elif stage == 2:
        # Height tracking: relaxed tolerance, moderate weight
        if height_reward_cfg:
            height_reward_cfg.params["std"] = math.sqrt(0.25)  # std = 0.5m (very relaxed)
            height_reward_cfg.weight = 4.0  # Increased to overcome upward habit
        
        # Orientation tracking: relaxed tolerance, moderate weight
        if orient_reward_cfg:
            orient_reward_cfg.params["std"] = math.sqrt(0.50)  # std ≈ 0.707rad (40°)
            orient_reward_cfg.weight = 4.0  # Increased to overcome upward habit
        
        # Disable locomotion penalties from Stage 2 onwards
        if lin_vel_z_l2_cfg:
            lin_vel_z_l2_cfg.weight = 0.0
        if ang_vel_xy_l2_cfg:
            ang_vel_xy_l2_cfg.weight = 0.0
        
        # CRITICAL: Disable upward reward from Stage 2 onwards (conflicts with pose tracking)
        if upward_cfg:
            upward_cfg.weight = 0.0
    
    # Stage 3: Strict tracking with high weight, upward remains disabled
    elif stage == 3:
        # Height tracking: strict tolerance, high weight
        if height_reward_cfg:
            height_reward_cfg.params["std"] = math.sqrt(0.05)  # std ≈ 0.22m
            height_reward_cfg.weight = 12.0  # Further increased (2x from Stage 2)
        
        # Orientation tracking: strict tolerance, high weight
        if orient_reward_cfg:
            orient_reward_cfg.params["std"] = math.sqrt(0.10)  # std ≈ 0.316rad (18°)
            orient_reward_cfg.weight = 12.0  # Further increased (2x from Stage 2)
        
        # Keep locomotion penalties disabled
        if lin_vel_z_l2_cfg:
            lin_vel_z_l2_cfg.weight = 0.0
        if ang_vel_xy_l2_cfg:
            ang_vel_xy_l2_cfg.weight = 0.0
        
        # Keep upward disabled
        if upward_cfg:
            upward_cfg.weight = 0.0
    
    # Stage 4: Very strict tracking with very high weight, upward remains disabled
    elif stage == 4:
        # Height tracking: very strict tolerance, very high weight
        if height_reward_cfg:
            height_reward_cfg.params["std"] = math.sqrt(0.05)  # std ≈ 0.22m (same as Stage 3)
            height_reward_cfg.weight = 16.0  # Further increased
        
        # Orientation tracking: very strict tolerance, very high weight
        if orient_reward_cfg:
            orient_reward_cfg.params["std"] = math.sqrt(0.10)  # std ≈ 0.316rad (same as Stage 3)
            orient_reward_cfg.weight = 16.0  # Further increased
        
        # Keep locomotion penalties disabled
        if lin_vel_z_l2_cfg:
            lin_vel_z_l2_cfg.weight = 0.0
        if ang_vel_xy_l2_cfg:
            ang_vel_xy_l2_cfg.weight = 0.0
        
        # Keep upward disabled
        if upward_cfg:
            upward_cfg.weight = 0.0


def _print_reward_parameters(env: ManagerBasedRLEnv):
    """Print current reward parameters for debugging."""
    import math
    
    print("  Reward Parameters:")
    
    # Print pose tracking rewards
    try:
        height_reward_cfg = env.reward_manager.get_term_cfg("track_height_exp")
        height_std = height_reward_cfg.params.get("std", 0.0)
        height_weight = height_reward_cfg.weight
        print(f"    track_height_exp:       weight={height_weight:.1f}, std={height_std:.3f} ({height_std:.2f}m)")
    except (AttributeError, KeyError):
        print("    track_height_exp:       Not configured")
    
    try:
        orient_reward_cfg = env.reward_manager.get_term_cfg("track_orientation_exp")
        orient_std = orient_reward_cfg.params.get("std", 0.0)
        orient_weight = orient_reward_cfg.weight
        print(f"    track_orientation_exp:  weight={orient_weight:.1f}, std={orient_std:.3f} ({math.degrees(orient_std):.1f}°)")
    except (AttributeError, KeyError):
        print("    track_orientation_exp:  Not configured")
    
    # Print locomotion penalty rewards
    try:
        lin_vel_z_cfg = env.reward_manager.get_term_cfg("lin_vel_z_l2")
        lin_vel_z_weight = lin_vel_z_cfg.weight
        print(f"    lin_vel_z_l2:           weight={lin_vel_z_weight:.2f}")
    except (AttributeError, KeyError):
        print("    lin_vel_z_l2:           Not configured")
    
    try:
        ang_vel_xy_cfg = env.reward_manager.get_term_cfg("ang_vel_xy_l2")
        ang_vel_xy_weight = ang_vel_xy_cfg.weight
        print(f"    ang_vel_xy_l2:          weight={ang_vel_xy_weight:.2f}")
    except (AttributeError, KeyError):
        print("    ang_vel_xy_l2:          Not configured")
    
    # Print upward reward (stability vs pose tracking trade-off)
    try:
        upward_cfg = env.reward_manager.get_term_cfg("upward")
        upward_weight = upward_cfg.weight
        print(f"    upward:                 weight={upward_weight:.2f}")
    except (AttributeError, KeyError):
        print("    upward:                 Not configured")


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
        - Roll: 0° (fixed), Pitch: 0° (fixed), Yaw: 0° (fixed)
        - Focus: Learn basic locomotion without pose control
        - Duration: 20,000 iterations
    
    Stage 2 (20,000-30,000 iterations):
        - Small range introduction
        - Height range: [0.30m, 0.36m] (±3cm from default 0.33m)
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
    
    # Calculate total iterations from environment step counter
    # CRITICAL FIX: Use environment's internal tracking to get true training iteration
    # 
    # When --resume is used, we need to get the TRUE iteration count that persists.
    # The environment's metrics or episode tracking can help us calculate this.
    # 
    # Method: Use the logged "Total timesteps" from metrics if available,
    # otherwise fall back to calculating from episode/step counters.
    
    # Try to get iteration from various sources (in priority order):
    if hasattr(env, '_curriculum_manual_iteration'):
        # Manually injected by a wrapper (most reliable)
        total_iterations = env._curriculum_manual_iteration  # type: ignore
        iteration_source = "manual_injection"
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '_rsl_rl_runner'):
        # Get iteration from RSL-RL runner (injected by train.py)
        runner = env.unwrapped._rsl_rl_runner  # type: ignore
        if hasattr(runner, 'current_learning_iteration'):
            total_iterations = runner.current_learning_iteration
            iteration_source = "rsl_rl_runner"
        else:
            # Fallback to calculation
            total_steps = env.common_step_counter
            steps_per_iteration = env.num_envs * 24
            total_iterations = total_steps // steps_per_iteration
            iteration_source = "rsl_rl_fallback"
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '_current_iteration'):
        # Check if the environment tracks current iteration
        total_iterations = env.unwrapped._current_iteration  # type: ignore
        iteration_source = "env_tracked_iteration"
    elif hasattr(env.unwrapped, 'episode_length_buf'):
        # Calculate from episode counter (for RSL-RL)
        total_steps = env.common_step_counter
        steps_per_iteration = env.num_envs * 24  # RSL-RL default: 24 steps per env per iteration
        total_iterations = total_steps // steps_per_iteration
        iteration_source = "rsl_rl_calculation"
    else:
        # Last resort fallback
        total_steps = env.common_step_counter  
        steps_per_iteration = env.num_envs * 24  # Assume RSL-RL default
        total_iterations = total_steps // steps_per_iteration
        iteration_source = "fallback_calculation"
        
        # WORKAROUND: If we detect we're likely resumed (curriculum stage doesn't match iteration),
        # try to infer the correct iteration from TensorBoard logged metrics
        if hasattr(env, '_curriculum_stage'):
            # Calculate expected stage based on current iteration count
            expected_stage_for_iter = (
                1 if total_iterations < 20000 else  # Stage 1: 0-20k
                2 if total_iterations < 30000 else  # Stage 2: 20k-30k
                3 if total_iterations < 45000 else  # Stage 3: 30k-45k
                4  # Stage 4: 45k+
            )
            if env._curriculum_stage != expected_stage_for_iter:  # type: ignore
                # Mismatch detected! We might be resumed.
                # Unfortunately, we can't reliably get the true iteration here.
                # Best we can do is keep the current stage and log a warning.
                if not hasattr(env, '_curriculum_resume_warning_shown'):
                    print(f"\n{'!'*80}")
                    print("[WARNING] Curriculum iteration mismatch detected!")
                    print(f"  Calculated iteration from steps: {total_iterations:,}")
                    print(f"  Current stage: {env._curriculum_stage}")  # type: ignore
                    print(f"  Expected stage for this iteration: {expected_stage_for_iter}")
                    print("  This likely means training was resumed with --resume.")
                    print("  Curriculum will maintain current stage until next boundary.")
                    print(f"{'!'*80}\n")
                    env._curriculum_resume_warning_shown = True  # type: ignore
    
    # DEBUG: Print to verify iteration counting
    if not hasattr(env, "_curriculum_debug_counter"):
        env._curriculum_debug_counter = 0  # type: ignore
        print(f"\n{'='*80}")
        print("[Curriculum] Initialization - 4-Stage Curriculum Enabled")
        print("  Stage 1: 0-20,000 iterations (Fixed pose at default)")
        print("  Stage 2: 20,000-30,000 iterations (±3cm, ±8° roll)")
        print("  Stage 3: 30,000-45,000 iterations (±10cm, ±20° roll, ±12° pitch/yaw)")
        print("  Stage 4: 45,000+ iterations (±15cm, ±30° roll, ±15° pitch/yaw)")
        print(f"  num_envs: {env.num_envs}")
        print(f"  Iteration source: {iteration_source}")
        print(f"  Current total_iterations: {total_iterations:,}")
        print(f"  common_step_counter: {env.common_step_counter:,}")
        print(f"{'='*80}\n")
        
    env._curriculum_debug_counter += 1  # type: ignore
    if env._curriculum_debug_counter % 1000 == 0:  # type: ignore
        print(f"\n[DEBUG Curriculum] Call {env._curriculum_debug_counter}:")  # type: ignore
        print(f"  Iteration source: {iteration_source}")
        print(f"  common_step_counter: {env.common_step_counter:,}")
        print(f"  total_iterations: {total_iterations:,}")
        print(f"  Current _curriculum_stage: {getattr(env, '_curriculum_stage', 'NOT SET')}")
    
    # Determine current stage based on total iterations
    # Stage 1: 0-20,000 (Base training with fixed pose)
    # Stage 2: 20,000-30,000 (Small range)
    # Stage 3: 30,000-45,000 (Medium range)
    # Stage 4: 45,000+ (Maximum range)
    
    # PLAY MODE DETECTION: If no runner is attached or marked as inference, use Stage 4
    # This check must happen BEFORE stage calculation to ensure it persists across resets
    is_inference_mode = (
        hasattr(env.unwrapped, '_is_inference_mode') or  # Explicitly marked by play.py
        not hasattr(env.unwrapped, '_rsl_rl_runner') or
        (hasattr(env.unwrapped, '_rsl_rl_runner') and env.unwrapped._rsl_rl_runner is None)  # type: ignore
    )
    
    if is_inference_mode:
        # ALWAYS use Stage 4 in inference mode, even after resets
        target_stage = 4
        height_range = (0.18, 0.43)  # [0.18m, 0.43m] range
        roll_range = (-0.524, 0.524)  # ±30° (π/6 rad)
        pitch_range = (-0.262, 0.262)  # ±15°
        yaw_range = (0.0, 0.0)  # Fixed at 0° (yaw not controlled - requires localization)
        
        # Print message only once per session
        if not hasattr(env, "_curriculum_inference_message_shown"):
            env._curriculum_inference_message_shown = True  # type: ignore
            print(f"\n{'='*80}")
            print("[Curriculum] INFERENCE MODE DETECTED")
            print("  Automatically setting to Stage 4 (Maximum Range)")
            print("  This allows full height and pose control capability")
            print("  NOTE: Yaw fixed at 0° (no yaw control to avoid localization dependency)")
            print(f"{'='*80}\n")
    elif total_iterations < 20000:  # Stage 1: Base training
        target_stage = 1
        height_range = (default_height, default_height)  # Fixed at 0.33m
        roll_range = (0.0, 0.0)  # Fixed at 0°
        pitch_range = (0.0, 0.0)  # Fixed at 0°
        yaw_range = (0.0, 0.0)  # Fixed at 0° (yaw not controlled - requires localization)
    elif total_iterations < 30000:  # Stage 2: Small range
        target_stage = 2
        height_range = (0.30, 0.36)  # ±3cm
        roll_range = (-0.14, 0.14)  # ±8°
        pitch_range = (0.0, 0.0)  # Keep pitch fixed
        yaw_range = (0.0, 0.0)  # Fixed at 0° (yaw not controlled - requires localization)
    elif total_iterations < 45000:  # Stage 3: Medium range
        target_stage = 3
        height_range = (0.23, 0.43)  # ±10cm
        roll_range = (-0.349, 0.349)  # ±20°
        pitch_range = (-0.21, 0.21)  # ±12°
        yaw_range = (0.0, 0.0)  # Fixed at 0° (yaw not controlled - requires localization)
    else:  # >= 45000, Stage 4: Maximum range
        target_stage = 4
        height_range = (0.18, 0.43)  # [0.18m, 0.43m] range
        roll_range = (-0.524, 0.524)  # ±30° (π/6 rad)
        pitch_range = (-0.262, 0.262)  # ±15°
        yaw_range = (0.0, 0.0)  # Fixed at 0° (yaw not controlled - requires localization)
    
    # Initialize curriculum state on first call, using the target_stage we just determined
    if not hasattr(env, "_curriculum_stage"):
        env._curriculum_stage = target_stage  # type: ignore
        env._curriculum_last_update = 0  # type: ignore
        
        # IMPORTANT: Set initial command ranges based on starting stage
        ranges.height = height_range
        ranges.roll = roll_range
        ranges.pitch = pitch_range
        ranges.yaw = yaw_range
        
        # Set initial reward parameters based on starting stage
        _update_reward_parameters(env, target_stage)
        
        print(f"\n{'='*80}")
        print(f"[Curriculum] Initialized at iteration {total_iterations}")
        print(f"  Starting Stage: {target_stage}")
        print(f"  Height Range: [{height_range[0]:.3f}, {height_range[1]:.3f}] m")
        print(f"  Roll Range:   [{roll_range[0]:.3f}, {roll_range[1]:.3f}] rad = [{math.degrees(roll_range[0]):.1f}, {math.degrees(roll_range[1]):.1f}]°")
        print(f"  Pitch Range:  [{pitch_range[0]:.3f}, {pitch_range[1]:.3f}] rad = [{math.degrees(pitch_range[0]):.1f}, {math.degrees(pitch_range[1]):.1f}]°")
        print(f"  Yaw Range:    [{yaw_range[0]:.3f}, {yaw_range[1]:.3f}] rad = [{math.degrees(yaw_range[0]):.1f}, {math.degrees(yaw_range[1]):.1f}]°")
        _print_reward_parameters(env)
        print(f"{'='*80}\n")
    
    # Update ranges if stage changed
    if target_stage != env._curriculum_stage:  # type: ignore
        env._curriculum_stage = target_stage  # type: ignore
        env._curriculum_last_update = total_iterations  # type: ignore
        
        # Update command ranges
        ranges.height = height_range
        ranges.roll = roll_range
        ranges.pitch = pitch_range
        ranges.yaw = yaw_range
        
        # Update reward parameters for new stage
        _update_reward_parameters(env, target_stage)
        
        # Print stage transition message
        print(f"\n{'='*80}")
        print(f"[Curriculum] Stage Transition at Iteration {total_iterations}")
        print(f"{'='*80}")
        print(f"  New Stage: {target_stage}")
        print(f"  Height Range: [{height_range[0]:.3f}, {height_range[1]:.3f}] m")
        print(f"  Roll Range:   [{roll_range[0]:.3f}, {roll_range[1]:.3f}] rad = [{math.degrees(roll_range[0]):.1f}, {math.degrees(roll_range[1]):.1f}]°")
        print(f"  Pitch Range:  [{pitch_range[0]:.3f}, {pitch_range[1]:.3f}] rad = [{math.degrees(pitch_range[0]):.1f}, {math.degrees(pitch_range[1]):.1f}]°")
        print(f"  Yaw Range:    [{yaw_range[0]:.3f}, {yaw_range[1]:.3f}] rad = [{math.degrees(yaw_range[0]):.1f}, {math.degrees(yaw_range[1]):.1f}]°")
        _print_reward_parameters(env)
        print(f"{'='*80}\n")
    
    # Log current stage every 100 iterations with detailed command ranges
    if total_iterations > 0 and total_iterations % 100 == 0 and total_iterations != env._curriculum_last_update:  # type: ignore
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
            15000                              # Stage 4: 45000+ (estimate 15k iterations for progress display)
        )
        progress = min(100.0, (iterations_in_stage / stage_total) * 100) if stage_total > 0 else 0.0
        
        print(f"\n{'='*80}")
        print(f"[Curriculum Progress] Iteration {total_iterations}")
        print(f"{'='*80}")
        print(f"  Current Stage: {target_stage}")
        print(f"  Stage Progress: {iterations_in_stage}/{stage_total} iterations ({progress:.1f}%)")
        print("  Command Ranges:")
        print(f"    Height: [{height_range[0]:.3f}, {height_range[1]:.3f}] m")
        print(f"    Roll:   [{roll_range[0]:.3f}, {roll_range[1]:.3f}] rad = [{math.degrees(roll_range[0]):.1f}, {math.degrees(roll_range[1]):.1f}]°")
        print(f"    Pitch:  [{pitch_range[0]:.3f}, {pitch_range[1]:.3f}] rad = [{math.degrees(pitch_range[0]):.1f}, {math.degrees(pitch_range[1]):.1f}]°")
        print(f"    Yaw:    [{yaw_range[0]:.3f}, {yaw_range[1]:.3f}] rad = [{math.degrees(yaw_range[0]):.1f}, {math.degrees(yaw_range[1]):.1f}]°")
        print(f"{'='*80}\n")
    
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
