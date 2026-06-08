# SPDX-License-Identifier: Apache-2.0
# Ported from walk-these-ways-go2 (IsaacGym) to IsaacLab 2.3.0

from __future__ import annotations

from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# from robot_lab.assets.unitree import UNITREE_GO2_CFG
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class Go2WalkTheseWaysEnvCfg(DirectRLEnvCfg):
    """Config for Go2 Walk-These-Ways velocity tracking environment (Direct RL)."""

    # ------------ sim params (dt=0.005 s, decimation=4 → 50 Hz policy) ---------
    decimation: int = 4
    episode_length_s: float = 20.0

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physx=PhysxCfg(
            solver_type=1,
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)
    terrain: TerrainImporterCfg = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    # ------------ robot asset -------------------------------------------------------
    # Actuator gains matching go2_config: Kp=25, Kd=0.6, clip at 23.5 Nm
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={
            "legs": DCMotorCfg(
                joint_names_expr=[".*"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=0.6,
                friction=0.0,
            ),
        },
    )

    # ------------ observation / action spaces ---------------------------------------
    # obs: gravity(3) + cmd(15)*scale + dof_pos(12) + dof_vel(12) + actions(12) + last_actions(12) + clock(4) = 70
    observation_space: int = 70
    num_scalar_observations: int = 70
    num_observation_history: int = 30
    num_privileged_obs: int = 2
    action_space: int = 12
    state_space: int = 2

    # ------------ commands ----------------------------------------------------------
    num_commands: int = 15
    # command limits (used by curriculum)
    lin_vel_x: tuple = (-1.0, 1.0)          # [m/s]
    lin_vel_y: tuple = (-0.6, 0.6)          # [m/s]
    ang_vel_yaw: tuple = (-1.0, 1.0)        # [rad/s]
    body_height_cmd: tuple = (-0.05, 0.05)  # [m] relative to nominal 0.30 m
    gait_frequency_cmd_range: tuple = (2.0, 4.0)   # [Hz]
    # For trot, _resample_commands maps raw phase to raw / 2 + 0.25; raw 0.5 gives phase 0.5.
    gait_phase_cmd_range: tuple = (0.5, 0.5)
    gait_offset_cmd_range: tuple = (0.0, 1.0)
    gait_bound_cmd_range: tuple = (0.0, 1.0)
    gait_duration_cmd_range: tuple = (0.5, 0.5)
    footswing_height_range: tuple = (0.06, 0.061)  # [m] fixed ~6 cm swing
    body_pitch_range: tuple = (-0.4, 0.4)
    body_roll_range: tuple = (-0.0, 0.0)
    stance_width_range: tuple = (0.10, 0.45)
    stance_length_range: tuple = (0.35, 0.45)
    aux_reward_coef_range: tuple = (0.0, 0.0)
    limit_vel_x: tuple = (-5.0, 5.0)
    limit_vel_y: tuple = (-0.6, 0.6)
    limit_vel_yaw: tuple = (-5.0, 5.0)
    limit_body_height: tuple = (-0.05, 0.05)
    limit_gait_frequency: tuple = (2.0, 4.0)
    limit_gait_phase: tuple = (0.5, 0.5)
    limit_gait_offset: tuple = (0.0, 1.0)
    limit_gait_bound: tuple = (0.0, 1.0)
    limit_gait_duration: tuple = (0.5, 0.5)
    limit_footswing_height: tuple = (0.06, 0.061)
    limit_body_pitch: tuple = (-0.4, 0.4)
    limit_body_roll: tuple = (-0.0, 0.0)
    limit_stance_width: tuple = (0.10, 0.45)
    limit_stance_length: tuple = (0.35, 0.45)
    limit_aux_reward_coef: tuple = (0.0, 0.0)

    resampling_time: float = 10.0  # [s] time between command resamples
    heading_command: bool = False

    # ------------ command curriculum ------------------------------------------------
    command_curriculum: bool = True
    # Velocity bins (curriculum expands over velocity space)
    num_bins_vel_x: int = 21
    num_bins_vel_y: int = 1
    num_bins_vel_yaw: int = 21
    num_bins_body_height: int = 1
    num_bins_gait_frequency: int = 1
    num_bins_gait_phase: int = 1
    num_bins_gait_offset: int = 1
    num_bins_gait_bound: int = 1
    num_bins_gait_duration: int = 1
    num_bins_footswing_height: int = 1
    num_bins_body_pitch: int = 1
    num_bins_body_roll: int = 1
    num_bins_stance_width: int = 1
    num_bins_stance_length: int = 1
    num_bins_aux_reward_coef: int = 1
    num_lin_vel_bins: int = 21  # compatibility alias
    num_ang_vel_bins: int = 21  # compatibility alias
    # Gait params: 1 bin each in pretrain (full range sampled uniformly, no curriculum expansion)
    # Distill stage increases these to enable gait-space curriculum
    num_gait_freq_bins: int = 1
    num_gait_phase_bins: int = 1
    num_gait_offset_bins: int = 1
    num_gait_bound_bins: int = 1
    num_gait_duration_bins: int = 1
    num_footswing_bins: int = 1
    num_stance_width_bins: int = 1
    num_stance_length_bins: int = 1
    curriculum_seed: int = 100

    # Gait categories (used for gaitwise curriculum)
    gaitwise_curricula: bool = True
    gait_categories: tuple[str, ...] = ("trot",)
    exclusive_phase_offset: bool = False
    binary_phases: bool = True
    pacing_offset: bool = False
    balance_gait_distribution: bool = True

    # Curriculum success thresholds (fraction of max reward per step)
    curriculum_tracking_lin_vel: float = 0.8
    curriculum_tracking_ang_vel: float = 0.7
    curriculum_tracking_contacts_shaped_force: float = 0.90
    curriculum_tracking_contacts_shaped_vel: float = 0.90

    # ------------ control -----------------------------------------------------------
    action_scale: float = 0.25
    hip_scale_reduction: float = 0.5
    joint_names: tuple[str, ...] = (
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    )
    foot_names: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    # ------------ visualization -----------------------------------------------------
    enable_debug_vis: bool = False  # set True to enable command/vel/pose markers at env startup

    # ------------ observation flags -------------------------------------------------
    observe_vel: bool = False
    observe_command: bool = True
    observe_clock_inputs: bool = True
    observe_two_prev_actions: bool = True
    observe_gait_commands: bool = True
    observe_yaw: bool = False
    observe_contact_states: bool = False

    # ------------ observation scales ------------------------------------------------
    obs_lin_vel_scale: float = 2.0
    obs_ang_vel_scale: float = 0.25
    obs_dof_pos_scale: float = 1.0
    obs_dof_vel_scale: float = 0.05

    # Command observation scales (per-command-dim)
    # [vx, vy, yaw, height, freq, phase, offset, bound, duration, swing_h,
    #  pitch, roll, stance_w, stance_l, aux]
    cmd_scale_lin_vel: float = 2.0
    cmd_scale_ang_vel: float = 0.25
    cmd_scale_body_height: float = 4.0
    cmd_scale_gait_freq: float = 1.0
    cmd_scale_gait_phase: float = 1.0
    cmd_scale_footswing_height: float = 0.15
    cmd_scale_body_pitch: float = 0.3
    cmd_scale_body_roll: float = 0.3
    cmd_scale_aux_reward: float = 1.0
    cmd_scale_stance_width: float = 1.0
    cmd_scale_stance_length: float = 1.0

    # ------------ observation noise -------------------------------------------------
    add_noise: bool = True
    noise_level: float = 1.0
    noise_gravity: float = 0.05
    noise_dof_pos: float = 0.01
    noise_dof_vel: float = 1.5
    noise_lin_vel: float = 0.1
    noise_ang_vel: float = 0.2

    # ------------ rewards ----------------------------------------------------------
    # Positive/negative reward handling (ji22-style)
    only_positive_rewards_ji22_style: bool = True
    reward_split_mode: str = "isaacgym"  # "isaacgym" matches original WTW term-level split; "per_env_clip" is previous port behavior
    sigma_rew_neg: float = 0.5
    only_positive_rewards: bool = False

    # Gait reward parameters
    kappa_gait_probs: float = 0.07
    gait_force_sigma: float = 100.0
    gait_vel_sigma: float = 10.0

    # Tracking sigmas
    tracking_sigma: float = 0.25
    tracking_sigma_yaw: float = 0.25

    # Soft limits
    soft_dof_pos_limit: float = 0.9

    # Height targets
    base_height_target: float = 0.30

    # Contact force penalty threshold
    max_contact_force: float = 500.0
    contact_threshold: float = 1.0
    collision_contact_threshold: float = 0.1

    # Terminal conditions
    use_terminal_body_height: bool = True
    terminal_body_height: float = 0.20
    use_terminal_roll_pitch: bool = True
    terminal_body_ori: float = 1.6  # [rad]

    # ------------ reward scales ----------------------------------------------------
    rew_tracking_lin_vel: float = 1.0
    rew_tracking_ang_vel: float = 0.5
    rew_lin_vel_z: float = -0.02
    rew_ang_vel_xy: float = -0.001
    rew_orientation: float = 0.0
    rew_orientation_control: float = -5.0
    rew_torques: float = -1e-5
    rew_dof_vel: float = -1e-4
    rew_dof_acc: float = -2.5e-7
    rew_action_rate: float = -0.01
    rew_action_smoothness_1: float = -0.1
    rew_action_smoothness_2: float = -0.1
    rew_collision: float = -5.0
    rew_dof_pos_limits: float = 0.0
    rew_jump: float = 10.0
    rew_tracking_contacts_shaped_force: float = 4.0
    rew_tracking_contacts_shaped_vel: float = 4.0
    rew_feet_clearance_cmd_linear: float = -30.0
    rew_feet_slip: float = -0.04
    rew_feet_impact_vel: float = 0.0
    rew_raibert_heuristic: float = -10.0
    rew_feet_contact_forces: float = 0.0
    rew_base_height: float = -30.0
    rew_dof_pos: float = 0.0
    rew_feet_air_time: float = 0.0

    # ------------ domain randomization ---------------------------------------------
    randomize_friction: bool = True
    friction_range: tuple = (0.1, 3.0)
    friction_obs_range: tuple = (0.0, 1.0)
    randomize_restitution: bool = True
    restitution_range: tuple = (0.0, 0.4)
    restitution_obs_range: tuple = (0.0, 1.0)
    randomize_base_mass: bool = True
    added_mass_range: tuple = (-1.0, 3.0)
    randomize_com_displacement: bool = False
    com_displacement_range: tuple = (-0.15, 0.15)
    randomize_motor_strength: bool = True
    motor_strength_range: tuple = (0.9, 1.1)
    randomize_motor_offset: bool = True
    motor_offset_range: tuple = (-0.02, 0.02)
    randomize_Kp_factor: bool = False
    Kp_factor_range: tuple = (0.8, 1.3)
    randomize_Kd_factor: bool = False
    Kd_factor_range: tuple = (0.5, 1.5)
    randomize_gravity: bool = True
    gravity_range: tuple = (-1.0, 1.0)
    gravity_rand_interval_s: float = 8.0
    gravity_impulse_duration: float = 0.99
    # Push robots (linear + angular impulse, synced with RoboDuet)
    push_robots: bool = True
    max_push_vel_xy: float = 1.0
    max_push_ang_vel: float = 0.6  # RoboDuet: random angular velocity perturbation on push
    push_interval_s: float = 15.0
    rand_interval_s: float = 4.0
    randomize_lag_timesteps: bool = True
    lag_timesteps: int = 6
    # Sub-step action delay: randomly choose which decimation step the new action takes effect
    # (RoboDuet: randomize_action_delay=True, mirrors sub-step jitter in real hardware)
    randomize_action_delay: bool = True
    randomize_rigids_after_start: bool = False
    strict_dr_writes: bool = True

    # ------------ init state -------------------------------------------------------
    init_pos_z: float = 0.34
    init_x_range: float = 0.2
    init_y_range: float = 0.2
    init_yaw_range: float = 3.14   # random yaw on reset
    init_pitch_range: float = 0.0  # random pitch on reset (0 = disabled; RoboDuet default ramps from 0 via curriculum)
    init_roll_range: float = 0.0   # random roll on reset  (0 = disabled)
    init_z_range: float = 0.0      # extra random z-offset on reset [m] (on top of init_pos_z)
    init_vel_range: float = 0.5    # random base velocity on reset

    # ------------ command dead zones / zero-cmd probability (synced with RoboDuet) --
    # Individual per-axis dead zones (vs. previous norm>0.2 on xy only)
    cmd_vel_x_deadzone: float = 0.07    # zero vx if |vx| < threshold
    cmd_vel_y_deadzone: float = 0.07    # zero vy if |vy| < threshold
    cmd_yaw_deadzone: float = 0.10      # zero yaw if |yaw_cmd| < threshold
    zero_cmd_prob: float = 0.1          # fraction of envs assigned zero velocity each resample

    clip_observations: float = 100.0
    clip_actions: float = 10.0
