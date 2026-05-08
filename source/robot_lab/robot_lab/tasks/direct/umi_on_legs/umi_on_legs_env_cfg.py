"""Configuration for UMI on Legs Direct RL environment.

This config matches the original IsaacGym UMI-on-Legs reaching task:
- Go2 quadruped + ARX5 arm (18 DOF: 12 leg + 6 arm)
- End-effector pose tracking with multi-time-horizon observations
- Curriculum learning on position and orientation error sigma
- Domain randomization (friction, mass, PD gains, pushes)
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from robot_lab.assets.go2_x5 import GO2_X5_UMI_CFG


from isaaclab.sensors import ContactSensorCfg


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Custom scene with ground plane, dome light, and contact sensor."""

    # Ground plane (must come first — physics assets need a ground to collide with)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(200.0, 200.0),
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    # Dome light for sky illumination
    dome_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # Contact sensor for all robot bodies (for termination and reward)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # Robot (patched in __post_init__)
    robot: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(usd_path=""),
    )


@configclass
class UmiOnLegsEnvCfg(DirectRLEnvCfg):
    """Configuration for the UMI on Legs reaching task."""

    # ---- General settings ----
    seed: int = 42
    decimation: int = 4  # 200Hz sim → 50Hz policy
    debug_vis: bool = True  # Set True to show EE target/current pose markers
    episode_length_s: float = 7.0
    is_finite_horizon: bool = False

    # ---- Action/Observation spaces ----
    action_space: int = 18
    observation_space: int = 96  # 60 + 36 (4 target times * 9D pose)
    state_space: int = 143  # privileged critic observations
    num_actions: int = 18
    num_observations: int = 96
    num_states: int = 143

    # ---- Simulation ----
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        gravity=(0.0, 0.0, -9.81),
    )

    # ---- Scene ----
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3.0)

    # ---- Robot asset ----
    robot: ArticulationCfg = GO2_X5_UMI_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # ---- Control parameters (matching original PositionController) ----
    action_scale: list = [0.25] * 18
    dof_offset: list = [
        0.1,  0.8, -1.5,   # FR leg
        -0.1, 0.8, -1.5,   # FL leg
        0.1,  1.0, -1.5,   # RR leg
        -0.1, 1.0, -1.5,   # RL leg
        0.0,  0.3,  0.5, 0.0, 0.0, 0.0,  # ARX5 arm
    ]

    kp: list = [40.0] * 12 + [100.0, 100.0, 100.0, 20.0, 20.0, 5.0]
    kd: list = [1.0] * 12 + [3.0, 3.0, 3.0, 2.0, 1.0, 0.5]
    torque_limit: list = [
        35.278, 35.278, 44.400, 35.278, 35.278, 44.400,
        35.278, 35.278, 44.400, 35.278, 35.278, 44.400,
        20.0, 20.0, 15.0, 7.0, 5.0, 5.0,
    ]

    ctrl_delay_steps: float = 4.0  # 20ms delay at 0.005s dt
    max_action_value: float = 100.0

    # ---- Observation parameters ----
    root_ang_vel_scale: float = 0.25
    root_ang_vel_noise: float = 0.05
    root_gravity_scale: float = 1.0
    root_gravity_noise: float = 0.05
    dof_pos_scale: float = 1.0
    dof_pos_noise: float = 0.01
    dof_vel_scale: float = 0.05
    dof_vel_noise: float = 0.075
    root_lin_vel_scale: float = 2.0  # privileged only

    obs_history_len: int = 1

    # ---- Task: Reaching ----
    task_link_name: str = "ee"
    pos_obs_scale: float = 10.0
    orn_obs_scale: float = 1.5
    pos_err_sigma: float = 0.5
    orn_err_sigma: float = 1.5
    pos_reward_scale: float = 0.0
    orn_reward_scale: float = 0.0
    pose_reward_scale: float = 4.0
    target_obs_times: list = [0.02, 0.04, 0.06, 1.0]
    position_obs_encoding: str = "linear"
    pos_obs_clip: float = None

    # Curriculum
    pos_sigma_curriculum: list = [
        (100.0, 2.0), (1.0, 1.0), (0.8, 0.5), (0.5, 0.1),
        (0.4, 0.05), (0.2, 0.01), (0.1, 0.005),
    ]
    orn_sigma_curriculum: list = [
        (100.0, 8.0), (1.0, 4.0), (0.8, 2.0), (0.6, 1.0), (0.2, 0.5),
    ]
    init_pos_curriculum_level: int = 1
    init_orn_curriculum_level: int = 1
    smoothing_dt_multiplier: float = 0.25

    pose_latency: float = 0.0
    target_relative_to_base: bool = False

    # ---- Constraint weights ----
    action_rate_weight: float = -0.05
    torque_weight: float = -1e-4
    torque_power: float = 2.0
    even_mass_distribution_weight: float = -1.0
    feet_under_hips_weight: float = -1.0
    aligned_body_ee_weight: float = -1.0
    root_height_weight: float = 0.0
    root_height_target: float = 0.35

    # ---- Domain randomization ----
    friction_range: tuple = (0.2, 2.0)
    num_friction_buckets: int = 64
    added_mass_range: tuple = (-2.0, 2.0)

    randomize_pd_params: bool = True
    kp_ratio_range: tuple = (0.5, 1.5)
    kd_ratio_range: tuple = (0.5, 1.5)

    push_robots: bool = True
    push_interval_s: float = 5.0
    max_push_vel_lin: float = 1.0
    max_push_vel_ang: float = 1.0

    dof_pos_reset_range_scale: float = 0.05

    init_pos_noise: list = [0.0, 0.0, 0.0]
    init_euler_noise: list = [0.0, 0.0, 0.0]
    init_lin_vel_noise: list = [0.0, 0.0, 0.0]
    init_ang_vel_noise: list = [0.0, 0.0, 0.0]

    # ---- Termination ----
    termination_contact_force_threshold: float = 1.0
    termination_contact_body_patterns: list = [
        "base", "hip", "Head", "thigh", "base_arm_link",
        "link1", "link2", "link3", "link4", "link5",
    ]
    safe_bounds_xy: float = 1e8

    # ---- Viewer ----
    viewer_pos: tuple = (3.0, 3.0, 3.0)
    viewer_lookat: tuple = (0.0, 0.0, 0.5)

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = self.robot
