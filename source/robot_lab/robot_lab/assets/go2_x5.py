"""Configuration for Unitree Go2 + ARX5 (x5 arm) robot.

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

# Joint name constants (order: FR, FL, RR, RL legs, then arm, then gripper)
DOG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["gripper_joint", "joint8"]
ALL_JOINT_NAMES = DOG_JOINT_NAMES + ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

# Ordered joint names matching the original Isaac Gym convention
# (FR hip, thigh, calf, FL hip, thigh, calf, RR hip, thigh, calf, RL hip, thigh, calf, arm1-6)
POLICY_JOINT_NAMES = DOG_JOINT_NAMES + ARM_JOINT_NAMES

# Default standing pose (offsets)
DEFAULT_DOF_POS = [
    # FR leg
    0.1, 0.8, -1.5,
    # FL leg
    -0.1, 0.8, -1.5,
    # RR leg
    0.1, 1.0, -1.5,
    # RL leg
    -0.1, 1.0, -1.5,
    # ARX5 arm
    0.0, 0.3, 0.5, 0.0, 0.0, 0.0,
]

# PD gains: legs use Kp=40, Kd=1.0; arm uses varying gains
KP_GAINS = [40.0] * 12 + [100.0, 100.0, 100.0, 20.0, 20.0, 5.0]
KD_GAINS = [1.0] * 12 + [3.0, 3.0, 3.0, 2.0, 1.0, 0.5]

# Torque limits (Nm)
TORQUE_LIMITS = [
    35.278, 35.278, 44.400,  # FR
    35.278, 35.278, 44.400,  # FL
    35.278, 35.278, 44.400,  # RR
    35.278, 35.278, 44.400,  # RL
    20.0, 20.0, 15.0, 7.0, 5.0, 5.0,  # ARX5
]

# Action scale (policy outputs in [-1, 1] -> scaled by this to get joint offset)
ACTION_SCALE = [0.25] * 18

# Joint limits (lower, upper) for legs + arm
JOINT_LIMITS_LOWER = [
    -1.0472, -1.5708, -2.7227,  # FR
    -1.0472, -1.5708, -2.7227,  # FL
    -1.0472, -0.5236, -2.7227,  # RR
    -1.0472, -0.5236, -2.7227,  # RL
    -3.141593, 0.0, 0.0, -1.570796, -1.570796, -1.570796,  # ARX5
]
JOINT_LIMITS_UPPER = [
    1.0472, 3.4907, -0.83776,  # FR
    1.0472, 3.4907, -0.83776,  # FL
    1.0472, 4.5379, -0.83776,  # RR
    1.0472, 4.5379, -0.83776,  # RL
    3.141593, 3.66519, 3.141593, 1.570796, 1.570796, 1.570796,  # ARX5
]

# Termination contact body name patterns
TERMINATION_CONTACT_BODIES = [
    "base",
    "hip",
    "Head",
    "thigh",
    "base_arm_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
]

# Force sensor link names (for feet)
FORCE_SENSOR_LINKS = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]


GO2_X5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/go2_x5_description/usd/go2_x5/go2_x5.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "gripper_joint": 0.0,
            "joint8": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[r"(FL|FR|RL|RR)_(hip|thigh|calf)_joint"],
            effort_limit=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
            min_delay=0,
            max_delay=5,
        ),
        "arm": DelayedPDActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit=20.0,
            velocity_limit=10.0,
            stiffness=25,
            damping=0.5,
            friction=0.01,
            min_delay=0,
            max_delay=5,
        ),
        "gripper": DelayedPDActuatorCfg(
            joint_names_expr=["gripper_joint", "joint8"],
            effort_limit=12.0,
            velocity_limit=5.0,
            stiffness=25,
            damping=0.5,
        ),
    },
)
"""Configuration of Unitree Go2 + ARX5 using DelayedPDActuator."""


# UMI variant: uses higher stiffness for legs (matching original IsaacGym),
# with explicit stiffness/damping per joint group
GO2_X5_UMI_CFG = GO2_X5_CFG.copy()
GO2_X5_UMI_CFG.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
)
GO2_X5_UMI_CFG.actuators = {
    "legs": DelayedPDActuatorCfg(
        joint_names_expr=[r"(FL|FR|RL|RR)_(hip|thigh|calf)_joint"],
        effort_limit=44.4,
        velocity_limit=21.0,
        stiffness=40.0,
        damping=1.0,
        friction=0.01,
        min_delay=0,
        max_delay=5,
    ),
    "arm": DelayedPDActuatorCfg(
        joint_names_expr=["joint[1-6]"],
        effort_limit=20.0,
        velocity_limit=10.0,
        stiffness={
            "joint1": 100.0, "joint2": 100.0, "joint3": 100.0,
            "joint4": 20.0, "joint5": 20.0, "joint6": 5.0,
        },
        damping={
            "joint1": 3.0, "joint2": 3.0, "joint3": 3.0,
            "joint4": 2.0, "joint5": 1.0, "joint6": 0.5,
        },
        friction=0.01,
        min_delay=0,
        max_delay=5,
    ),
    "gripper": DelayedPDActuatorCfg(
        joint_names_expr=["gripper_joint", "joint8"],
        effort_limit=12.0,
        velocity_limit=5.0,
        stiffness=25,
        damping=0.5,
    ),
}
"""UMI variant with PD gains matching original IsaacGym training."""
