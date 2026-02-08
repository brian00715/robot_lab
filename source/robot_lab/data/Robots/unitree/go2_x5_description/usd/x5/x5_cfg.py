import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg


JOINT_CONTROLLER_ACTUATORS = {
    "joint_controller": DelayedPDActuatorCfg(
        joint_names_expr=["joint[1-6]"],
        effort_limit=30.0,
        velocity_limit=5.0,
        stiffness={
            "joint1": 80.0,
            "joint2": 70.0,
            "joint3": 70.0,
            "joint4": 30.0,
            "joint5": 30.0,
            "joint6": 20.0,
        },
        damping={
            "joint1": 2.0,
            "joint2": 2.0,
            "joint3": 2.0,
            "joint4": 1.0,
            "joint5": 1.0,
            "joint6": 0.7,
        },
        friction=0.01,
        min_delay=0,
        max_delay=5,
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["gripper_joint_left"],
        effort_limit=1.5,
        velocity_limit=0.3,
        friction=0.01,
        stiffness=5.0,
        damping=0.2,
    ),
}

CARTESIAN_CONTROLLER_ACTUATORS = {
    "cartesian_controller": DelayedPDActuatorCfg(
        joint_names_expr=["joint[1-6]"],
        effort_limit=30.0,
        velocity_limit=5.0,
        stiffness={
            "joint1": 200.0,
            "joint2": 200.0,
            "joint3": 200.0,
            "joint4": 120.0,
            "joint5": 80.0,
            "joint6": 60.0,
        },
        damping={
            "joint1": 5.0,
            "joint2": 5.0,
            "joint3": 5.0,
            "joint4": 1.0,
            "joint5": 1.0,
            "joint6": 1.0,
        },
        friction=0.01,
        min_delay=0,
        max_delay=5,
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["gripper_joint_left"],
        effort_limit=1.5,
        velocity_limit=0.3,
        friction=0.01,
        stiffness=5.0,
        damping=0.2,
    ),
}

X5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "x5.usd"),
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "gripper_joint_left": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=CARTESIAN_CONTROLLER_ACTUATORS,
)
