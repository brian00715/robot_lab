import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# ARX X5 robot configuration
ARX_X5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "arx5.usd"),
        activate_contact_sensors=False,
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
            "gripper_joint_right": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            damping=40.0,
            stiffness=400.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint_left"],
            damping=40.0,
            stiffness=400.0,
        ),
    },
)
