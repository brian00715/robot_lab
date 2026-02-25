# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Configuration for Unitree GO2 + ARX5 Arm - Stage 2 Training (IK-based)

This configuration implements Stage 2 training with IK-based arm control:
- Dog learns optimal pose compensation for arm motion
- Arm uses IK to reach random workspace targets  
- Dog policy only controls 12 leg joints (same as Stage 1)
- Combined learning: locomotion + pose optimization
"""

import math

from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg, RewardTermCfg as RewTerm

# Import base configuration
from .rough_env_cfg import (
    UnitreeGo2X5VelocityPoseRoughEnvCfg,
    DOG_JOINT_NAMES,
    ARM_JOINT_NAMES,
)

# Import IK-based composite action
from robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp.composite_actions_ik import (
    DogArmIKCompositeAction,
)

# Import IK tracking rewards
from robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp import (
    ik_rewards,
)


@configclass
class UnitreeGo2X5VelocityPoseIKRoughEnvCfg(
    UnitreeGo2X5VelocityPoseRoughEnvCfg
):
    """Configuration for GO2+ARX5 with IK-based arm control (Stage 2)."""
    
    def __post_init__(self):
        # Call parent post_init first
        super().__post_init__()
        
        # ========================================
        # Replace Action Configuration with IK-based Composite Action
        # ========================================
        
        # CRITICAL: Override the dog_arm_composite action to use IK controller
        self.actions.dog_arm_composite = ActionTermCfg(
            class_type=DogArmIKCompositeAction,
            asset_name="robot",
            scale={
                ".*_hip_joint": 0.5,     # Hip joints - larger range
                ".*_thigh_joint": 0.5,   # Thigh joints
                ".*_calf_joint": 0.5,    # Calf joints
            },
        )
        
        # ========================================
        # Add IK Tracking Rewards
        # ========================================
        
        # End-effector position tracking (primary IK reward)
        self.rewards.tracking_ee_world = RewTerm(
            func=ik_rewards.tracking_ee_world,
            weight=0.8,  # From B1Z1 config
            params={"std": 1.0}
        )
        
        # End-effector tracking in spherical coordinates (optional)
        self.rewards.tracking_ee_sphere = RewTerm(
            func=ik_rewards.tracking_ee_sphere,
            # Disabled by default, can enable for relative tracking
            weight=0.0,
            params={"std": 1.0}
        )
        
        # Conditional tracking rewards (walking/standing)
        self.rewards.tracking_ee_sphere_walking = RewTerm(
            func=ik_rewards.tracking_ee_sphere_walking,
            weight=0.0,  # Can enable for walking-specific tracking
            params={"std": 1.0, "speed_threshold": 0.1}
        )
        
        self.rewards.tracking_ee_sphere_standing = RewTerm(
            func=ik_rewards.tracking_ee_sphere_standing,
            weight=0.0,  # Can enable for standing-specific tracking
            params={"std": 1.0, "speed_threshold": 0.1}
        )
        
        # End-effector orientation tracking
        self.rewards.tracking_ee_orn = RewTerm(
            func=ik_rewards.tracking_ee_orn,
            weight=0.0,  # Disabled by default (B1Z1 also uses 0.0)
            params={"std": 1.0}
        )
        
        # Arm energy penalty
        self.rewards.arm_energy_abs_sum = RewTerm(
            func=ik_rewards.arm_energy_abs_sum,
            weight=0.0,  # Can enable to encourage energy efficiency
        )
        
        print(
            "[UnitreeGo2X5VelocityPoseIKRoughEnvCfg] "
            "Configured with IK-based arm control"
        )
        print("  - Dog policy: 12D (legs only)")
        print("  - Arm control: IK to workspace targets")
        print("  - Total robot DOF: 18 (12 dog + 6 arm)")
        print("  - IK tracking rewards: tracking_ee_world (weight=0.8)")
