# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2VelocityPoseRoughEnvCfg


@configclass
class UnitreeGo2VelocityPoseFlatEnvCfg(UnitreeGo2VelocityPoseRoughEnvCfg):
    """Configuration for the Unitree Go2 locomotion velocity-pose tracking environment on flat terrain.
    
    This configuration keeps the 6D command-aware reward functions (stand_still_full_cmd, 
    joint_pos_penalty_full_cmd, etc.) but disables height/orientation tracking and acceleration penalties.
    
    Goal: Validate the 6D command awareness works correctly with basic velocity tracking only.
    """
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Terrain and Sensors------------------------------
        # Change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # No height scan needed on flat terrain
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        
        # No terrain curriculum on flat terrain
        self.curriculum.terrain_levels = None
        
        # ------------------------------Disable VelocityPose-specific Rewards------------------------------
        # Disable height tracking reward (VelocityPose-specific)
        # Keep 6D command awareness, but don't actively track height
        if hasattr(self.rewards, "track_height_exp"):
            reward_term = getattr(self.rewards, "track_height_exp", None)
            if reward_term is not None:
                reward_term.weight = 0
        
        # Disable orientation tracking reward (VelocityPose-specific)
        # Keep 6D command awareness, but don't actively track roll/pitch
        if hasattr(self.rewards, "track_orientation_exp"):
            reward_term = getattr(self.rewards, "track_orientation_exp", None)
            if reward_term is not None:
                reward_term.weight = 0
        
        # Disable z-axis linear acceleration penalty (VelocityPose-specific)
        # On flat terrain, we don't need to worry about smooth height changes
        if hasattr(self.rewards, "base_lin_acc_z_l2"):
            reward_term = getattr(self.rewards, "base_lin_acc_z_l2", None)
            if reward_term is not None:
                reward_term.weight = 0
        
        # Disable roll/pitch angular acceleration penalty (VelocityPose-specific)
        # On flat terrain, we don't need to worry about smooth orientation changes
        if hasattr(self.rewards, "base_ang_acc_xy_l2"):
            reward_term = getattr(self.rewards, "base_ang_acc_xy_l2", None)
            if reward_term is not None:
                reward_term.weight = 0
        
        # Disable conditional velocity penalties if they exist (replaced by acceleration penalties)
        if hasattr(self.rewards, "lin_vel_z_penalty_conditional"):
            reward_term = getattr(self.rewards, "lin_vel_z_penalty_conditional", None)
            if reward_term is not None:
                reward_term.weight = 0
        
        if hasattr(self.rewards, "ang_vel_xy_penalty_conditional"):
            reward_term = getattr(self.rewards, "ang_vel_xy_penalty_conditional", None)
            if reward_term is not None:
                reward_term.weight = 0
        
        # ------------------------------Keep 6D Command-Aware Rewards------------------------------
        # These rewards are KEPT (they use 6D commands but only penalize when appropriate):
        # - stand_still_full_cmd: Uses all 6D commands to determine "truly static"
        # - joint_pos_penalty_full_cmd: Uses all 6D commands to determine motion state
        # - track_lin_vel_xy_exp: Basic velocity tracking (weights from parent)
        # - track_ang_vel_z_exp: Basic angular velocity tracking (weights from parent)
        
        # Note: The base_height_l2 reward uses world coordinates on flat terrain
        self.rewards.base_height_l2.params["sensor_cfg"] = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2VelocityPoseFlatEnvCfg":
            self.disable_zero_weight_rewards()
