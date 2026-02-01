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

        # ------------------------------Command Ranges for Inference------------------------------
        import math
        self.commands.base_velocity_pose.ranges.height = (0.20, 0.46)
        self.commands.base_velocity_pose.ranges.roll = (-math.pi/4, math.pi/4)  
        self.commands.base_velocity_pose.ranges.pitch = (-0.436, 0.436)  
        self.commands.base_velocity_pose.default_height = 0.35
        
        
        print("[Config] Curriculum enabled for flat terrain training (Stage 1-4)")

        # ------------------------------Terrain and Sensors------------------------------
        # Change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # No height scan needed on flat terrain
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        
        self.curriculum.terrain_levels = None
        
        # ------------------------------Disable VelocityPose-specific Rewards for Stage 1------------------------------
        if hasattr(self.rewards, "track_height_exp"):
            reward_term = getattr(self.rewards, "track_height_exp", None)
            if reward_term is not None:
                pass  
        
        if hasattr(self.rewards, "track_orientation_exp"):
            reward_term = getattr(self.rewards, "track_orientation_exp", None)
            if reward_term is not None:
                pass 
        
        if hasattr(self.rewards, "base_lin_acc_z_l2"):
            reward_term = getattr(self.rewards, "base_lin_acc_z_l2", None)
            if reward_term is not None:
                pass  
        
        if hasattr(self.rewards, "base_ang_acc_xy_l2"):
            reward_term = getattr(self.rewards, "base_ang_acc_xy_l2", None)
            if reward_term is not None:
                pass  
        
        # Disable conditional velocity penalties if they exist (replaced by acceleration penalties)
        if hasattr(self.rewards, "lin_vel_z_penalty_conditional"):
            reward_term = getattr(self.rewards, "lin_vel_z_penalty_conditional", None)
            if reward_term is not None:
                reward_term.weight = 0
        
        if hasattr(self.rewards, "ang_vel_xy_penalty_conditional"):
            reward_term = getattr(self.rewards, "ang_vel_xy_penalty_conditional", None)
            if reward_term is not None:
                reward_term.weight = 0
        
        # ------------------------------Keep Command-Aware Rewards------------------------------
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        
        if hasattr(self.rewards, "track_height_exp"):
            self.rewards.track_height_exp.params["sensor_cfg"] = None
        if hasattr(self.rewards, "track_height_exp"):
            print(f"[Config] track_height_exp weight: {self.rewards.track_height_exp.weight}")
        if hasattr(self.rewards, "track_orientation_exp"):
            print(f"[Config] track_orientation_exp weight: {self.rewards.track_orientation_exp.weight}")

        if self.__class__.__name__ == "UnitreeGo2VelocityPoseFlatEnvCfg":
            height_reward = getattr(self.rewards, "track_height_exp", None)
            orient_reward = getattr(self.rewards, "track_orientation_exp", None)
            
            self.disable_zero_weight_rewards()
            
            # Restore VelocityPose rewards if they were removed
            if height_reward is not None and not hasattr(self.rewards, "track_height_exp"):
                self.rewards.track_height_exp = height_reward
                print("[Config] Restored track_height_exp after disable_zero_weight_rewards")
            if orient_reward is not None and not hasattr(self.rewards, "track_orientation_exp"):
                self.rewards.track_orientation_exp = orient_reward
                print("[Config] Restored track_orientation_exp after disable_zero_weight_rewards")
