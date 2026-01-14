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
        # Adjust height range: 0.20-0.46m (Go2 default is 0.35m)
        # Adjust roll range: ±30 degrees (±0.524 radians)
        import math
        self.commands.base_velocity_pose.ranges.height = (0.20, 0.46)
        self.commands.base_velocity_pose.ranges.roll = (-math.pi/6, math.pi/6)  # ±30°
        # Keep default height at 0.35m (middle of range)
        self.commands.base_velocity_pose.default_height = 0.35
        
        # Disable curriculum for flat terrain (for inference with full range)
        # Curriculum is useful for training but not needed for inference/testing
        self.curriculum.command_curriculum_height_pose = None
        print("[Config] Disabled command_curriculum_height_pose for flat terrain inference")

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
        # But KEEP the height/pose curriculum for Stage 2+
        self.curriculum.terrain_levels = None
        # curriculum.command_curriculum_height_pose is kept from parent config
        
        # ------------------------------Disable VelocityPose-specific Rewards for Stage 1------------------------------
        # Stage 1 (0-14,500): These rewards are disabled for basic training
        # Stage 2+ (14,500+): Will be automatically enabled by curriculum
        
        # Height tracking reward - will be enabled in Stage 2
        if hasattr(self.rewards, "track_height_exp"):
            reward_term = getattr(self.rewards, "track_height_exp", None)
            if reward_term is not None:
                # Keep enabled for curriculum learning
                pass  # Don't disable, let it work from the start
        
        # Orientation tracking reward - will be enabled in Stage 2
        if hasattr(self.rewards, "track_orientation_exp"):
            reward_term = getattr(self.rewards, "track_orientation_exp", None)
            if reward_term is not None:
                # Keep enabled for curriculum learning
                pass  # Don't disable, let it work from the start
        
        # Z-axis linear acceleration penalty - enable for smooth height changes
        if hasattr(self.rewards, "base_lin_acc_z_l2"):
            reward_term = getattr(self.rewards, "base_lin_acc_z_l2", None)
            if reward_term is not None:
                # Keep enabled for smooth motion
                pass  # Don't disable
        
        # Roll/pitch angular acceleration penalty - enable for smooth orientation changes
        if hasattr(self.rewards, "base_ang_acc_xy_l2"):
            reward_term = getattr(self.rewards, "base_ang_acc_xy_l2", None)
            if reward_term is not None:
                # Keep enabled for smooth motion
                pass  # Don't disable
        
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
        
        # VelocityPose-specific: track_height_exp also needs to use world coordinates on flat terrain
        if hasattr(self.rewards, "track_height_exp"):
            self.rewards.track_height_exp.params["sensor_cfg"] = None
        
        # IMPORTANT: Explicitly ensure VelocityPose tracking rewards are NOT disabled
        # Force-enable them even if disable_zero_weight_rewards might interfere
        if hasattr(self.rewards, "track_height_exp"):
            print(f"[Config] track_height_exp weight: {self.rewards.track_height_exp.weight}")
        if hasattr(self.rewards, "track_orientation_exp"):
            print(f"[Config] track_orientation_exp weight: {self.rewards.track_orientation_exp.weight}")

        # If the weight of rewards is 0, set rewards to None
        # BUT: Skip VelocityPose tracking rewards to prevent disabling them
        if self.__class__.__name__ == "UnitreeGo2VelocityPoseFlatEnvCfg":
            # Store VelocityPose rewards before calling disable
            height_reward = getattr(self.rewards, "track_height_exp", None)
            orient_reward = getattr(self.rewards, "track_orientation_exp", None)
            
            # Call the disable function
            self.disable_zero_weight_rewards()
            
            # Restore VelocityPose rewards if they were removed
            if height_reward is not None and not hasattr(self.rewards, "track_height_exp"):
                self.rewards.track_height_exp = height_reward
                print("[Config] Restored track_height_exp after disable_zero_weight_rewards")
            if orient_reward is not None and not hasattr(self.rewards, "track_orientation_exp"):
                self.rewards.track_orientation_exp = orient_reward
                print("[Config] Restored track_orientation_exp after disable_zero_weight_rewards")
