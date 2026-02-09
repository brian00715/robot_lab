# Copyright (c) 2024-2025 Ziqi Fan  
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL PPO configuration for Unitree GO2 + ARX5 (Stage 1)."""

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity_pose.config.quadruped.unitree_go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2VelocityPoseRoughPPORunnerCfg,
)


@configclass
class UnitreeGo2X5VelocityPoseRoughPPORunnerCfg(UnitreeGo2VelocityPoseRoughPPORunnerCfg):
    """PPO runner configuration for GO2+X5 Stage 1."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Adjust network size for 76D observations
        # Policy network: 76D input → 512 → 256 → 128 → 12D output (dog joints only)
        self.policy.class_name = "ActorCritic"
        
        # Stage 1: 50k iterations (approximately 7-8 days on 4090)
        self.max_iterations = 50000
        
        # Learning rate schedule
        self.policy.init_noise_std = 1.0
        self.algorithm.learning_rate = 5e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.gamma = 0.99
        self.algorithm.lam = 0.95
        self.algorithm.desired_kl = 0.01
        
        # PPO-specific
        self.algorithm.entropy_coef = 0.01
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4


@configclass
class UnitreeGo2X5VelocityPoseFlatPPORunnerCfg(UnitreeGo2X5VelocityPoseRoughPPORunnerCfg):
    """PPO runner configuration for flat terrain."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Flat terrain can be slightly easier
        self.algorithm.learning_rate = 5e-4
