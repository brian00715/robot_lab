# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL PPO configuration for GO2 + ARX X5 WBC."""

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
)


@configclass
class ArxX5WbcRoughPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_x5_wbc_rough"
        self.max_iterations = 50000
        self.policy.init_noise_std = 1.0
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.01


@configclass
class ArxX5WbcFlatPPORunnerCfg(ArxX5WbcRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_x5_wbc_flat"
