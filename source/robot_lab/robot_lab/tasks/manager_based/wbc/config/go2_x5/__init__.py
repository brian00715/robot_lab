# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Gym registrations for Unitree GO2 + ARX X5 whole-body-control tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="RobotLab-Isaac-WBC-Flat-Unitree-Go2-X5-v0",
    entry_point="robot_lab.tasks.manager_based.wbc:WbcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ArxX5WbcFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArxX5WbcFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-WBC-Flat-Unitree-Go2-X5-Play-v0",
    entry_point="robot_lab.tasks.manager_based.wbc:WbcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ArxX5WbcFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArxX5WbcFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-WBC-Rough-Unitree-Go2-X5-v0",
    entry_point="robot_lab.tasks.manager_based.wbc:WbcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ArxX5WbcRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArxX5WbcRoughPPORunnerCfg",
    },
)
