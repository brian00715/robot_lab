# SPDX-License-Identifier: Apache-2.0
"""Go2 Walk-These-Ways direct RL environment."""

import gymnasium as gym

from . import agents

gym.register(
    id="RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0",
    entry_point=f"{__name__}.go2_wtw_env:Go2WalkTheseWaysEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_wtw_env_cfg:Go2WalkTheseWaysEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WalkTheseWaysPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:Go2WalkTheseWaysDistillationRunnerCfg",
    },
)
