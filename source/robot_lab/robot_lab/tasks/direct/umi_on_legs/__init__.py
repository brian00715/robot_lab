"""Registration for UMI-on-Legs Direct RL environments."""

import gymnasium as gym

from . import agents


gym.register(
    id="RobotLab-Isaac-UMI-on-Legs-Direct-v0",
    entry_point=f"{__name__}.umi_on_legs_env:UmiOnLegsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.umi_on_legs_env_cfg:UmiOnLegsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UmiOnLegsPPORunnerCfg",
    },
)
