# SPDX-License-Identifier: Apache-2.0
"""PPO config for Go2 + ARX5 Walk-These-Ways with arm disturbance curriculum."""

from isaaclab.utils import configclass
from .rsl_rl_ppo_cfg import (
    Go2WTWActorCriticCfg,
    Go2WTWPPOAlgorithmCfg,
    Go2WalkTheseWaysPPORunnerCfg,
)


@configclass
class Go2X5WTWActorCriticCfg(Go2WTWActorCriticCfg):
    """Actor-Critic for Go2+X5 WTW.

    Same RMA architecture (adaptation_module + actor_body + critic_body)
    but with 82D-per-step observations (70 dog + 12 arm).

    obs_history input: 82 × 30 = 2460D
    privileged input:  2D (friction + restitution)
    action output:     12D (dog joints only)
    """
    adaptation_module_branch_hidden_dims = [256, 128]


@configclass
class Go2X5ArmDisturbancePPORunnerCfg(Go2WalkTheseWaysPPORunnerCfg):
    """PPO runner config for Go2+X5 arm disturbance locomotion training."""

    experiment_name = "go2_x5_arm_disturbance"
    max_iterations = 50000

    policy = Go2X5WTWActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = Go2WTWPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
