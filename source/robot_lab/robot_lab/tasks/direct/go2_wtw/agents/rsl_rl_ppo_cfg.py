# SPDX-License-Identifier: Apache-2.0
# PPO config for Go2 Walk-These-Ways (mirroring walk-these-ways ppo_cse settings)

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2WTWActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "Go2WTWActorCritic"
    adaptation_module_branch_hidden_dims = [256, 128]


@configclass
class Go2WTWPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "Go2WTWPPO"
    adaptation_module_learning_rate = 1.0e-3
    num_adaptation_module_substeps = 1
    selective_adaptation_module_loss = False


@configclass
class Go2WalkTheseWaysPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 200
    experiment_name = "go2_walk_these_ways"
    empirical_normalization = False
    clip_actions = 10.0
    obs_groups = {"policy": ["obs_history"], "critic": ["obs_history", "privileged"]}

    policy = Go2WTWActorCriticCfg(
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
