"""RSL-RL PPO agent configuration for UMI-on-Legs.

Matches the original IsaacGym UMI training recipe:
- 24 steps per env per iteration
- gamma=0.99, lambda=0.95
- learning_rate=1e-3 (adaptive KL, desired_kl=0.01)
- 32 epochs, 4 mini-batches
- entropy_coef=0.001
"""

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class UmiOnLegsPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for UMI-on-Legs."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "umi_on_legs_direct"
    run_name = ""
    logger = "wandb"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=32,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )
