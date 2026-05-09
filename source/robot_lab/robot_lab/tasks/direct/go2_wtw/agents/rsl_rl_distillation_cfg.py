# SPDX-License-Identifier: Apache-2.0
# Distillation config for Go2 Walk-These-Ways teacher→student second stage.
#
# Teacher: policy obs (70) + privileged obs (friction + restitution = 2) = 72 total
# Student: policy obs only (70)
#
# Usage:
#   python scripts/reinforcement_learning/rsl_rl/train.py \
#     --task RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0 \
#     --agent rsl_rl_distillation_cfg_entry_point \
#     --checkpoint logs/rsl_rl/go2_walk_these_ways/<run>/model_199.pt \
#     --headless --num_envs 4096 --max_iterations 200

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class Go2WalkTheseWaysDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 200
    experiment_name = "go2_walk_these_ways"

    # student gets "policy" obs; teacher gets "policy" + "privileged" obs
    obs_groups = {"policy": ["policy"], "teacher": ["policy", "privileged"]}

    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=24,
    )
