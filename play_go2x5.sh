#!/bin/bash
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab230

# mode_indices = {
#   "circular": 0,
#   "figure_eight": 1,
#   "sinusoidal": 2,
#   "random_walk": 3,
#   "reach_points": 4,
#   "fishing": 5,
#   "grasping": 6,
#   "swinging": 7,
#   "probing": 8
#   }

python scripts/reinforcement_learning/rsl_rl/play.py \
--task=RobotLab-Isaac-VelocityPose-Flat-Unitree-Go2-X5-v0 \
--checkpoint=logs/rsl_rl/unitree_go2_x5_velocity_pose_flat/2026-02-17_03-26-24_go2x5_curriculum_stage_5/model_20000.pt \
--num_envs=64 \
--device=cpu \
--curriculum_stage=1 \
# --arm_actions_idx=6
