#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab230

python scripts/reinforcement_learning/rsl_rl/play.py \
--task=RobotLab-Isaac-VelocityPose-Flat-Unitree-Go2-X5-v0 \
--checkpoint=logs/rsl_rl/unitree_go2_x5_velocity_pose_flat/2026-02-14_21-19-53_go2x5_improved_gait/model_3000.pt \
--num_envs=4 \
--device=cpu \
--curriculum_stage=4
