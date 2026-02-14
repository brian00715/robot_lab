#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab230

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-VelocityPose-Flat-Unitree-Go2-X5-v0 \
  --num_envs=4096 \
  --headless \
  --video \
  --video_interval=24000 \
  --run_name=go2x5_curriculum_fixed \
  --max_iterations=36000

