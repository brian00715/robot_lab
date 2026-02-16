#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab230

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-VelocityPose-Flat-Unitree-Go2-X5-v0 \
  --num_envs=4096 \
  --headless \
  --video \
  --video_interval=24000 \
  --resume \
  --load_run=2026-02-16_01-03-18_go2x5_curriculum_stage_5 \
  --checkpoint=model_25000.pt \
  --run_name=go2x5_stage_5_continue \
  --max_iterations=15000 

