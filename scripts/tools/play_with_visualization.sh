#!/bin/bash
# 运行VelocityPose推理并可视化指令

cd /home/adam/robot_lab

# 激活环境
source /home/adam/anaconda3/bin/activate isaaclab230

# 运行play脚本
/home/adam/anaconda3/envs/isaaclab230/bin/python \
    scripts/reinforcement_learning/rsl_rl/play.py \
    --task=RobotLab-Isaac-VelocityPose-Flat-Unitree-Go2-v0 \
    --checkpoint=logs/rsl_rl/unitree_go2_velocity_pose_flat/2026-01-14_02-53-41/model_64499.pt \
    --num_envs=4 \
    "$@"
