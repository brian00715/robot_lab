#!/usr/bin/env bash
# VK_ICD_FILENAMES: 只加载 NVIDIA ICD，避免 RTX scenedb plugin 枚举 Intel/Mesa ICD 崩溃
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-WBC-Flat-Unitree-Go2-X5-H2PFG-Play-v0 \
  --num_envs 1 \
  --checkpoint /home/user/Self_Projects/robot_lab/logs/rsl_rl/unitree_go2_x5_wbc_flat_h2pfg/2026-05-07_19-34-44/model_44999.pt \
  --device cpu
