python scripts/reinforcement_learning/rsl_rl/play_wtw_annotated.py \
  --task RobotLab-Isaac-Go2X5-ArmDisturbance-Direct-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/go2_x5_arm_disturbance/2026-05-23_11-43-08/model_49999.pt \
  --video --video_length 1500 --env_id 0 \
  --output_name play_30s_v2