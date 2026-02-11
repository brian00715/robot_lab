cd /home/yzy/MyProject/robot_lab && \
conda activate isaaclab230 && \
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-VelocityPose-Flat-Unitree-Go2-X5-v0 \
  --num_envs=4096 \
  --headless
  --video