# VelocityPose Task for Unitree Go2

## 概述

VelocityPose任务是在原有Velocity任务的基础上扩展的，增加了对机器人base的高度和姿态（roll和pitch角）控制。

### 扩展的命令维度

原始Velocity任务的命令维度：
- `lin_vel_x`: 前后线速度
- `lin_vel_y`: 左右线速度  
- `ang_vel_z`: 偏航角速度

VelocityPose任务新增命令维度：
- `height`: 机器人base质心的目标高度
- `roll`: 机器人base的目标roll角
- `pitch`: 机器人base的目标pitch角

**总命令维度**: 从3维扩展到6维 `[lin_vel_x, lin_vel_y, ang_vel_z, height, roll, pitch]`

## 课程学习策略

为了实现课程学习，新增的height、roll和pitch命令初始阶段保持默认值：

- **height**: 初始值为Go2的默认高度 `0.33m`，范围设置为 `(0.0, 0.0)`
- **roll**: 初始值为 `0.0` 弧度，范围设置为 `(0.0, 0.0)`
- **pitch**: 初始值为 `0.0` 弧度，范围设置为 `(0.0, 0.0)`

这意味着在训练初期，机器人会保持默认高度和水平姿态，只需要学习速度跟踪。

## 使用方法

### 1. 激活环境

```bash
conda activate isaaclab230
```

### 2. 训练任务

```bash
# 基础训练命令
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-VelocityPose-Rough-Unitree-Go2-v0 \
    --headless

# 使用更少的环境数量（用于调试）
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-VelocityPose-Rough-Unitree-Go2-v0 \
    --headless \
    --num_envs=512

# 启用可视化（不使用headless模式）
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-VelocityPose-Rough-Unitree-Go2-v0 \
    --num_envs=64
```

### 3. 播放训练好的策略

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=RobotLab-Isaac-VelocityPose-Rough-Unitree-Go2-v0 \
    --num_envs=16
```

## 任务配置文件位置

```
source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/
├── __init__.py                                 # 任务包初始化
├── velocity_pose_env_cfg.py                    # 基础环境配置
├── mdp/                                        # MDP组件
│   ├── __init__.py
│   └── commands.py                             # 扩展的命令生成器
└── config/
    └── quadruped/
        └── unitree_go2/
            ├── __init__.py                     # 任务注册
            ├── rough_env_cfg.py               # Go2特定配置
            └── agents/
                ├── __init__.py
                └── rsl_rl_ppo_cfg.py          # PPO算法配置
```

## 后续扩展

如果需要启用课程学习，可以在配置文件中修改命令范围：

```python
# 在 rough_env_cfg.py 中修改
self.commands.base_velocity_pose.ranges.height = (-0.05, 0.05)  # 允许±5cm的高度变化
self.commands.base_velocity_pose.ranges.roll = (-0.1, 0.1)      # 允许±0.1弧度的roll角
self.commands.base_velocity_pose.ranges.pitch = (-0.1, 0.1)     # 允许±0.1弧度的pitch角
```

## 观察空间变化

观察空间中的`velocity_commands`项现在包含6个元素而不是原来的3个：
- 原始: `[lin_vel_x, lin_vel_y, ang_vel_z]` (3维)
- 现在: `[lin_vel_x, lin_vel_y, ang_vel_z, height, roll, pitch]` (6维)

策略网络会自动适应新的观察空间维度。

## 注意事项

1. **命令名称**: 所有与命令相关的reward和observation项都使用 `base_velocity_pose` 而不是 `base_velocity`
2. **默认高度**: Go2的默认高度设置为 `0.33m`，可以根据实际机器人调整
3. **课程学习**: 初始阶段保持height/roll/pitch为默认值，后续可以逐步增加范围
4. **奖励函数**: 继承了所有Velocity任务的奖励函数，可以后续添加height和pose跟踪奖励

## 与原始Velocity任务的区别

| 特性 | Velocity任务 | VelocityPose任务 |
|------|-------------|-----------------|
| 命令维度 | 3 (线速度x, 线速度y, 角速度z) | 6 (线速度x, 线速度y, 角速度z, 高度, roll, pitch) |
| 控制目标 | 仅速度跟踪 | 速度跟踪 + 高度控制 + 姿态控制 |
| 观察空间 | velocity_commands (3维) | velocity_commands (6维) |
| 课程学习 | 速度范围可调 | 速度、高度、姿态范围均可调 |
| 任务名称 | RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 | RobotLab-Isaac-VelocityPose-Rough-Unitree-Go2-v0 |
