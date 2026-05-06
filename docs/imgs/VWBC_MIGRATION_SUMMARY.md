# visual_whole_body 到 robot_lab_raw 移植总结

## 背景

本次移植目标是把 `visual_whole_body` 中的 GO2 + ARX X5 whole-body-control 任务集成到 `robot_lab_raw` 的 Isaac Lab manager-based 任务体系中，形成独立的 `manager_based/wbc/` 任务包，并提供 GO2 + X5 的训练与播放配置。

当前工作分支：

```bash
feat/migrate-vwbc
```

## 新增任务入口

新增 manager-based WBC 任务包：

```text
source/robot_lab/robot_lab/tasks/manager_based/wbc/
├── __init__.py
├── wbc_env.py
├── wbc_env_cfg.py
├── config/
│   └── go2_x5/
│       ├── __init__.py
│       ├── flat_env_cfg.py
│       ├── rough_env_cfg.py
│       └── agents/
│           └── rsl_rl_ppo_cfg.py
└── mdp/
    ├── actions.py
    ├── arm_controller.py
    ├── commands.py
    ├── composite_actions.py
    ├── curriculums.py
    ├── events.py
    ├── observations.py
    ├── rewards.py
    └── visualizers.py
```

核心入口：

- `WbcEnv` 继承 Isaac Lab `ManagerBasedRLEnv`。
- `WbcRoughEnvCfg` 继承现有 locomotion velocity env cfg，并替换 command、reward、curriculum 中与 `base_velocity` 绑定的部分。
- `config/go2_x5/` 提供 GO2 + ARX X5 的 rough、flat、play 配置。

## Gym 任务注册

新增 Gym id：

```text
RobotLab-Isaac-WBC-Flat-Unitree-Go2-X5-v0
RobotLab-Isaac-WBC-Rough-Unitree-Go2-X5-v0
RobotLab-Isaac-WBC-Flat-Unitree-Go2-X5-Play-v0
```

其中 `Play-v0` 是专用播放环境，不修改通用 `scripts/reinforcement_learning/rsl_rl/play.py`。

## 机器人资产

新增 GO2 + X5 asset cfg：

```text
source/robot_lab/robot_lab/assets/go2_x5.py
```

新增本地 USD/mesh/URDF 资源：

```text
source/robot_lab/data/Robots/unitree/go2_x5_description/
```

`GO2_X5_CFG` 指向：

```text
source/robot_lab/data/Robots/unitree/go2_x5_description/usd/go2_x5/go2_x5.usd
```

asset 配置中包含：

- GO2 12 个腿部关节 actuator。
- X5 `joint1` 到 `joint6` 六个机械臂关节 actuator。
- gripper actuator。
- 初始姿态、默认关节角、PD 参数与 delayed actuator 配置。

## MDP 迁移内容

迁移并适配了以下 WBC 逻辑：

- 7D command：`lin_vel_x, lin_vel_y, ang_vel_z, height, roll, pitch, yaw placeholder`
- `UniformVelocityPoseCommandCfg`
- 高度/姿态 command curriculum
- GO2 + arm 组合 action
- ARX5 轨迹控制器
- arm motion modes：
  - circular
  - figure_eight
  - sinusoidal
  - random_walk
  - reach_points
  - fishing
  - grasping
  - swinging
  - probing
- arm joint observation
- arm end-effector relative position observation
- combined center-of-mass offset observation
- height tracking reward
- roll/pitch orientation tracking reward
- standing angular velocity accumulation reward
- WBC command visualizer
- WBC 专用 events、observations、rewards、curriculums

当前 GO2 + X5 policy observation 是 70D：

```text
3 base lin vel
+ 3 base ang vel
+ 3 projected gravity
+ 7 command
+ 12 dog joint pos
+ 12 dog joint vel
+ 12 last action
+ 6 arm joint pos
+ 6 arm joint vel
+ 3 arm ee relative pos
+ 3 combined CoM offset
= 70
```

policy action 维度仍是 12D，只控制 GO2 腿部。机械臂动作由内部轨迹控制器生成，并由 composite action term 合并成完整 joint target。

## 关键适配点

### command 名称

原 locomotion velocity task 使用 `base_velocity`。WBC 任务改为：

```text
base_velocity_pose
```

因此所有 command-aware reward、observation、curriculum 都需要指向 `base_velocity_pose`。

已处理项包括：

- `velocity_commands`
- `track_lin_vel_xy_exp`
- `track_ang_vel_z_exp`
- `stand_still`
- `joint_pos_penalty`
- feet 相关 command-aware rewards
- WBC height/pose curriculum

### 继承 curriculum 清理

`robot_lab_raw v2.3.0` 的 inherited `command_levels` 会硬编码读取 `base_velocity`，在 WBC 任务中会触发：

```text
KeyError: 'base_velocity'
```

处理方式：

- 在 `WbcRoughEnvCfg` 中禁用 inherited `command_levels`。
- 在 GO2 + X5 rough cfg 中再次显式设置：

```python
self.curriculum.command_levels = None
self.curriculum.command_levels_lin_vel = None
self.curriculum.command_levels_ang_vel = None
```

WBC 训练任务使用：

```python
self.curriculum.command_curriculum_height_pose
```

Play 环境中则禁用该 curriculum，避免播放时 reset 把 arm motion scale 覆盖回训练 Stage 1。

### 机械臂动作

迁移后发现 play 中机械臂看起来不动，原因有两个：

1. 训练 Stage 1 原始逻辑中 `arm_motion_scale = 0.0`，机械臂会保持静止。
2. 通用 `play.py` 不负责设置 WBC inference stage，不能在通用脚本里硬编码 WBC 逻辑。

处理方式：

- 训练 Stage 1 改为低强度机械臂扰动：

```python
stage1_arm_motion_scale = 0.3
```

- 新增专用 Play Env：

```python
class ArxX5WbcFlatPlayEnvCfg(ArxX5WbcFlatEnvCfg):
    inference_stage = 4
    fixed_arm_mode_idx = 0
```

- Play Env 默认使用 Stage 4，arm controller scale 为 `2.5`，机械臂动作肉眼可见。

### composite action 映射

GO2 + X5 USD 中 joint 顺序不是简单的 dog joints 后接 arm joints，并且存在 gripper joint。

运行时 joint names 示例：

```text
['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
 'joint1', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint',
 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper_joint', 'joint8']
```

因此 `DogArmCompositeAction` 做了显式映射：

- policy 输出顺序固定为 `FR, FL, RR, RL` 的 12 个狗腿关节。
- arm trajectory 只写入 `joint1` 到 `joint6`。
- gripper 等非狗腿、非 arm trajectory 关节作为 passive joints。
- full joint target 默认从 `default_joint_pos` clone，再覆盖 dog 和 arm target，避免 passive joints 被置零。

### inherited reward 清理

`wheel_vel_penalty` 是 inherited wheeled reward，GO2 + X5 没有 wheel joints。该 reward 在 base cfg 中 weight 为 0，但 Isaac Lab manager 在 play/parse 阶段仍可能解析其空 joint regex。

处理方式：

- GO2 + X5 配置调用 `disable_zero_weight_rewards()`。
- Play Env 也纳入 zero-weight reward 清理范围，避免 `wheel_vel_penalty:asset_cfg` 解析失败。

## Play 用法

推荐使用专用 Play Env：

```bash
conda run -n isaaclab230 python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-WBC-Flat-Unitree-Go2-X5-Play-v0 \
  --num_envs 1
```

该 task 默认：

- `inference_stage = 4`
- `fixed_arm_mode_idx = 0`
- arm motion mode 为 circular
- `command_curriculum_height_pose = None`
- `controller_scale = 2.5`

## 训练用法

Flat 训练 task：

```bash
conda run -n isaaclab230 python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-WBC-Flat-Unitree-Go2-X5-v0
```

Rough 训练 task：

```bash
conda run -n isaaclab230 python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-WBC-Rough-Unitree-Go2-X5-v0
```

## 验证结果

已完成的验证：

- WBC Python 文件 `py_compile` 通过。
- Gym task 注册可见。
- `ArxX5WbcFlatEnvCfg`、`ArxX5WbcRoughEnvCfg`、`ArxX5WbcFlatPlayEnvCfg` 可实例化。
- Play Env 创建、reset、step 探针通过。
- Play Env arm controller 确认为 Stage 4：

```text
registered True
cfg_inference_stage 4
cfg_fixed_arm_mode_idx 0
curriculum_height_pose None
controller_scale 2.5
arm_names ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
target_abs_mean ~= 0.714
delta_abs_mean over 80 steps ~= 0.208
```

这说明 Play Env 中机械臂 target 非零，并且实际 arm joint position 随 step 发生明显变化。

## 当前注意事项

- `config/go2_x5/` 是当前代码实际命名；用户侧口头提到的 ARX_X5 体现在类名 `ArxX5Wbc*` 和 GO2 + ARX X5 任务语义中。
- 通用 `play.py` 未修改，WBC 播放行为由专用 `Play-v0` task 控制。
- Play Env 默认固定 `arm_actions_idx = 0`，如需支持运行时切换 motion mode，可以后续新增多个 Play cfg 或专用 wrapper，而不是污染通用 play 脚本。
- Rough 训练的端到端长时间训练还需要正式跑完整流程；当前验证覆盖注册、配置解析、环境创建、reset、step 和 arm motion。
