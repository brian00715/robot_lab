# Go2 Walk These Ways IsaacLab Migration Summary

本文档总结 `walk-these-ways-go2` IsaacGym 任务迁移到 `robot_lab` IsaacLab Direct RL 任务后的关键改动、语义对齐点、训练方式和剩余缺口。

目标任务：

- Gym id: `RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0`
- Env cfg: `robot_lab.tasks.direct.go2_wtw.go2_wtw_env_cfg:Go2WalkTheseWaysEnvCfg`
- Env: `robot_lab.tasks.direct.go2_wtw.go2_wtw_env:Go2WalkTheseWaysEnv`
- RSL-RL cfg: `robot_lab.tasks.direct.go2_wtw.agents.rsl_rl_ppo_cfg:Go2WalkTheseWaysPPORunnerCfg`

## Migration Scope

迁移目标是让 IsaacLab Direct RL 版本尽量保持原 IsaacGym 版本的算法语义，包括：

- MDP observation/action/reward 建模
- command curriculum 和 gait command 采样
- reward recipe 和 reward scale 语义
- domain randomization
- CSE/RMA 风格训练 recipe
- RSL-RL 训练和回放入口

明确未完成的主要缺口：

- 原版 `actuator_net` 控制分支尚未迁移。当前仍使用 IsaacLab `DCMotorCfg` 位置控制方式，虽然 PD 参数和 action scale 对齐了原配置，但不等价于原版 actuator network torque model。

## File Layout

核心文件：

- `go2_wtw_env_cfg.py`
  - Direct RL 环境参数、命令范围、奖励参数、随机化参数、初始化范围、观测空间定义。
- `go2_wtw_env.py`
  - Direct RL 环境实现，包括 observation、reward、command curriculum、domain randomization、reset 和 step callbacks。
- `go2_wtw_curriculum.py`
  - 从原版 `RewardThresholdCurriculum` 迁移的 command curriculum grid。
- `agents/rsl_rl_ppo_cfg.py`
  - WTW 默认 RSL-RL/CSE 训练配置。
- `agents/rsl_rl_rma.py`
  - 自定义 `Go2WTWActorCritic` 和 `Go2WTWPPO`，用于复刻原版 `ppo_cse` 的 adaptation module 训练。
- `scripts/reinforcement_learning/rsl_rl/train.py`
  - 训练入口，WTW 任务会注册自定义 RMA classes。
- `scripts/reinforcement_learning/rsl_rl/play_wtw.py`
  - WTW 专用回放入口，兼容新 RMA checkpoint 和旧标准 ActorCritic checkpoint。

## Environment Semantics

### Simulation Rate

当前配置保持原版 policy rate：

- sim dt: `0.005`
- decimation: `4`
- policy dt: `0.02`

这对应原版 IsaacGym 中 `self.dt = decimation * sim.dt`。

### Action

动作维度为 12，对应 12 个关节。

当前 Direct RL 环境中：

- action clip: `10.0`
- action scale: `0.25`
- hip joint scale reduction: `0.5`
- PD stiffness: `25.0`
- PD damping: `0.6`
- effort limit: `23.5`

注意：这里仍是位置目标控制，不是原版 `actuator_net`。

### Observation

单帧 observation 维度为 70：

- projected gravity: 3
- scaled commands: 15
- joint position delta: 12
- joint velocity: 12
- current action: 12
- previous action: 12
- gait clock inputs: 4

新增并启用 CSE/RMA 训练所需的 history：

- `num_scalar_observations = 70`
- `num_observation_history = 30`
- `obs_history` 维度为 `30 * 70 = 2100`

环境返回 observation dict：

- `policy`: 当前 70 维单帧 obs
- `obs_history`: 2100 维历史 obs
- `privileged`: 2 维 teacher/system-id target

### Privileged Observation

当前 privileged obs 为：

- friction
- restitution

并按原版 `get_scale_shift` 风格映射到 normalized range：

- friction obs range: `[0.0, 1.0]`
- restitution obs range: `[0.0, 1.0]`

## Reward Semantics

原版 IsaacGym 在 `_prepare_reward_function()` 中会将所有非零 reward scale 乘以 policy dt：

```python
self.reward_scales[key] *= self.dt
```

迁移版本已恢复这一点：

```python
self.reward_scales = {name: scale * self.step_dt for name, scale in self.reward_scales.items()}
```

因此 reward scale 表示“每秒权重”，实际每步 reward 是积分后的 policy-step reward。这一点对 PPO return/advantage 尺度和 curriculum success threshold 都有影响。

同时恢复原版 ji22 style reward setting：

- `only_positive_rewards_ji22_style = True`
- `sigma_rew_neg = 0.02`

## Command Curriculum

当前 command 维度为 15，顺序对齐原版：

1. `x_vel`
2. `y_vel`
3. `yaw_vel`
4. `body_height`
5. `gait_frequency`
6. `gait_phase`
7. `gait_offset`
8. `gait_bound`
9. `gait_duration`
10. `footswing_height`
11. `body_pitch`
12. `body_roll`
13. `stance_width`
14. `stance_length`
15. `aux_reward_coef`

恢复的训练 command ranges：

- `lin_vel_x = [-1.0, 1.0]`
- `lin_vel_y = [-0.6, 0.6]`
- `ang_vel_yaw = [-1.0, 1.0]`
- `body_height_cmd = [-0.25, 0.15]`
- `gait_frequency = [2.0, 4.0]`
- `gait_phase/offset/bound = [0.0, 1.0]`
- `gait_duration = [0.5, 0.5]`
- `footswing_height = [0.03, 0.35]`
- `body_pitch = [-0.4, 0.4]`
- `body_roll = [-0.0, 0.0]`
- `stance_width = [0.10, 0.45]`
- `stance_length = [0.35, 0.45]`
- `aux_reward_coef = [0.0, 0.0]`

恢复的 curriculum limit ranges include：

- `limit_vel_x = [-5.0, 5.0]`
- `limit_vel_y = [-0.6, 0.6]`
- `limit_vel_yaw = [-5.0, 5.0]`
- 其他 gait/body/stance/aux limit 与原训练脚本对齐

恢复的 gait category order：

```python
["pronk", "trot", "pace", "bound"]
```

恢复的 gait 处理逻辑：

- gaitwise curricula
- category-specific phase structure
- binary phase quantization
- small XY command dead-zone
- curriculum local expansion range
- contact shaped force/velocity success threshold: `0.90`

## Domain Randomization

已迁移并补强的 randomization：

- friction
- restitution
- base mass payload
- COM displacement buffer and PhysX writeback
- motor strength
- motor offset
- Kp/Kd factor buffers
- action lag
- gravity randomization
- optional robot push

重要修复：

- base mass / COM 不再只写 Python buffer，会尝试通过 `root_physx_view` 写回 PhysX。
- friction / restitution 会尝试写回 material properties。
- gravity randomization 会同时更新 reward 使用的 `gravity_vec`，并尝试写回 USD physics scene。

这些 PhysX writeback API 在不同 IsaacLab/IsaacSim 组合中可能有差异，因此代码里保留了兼容性 fallback。

## Reset And Initial State

恢复原版 reset 随机化：

- `init_x_range = 0.2`
- `init_y_range = 0.2`
- `init_yaw_range = 3.14`
- `init_vel_range = 0.5`
- `init_pos_z = 0.34`

reset 时还会清空：

- action history
- joint target history
- gait index
- feet air time
- obs history
- lag buffer

## CSE/RMA Training

当前默认训练配置已经切到 CSE/RMA 风格，不需要额外指定 agent。

RSL-RL obs groups：

```python
obs_groups = {
    "policy": ["obs_history"],
    "critic": ["obs_history", "privileged"],
}
```

网络结构：

- adaptation module: `2100 -> 256 -> 128 -> 2`
- actor body: `2102 -> 512 -> 256 -> 128 -> 12`
- critic body: `2102 -> 512 -> 256 -> 128 -> 1`
- activation: `elu`
- initial action std: `1.0`

训练逻辑：

- rollout 时 actor 使用 student path：
  - `latent = adaptation_module(obs_history)`
  - `action_mean = actor_body(obs_history, latent)`
- critic 使用 teacher privileged info：
  - `value = critic_body(obs_history, privileged)`
- PPO update 后额外执行 adaptation loss：
  - target: privileged obs
  - prediction: `adaptation_module(obs_history)`
  - loss: MSE
  - adaptation learning rate: `1e-3`
  - substeps: `1`

这对应原版 `go2_gym_learn/ppo_cse` 的核心训练范式。

## Train Command

在 `robot_lab` 根目录：

```bash
cd /home/simon/Projects/Locomotion/walk-these-ways-go2-migrate/robot_lab
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab230
export PYTHONPATH=$PWD/source/robot_lab:$PYTHONPATH
```

小规模 smoke test：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --headless
```

正式训练：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0 \
  --num_envs 4096 \
  --headless
```

日志和 checkpoint：

```text
logs/rsl_rl/go2_walk_these_ways/<timestamp>/model_*.pt
```

## Play Command

建议对新 CSE/RMA checkpoint 显式指定 checkpoint：

```bash
python scripts/reinforcement_learning/rsl_rl/play_wtw.py \
  --task RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0 \
  --num_envs 4 \
  --checkpoint logs/rsl_rl/go2_walk_these_ways/<run>/model_<iter>.pt
```

`play_wtw.py` 还兼容旧 checkpoint：

- 如果 checkpoint 中是标准 RSL-RL `actor.* / critic.*` key，会自动切回旧 `ActorCritic` replay config。
- 如果 checkpoint 中有 `adaptation_module.*`，则使用新的 WTW RMA policy。

这只解决旧 checkpoint 回放，不代表旧 checkpoint 已经具备 CSE/RMA 语义。要得到新范式策略，需要重新训练。

## Validation Performed

已执行的静态验证：

```bash
python3 -m py_compile \
  source/robot_lab/robot_lab/tasks/direct/go2_wtw/go2_wtw_env.py \
  source/robot_lab/robot_lab/tasks/direct/go2_wtw/go2_wtw_env_cfg.py \
  source/robot_lab/robot_lab/tasks/direct/go2_wtw/agents/rsl_rl_ppo_cfg.py \
  source/robot_lab/robot_lab/tasks/direct/go2_wtw/agents/rsl_rl_rma.py \
  scripts/reinforcement_learning/rsl_rl/train.py \
  scripts/reinforcement_learning/rsl_rl/play_wtw.py
```

已执行的 diff 检查：

```bash
git diff --check
```

还做过 RMA policy 小型 smoke test，确认：

- `act(obs)` 输出 shape 为 `[N, 12]`
- `evaluate(obs)` 输出 shape 为 `[N, 1]`
- exporter actor path 可接受 2100 维 history 输入

尚未在本文档生成时记录完整 Isaac Sim runtime 训练收敛验证。

## Remaining Gaps

### 1. Actuator Net

原版使用：

```python
Cfg.control.control_type = "actuator_net"
```

当前迁移版本尚未实现 actuator network torque inference，因此控制动力学仍不严格等价。

这是目前最主要的剩余算法语义缺口。

### 2. Terrain Fidelity

当前通过 IsaacLab `TerrainImporterCfg` 使用 plane terrain。原版训练脚本中 terrain mesh type 是 `trimesh`，但当前训练配置下有效 terrain noise 为 0，行为接近平地。

如果后续需要严格复刻原版 terrain mesh generation、terrain origins、teleport/center robots 等行为，需要单独迁移原版 terrain generation。

### 3. Runtime API Compatibility

PhysX writeback for mass / COM / material / gravity 在不同 IsaacLab 和 IsaacSim 版本中 API 可能有差异。当前实现采用 best-effort writeback 和 fallback，不修改 IsaacLab 源码。

