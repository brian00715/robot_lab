# VWBC IsaacGym → IsaacLab 移植总结

源仓库：`visual_wholebody/low-level/`（IsaacGym，legged_gym 框架）  
目标：`robot_lab_raw/`（IsaacLab 2.3.0，manager-based 框架）  
资产：Unitree GO2 + ARX X5

---

## 来源说明

原始 `visual_wholebody` 不是 IsaacLab 项目，而是 IsaacGym/legged_gym 风格代码：

- low-level 入口：`visual_wholebody/low-level/legged_gym/envs/manip_loco/`
  - 核心类：`ManipLoco`（`manip_loco.py`）
  - 配置：`B1Z1RoughCfgPPO`（`b1z1_config.py`）
  - 奖励：`ManipLoco_rewards`（`rewards/maniploco_rewards.py`）
- high-level 入口：`visual_wholebody/high-level/envs/`
- 典型机器人：**B1 + Z1**（非 GO2 + ARX X5）

原始机械臂运动机制是 EE goal + Jacobian damped-least-squares IK：

- `ManipLoco.step()` 中把 `actions[:, 12:] = 0.`，腿部策略 action 与机械臂 IK 完全分离。
- EE 目标由 `curr_ee_goal_cart / curr_ee_goal_sphere / ee_goal_orn_quat` 维护，由
  `EEGoalSphereCommand` 以球坐标轨迹插值驱动。
- `_control_ik()` 用 damped least-squares Jacobian IK 把 EE pose error 转成 6 个
  arm joint position targets。

原始 `visual_wholebody` **没有** `ARX5TrajectoryController`，也没有
`circular / figure_eight / fishing / grasping` 等预定义 motion mode。

---

## 任务文件结构

```text
source/robot_lab/robot_lab/tasks/manager_based/wbc/
├── __init__.py
├── wbc_env.py
├── wbc_env_cfg.py
├── config/
│   └── go2_x5/
│       ├── __init__.py          # gym 注册（3 个任务 ID）
│       ├── flat_env_cfg.py
│       ├── rough_env_cfg.py
│       └── agents/
│           └── rsl_rl_ppo_cfg.py
├── learning/
│   ├── __init__.py              # VWBCActorCritic / VWBCPPO 注入 rsl-rl globals
│   ├── actor_critic.py
│   └── ppo.py
└── mdp/
    ├── __init__.py
    ├── actions.py
    ├── commands.py
    ├── composite_actions.py
    ├── curriculums.py
    ├── events.py
    ├── observations.py
    ├── rewards.py
    ├── terminations.py
    └── visualizers.py

source/robot_lab/robot_lab/assets/
└── go2_x5.py                   # GO2_X5_CFG（DelayedPD，legs/arm/gripper）
```

---

## Batch 1 — MDP 语义移植

### 观测（`mdp/observations.py`）

`vwbc_full_observation` 单函数产出完整 744 维观测向量，布局与
`manip_loco.compute_observations` 完全对应：

```
proprio  (66)  = roll/pitch(2) + ang_vel(3) + dog_pos_rel(12) + arm_pos_rel(6)
               + dog_vel(12) + arm_vel(6) + last_leg_action(12)
               + foot_contacts(4) + commands(3) + ee_goal_local(3) + ee_orn_zero(3)
priv     (18)  = mass_params(5) + friction(1) + leg_motor_strength−1(12)
history (660)  = 10 × proprio
```

**有意保留的差异**：省略 `reindex_all` / `reindex_feet` 重映射——IsaacLab 中关节/body
顺序由 USD 资产决定，策略通过名称语义学习，不依赖固定 SDK 索引。

### 动作（`mdp/composite_actions.py`）

`VisualWholeBodyAction`：
- **action_dim = 12**（仅狗腿，arm 不在策略输出内）
- arm 由环境通过 IK 驱动（`EEGoalSphereCommand` → damped-LS Jacobian IK）
- 3-step action delay 缓冲区，含课程切换（`delay_curriculum_switch_steps`）
- `motor_strength` 乘子由 event term 注入，用于特权观测

### 指令（`mdp/commands.py`）

| 指令 | 对应原始 | 维度 |
|------|----------|------|
| `VWBCVelocityCommand` | `_resample_commands` | 3D (vx, vy=0, wz) |
| `EEGoalSphereCommand` | `_resample_ee_goal` + `_update_curr_ee_goal` | 球坐标插值 → 笛卡尔 |

### 奖励（`mdp/rewards.py`，`wbc_env_cfg.py`）

共 23 项，全部乘以 `1/100`（对应 `compute_reward` 末尾的 `rew_buf /= 100`）：

| 函数 | b1z1 权重 | 说明 |
|------|-----------|------|
| `tracking_lin_vel_max` | 2.0 | |
| `tracking_ang_vel_yaw` | 0.5 | |
| `lin_vel_z_square` | −1.5 | |
| `ang_vel_xy_square` | −0.2 | |
| `roll_abs` | −2.0 | |
| `base_height_l1` | −5.0 | target=0.55 m |
| `torques_l2_full` | −2.5e-5 | |
| `dof_acc_leg` | −7.5e-7 | |
| `delta_torques_leg` | −1e-7 | |
| `action_rate_leg` | −0.015 | |
| `dof_pos_limits_leg` | −10.0 | |
| `hip_pos_l2` | −0.3 | |
| `work_leg` | −0.003 | |
| `stand_still_exp` | 1.0 | |
| `walking_dof_exp` | 1.5 | |
| `alive` | 1.0 | |
| `collision` | −10.0 | |
| `feet_contact_forces` | −0.001 | max=40 N |
| `feet_drag` | −0.08 | |
| `feet_jerk` | −2e-4 | |
| `feet_air_time` | 2.0 | threshold=0.5 s |
| `feet_height_l2` | 1.0 | target_norm=0.3 m |
| `tracking_ee_world` | 0.8 | arm EE 位置误差 |

**有意省略的原始奖励项**：`tracking_contacts_shaped_force/vel (−2.0 each)`——原始实现在
`observe_gait_commands=False`（b1z1 配置值）时直接 `return 0, 0`，实际无效。

### 终止条件（`mdp/terminations.py`）

| 条件 | 阈值 |
|------|------|
| `bad_orientation` | \|roll\| 或 \|pitch\| > 0.8 rad |
| `base_height_low` | z < 0.1 m |
| `time_out` | episode_length_s = 10 s |

### 事件（domain randomization，`mdp/events.py`）

| 事件 | 参数（对应 b1z1） |
|------|------------------|
| `randomize_friction` | [0.3, 3.0] |
| `randomize_base_mass_and_com` | mass add [0, 15] kg；COM ±0.15 m；gripper [0, 0.1] kg |
| `randomize_motor_strength` | leg [0.7, 1.3] |
| `reset_root_state` | xy ±0.5 m，yaw ±π/2，vel ±0.1 |
| `reset_joints` | ×[0.8, 1.2] |
| `push_robot` | 间隔 8 s，±0.5 m/s，零指令时 2.5× 加强 |

---

## Batch 2 — 自定义 ActorCritic + DAgger

### 网络架构（`learning/actor_critic.py`）

对应 `visual_wholebody/third_party/rsl_rl/.../actor_critic.py`：

```
priv_encoder:    Linear(18→64) → ELU → Linear(64→20) → ELU        # latent_dim=20
history_encoder: StateHistoryEncoder(10 frames × 66 → 20)
                   Linear(66→30) → Conv1d(30→20,k=4,s=2)
                                 → Conv1d(20→10,k=2,s=1) → Linear(30→20)
actor_backbone:  Linear(66+20→128) → ELU
leg_head:        Linear(128→128) → ELU → Linear(128→128) → ELU → Linear(128→12)
critic_backbone: Linear(66+18→128) → ELU
value_head:      Linear(128→128) → ELU → Linear(128→128) → ELU → Linear(128→1)
```

训练时 actor 使用 `priv_encoder`；部署时切换为 `history_encoder`
（通过 `_inference_hist_encoding` 标志）。

### DAgger 训练（`learning/ppo.py`，`VWBCPPO`）

| 机制 | 实现 |
|------|------|
| priv_reg_loss | PPO 主循环内：`‖priv_latent − hist_latent.detach()‖₂`，把 priv encoder 拉向 hist 的可解码域 |
| DAgger pass | 每 `dagger_update_freq=20` 轮：单独优化 `history_encoder`，目标为 `priv_latent.detach()` |
| min_std 下界 | 每轮更新后：`policy.std ← max(std, min_policy_std)` |

### 超参数（`agents/rsl_rl_ppo_cfg.py`，对应 b1z1）

| 参数 | 值 |
|------|----|
| `init_noise_std` | [0.8, 1.0, 1.0] × 4（12 维，arm 已移除） |
| `min_policy_std` | [0.15, 0.25, 0.25] × 4 |
| `priv_encoder_dims` | [64, 20] |
| `priv_reg_coef_schedual` | [0.0, 0.1, 3000, 7000] |
| `dagger_update_freq` | 20 |
| `learning_rate` | 2e-4 |
| `num_steps_per_env` | 24 |
| `gamma / lam` | 0.99 / 0.95 |
| `clip_param` | 0.2 |
| `entropy_coef` | 0.0 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |

### 类注册（`learning/__init__.py`）

`VWBCActorCritic` 和 `VWBCPPO` 注入 `rsl_rl.runners.on_policy_runner` 模块全局命名空间
（`eval()` 作用域），使 `OnPolicyRunner` 能通过 `class_name` 字符串找到它们。

**有意省略的原始特性**：

- `mixing_schedule`：原始将 leg/arm 奖励堆叠为 2D 张量并交叉混合优势值。IsaacLab 奖励流
  为标量，移植需子类化 `RolloutStorage`/`Transition`。已跳过。
- `torque_supervision`：b1z1 中 `torque_supervision=False`。已跳过。

---

## Bug 修复

### 1. `trunk` body 不存在（`config/go2_x5/rough_env_cfg.py`）

**现象**：
```
Not all regular expressions are matched ... trunk: []
```
**原因**：GO2 的躯干 body 名为 `base`，无 `trunk`。  
**修复**：从 collision `body_names` 列表中删除 `"trunk"`。

---

### 2. 观测维度 909 ≠ 744（`mdp/observations.py`）

**现象**：
```
VWBCActorCritic obs dim mismatch: expected 744, got actor=909 critic=909
```
**原因**：`vwbc_full_observation` 内部动态构造
`SceneEntityCfg(contact_sensor_name, body_names=foot_body_pattern)` 后直接传入
`foot_contacts_from_sensor`。该 cfg 未经 manager 的 `prepare_terms()` 解析，
`body_ids = slice(None)`（匹配所有 body）。GO2 contact sensor 追踪 19 个 body，
返回 `(N, 19)` 而非 `(N, 4)`，每帧 proprio 多出 15 维，共多 `15 × 11 = 165` 维。

**修复**：在 `env` 上一次性缓存脚的 body 索引，通过 `sensor.body_names + re.fullmatch`
直接匹配，绕过 SceneEntityCfg 解析流程：

```python
_sensor: ContactSensor = env.scene[contact_sensor_name]
if not hasattr(env, "_vwbc_foot_body_ids"):
    env._vwbc_foot_body_ids = [
        i for i, n in enumerate(_sensor.body_names) if re.fullmatch(foot_body_pattern, n)
    ]
_forces = _sensor.data.net_forces_w_history[:, 0, env._vwbc_foot_body_ids, :]
foot_contacts = (torch.norm(_forces, dim=-1) > contact_threshold).float()
```

---

### 3. `permute` 维度错误（`mdp/commands.py`，`_collision_check`）

**现象**：
```
RuntimeError: permute(sparse_coo): input.dim() = 2 is not equal to len(dims) = 3
```
**原因**：`torch.lerp(...).squeeze(0)` 在 `n=1` 时把 batch 维度压掉，
`(1, 3, T)` → `(3, T)`，后续 `.permute(2, 0, 1)` 要求 3D 张量。  
**修复**：删除多余的 `.squeeze(0)`。

---

### 4. `quat_rotate_inverse` 弃用（`mdp/observations.py`）

`quat_rotate_inverse` → `quat_apply_inverse`（IsaacLab 2.3 重命名）。

---

## 资产清理

### 删除 UMI-on-legs 相关内容

UMI-on-legs 是另一套系统（`mani-centric-wbc`），与 visual_wholebody 无关。

| 操作 | 对象 |
|------|------|
| 删除文件 | `config/go2_x5/umi_locomotion6d_env_cfg.py` |
| 删除 | `assets/go2_x5.py` 中的 `GO2_X5_UMI_CFG` 定义及 umi 专用 PD 增益注释 |
| 删除 | `assets/go2_x5.py` 中注释掉的 `joint_sdk_names` 块 |
| 删除 | `config/go2_x5/__init__.py` 中的 umi 说明注释 |

---

## 任务注册与用法

```
RobotLab-Isaac-WBC-Flat-Unitree-Go2-X5-v0       # 平地训练（推荐调试用）
RobotLab-Isaac-WBC-Flat-Unitree-Go2-X5-Play-v0  # 平地推演（num_envs=1）
RobotLab-Isaac-WBC-Rough-Unitree-Go2-X5-v0      # 粗糙地形训练
```

训练：
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task RobotLab-Isaac-WBC-Flat-Unitree-Go2-X5-v0 \
    --num_envs 64 --max_iterations 2 --headless
```

推演：
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task RobotLab-Isaac-WBC-Flat-Unitree-Go2-X5-Play-v0 \
    --num_envs 1
```
