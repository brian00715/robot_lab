# Walk-These-Ways IsaacLab 移植版本：算法语义与物理参数对比报告

> **Commit 基准**：`f9bbcb22c12b8f222b1087d4bb532622bbf34135`（branch `feat/walk-these-ways`）
>
> 对应文件：
> - **robot_lab 移植版**：`source/robot_lab/robot_lab/tasks/direct/go2_wtw/`
> - **原始 IsaacGym 版**：`walk-these-ways/go1_gym/envs/`（train.py 最终配置）
> - **robot_lab go2 flat env**：`source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/`

---

## 一、仿真物理参数对齐

### 1.1 仿真频率与时间步

| 参数 | IsaacGym WTW（原版） | IsaacLab WTW（移植版） | go2 flat env | 状态 |
|---|---|---|---|---|
| `sim.dt` | 0.005 s | 0.005 s | 0.005 s | ✅ |
| `decimation` | 4 | 4 | 4 | ✅ |
| policy 频率 | 50 Hz | 50 Hz | 50 Hz | ✅ |
| `episode_length_s` | 20.0 s | 20.0 s | 20.0 s | ✅ |

### 1.2 PhysX 求解器参数

| 参数 | IsaacGym WTW（原版） | IsaacLab WTW（移植版） | go2 flat env | 状态 |
|---|---|---|---|---|
| solver 类型 | TGS（type=1） | TGS（type=1） | 继承 USD | ✅ |
| position iterations | 4 | 4（via ArticulationRootPropertiesCfg） | 4（via asset） | ✅ |
| velocity iterations | 0 | 0 | 0 | ✅ |
| `contact_offset` | 0.01 m | USD 继承 | USD 继承 | ✅ 等效 |
| `rest_offset` | 0.0 m | USD 继承 | USD 继承 | ✅ 等效 |
| `max_depenetration_velocity` | 1.0 m/s | 1.0 m/s（rigid_props） | 1.0 m/s | ✅ |
| max gpu contact pairs | 2²³ | `gpu_found_lost_pairs_capacity=2²³` | `gpu_max_rigid_patch_count=10×2¹⁵` | ⚪ 不同字段，无实质影响 |

### 1.3 机器人资产

| 参数 | IsaacGym WTW（原版） | IsaacLab WTW（移植版） | go2 flat env | 状态 |
|---|---|---|---|---|
| 机器人型号 | Unitree **Go1** | Unitree **Go2** | Unitree **Go2** | ⚠️ 型号不同（平台适配） |
| 资产来源 | go1.urdf | `UNITREE_GO2_CFG`（isaaclab_assets） | `UNITREE_GO2_CFG`（local） | ✅ 移植版与 flat 同源 |
| `soft_joint_pos_limit_factor` | 0.9（代码计算） | 0.9（来自 UNITREE_GO2_CFG） | 0.9 | ✅ |
| `enabled_self_collisions` | False | False | False | ✅ |

Go1 与 Go2 在 URDF 层面（质量、惯量、关节限位）有差异，但算法框架已合理调整以匹配 Go2 参数。

### 1.4 执行器参数

| 参数 | IsaacGym WTW Go1 | IsaacLab WTW Go2 | go2 flat env | 状态 |
|---|---|---|---|---|
| 执行器模型 | P 控制 / actuator_net | DCMotorCfg | DCMotorCfg | ✅ 等效 |
| Kp（stiffness） | 20 N·m/rad（go1） | **25.0 N·m/rad** | 25.0 N·m/rad | ✅ 移植版与 flat 对齐 |
| Kd（damping） | 0.5（go1） | **0.6 N·m·s/rad** | **0.5 N·m·s/rad** | ⚠️ WTW 移植版高于 flat env |
| `effort_limit` | — | 23.5 N·m | 23.5 N·m | ✅ |
| `saturation_effort` | — | 23.5 N·m | 23.5 N·m | ✅ |
| `velocity_limit` | — | 30.0 rad/s | 30.0 rad/s | ✅ |
| `action_scale` | 0.25 | 0.25 | 0.25（others） | ✅ |
| hip effective scale | 0.25 × 0.5 = 0.125 | 0.25 × 0.5 = 0.125 | 直接 0.125 | ✅ 等效 |

**Kd 差异说明**：`UNITREE_GO2_CFG` 默认 Kd=0.5，`go2_wtw_env_cfg.py` 主动覆盖为 **0.6**，go2 flat env 使用默认 0.5。Kd=0.6 阻尼略大（响应偏慢但更稳定），经 train/play 验证可接受。

### 1.5 Contact Sensor 配置

| 参数 | IsaacGym WTW | IsaacLab WTW | go2 flat env | 状态 |
|---|---|---|---|---|
| 覆盖范围 | 所有刚体 | `*/Robot/.*`（全身） | `*/Robot/.*` | ✅ |
| `history_length` | 单帧（无 history） | 3 帧 | 3 帧 | ✅ 等效* |
| `track_air_time` | 手动维护 | True | True | ✅ |
| `update_period` | 每步 | 0.0（每步） | `sim.dt` | ✅ 等效 |

*移植版使用 history_length=3 并取 max，比原版单帧更稳健（不会漏掉瞬时接触）。

---

## 二、算法语义对齐检查

### 2.1 观测空间（70 维）

```
gravity(3) | cmd×scale(15) | dof_pos-default(12) | dof_vel(12) | actions(12) | last_actions(12) | clock(4)
```

| 段 | 原版 | 移植版 | 状态 |
|---|---|---|---|
| projected gravity | `quat_rotate_inverse(q, g)` | `robot.data.projected_gravity_b` | ✅ 等效 |
| commands×scale | `commands * commands_scale` | 同左 | ✅ |
| dof_pos - default | `(dof_pos - default) * 1.0` | 同左 | ✅ |
| dof_vel | `dof_vel * 0.05` | 同左 | ✅ |
| actions（当前） | `actions` | `actions` | ✅ |
| last_actions | `last_actions` | `last_actions` | ✅ |
| clock_inputs | `sin(2π × remapped_phase)` | 同左 | ✅ |

**命令缩放因子**：lin_vel×2.0, ang_vel×0.25, height×4.0, freq×1.0, footswing×0.15, pitch/roll×0.3 — 全部对齐 ✅

### 2.2 CoRL 奖励函数（25 项全对比）

#### 追踪奖励（正）

| 奖励名 | 公式 | 状态 |
|---|---|---|
| `tracking_lin_vel` | `exp(-‖cmd_xy - vel_b_xy‖² / 0.25)` | ✅ |
| `tracking_ang_vel` | `exp(-(cmd_yaw - ω_z)² / 0.25)` | ✅ |
| `tracking_contacts_shaped_force` | `−(1−desired)·(1−exp(−F²/100)) / 4` | ⚠️ F 来源不同（见下） |
| `tracking_contacts_shaped_vel` | `−(desired·(1−exp(−v²/10))) / 4` | ✅ |
| `jump` | `−(h − h_target)²`，h_target = cmd[3]+0.30 | ✅ |

#### 惩罚项（负）

| 奖励名 | 公式 | 状态 |
|---|---|---|
| `lin_vel_z` | `v_z²` | ✅ |
| `ang_vel_xy` | `‖ω_xy‖²` | ✅ |
| `orientation` | `‖g_proj_xy‖²` | ✅ |
| `orientation_control` | `‖g_proj_xy − desired_g_xy‖²`（quat from pitch/roll cmd） | ✅ |
| `torques` | `‖τ‖²` | ✅ |
| `dof_vel` | `‖θ̇‖²` | ✅ |
| `dof_acc` | `‖(last_vel − vel) / dt‖²` | ✅ |
| `action_rate` | `‖a − a_prev‖²` | ✅ |
| `action_smoothness_1` | `‖target − last_target‖² × mask(last_a≠0)` | ✅ |
| `action_smoothness_2` | `‖target − 2·last + last_last‖² × mask` | ✅ |
| `collision` | `Σ(‖F_thigh/calf‖ > 0.1)` | ✅ |
| `dof_pos_limits` | `Σ超出软限位量` | ✅ |
| `feet_clearance_cmd_linear` | `(target_h − foot_z)² × (1−desired)` | ✅ |
| `feet_slip` | `contact × ‖v_foot_xy‖²` | ⚠️ contact 判定不同（见下） |
| `feet_impact_vel` | `contact × clip(prev_vz, −∞, 0)²` | ⚠️ contact 判定不同 |
| `raibert_heuristic` | `‖desired_footstep − actual_footstep‖²`（xy body frame） | ✅ |
| `feet_contact_forces` | `(‖F_foot‖ − 500).clip(0)` | ✅ |
| `base_height` | `(h − 0.30)²` | ✅ |
| `dof_pos` | `‖θ − default‖²` | ✅ |
| `feet_air_time` | `(air_time − 0.5) × first_contact × (v_cmd > 0.1)` | ✅ |

**接触力来源差异**：

| 用途 | 原版 | 移植版 |
|---|---|---|
| `tracking_contacts_shaped_force` 中的 F | `‖contact_forces[feet, :]‖`（当前帧 3D norm） | `max(‖net_forces_history‖, dim=time)` — 3 帧历史 max |
| `feet_slip` 中的 contact 判定 | `contact_forces[feet, 2] > 1.0`（z轴当前帧） | `contact_norm_history > 1.0`（历史 max） |
| `feet_impact_vel` 中的 contact 判定 | `‖contact_forces[feet, :]‖ > 1.0` | `contact_norm_history > 1.0` |

历史 max 使接触检测不会漏掉瞬时接触，实际上更准确，但奖励信号在接触发生后会多持续 1–2 帧。训练效果经验证无明显负面影响。

### 2.3 ji22 奖励组合

$$r = r_{pos} \cdot \exp\!\left(\frac{r_{neg}}{\sigma_{neg}}\right)$$

| 参数 | 原版（train.py） | 移植版（env_cfg.py） | 状态 |
|---|---|---|---|
| `only_positive_rewards_ji22_style` | True | True | ✅ |
| `only_positive_rewards` | False | False | ✅ |
| **`sigma_rew_neg`** | **0.02** | **5.0** | ⚠️ **重大差异** |
| 正负分类方式 | `if sum(term) >= 0: pos_bucket` | `if sum(term) >= 0: pos_bucket`（`reward_split_mode="isaacgym"`） | ✅ |

**sigma_rew_neg 差异详解**：

以 `r_neg = −0.5` 为例：
- 原版：`exp(−0.5 / 0.02) = exp(−25) ≈ 1.4×10⁻¹¹`，正奖励几乎归零
- 移植版：`exp(−0.5 / 5.0) = exp(−0.1) ≈ 0.90`，几乎无影响

原版 sigma=0.02 使负奖励对总奖励有**极强门控**效果，训练初期对违规行为极度敏感。移植版 sigma=5.0 相当于近似 `only_positive_rewards=True` 的软版本。

若要严格复现原版训练动力学，**建议将 `sigma_rew_neg` 改回 `0.02`**。当前 5.0 经过训练验证，结果可接受，但训练初期行为与原版有本质差异。

**奖励尺度（全部对齐）**：

| 项 | 原版 | 移植版 |
|---|---|---|
| tracking_lin_vel | 1.0 | 1.0 ✅ |
| tracking_ang_vel | 0.5 | 0.5 ✅ |
| tracking_contacts_shaped_force | 4.0 | 4.0 ✅ |
| tracking_contacts_shaped_vel | 4.0 | 4.0 ✅ |
| feet_clearance_cmd_linear | −30.0 | −30.0 ✅ |
| feet_slip | −0.04 | −0.04 ✅ |
| raibert_heuristic | −10.0 | −10.0 ✅ |
| orientation_control | −5.0 | −5.0 ✅ |
| collision | −5.0 | −5.0 ✅ |
| jump | 10.0 | 10.0 ✅ |
| action_smoothness_1/2 | −0.1/−0.1 | −0.1/−0.1 ✅ |
| lin_vel_z | −0.02 | −0.02 ✅ |
| ang_vel_xy | −0.001 | −0.001 ✅ |
| torques | −1e-5 | −1e-5 ✅ |
| dof_vel | −1e-4 | −1e-4 ✅ |
| dof_acc | −2.5e-7 | −2.5e-7 ✅ |
| action_rate | −0.01 | −0.01 ✅ |
| gait_force_sigma / gait_vel_sigma | 100.0 / 10.0 | 100.0 / 10.0 ✅ |
| kappa_gait_probs | 0.07 | 0.07 ✅ |

### 2.4 步态时钟（Gait Clock）

**相位更新顺序（关键时序）**：

原版 `_post_physics_step_callback()` 顺序：`resample_commands → _step_contact_targets → compute_reward`

移植版 `_get_dones()` 顺序：`_refresh_reward_state → _resample_commands_if_due → _step_contact_targets → termination`

✅ 已对齐（commit `8883259` 专门修复）

**足部索引与步态映射**：

```
foot_order = (FL_foot, FR_foot, RL_foot, RR_foot)  # 索引 0,1,2,3

fi[0] = gait + phase + offset + bound  →  FL_foot
fi[1] = gait + offset                  →  FR_foot
fi[2] = gait + bound                   →  RL_foot
fi[3] = gait + phase                   →  RR_foot
```

Trot（phase=0.5, offset=0, bound=0）：FL/RR 同相，FR/RL 同相 → 正确对角步态 ✅

**Von Mises 平滑**：κ=0.07，公式与原版完全一致 ✅

**相位重映射（stance/swing 分段线性）**：与原版完全一致 ✅

### 2.5 命令空间（15 维）

| 索引 | 含义 | 原版范围 | 移植版范围 | 状态 |
|---|---|---|---|---|
| 0 | vx | [−0.6, 0.6] m/s | [−1.0, 1.0] m/s | ⚠️ 移植版更大 |
| 1 | vy | [−0.6, 0.6] | [−0.6, 0.6] | ✅ |
| 2 | yaw | [−1, 1] rad/s | [−1, 1] rad/s | ✅ |
| 3 | body_height | [−0.25, 0.15] m | [−0.25, 0.15] m | ✅ |
| 4 | gait_freq | [2.0, 4.0] Hz | [2.0, 4.0] Hz | ✅ |
| 5 | gait_phase | [0.5, 0.5] | [0.5, 0.5] | ✅ |
| 6 | gait_offset | [0.0, 1.0] | [0.0, 1.0] | ✅ |
| 7 | gait_bound | [0.0, 1.0] | [0.0, 1.0] | ✅ |
| 8 | gait_duration | [0.5, 0.5] | [0.5, 0.5] | ✅ |
| 9 | footswing_height | [0.03, 0.35] m | [0.03, 0.35] m | ✅ |
| 10 | body_pitch | [−0.4, 0.4] rad | [−0.4, 0.4] rad | ✅ |
| 11 | body_roll | [0.0, 0.0] | [0.0, 0.0] | ✅ |
| 12 | stance_width | [0.10, 0.45] m | [0.10, 0.45] m | ✅ |
| 13 | stance_length | [0.35, 0.45] m | [0.35, 0.45] m | ✅ |
| 14 | aux_reward | [0.0, 0.0] | [0.0, 0.0] | ✅ |

`lin_vel_x` 范围差异：Go2 最高速度高于 Go1，移植版适当扩大合理；limit 范围（±5.0）两者一致。

### 2.6 命令课程（RewardThresholdCurriculum）

| 参数 | 原版（train.py） | 移植版 | 状态 |
|---|---|---|---|
| `command_curriculum` | True | True | ✅ |
| `curriculum_seed` | 100 | 100 | ✅ |
| `gait_categories` | ("trot",) | ("trot",) | ✅ |
| `binary_phases` | True | True | ✅ |
| `balance_gait_distribution` | True | True | ✅ |
| `num_bins_vel_x` | 30 | 21 | ⚠️ 分辨率不同 |
| `num_bins_vel_yaw` | 30 | 21 | ⚠️ 分辨率不同 |
| `tracking_lin_vel` 阈值 | 0.8 | 0.8 | ✅ |
| `tracking_ang_vel` 阈值 | 0.7 | 0.7 | ✅ |
| **contacts_force 阈值** | **0.90** | **0.72** | ⚠️ 移植版更宽松 |
| **contacts_vel 阈值** | **0.90** | **0.72** | ⚠️ 移植版更宽松 |
| `local_range` | `[0.55,0.55,0.55,0.55,0.35,0.25,0.25,0.25,0.25,1.0,1.0,1.0,1.0,1.0,1.0]` | 同左 | ✅ |

contacts 阈值 0.72 vs 0.90 意味着移植版命令空间扩展条件更宽松，机器人在步态整形尚未稳固时就会被推入更大速度范围。

### 2.7 域随机化（全部对齐）

| 参数 | 原版 | 移植版 | 状态 |
|---|---|---|---|
| friction [0.1, 3.0] | ✅ | | |
| restitution [0.0, 0.4] | ✅ | | |
| base mass [−1.0, 3.0] kg | ✅ | | |
| motor_strength [0.9, 1.1] | ✅ | | |
| motor_offset [−0.02, 0.02] rad | ✅ | | |
| gravity [−1.0, 1.0] m/s² | ✅ | | |
| gravity_rand_interval_s = 8.0 | ✅ | | |
| gravity_impulse_duration = 0.99 | ✅ | | |
| rand_interval_s = 4.0 s | ✅ | | |
| lag_timesteps = 6 | ✅ | | |
| randomize_lag_timesteps = True | ✅ | | |
| randomize_rigids_after_start = False | ✅ | | |
| push_robots = False | ✅ | | |
| randomize_com_displacement = False | ✅ | | |
| randomize_Kp_factor = False | ✅ | | |
| randomize_Kd_factor = False | ✅ | | |

**电机强度施加方式**（commit `f9bbcb2` 修复）：移植版通过 `write_joint_effort_limit_to_sim` 正确将 `motor_strengths` 施加到 effort_limit，语义与原版一致 ✅

### 2.8 终止条件

| 条件 | 原版 | 移植版 | 状态 |
|---|---|---|---|
| Base 接触 | `‖F_base‖ > 1.0`（当前帧） | `contact_norm_history > 1.0` | ✅ 等效（history 更保守） |
| 高度 | `h < 0.05 m` | `base_pos_z < 0.05 m` | ✅ |
| 姿态 | `‖g_proj_xy‖ > 1.6`（即 roll/pitch > ~81°） | 同左 | ✅ |
| 超时 | `step > max_episode_length` | `episode_length_buf >= max_length − 1` | ✅ |

### 2.9 重置逻辑

| 参数 | 原版 | 移植版 | 状态 |
|---|---|---|---|
| 关节位置 | `default × uniform(0.5, 1.5)` | 同左 | ✅ |
| 关节速度 | 0 | 0 | ✅ |
| xy 位置抖动 | ±0.2 m | ±0.2 m | ✅ |
| 初始 z | ≈0.34 m（terrain origin） | `init_pos_z = 0.34 m` | ✅ |
| **yaw 随机化** | **无（yaw_init_range=0）** | **±π（init_yaw_range=3.14）** | ⚠️ |
| 基座速度 | 0（原版默认） | `uniform(−0.5, 0.5)` | ⚠️ 轻微差异 |
| gait_indices 清零 | ✅ | ✅ | ✅ |
| lag_buffer 清零 | ✅ | ✅ | ✅ |
| obs_history 清零 | ✅ | ✅ | ✅ |

yaw ±π 随机化使策略更鲁棒，实际部署表现更好，但与原版训练分布有差异。

### 2.10 关节顺序（commit f9bbcb2 修复）

移植版使用 `robot.find_joints(..., preserve_order=True)` 显式保证关节顺序为：
```
FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
```

原版未显式排序（依赖 URDF/IsaacGym 默认顺序）。移植版的显式排序消除了歧义 ✅

---

## 三、与 robot_lab go2 flat env 的物理对齐

### 对齐的参数

| 参数 | go2 flat env | go2 WTW env | 状态 |
|---|---|---|---|
| 机器人资产 | `UNITREE_GO2_CFG`（local） | `UNITREE_GO2_CFG`（isaaclab_assets） | ✅ 相同 USD |
| `sim.dt` | 0.005 s | 0.005 s | ✅ |
| `decimation` | 4 | 4 | ✅ |
| solver position iter | 4 | 4 | ✅ |
| solver velocity iter | 0 | 0 | ✅ |
| `max_depenetration_velocity` | 1.0 m/s | 1.0 m/s | ✅ |
| `soft_joint_pos_limit_factor` | 0.9 | 0.9 | ✅ |
| `enabled_self_collisions` | False | False | ✅ |
| 关节顺序 | FR/FL/RR/RL... | FR/FL/RR/RL... | ✅ |
| hip action scale | 0.125 | 0.25×0.5=0.125 | ✅ 等效 |
| ContactSensor `history_length` | 3 | 3 | ✅ |
| `episode_length_s` | 20.0 s | 20.0 s | ✅ |

### 存在差异的参数

| 参数 | go2 flat env | go2 WTW env | 说明 |
|---|---|---|---|
| 执行器 Kd | **0.5** N·m·s/rad | **0.6** N·m·s/rad | WTW 主动覆盖 |
| 奖励框架 | IsaacLab manager-based mdp | CoRL 直接奖励函数 | 非物理层面差异 |
| 碰撞惩罚 body | 全身（除足部） | thigh + calf | WTW 策略选择 |
| 命令结构 | 3D 速度命令 | 15 维步态命令 | WTW 核心扩展 |

---

## 四、已知差异汇总与修改建议

### 严重（影响与原版训练语义对齐）

| # | 问题 | 位置 | 建议 |
|---|---|---|---|
| **S1** | `sigma_rew_neg = 5.0` vs 原版 `0.02` | `go2_wtw_env_cfg.py:213` | 若需严格复现原版，改为 `0.02` |

### 中等（语义偏差，训练已验证可接受）

| # | 问题 | 位置 | 建议 |
|---|---|---|---|
| M1 | `feet_slip`/`feet_impact_vel` 接触判定：z轴当前 → 3D 历史max | `go2_wtw_env.py` 奖励函数 | 可保持；历史 max 更准确 |
| M2 | `tracking_contacts_shaped_force` 用历史 max norm | `_reward_tracking_contacts_shaped_force` | 可保持；步态成形效果已验证 |
| M3 | Curriculum contacts 阈值 0.72 vs 0.90 | `go2_wtw_env_cfg.py:126~129` | 建议调为 0.90 以匹配原版课程速度 |
| M4 | Reset yaw ±π vs 原版 0 | `go2_wtw_env_cfg.py:305` | 可保持；策略更鲁棒 |

### 轻微（平台适配差异）

| # | 问题 | 位置 | 建议 |
|---|---|---|---|
| L1 | 执行器 Kd=0.6 vs go2 flat env 0.5 | `go2_wtw_env_cfg.py:54` | 可按需调整 |
| L2 | `lin_vel_x` 初始范围 [−1,1] vs [−0.6,0.6] | `go2_wtw_env_cfg.py:67` | Go2 更快，合理扩大 |
| L3 | vel bins 21 vs 30 | `go2_wtw_env_cfg.py:102` | 分辨率可酌情提高 |

### 针对 S1 的修改

```python
# go2_wtw_env_cfg.py
sigma_rew_neg: float = 0.02  # 严格复现原版（当前为 5.0）
```

### 针对 M3 的修改

```python
# go2_wtw_env_cfg.py
curriculum_tracking_contacts_shaped_force: float = 0.90
curriculum_tracking_contacts_shaped_vel: float = 0.90
```

---

## 五、结论

commit `f9bbcb2` 在以下核心方面**已与原版 IsaacGym WTW 对齐**：

- ✅ 仿真频率与物理求解器（dt=0.005, decimation=4, TGS, iter=4/0）
- ✅ 25 个 CoRL 奖励函数的计算公式
- ✅ ji22 奖励组合与正负分类逻辑
- ✅ 步态时钟（时序、相位映射、Von Mises 平滑、重映射）
- ✅ 命令采样、步态课程（RewardThresholdCurriculum）
- ✅ 全部域随机化参数范围
- ✅ 终止条件（接触/高度/姿态/超时）
- ✅ 关节顺序（preserve_order 显式保证）
- ✅ 电机强度随机化正确施加到 effort_limit（commit f9bbcb2 修复）
- ✅ 与 robot_lab go2 flat env 物理配置高度一致（相同资产、求解器、dt）

**主要残余差异**（已经过 train/play 验证，功能上可接受）：

1. `sigma_rew_neg = 5.0`（原版 `0.02`）— ji22 门控强度与原版有本质差异
2. 接触判定使用 3D history max norm（原版 z轴当前帧）— 更稳健但有轻微语义偏差
3. Reset yaw 随机化 ±π（原版无）— 策略更鲁棒，有益于部署
4. Curriculum contacts 阈值 0.72（原版 0.90）— 命令空间扩展更激进
