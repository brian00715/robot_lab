# H2 实现报告与方案：在当前 IK+PPO 中移植 FPG 风格 IK Feasibility Reward

日期：2026-05-07  
项目：robot_lab (IsaacLab, GO2 + ARX X5)

---

## 1. 目标与原子假设对齐

本报告对应 CoRL 规划文档中的 H2：

- 假设：将 pointwise 的 IK feasibility 作为 dense geometric prior 注入到当前 IK+PPO，可提升 EE 对目标位姿（至少先是位置）的跟踪精度与样本效率。
- 对照：
  - Baseline：当前 VWBC 移植版本（已有 `tracking_ee_world` + locomotion regularization）。
  - H2：Baseline + `r_ik_feasibility`。

H2 的判据沿用主文档定义：

- tracking error 改善 >= 20%，或
- 达到 80% 最终性能所需 step 数减少 >= 30%。

---

## 2. 当前代码实现现状（与 H2 直接相关）

### 2.1 控制与策略职责划分

当前环境是“腿策略 + 机械臂 IK”的组合：

- 策略 action 仅 12 维（狗腿），定义在 `mdp/composite_actions.py` 的 `VisualWholeBodyAction`。
- 机械臂 6 关节不由策略直接输出，而是在 `process_actions()` 内通过 DLS Jacobian IK 实时求目标关节：
  - 当前 EE 误差：`compute_pose_error(...)` 得到 `(pos_error, rot_error)`。
  - IK 求解：
    - `delta_q = J^T (J J^T + lambda^2 I)^(-1) delta_pose`
    - `arm_target = arm_pos + delta_q`。

这意味着：系统已经具备实现 FPG 风格 feasibility reward 的全部核心量（`J`、`delta_pose`、`delta_q`），无需引入新求解器。

### 2.2 现有 EE 跟踪奖励

`mdp/rewards.py` 中已有：

- `tracking_ee_world(env, ee_goal_command_name="ee_goal", sigma=1.0, ee_body_name="ee")`
- 形式：
  - `err = sum(abs(ee_pos_w - ee_goal_w))`
  - `r = exp(-err / sigma * 2)`

`wbc_env_cfg.py` 中配置：

- `tracking_ee_world` 权重是 `0.8 * (1/100)`。
- 整体奖励采用与 VWBC 一致的 `1/100` 缩放（`_VWBC_REW_NORM = 1/100`）。

结论：当前奖励已鼓励“结果误差小”，但尚未显式鼓励“当前姿态在几何上容易 IK 可解”。

### 2.3 命令与目标来源

EE 目标由 `mdp/commands.py` 中的 `EEGoalSphereCommand` 发布：

- `curr_ee_goal_cart_world`
- `ee_goal_orn_quat`

这两个字段已经被 IK 与 `tracking_ee_world` 使用，可直接复用到 feasibility reward。

---

## 3. H2 奖励设计：FPG 风格在本代码库的可落地版本

## 3.1 核心定义

按 H2 目标，我们将可行性定义为“当前关节到 IK 解的距离越小越好，不可解时给 0”：

- 设当前 arm 关节为 $q_{arm}$，一次 DLS 解得到 $q_{IK} = q_{arm} + \Delta q$。
- feasibility 主体：
  $$
  f = \exp\left(-\frac{\|\Delta q_{norm}\|_2^2}{\sigma_q^2}\right)
  $$
- 若不可解，置 0：
  $$
  f = 0
  $$

最终奖励：
$$
r_{ik\_feas} = w_{ik} \cdot f
$$

其中 $\Delta q_{norm}$ 是按关节范围归一化后的增量，避免某个关节量纲主导：
$$
\Delta q_{norm,i} = \frac{\Delta q_i}{q_{max,i}-q_{min,i}+\epsilon}
$$

## 3.2 “可解/不可解”判定（面向当前工程约束）

由于当前 reward 路径没有独立多步 IK 迭代器，也不应在每步引入重计算，建议使用线性化残差门控：

1. 用当前代码同款 DLS 算法得到 `delta_q`。  
2. 计算线性重建残差：
   $$
   r_{lin} = \|J\Delta q - \Delta x\|_2
   $$
   其中 $\Delta x = [\Delta p, \Delta \theta]$。  
3. 判定：
   - 若 `r_lin <= residual_tol` 且 `||pos_error|| <= pos_tol`，认为可解；
   - 否则记为不可解（reward=0）。

建议默认阈值：

- `residual_tol = 0.08`
- `pos_tol = 0.20` (m)

说明：这不是严格全局 IK 可达性证明，但对 PPO dense reward 足够稳定，且与现有单步 DLS 控制链一致。

## 3.3 与现有 tracking_ee_world 的互补关系

- `tracking_ee_world`：结果导向（误差小就好）。
- `ik_feasibility`：过程导向（鼓励进入“容易解”的局部几何区域）。

两者叠加符合 H2 要验证的“几何 prior 是否改善学习效率与精度”。

---

## 4. 代码接入设计（最小侵入）

### 4.1 需要修改的文件

1. `source/robot_lab/robot_lab/tasks/manager_based/wbc/mdp/rewards.py`
- 新增函数 `ik_feasibility_dls(...)`。
- 复用与 `VisualWholeBodyAction._compute_arm_targets()`一致的雅可比提取与 DLS 公式。
- 将新函数加入 `__all__`。

2. `source/robot_lab/robot_lab/tasks/manager_based/wbc/wbc_env_cfg.py`
- 在 `WbcRewardsCfg` 中新增 reward term：
  - 名称：`ik_feasibility`
  - 权重：`w_ik * _VWBC_REW_NORM`
  - 默认建议 `w_ik=1.0`
  - 配置参数：`sigma_q, residual_tol, pos_tol, ik_damping, ee_body_name, ee_goal_command_name, arm_joint_pattern`

3. （可选但强烈建议）新增一个对照任务 ID
- 位置：`source/robot_lab/robot_lab/tasks/manager_based/wbc/config/go2_x5/__init__.py`
- 增加专用实验任务，例如：
  - `RobotLab-Isaac-WBC-Flat-Unitree-Go2-X5-H2PFG-v0`
- 其 env cfg 仅切换 `ik_feasibility` 开/关与权重，其他保持一致。

### 4.2 推荐函数签名

```python
def ik_feasibility_dls(
    env,
    ee_goal_command_name="ee_goal",
    ee_body_name="ee",
    arm_joint_pattern="joint[1-6]",
    asset_cfg=SceneEntityCfg("robot"),
    ik_damping=0.05,
    sigma_q=0.35,
    residual_tol=0.08,
    pos_tol=0.20,
    eps=1e-6,
):
    ...
```

### 4.3 伪代码

```python
asset = env.scene[asset_cfg.name]
arm_ids = resolve_arm_joint_ids_once(env, asset, arm_joint_pattern)
ee_id = resolve_ee_body_once(env, asset, ee_body_name)
cmd = env.command_manager.get_term(ee_goal_command_name)

# current pose and desired pose
p = asset.data.body_pos_w[:, ee_id, :]
q = normalize(asset.data.body_quat_w[:, ee_id, :])
p_des = cmd.curr_ee_goal_cart_world
q_des = cmd.ee_goal_orn_quat

pos_err, rot_err = compute_pose_error(p, q, p_des, q_des, rot_error_type="axis_angle")
delta_x = cat([pos_err, rot_err], dim=-1)

J = get_arm_jacobian(asset, ee_id, arm_ids)
A = J @ J.transpose(1,2) + (ik_damping**2) * I6
delta_q = J.transpose(1,2) @ solve(A, delta_x.unsqueeze(-1))
delta_q = delta_q.squeeze(-1)

# normalized joint-space distance
joint_limits = asset.data.soft_joint_pos_limits[:, arm_ids]
joint_range = joint_limits[...,1] - joint_limits[...,0]
dq_norm = delta_q / (joint_range + eps)

# feasibility gate
lin_res = norm((J @ delta_q.unsqueeze(-1)).squeeze(-1) - delta_x, dim=-1)
pos_norm = norm(pos_err, dim=-1)
solvable = (lin_res <= residual_tol) & (pos_norm <= pos_tol)

f = exp(-sum(dq_norm**2, dim=-1) / (sigma_q**2))
return where(solvable, f, zeros_like(f))
```

---

## 5. 实验设计与日志建议（用于验证 H2）

## 5.1 必做对照

- B0: baseline（当前主线）
- B1: baseline + `ik_feasibility`（`w_ik=1.0`）
- B2/B3/B4: `w_ik in {0.1, 0.5, 2.0}`

每组 3 seeds。

## 5.2 关键指标

训练阶段：

- `train/reward_tracking_ee_world`
- `train/reward_ik_feasibility`
- `train/ik_solvable_ratio`（每步 solvable 的 env 比例）
- `train/ik_delta_q_norm`

评估阶段：

- EE position RMSE（主指标）
- EE orientation error（若开始纳入姿态 tracking）
- 达到固定误差阈值的样本效率（steps）

## 5.3 成功信号

- 同样训练步数下，B1-B4 至少一组显著降低 EE RMSE；
- 或在达到同样 RMSE 时，步数明显下降（>=30%）。

---

## 6. 风险与规避

1. 奖励冲突（过度偏向“可解”而牺牲 locomotion 稳定）
- 规避：从小权重启动（0.1/0.5），并监控 base 稳定相关 reward 是否塌陷。

2. 门控过严导致奖励稀疏
- 规避：先放宽 `residual_tol` 与 `pos_tol`，保证 `ik_solvable_ratio` 在训练初期不是接近 0。

3. 数值尺度不匹配
- 规避：保持与现工程一致的 `1/100` reward norm，并以 `w_ik` 扫描做主调参轴。

---

## 7. 建议的实施顺序

1. 在 `rewards.py` 增加 `ik_feasibility_dls`，先不注册。  
2. 在单元运行中打印/记录 `ik_solvable_ratio` 与 `delta_q_norm` 分布，确认数值稳定。  
3. 在 `wbc_env_cfg.py` 注册 `ik_feasibility`，默认 `w_ik=0.5` 先跑短程 smoke test。  
4. 建立 H2 专用 task id，跑 3-5 个短实验（每个 2k~5k iteration）筛选 `w_ik`。  
5. 固定最优权重后跑完整 3 seeds，对照 baseline 出结论。

---

## 8. 结论

当前代码基础非常适合做 H2：

- 机械臂 IK 链路已存在；
- EE 目标命令已标准化；
- 奖励管理可直接插入新项；
- PPO 训练链无需结构性改造。

因此，H2 的工程复杂度属于“低到中等”，主要工作是：

- 实现并稳定 `ik_feasibility_dls` 的门控与归一化；
- 做小规模权重与阈值扫描；
- 以 RMSE 与 sample efficiency 做严格对照验证。
