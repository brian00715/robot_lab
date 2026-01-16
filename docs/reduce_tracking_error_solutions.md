# 减少跟踪误差的优化方案

## 当前状态
- **Stage 3**: 0-15k iterations (±10cm height, ±20° roll, ±12° pitch/yaw)
- **Stage 4**: 15k+ iterations (±15cm height, ±30° roll, ±15° pitch/yaw)
- **Current Weights**:
  - `track_height_exp`: weight=2.0, std=0.5 (tolerance ±50cm)
  - `track_orientation_exp`: weight=1.0, std=0.707 (tolerance ±40°)

## 🎯 方案优先级

### ⭐ 方案 1: 调整奖励标准差（最有效，推荐优先尝试）

**原理**: 减小 `std` 参数会让奖励函数对误差更敏感，惩罚更严格。

**当前问题**:
- Height std=0.5 → 50cm 误差时奖励仍有 37% (太宽容)
- Orientation std=0.707rad (40°) → 40° 误差时奖励仍有 37% (太宽容)

**建议修改**:

```python
# In rough_env_cfg.py, line ~189
self.rewards.track_height_exp = RewTerm(
    func=mdp.track_height_exp,
    weight=2.0,  # Keep current weight
    params={
        "command_name": "base_velocity_pose",
        "std": math.sqrt(0.05),  # 改小! 从 0.25 → 0.05 (std从0.5→0.22m)
        # 新的容忍度: ±5cm误差 → 79%奖励, ±10cm → 61%, ±22cm → 37%
        "sensor_cfg": SceneEntityCfg("height_scanner_base"),
    }
)

# Line ~197
self.rewards.track_orientation_exp = RewTerm(
    func=mdp.track_orientation_exp,
    weight=1.0,  # Keep current weight
    params={
        "command_name": "base_velocity_pose",
        "std": math.sqrt(0.10),  # 改小! 从 0.5 → 0.10 (std从0.707→0.316rad ≈18°)
        # 新的容忍度: ±5°误差 → 76%奖励, ±10° → 58%, ±18° → 37%
    }
)
```

**预期效果**:
- Height 跟踪误差从 ±10-20cm 降到 ±2-5cm
- Orientation 跟踪误差从 ±10-15° 降到 ±3-8°

**测试步骤**:
```bash
# 1. 修改 rough_env_cfg.py
# 2. 重新训练 (--resume 继续)
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-VelocityPose-Flat-Unitree-Go2-v0 \
    --num_envs=4096 \
    --max_iterations=50000 \
    --resume \
    --load_run=2026-01-15_15-09-03 \
    --checkpoint=model_37000.pt \
    --headless

# 3. 观察 TensorBoard 中的 track_height_exp 和 track_orientation_exp 奖励变化
```

---

### ⭐⭐ 方案 2: 增加奖励权重（简单但可能不够精细）

**原理**: 增加权重会让智能体更重视这些奖励项。

**建议修改**:
```python
# Height tracking
self.rewards.track_height_exp.weight = 4.0  # 从 2.0 → 4.0 (2倍)

# Orientation tracking  
self.rewards.track_orientation_exp.weight = 2.0  # 从 1.0 → 2.0 (2倍)
```

**注意事项**:
- 权重太高可能导致其他行为（如步态、稳定性）变差
- 建议先尝试方案1（调整std），如果效果不够再增加权重

---

### ⭐⭐⭐ 方案 3: Stage-based 渐进式奖励参数（最精细）

**原理**: 在不同 Stage 使用不同的奖励参数，逐步收紧容忍度。

**实现**: 在 `curriculums.py` 中添加奖励参数调整逻辑

```python
# In command_curriculum_height_pose function, after setting ranges

# Also adjust reward tolerance based on stage
if hasattr(env, "reward_manager"):
    # Get reward terms
    height_reward_cfg = env.reward_manager.get_term_cfg("track_height_exp")
    orient_reward_cfg = env.reward_manager.get_term_cfg("track_orientation_exp")
    
    if target_stage == 3:  # Medium range, medium tolerance
        height_reward_cfg.params["std"] = math.sqrt(0.08)  # std≈0.28m (±8cm → 70% reward)
        orient_reward_cfg.params["std"] = math.sqrt(0.15)  # std≈0.39rad (±22° → 70% reward)
    elif target_stage == 4:  # Maximum range, strict tolerance
        height_reward_cfg.params["std"] = math.sqrt(0.05)  # std≈0.22m (±5cm → 78% reward)
        orient_reward_cfg.params["std"] = math.sqrt(0.10)  # std≈0.32rad (±18° → 70% reward)
```

---

### 方案 4: 添加误差惩罚项（补充方案）

**原理**: 对大误差额外惩罚，迫使策略更保守。

**实现**: 在 `rewards.py` 中添加新的惩罚项

```python
def height_error_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.05,  # 5cm threshold
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for height tracking error exceeding threshold."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_height = command[:, 3]
    current_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    error = torch.abs(current_height - target_height)
    # Only penalize if error > threshold
    penalty = torch.clamp(error - threshold, min=0.0)
    return -penalty  # Negative reward

def orientation_error_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.174,  # 10° threshold (0.174 rad)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for orientation tracking error exceeding threshold."""
    from isaaclab.utils.math import quat_error_magnitude
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # ... (similar to track_orientation_exp but with threshold)
    angle_error = ...  # Calculate angle error
    penalty = torch.clamp(angle_error - threshold, min=0.0)
    return -penalty
```

**配置**:
```python
# In rough_env_cfg.py
self.rewards.height_error_penalty = RewTerm(
    func=mdp.height_error_penalty,
    weight=1.0,
    params={"command_name": "base_velocity_pose", "threshold": 0.05}
)
self.rewards.orientation_error_penalty = RewTerm(
    func=mdp.orientation_error_penalty,
    weight=0.5,
    params={"command_name": "base_velocity_pose", "threshold": 0.174}
)
```

---

### 方案 5: 调整其他相关参数

#### 5.1 增加加速度平滑惩罚

```python
# Make height/orientation changes smoother
self.rewards.base_lin_acc_z_l2.weight = -0.05  # 从 -0.02 增加到 -0.05
self.rewards.base_ang_acc_xy_l2.weight = -0.02  # 从 -0.01 增加到 -0.02
```

#### 5.2 调整 PD 控制增益（如果使用 PD controller）

```python
# In actuator config (if using PD controller)
# Increase stiffness/damping for more responsive tracking
stiffness: 50.0  # Increase from default
damping: 2.0     # Increase proportionally
```

#### 5.3 减小命令重采样时间

```python
# In velocity_pose_env_cfg.py
base_velocity_pose = mdp.UniformVelocityPoseCommandCfg(
    resampling_time_range=(5.0, 8.0),  # 从 (10.0, 10.0) 改为更短
    # 更频繁的命令变化会迫使策略更快响应
)
```

---

## 🎯 推荐实施顺序

### Phase 1: 快速验证（1-2小时训练）
1. **实施方案1**: 调整 std 参数
   - Height std: 0.5 → 0.22 (改小 `math.sqrt(0.05)`)
   - Orientation std: 0.707 → 0.316 (改小 `math.sqrt(0.10)`)
2. 用 `--resume` 继续训练 5000 iterations
3. 观察 TensorBoard:
   - `Rewards/track_height_exp` 应该**下降**（因为标准更严格）
   - `Rewards/track_orientation_exp` 应该**下降**
   - 但最终会逐渐**回升**（策略适应新标准）

### Phase 2: 如果效果不够（再训练2-3小时）
4. **叠加方案2**: 增加权重
   - Height weight: 2.0 → 3.0
   - Orientation weight: 1.0 → 1.5
5. 继续训练 5000 iterations

### Phase 3: 精细调优（可选）
6. **实施方案3**: Stage-based 参数
7. **实施方案4**: 添加误差惩罚项（如果还不够严格）

---

## 📊 监控指标

在 TensorBoard 中重点观察：

### 关键奖励指标
- `Rewards/track_height_exp`: 应该从低谷逐渐回升到 0.8-0.95
- `Rewards/track_orientation_exp`: 应该从低谷逐渐回升到 0.7-0.9
- `Rewards/total`: 总奖励可能暂时下降，但会恢复

### 调试信息（Terminal输出）
```
[DEBUG] Orientation Tracking Reward Statistics:
  Quaternion error angle (deg): mean=5.0, max=15.0  ← 目标: mean<5°, max<12°
  Final reward: mean=0.85  ← 目标: mean>0.80

[DEBUG] Height Tracking Reward Statistics:
  Height error (m): mean=0.02, max=0.08  ← 目标: mean<0.03, max<0.06
  Final reward: mean=0.90  ← 目标: mean>0.85
```

---

## ⚠️ 注意事项

1. **不要一次改太多**: 先试方案1，观察效果再叠加其他方案
2. **监控副作用**: 
   - 步态质量 (`feet_air_time`, `feet_contact`)
   - 稳定性 (`base_height`, `base_orientation`)
   - 能耗 (`action_rate`, `joint_torques`)
3. **保存checkpoint**: 每次修改前保存好的 checkpoint
4. **对比测试**: 用 play.py 对比修改前后的实际表现

---

## 📁 需要修改的文件

### 方案1（推荐先试）:
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/config/quadruped/unitree_go2/rough_env_cfg.py`
  * Line ~189: `track_height_exp` 的 `std` 参数
  * Line ~197: `track_orientation_exp` 的 `std` 参数

### 方案2（叠加）:
- 同上文件
  * Line ~186: `track_height_exp.weight`
  * Line ~196: `track_orientation_exp.weight`

### 方案3（进阶）:
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/mdp/curriculums.py`
  * 在 `command_curriculum_height_pose` 函数中添加奖励参数调整逻辑

### 方案4（补充）:
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/mdp/rewards.py`
  * 添加新的惩罚函数
- `rough_env_cfg.py`
  * 注册新的奖励项

---

## 🚀 立即可用的修改命令

```bash
# 1. 备份当前配置
cp source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/config/quadruped/unitree_go2/rough_env_cfg.py \
   source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/config/quadruped/unitree_go2/rough_env_cfg.py.backup

# 2. 使用 Copilot 修改 rough_env_cfg.py 中的 std 参数（方案1）

# 3. 继续训练
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-VelocityPose-Flat-Unitree-Go2-v0 \
    --num_envs=4096 \
    --max_iterations=50000 \
    --resume \
    --load_run=2026-01-15_15-09-03 \
    --checkpoint=model_37000.pt \
    --headless
```

---

*Created: 2026-01-15*
*For: VelocityPose Stage 3/4 Training Optimization*
