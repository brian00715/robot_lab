# Go2 WTW IsaacGym 等价性恢复记录与实验计划

本文记录当前 `go2_wtw` IsaacLab 迁移版相对原 IsaacGym Walk-These-Ways 的已做改动、仍可疑问题，以及后续实验顺序。目标不是单纯调到“能走”，而是逐步逼近原 IsaacGym 训练出来的整体效果和步态/高度命令响应。

## 当前观察

- 早期错误版本在训练后期会趴下、抖腿或原地不走。
- 恢复 `sigma_rew_neg=5.0` 并取消 `reward_scales *= step_dt` 后，策略能重新正常走。
- 最新一组明显好很多，但仍有两个主要问题：
  - 高度命令跟踪不够明显。
  - gait selector 的步态区分不如 IsaacGym 版本清晰，容易变成 bound/pronk 混合或一种通用步态。

## 已做改动要点

### 1. Reward 尺度与 ji22-style 正负奖励组合

当前 IsaacLab 版使用：

```python
rew_total = rew_pos * exp(rew_neg / sigma_rew_neg)
```

其中 `rew_neg <= 0`。`sigma_rew_neg` 控制负 reward 对正 reward 的门控强度。

- `sigma_rew_neg=0.02` 时，稍大的负项就会让 `exp(rew_neg / sigma)` 接近 0，策略容易学成少动、低动作幅度、四腿一起小幅蹦。
- `sigma_rew_neg=5.0` 时，负项门控放松，训练明显恢复。

当前 workaround：

- 去掉了 `reward_scales *= step_dt`。
- `sigma_rew_neg` 改为 `5.0`。
- 这让训练恢复可用，但还没有证明和原 IsaacGym 的 reward 量纲等价。

### 2. CSE/RMA 路径

已恢复 Go2 WTW 自定义 RMA/CSE 训练路径：

- `scripts/reinforcement_learning/rsl_rl/train.py` 重新注册 `Go2WTWActorCritic` 和 `Go2WTWPPO`。
- `source/robot_lab/robot_lab/tasks/direct/go2_wtw/agents/rsl_rl_ppo_cfg.py` 使用 `Go2WTWActorCriticCfg` 和 `Go2WTWPPOAlgorithmCfg`。
- policy 输入使用 `obs_history`，critic 使用 `obs_history + privileged`。

### 3. Observation history 语义

已恢复原 WTW 的 history timing：

- 先构造当前 observation。
- 把当前 observation 写入 `obs_history`。
- 再更新 `last_actions`、`last_last_actions`、`last_joint_pos_target`、`last_dof_vel` 等 buffer。

这个顺序很重要。原 WTW 的 `HistoryWrapper` 记录的是已经 emit 出去的 obs，而不是提前更新过 history buffer 的 obs。

### 4. Foot/contact 顺序

已显式固定脚顺序：

```text
FL_foot, FR_foot, RL_foot, RR_foot
```

原因是 IsaacLab 中 `contact_sensor.find_bodies()` 和 `robot.find_bodies()` 的 body 顺序不一定一致。contact reward 和 foot kinematics 如果脚顺序错，会导致 desired contact phase 对错腿。

### 5. 高度命令与 contact reward 的临时增强

当前配置做了折中：

- `body_height_cmd = (-0.10, 0.10)`
- `limit_body_height = (-0.10, 0.10)`
- `cmd_scale_body_height = 4.0`
- `rew_jump = 20.0`
- `rew_tracking_contacts_shaped_force = 6.0`
- `rew_tracking_contacts_shaped_vel = 6.0`
- contact curriculum threshold 从 `0.90` 降到 `0.8`

这些是为了让当前 IsaacLab 版本先稳定 work，不代表最终等价配置。

### 6. Play 侧修复

- `play_wtw.py` 和 `play_wtw_keyboard.py` 增加 CSE actor 的 ONNX export fallback，避免 `_StudentActor` 不是 subscriptable 导致 play 崩溃。
- `play_wtw_keyboard.py` 默认 gait 改为 trot，便于视觉检查。
- 增加 keyboard 写 command 的路径，方便检查 height、pitch、gait 参数是否被 policy 响应。

### 7. 训练日志增强

已增加：

- `command_area`
- `command_area_pronk`
- `command_area_trot`
- `command_area_pace`
- `command_area_bound`
- command min/max：height、freq、phase、offset、bound
- action min/max

这些用于判断 curriculum 是没有展开，还是展开了但 policy 不响应。

### 8. Play benchmark 量化评估工具

已新增固定 command 的 play benchmark：

```text
scripts/reinforcement_learning/rsl_rl/play_wtw_benchmark.py
```

目的：把 play 阶段的肉眼判断量化，作为 A1.1b/A2/A3/A4 等实验的统一评估工具。它会加载指定 checkpoint，按固定 command suite 运行 policy，并输出每个 case 的速度、姿态、高度、接触相位和步态指标。

推荐运行方式：

```bash
python scripts/reinforcement_learning/rsl_rl/play_wtw_benchmark.py \
  --task RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0 \
  --num_envs 8 \
  --headless \
  --checkpoint logs/rsl_rl/go2_walk_these_ways/<run>/model_<iter>.pt
```

输出位置默认在 checkpoint 所在 run 目录下：

```text
<run_dir>/benchmark_play/<timestamp>/summary.csv
<run_dir>/benchmark_play/<timestamp>/summary.json
```

内置 suite：

- `--suite standard`：默认，包含 gait、height、posture 测试。
- `--suite gaits`：只测 trot/pace/bound/pronk 和 yaw。
- `--suite height`：统一公平高度测试，height 为 `-0.10/0.00/0.10`。
- `--suite height_full`：原版完整高度范围压力测试，height 为 `-0.25/0.00/0.15`。
- `--suite posture`：只测 pitch/roll command。

也可以用 `--cases_json` 自定义 case：

```json
{
  "cases": [
    {"name": "trot_fast_low", "gait": "trot", "vx": 0.8, "height": -0.1},
    {"name": "pace_turn", "gait": "pace", "vx": 0.3, "yaw_rate": 0.5}
  ]
}
```

核心指标：

- 稳定性：`done_rate`。理想值为 `0`。如果不摔但趴走，`done_rate` 可能仍为 `0`，需要结合高度和 contact duty 看。
- 速度跟踪：`vx_mean`、`vx_rmse`、`yaw_rate_rmse`、`tracking_score`。
- 高度跟踪：`target_height`、`height_mean`、`height_rmse`、`height_score`。
- 姿态跟踪：`pitch_mean/rmse`、`roll_mean/rmse`。当前环境 reward/debug 约定下，pitch/roll 的跟踪目标是 `-cmd`。
- 接触匹配：`contact_match_mean`、`contact_prob_match_mean`。后者使用连续的 `desired_contact_states`，比单纯 bool match 更稳。
- 接触 duty：`contact_duty_*`、`desired_duty_*`、`contact_duty_error_mean`。如果实际 duty 明显高于 desired，说明策略偏向贴地/拖地/撑着走。
- 触地频率：`contact_freq_*`、`contact_freq_mean`、`contact_freq_ratio`、`contact_freq_error`。脚本默认用 `--min_contact_state_s=0.06` 对 contact 做去抖，避免 force threshold 抖动把频率算得过高。
- 相位集中度：`contact_phase_r_*`、`contact_phase_r_mean`。越接近 `1` 表示触地集中在稳定相位；很低表示接触相位分散，步态不干净。
- 步态评分：`gait_score_v2`。这是当前推荐看的 gait 指标；旧的 `gait_score` 保留用于历史对比。
- 综合评分：`overall_score = tracking + gait_score_v2 + height` 的加权结果，只用于粗略排序，不应替代分项分析。

解读规则：

- 如果 `tracking_score` 高、`height_score` 低，说明速度会跟但高度命令没有学好。
- 如果 `gait_score_v2` 低，同时 `contact_freq_ratio > 1.2`，说明实际触地过频，可能有抖脚、多次触地或不稳定换相。
- 如果 `contact_duty_error_mean` 高，且多只脚 `contact_duty_*` 明显大于 `desired_duty_*`，说明策略偏贴地走，视觉上容易表现为低趴、拖腿或步态不清晰。
- 如果 `contact_phase_r_mean` 很低，说明触地相位分散，即使 same/opp 关系偶尔看起来对，实际步态也不够稳定。
- 如果 `done_rate=0` 但 `height_mean` 明显低、`contact_duty_*` 明显高，说明不是摔倒问题，而是低姿态局部最优。

A 系列实验建议统一保存并横向比较：

```text
A1.1b / A2 / A3 / A4
  standard suite: 整体速度、基础高度、姿态、四种 gait
  height_full suite: 原版 height range 压力测试
```

重点比较列：

- `trot_height_high`: `height_mean`、`height_rmse`、`height_score`
- `trot_height_low`: 是否出现 `height_mean` 过低或 contact duty 变高
- gait cases: `gait_score_v2`、`contact_freq_ratio`、`contact_duty_error_mean`、`contact_phase_r_mean`
- posture cases: pitch/roll 是否单调响应 command
- all cases: `done_rate`

当前 A1.1b 的 benchmark 结论：

- `done_rate = 0`，稳定性好。
- `tracking_score` 约 `0.98+`，速度跟踪不是主要问题。
- height 高命令明显不足：`target_height=0.40` 时，`height_mean` 约 `0.28`，`height_score` 明显下降。
- pitch 有明显响应，但 roll 基本无响应。
- `gait_score_v2` 约在 `0.5~0.57`，步态区分仍不够强。
- `contact_freq_ratio` 大多大于 `1.3`，实际触地频率高于 command frequency。
- 实际 `contact_duty_*` 普遍高于 `desired_duty_*`，说明策略偏向贴地/长支撑，而不是干净摆腿。

横向可视化比较工具：

```text
scripts/reinforcement_learning/rsl_rl/compare_wtw_benchmarks.py
```

这个脚本只读取 `play_wtw_benchmark.py` 生成的 `summary.json`，不会启动 IsaacLab。它适合把不同机器、不同实验的 benchmark 结果拷到一起后离线比较。

推荐用法：

```bash
python scripts/reinforcement_learning/rsl_rl/compare_wtw_benchmarks.py \
  --inputs logs/rsl_rl/go2_walk_these_ways/*/benchmark_play/*/summary.json \
  --output_dir logs/rsl_rl/go2_walk_these_ways/benchmark_compare/<name>
```

如果只比较几组明确实验，可以手动指定 label：

```bash
python scripts/reinforcement_learning/rsl_rl/compare_wtw_benchmarks.py \
  --inputs \
    logs/rsl_rl/go2_walk_these_ways/<A1_RUN>/benchmark_play/<stamp>/summary.json \
    logs/rsl_rl/go2_walk_these_ways/<A2_RUN>/benchmark_play/<stamp>/summary.json \
  --labels A1 A2 \
  --output_dir logs/rsl_rl/go2_walk_these_ways/benchmark_compare/A1_vs_A2
```

输出内容：

- `aggregate.csv`：每个实验一行，包含关键指标的 mean/min/max。
- `cases_long.csv`：每个实验每个 case 一行，方便继续筛选或自己画图。
- `report.md`：按 `overall_score_mean` 排名的简要报告。
- `aggregate_scores.png`：总分、速度、高度、步态四个主指标。
- `gait_score_breakdown.png`：trot/pace/bound/pronk 的 `gait_score_v2` 分项对比。
- `height_response.png`：`target_height` 与 `height_mean` 的散点图，用来快速看高度跟踪是否单调、是否压缩。
- `case_heatmap_*.png`：每个 case 的关键指标热力图，用来定位是哪一个 case 拉低整体结果。

## 核心机制说明

### `binary_phases`

`binary_phases=True` 会把 command 的 gait phase 参数离散到 `0` 或 `0.5`：

```text
commands[:, 5] = gait_phase
commands[:, 6] = gait_offset
commands[:, 7] = gait_bound
```

目的：让 gait 更接近经典四足步态的半周期结构，减少连续 phase 造成的混合步态。

### gaitwise category

训练时每个 env 会被随机分到一种 gait category：

```text
pronk, trot, pace, bound
```

category 本身不会直接输入 policy。代码只是根据 category 改写 command 的 phase/offset/bound：

- trot：对角腿交替。
- pace：同侧腿交替。
- bound：前后腿交替。
- pronk：四腿同相。

policy 最终看到的是 command 和 clock inputs，而不是 category label。

### 多维 limit/bin/curriculum

每个 command 维度有两层范围：

- `limit_*`：curriculum 网格的最大边界。
- `*_range`：初始可采样范围。

每个维度再由 `num_bins_*` 切成若干格子。每个高维 bin 有一个 weight，训练成功后会增加当前 bin 和附近 bin 的权重，`command_area` 反映当前可采样区域占整个 grid 的比例。

如果 `command_area_*` 长期很小，说明训练还卡在很小的 command 空间里，gait 不清晰可能只是因为复杂命令没展开。

## 仍然可疑的问题

### 1. ji22 reward split 和 IsaacGym 不完全一致

原 IsaacGym 代码按整个 reward term 的 `torch.sum(rew)` 判断该 term 是正项还是负项：

```python
if torch.sum(rew) >= 0:
    rew_buf_pos += rew
elif torch.sum(rew) <= 0:
    rew_buf_neg += rew
```

当前 IsaacLab 版是 per-env clip：

```python
rew_buf_pos += torch.clip(rew, min=0.0)
rew_buf_neg += torch.clip(rew, max=0.0)
```

这会改变 `sigma_rew_neg` 的实际效果，可能解释为什么原版 `reward *= dt + sigma_rew_neg=0.02` 能训好，而当前一照搬就坏。

### 2. Reward 原始量级不清楚

当前 TensorBoard 主要看加权后的 reward。还缺少：

- raw reward mean
- scaled reward mean
- `rew_buf_pos`
- `rew_buf_neg`
- `exp(rew_buf_neg / sigma_rew_neg)`

没有这些量，很难判断是哪个负项把策略压成少动或趴下。

### 3. Contact reward 在 IsaacLab 中可能量级不同

即使公式相同，IsaacLab 的 contact force、foot velocity、contact sensor 更新时序可能和 IsaacGym 不完全一致。

重点检查：

- `desired_contact_states`
- actual contact bool
- foot force norm
- foot speed
- desired/actual contact correlation by foot and by gait category

### 4. Height tracking 可能被 reward/target 配方压住

当前用 `jump` reward 跟踪：

```text
target_height = commands[:, 3] + base_height_target
```

如果 height 命令不明显，需要单独固定速度和 gait 后扫 height command，而不是在完整 command distribution 中肉眼判断。

### 5. Zero-speed stand 不是原始 gait 训练分布的一部分

原 WTW gait frequency 训练范围是 `2-4 Hz`。所以速度为 0 时仍踏步并不一定是 bug。

如果希望速度为 0 时站住，需要额外加入：

- stand category
- 或 zero-frequency command distribution
- 或 play/deploy 侧 command gating

这是功能增强，不应混进等价性恢复实验。

## 实验清单

### E0 当前基线

目的：记录当前 workaround 的表现，作为后续对照。

配置：

- 不改代码。
- 使用当前 `sigma_rew_neg=5.0`。
- 不使用 `reward_scales *= step_dt`。
- height range 为 `(-0.10, 0.10)`。
- `rew_jump=20.0`，contact `6/6`。

观察指标：

- gait 是否能稳定走。
- height command 是否有明显响应。
- `command_area_*` 是否展开。
- `rew_tracking_contacts_shaped_force/vel` 是否仍很负。
- action min/max 是否正常。

### E1 Reward split 等价实验

目的：验证 IsaacGym-style reward split 是否是关键差异。

改动：

- 增加开关 `reward_split_mode = "isaacgym" | "per_env_clip"`。
- 先使用 `"isaacgym"`。
- 其他保持 E0 不变。

当前状态：

- 已实现，默认值为 `"isaacgym"`。
- `"per_env_clip"` 可用于回退到之前 IsaacLab port 的行为。
- 训练反馈：E1 步态清晰很多，高度跟踪有效。当前基线已回到 E1 配置。

成功判据：

- gait/contact reward 改善。
- `command_area_*` 展开更快。
- 不出现少动、小幅蹦或后期趴下。

失败判据：

- 表现无变化，说明主要问题不在 positive/negative split。

### E2 原版 ji22 量纲实验

目的：验证原始 `dt scaling + sigma=0.02` 能否在修正 reward split 后恢复。

改动：

- 在 E1 成功基础上恢复 `reward_scales *= step_dt`。
- `sigma_rew_neg = 0.02`。

当前状态：

- 已实现，默认使用 `reward_split_mode="isaacgym"`。
- 已恢复 `reward_scales *= step_dt`。
- 已恢复 `sigma_rew_neg=0.02`。
- 已新增 `raw_rew_*`、`rew_ji22_pos`、`rew_ji22_neg`、`rew_ji22_exp_gate` 日志。
- 训练反馈：E2 后期再次出现原地趴下，说明原版 `dt scaling + sigma=0.02` 在当前 IsaacLab reward 量级下仍不成立。
- 处理结果：代码已回退到 E1 的稳定配置，但保留诊断日志。

必须新增日志：

- `rew_buf_pos`
- `rew_buf_neg`
- `exp_gate = exp(rew_buf_neg / sigma_rew_neg)`
- 每个 reward raw/scaled mean。

成功判据：

- `exp_gate` 不长期接近 0。
- 策略不再学成小动作蹦或趴下。

失败判据：

- `exp_gate` 长期接近 0。
- 说明某些负项在 IsaacLab 中量级过大，需要查 raw reward。

### E3 原版 reward 权重恢复

目的：判断当前增强版 contact/height 权重是否只是 workaround。

改动：

- `rew_jump = 10.0`
- `rew_tracking_contacts_shaped_force = 4.0`
- `rew_tracking_contacts_shaped_vel = 4.0`

当前状态：

- 暂停作为主线。
- 由于 E2 已失败，不建议在 E2 配置上继续验证 E3。
- 如需继续恢复原版权重，应在 E1 稳定配置上单独做 `jump/contact` 权重 ablation。

成功判据：

- gait 仍清晰。
- height 不明显变差。

失败判据：

- gait 混合或 category 响应弱。
- 如果失败，优先只增强 contact，不急着增强 height。

### E4 原版 height range 恢复

目的：验证 `(-0.25, 0.15)` 是否能在等价 reward 量纲下工作。

改动：

- `body_height_cmd = (-0.25, 0.15)`
- `limit_body_height = (-0.25, 0.15)`

观察：

- actual base z 和 target z 的误差分布。
- 低高度命令时是否变成趴地局部最优。
- 高高度命令是否真的抬高。

成功判据：

- 不趴。
- base height 对 command 有单调响应。

失败判据：

- 再次后期趴下。
- 说明 height reward 或 IsaacLab base/contact 物理量级仍不等价。

### E5 Contact/gait 诊断实验

目的：确认 gait 不清晰是 command 没展开，还是 contact target/reward 没学到。

改动：

- 加 debug log，不一定需要训练完整。
- 按 gait category 统计：
  - desired contact state
  - actual contact bool
  - foot force norm
  - foot velocity
  - desired/actual contact correlation

成功判据：

- trot/pace/bound/pronk 的 contact pattern 有清楚差异。

失败判据：

- 不同 category 的 actual contact pattern 高度相似。
- 或 desired/actual contact correlation 很低。

可能后续改动：

- 调 `gait_force_sigma`、`gait_vel_sigma`。
- 增强 contact reward。
- 尝试把 `desired_contact_states` 加入 observation 做 ablation。

### E6 Stand mode 功能增强

目的：解决速度为 0 仍踏步的问题。

注意：这不是 IsaacGym 等价性恢复，而是功能增强。

可能方案：

- 增加 `stand` category。
- 训练中采样 `gait_frequency=0` 的 zero-speed command。
- deploy/play 侧当 `vx=vy=yaw=0` 时强制写 standing command。

成功判据：

- `vx=vy=yaw=0` 时稳定站立。
- 非零速度时 trot/pace/bound/pronk 仍清晰。

## E2 后的 A 系列 ablation

E1 训练反馈较好，E2 后期再次趴下。因此当前主线回到 E1 稳定配置，不再继续在 E2 配置上做 E3/E4。A 系列实验只在 E1 基线上逐项恢复原版权重。

### A1 Contact 权重恢复

目的：判断 E1 的步态清晰是否依赖增强后的 contact shaped reward。

改动：

- `rew_tracking_contacts_shaped_force = 4.0`
- `rew_tracking_contacts_shaped_vel = 4.0`
- `rew_jump` 保持 `20.0`
- 保持 `reward_split_mode="isaacgym"`
- 保持 `sigma_rew_neg=5.0`
- 保持不使用 `reward_scales *= step_dt`

当前状态：

- 已实现。

观察指标：

- gait selector 切换是否仍清晰。
- `command_area_pronk/trot/pace/bound` 是否仍展开。
- `raw_rew_tracking_contacts_shaped_force/vel` 和 `rew_tracking_contacts_shaped_force/vel` 是否明显恶化。
- 是否再次偏向 bound/pronk 混合。

下一步：

- 若 A1 仍好，再做 A2：只把 `rew_jump=20.0` 降到 `10.0`。
- 若 A1 变差，不继续降 `rew_jump`，先做 contact/gait 诊断日志。

### A1.1 Contact curriculum 阈值放宽

目的：A1 训练反馈显示整体效果更好，高度跟踪明显有效，pronk/bound/pace 都还可以，但 trot 不够对称且走起来偏跳；同时 mean reward 很早收敛，而 command region 增长缓慢。该实验只放宽 contact curriculum 的成功门槛，验证 command region 是否可以更快展开，尤其是 `command_area_trot`。

改动：

- `curriculum_tracking_contacts_shaped_force = 0.65`
- `curriculum_tracking_contacts_shaped_vel = 0.65`
- 保持 `curriculum_tracking_lin_vel = 0.8`
- 保持 `curriculum_tracking_ang_vel = 0.7`
- 保持 A1 的 reward 权重：`rew_jump=20.0`，contact `4.0/4.0`
- 保持 `reward_split_mode="isaacgym"`，`sigma_rew_neg=5.0`，不使用 `reward_scales *= step_dt`

当前状态：

- 已实现。
- 训练反馈待补充。若 `0.65/0.65` 展开过快或步态变差，使用 A1.1b 的温和阈值继续对照。

观察指标：

- `command_area_trot` 是否比 A1 增长更快。
- `command_area_pronk/trot/pace/bound` 是否出现明显失衡。
- trot 是否更对称，是否减少跳跳的表现。
- 若 command region 增长快但步态变差，说明阈值过松导致 curriculum 放开太早。

### A1.1b Contact curriculum 温和放宽

目的：作为 A1.1 的保守版本，避免 `0.65/0.65` 过早放开 command region，同时仍比 A1 的 `0.8/0.8` 更激进。该实验用于判断 trot 改善来自更快 curriculum 展开，还是来自过松阈值带来的不稳定。

改动：

- `curriculum_tracking_contacts_shaped_force = 0.72`
- `curriculum_tracking_contacts_shaped_vel = 0.72`
- 保持 `curriculum_tracking_lin_vel = 0.8`
- 保持 `curriculum_tracking_ang_vel = 0.7`
- 保持 A1 的 reward 权重：`rew_jump=20.0`，contact `4.0/4.0`
- 保持 `reward_split_mode="isaacgym"`，`sigma_rew_neg=5.0`，不使用 `reward_scales *= step_dt`

当前状态：

- 已实现。

观察指标：

- `command_area_trot` 增长是否明显快于 A1，但不出现 A1.1 可能的过早扩张。
- trot 是否比 A1 更对称、更少跳，同时不牺牲 pronk/bound/pace。
- 后期是否仍保持站高和步态稳定，不重新出现趴下或四腿同相化。

### A2 Jump 权重恢复

目的：在 A1/A1.1b 已经能基本 work 的基础上，单独验证当前高度跟踪是否依赖增强后的 `rew_jump=20.0`。该实验只恢复原 IsaacGym 的 jump 权重，不同时恢复 height range，避免把“高度命令范围变大”和“高度 reward 变弱”混在一起。

改动：

- `rew_jump = 10.0`
- 保持 A1 的 contact 权重：`rew_tracking_contacts_shaped_force = 4.0`，`rew_tracking_contacts_shaped_vel = 4.0`
- 保持 A1.1b 的 contact curriculum 阈值：`0.72/0.72`
- 保持当前 height range：`body_height_cmd = (-0.10, 0.10)`，`limit_body_height = (-0.10, 0.10)`
- 保持 `reward_split_mode="isaacgym"`，`sigma_rew_neg=5.0`，不使用 `reward_scales *= step_dt`

当前状态：

- 已实现。

观察指标：

- height command 响应是否仍明显，尤其是高低两端是否还有单调响应。
- 后期是否重新出现低高度、趴走或动作幅度变小。
- gait selector 是否保持 A1.1b 的清晰度，避免因高度 reward 变弱又退化到一种通用跳步。

下一步：

- 若 A2 高度跟踪仍好，再做 A3：只恢复原版 height range `(-0.25, 0.15)`。
- 若 A2 高度跟踪明显变弱或后期变低，先不要恢复 height range，考虑 `rew_jump=15.0` 或补充 base height tracking 诊断。

### A3 Height range 恢复

目的：在 A2 基础上单独恢复原 IsaacGym 的 height command 范围，验证当前 reward/command 量纲下是否能承受完整高度采样范围。该实验只扩大高度范围，不再同时改 `rew_jump`、contact 权重或 curriculum 阈值。

改动：

- `body_height_cmd = (-0.25, 0.15)`
- `limit_body_height = (-0.25, 0.15)`
- 保持 `rew_jump = 10.0`
- 保持 contact 权重：`rew_tracking_contacts_shaped_force = 4.0`，`rew_tracking_contacts_shaped_vel = 4.0`
- 保持 A1.1b 的 contact curriculum 阈值：`0.72/0.72`
- 保持 `reward_split_mode="isaacgym"`，`sigma_rew_neg=5.0`，不使用 `reward_scales *= step_dt`

当前状态：

- 已实现。

观察指标：

- 低 height command 是否重新诱发趴走或后期高度塌陷。
- actual base height 对 command 是否仍单调，尤其是 `-0.25` 附近是否只是贴地求稳。
- gait selector 是否仍有清晰区分，还是因为高度范围变大而退化成低幅度通用步态。

下一步：

- 若 A3 稳定，再考虑继续恢复其它原版设置或做多 seed 确认。
- 若 A3 失败而 A2 稳定，优先尝试 A4：原版 height range + `rew_jump=15.0`，或引入 height curriculum/分阶段扩大范围。

## 推荐执行顺序

1. 固化 E0 当前基线日志。
2. 实现 E1：IsaacGym-style reward split。
3. 如果 E1 改善，做 E2：恢复 `dt scaling + sigma=0.02`。
4. 如果 E2 稳定，做 E3：恢复原版 `jump/contact` 权重。
5. 如果 E3 稳定，做 E4：恢复原版 height range。
6. 并行准备 E5 诊断日志，用于解释 gait 不清晰。
7. 等等价性恢复后，再做 E6 stand mode。

## 当前优先级判断

最高优先级是 E1 和 E2。

原因：原 IsaacGym 训练脚本确实使用 `reward *= dt` 和 `sigma_rew_neg=0.02`，但当前 IsaacLab 直接照搬会崩。这说明当前 reward 正负拆分、reward 原始量级或 step timing 至少有一个还不等价。先解决这个，比继续手调 height/contact 权重更有信息量。
