# CoRL 2026 项目文档 v1.3：Trajectory-Aware Whole-Body Loco-Manipulation



### 0.1 关键概念

| 术语 | 一句话定义 | 具体例子 |
|---|---|---|
| **EE / End-Effector** | 机械臂末端（夹爪基座或抓取点） | 机械臂手腕中心的 6D pose（x, y, z + roll, pitch, yaw） |
| **Trajectory** | 一段时间内 EE 的 6D pose 序列 | 50 步 × 50ms 的 pose 序列，总时长 2.5 秒 |
| **Trajectory tracking** | 让 EE 实际运动尽量贴合给定 trajectory（含 pose、速度、时序） | 比 pose reaching 更难——要求"什么时候到哪里"，而不只是"最终到哪里" |
| **Whole-body / WBC** | 同时控制四足 12 个腿关节 + 机械臂 6 个关节 | 18 维 action，无关节固定 |
| **Reachability** | 给定一个 base pose，机械臂能到达的 EE pose 集合 | base 站直时手能伸到 0.6m 外；下蹲时能伸到地面 |
| **Manipulability** | 给定机械臂构型，EE 各方向的"操作灵活度"（雅可比奇异值乘积） | 手臂完全伸直时 manipulability 几乎为 0（奇异），半伸直时较高 |
| **GORM** (QuadWBG) | "给一个 EE 抓取 pose，反推 base 应该站在哪些 pose"的预计算几何表 | 输入 grasp pose → 输出"站这里抓取成功率高"的 base pose 分布 |
| **TRT**（我们提的）| GORM 的轨迹版本：给一段 EE 轨迹，反推 base pose 应该走的"管道" | 输入 EE 轨迹 → 输出每个时刻的 reachable base pose 序列 + 时间维度连续性约束 |
| **Anticipatory** | base 提前知道未来 EE 命令，在 arm 接近 reachable 边界**之前**就开始移动 | reactive baseline：等 EE 偏差变大才动 base；anticipatory：提前 200ms 就开始 |
| **Reactive baseline** | policy 只看当前与过去状态，没有 future trajectory 信息 | 类似 ETH multi-critic，输入 current twist + current pose |
| **Multi-critic**（ETH CoRL 2025） | 一个共享 actor + 多个 critic 分别评估不同 reward 组（loco / mani / contact） | 每个 critic 独立算 advantage 后归一化求和，避免 reward scale 互相压制 |
| **AMP**（Adversarial Motion Priors）| 用 GAN-style discriminator 让 RL policy 输出的 motion 接近一组 reference motion | reference 来自 mocap 或 trajopt，无需手设 reward |
| **PFG**（Hou 2025） | 把"当前 base pose 是否让 arm 能 IK 解出来"作为 dense reward | 解得出 → 给 base 奖励；解不出 → 不给 |
| **OCS2** | ETH 开发的 trajectory optimization 软件，可解四足 + arm 的 MPC 问题 | 离线在 sim 中生成 reference trajectory |
| **Cross-disturbance training** | 训练 loco expert 时给机械臂加扰动；训练 manip expert 时给 base 加扰动 | 让两个 expert 在对方的扰动范围内仍稳定，便于后续整合 |

---

## 1. 解决什么问题

### 1.1 目标

给一台带机械臂的四足机器人（约 18 DoF）一段 6D EE 轨迹，让它通过 whole-body coordination 把这段轨迹尽量精确地"画"出来——而且要能在真机上做到。

形式化讲：

- **输入**：world frame 下的 EE 6D pose 序列 `{p_t, R_t}_{t=0..T}`，每步 50ms，T 通常 50–100 步。
- **输出**：18 维关节命令 `q_t`（12 腿 + 6 臂）。
- **评估**：三类指标——pose tracking error（位置 cm + 角度 rad）、velocity profile fidelity（轨迹时序贴合度）、success rate（完成率）。

### 1.2 动机

quad-arm 系统比固定底座机械臂多出 12 DoF（base 6 + 腿冗余）。这是双刃剑：扩大了工作空间，但也带来了动力学耦合和分层协调的麻烦。

更重要的是，现有方法的实验绝大多数停留在 **pose reaching**（到一个目标点）或 **single-target grasping**（抓一个物体）。但真实下游任务——diffusion policy rollout、teleop replay、TAMP 输出——天然产生的是**一段轨迹**，不是一个目标点。pose-level 的方法在这个层级上是断的。

trajectory tracking 还有一个工程上的好处：它把任务语义和控制解耦——上层 planner 输出轨迹，下层 controller 执行。这是工业界和研究界都会用到的稳定接口。



---

## 2. 文献调研

### 2.1 直接相关工作

下面的论文按"对我们故事的相关度"排，每篇一句话说它在做什么、跟我们的关系。

| 论文 | 它做了什么 | 跟我们的关系 |
|---|---|---|
| **QuadWBG** (ICRA 2025) | 提出 GORM——预计算 6-DoF base reachability map 用于 grasping | **核心 inspiration**——我们把 GORM 推广到 trajectory 层面 |
| **MLM** (2025) | 用真实采集的 TCP 轨迹训 RL whole-body tracker；NAE 从历史 TCP 预测未来 | 关键 baseline——证明 commanded future > predicted future |
| **Multi-critic** (CoRL 2025, ETH) | 三个 critic 拆 reward（loco / mani / contact），EE 用 twist 而非 pose tracking | **拥抱不对抗**：直接采纳为我们的 critic architecture |
| **WBPT** (ICRA 2025, ETH) | 用 3 个 keypoint 表示 SE(3) 避免 rotation reward 调权重；workspace sampling | 直接采纳 keypoint 表示；workspace sampling 启发 reachability-graded curriculum |
| **PFG** (Hou 2025) | 把 IK feasibility 作为 dense reward 引导 RL 探索"姿态使 arm 可达"的状态 | 重要对比——pointwise 几何 prior，我们做 trajectory-level |
| **RFM** (RA-L 2024) | reward algebra：乘法 gating、power 项 boost 小误差、sigmoid blending | 直接借鉴 reward 组合方式 |
| **ODYSSEY** (AAAI 2026) | 长程 mobile manipulation；EE target 在 world frame 采样避免 base 抖动污染 | 借鉴 world-frame EE sampling |
| **Dadiotis 2025** | RL 推动物体；asymmetric actor-critic（critic 看物理参数，actor 不看） | 借鉴 asymmetric arch |

经典基线（Deep WBC / VBC / RoboDuet）作为常规对比，复现成本视情况而定。WildLMa 的 skill library 路线 scope 不同，仅作系统集成参考。

### 2.2 现有工作留下的空白

把上面这些工作横切再纵切，我们要切入的位置就清楚了：

| 维度 | 现有工作做到哪里 | 还没人做的 gap |
|---|---|---|
| Task formulation | pose reaching；trajectory tracking（MLM、multi-critic）但都没显式几何 prior | trajectory tracking + **explicit reachability prior** |
| 几何 prior | GORM 单时刻、6-DoF base | **trajectory-level reachability tube** |
| Reward 设计 | RFM 的 algebra；PFG 的 IK reward | RFM 思想 + **基于 reachability 几何而非纯 distance** |
| Future 信息 | NAE 历史预测；twist 瞬时 | **commanded future trajectory** + manipulability encoding |
| Critic 架构 | Multi-critic | （已被占领，我们采纳） |
| 训练分布 | WBPT workspace + body perturbation | **reachability-graded curriculum** + cross-disturbance |



---

## 3. 核心 Insight 与 Motivation

### 3.1 第一性原理：base 不是 disturbance，是 arm capability 的一部分

最朴素但最重要的观察：

> EE pose = base pose ⊕ arm pose

案例：

- 机械臂只能伸 0.6m，EE 目标在 1.5m 外——**必须**靠 base 走过去。
- EE 目标在地面附近，base 站直会让 arm 进入"几乎完全伸直"的奇异构型，manipulability 极低；下蹲后 base + arm 都能舒展。
- EE 要快速画一个直径 0.4m 的圆，arm 单独可以做但 base 不动会限制速度；base + arm 协调可以让圆又快又精确。

这意味着 base 不是控制器要去对抗的"扰动"，而是 arm 能力的延伸。**当任务从 pose reaching 推广到 trajectory tracking，base 的角色从"找一个好位置"变为"走一条好路径"——也就是说，需要的是 sequential geometric prior，而不是 pointwise 的。**

现有方法没有显式提供这个 prior，全都让 RL 从 reward 里隐式发现。这就是我们的切入点。

### 3.2 Core hypothesis

> 给 RL 同时显式提供两类信息：(1) future trajectory（policy 知道接下来 0.4 秒 EE 要去哪），(2) base 应该走的 reachability tube（一段时间内 base 应该在哪些 pose 范围内），它能学到比纯 reward shaping 更鲁棒、更精确、sample efficiency 更高的 whole-body trajectory tracking 策略。

这个总假设可以拆成 4 个 atomic hypothesis（见 §4），每个独立验证。可证伪的关键是：四个加起来都不显著 → 整个故事死，认真考虑改投。

---

## 4. 故事候选

### 4.2 四个原子假设

#### H1：Commanded future trajectory in observation 显著优于 reactive baseline

**核心问题**：当 policy 输入加入 future EE trajectory 后，base 是否会"提前"准备移动，从而改善 dynamic trajectory tracking？

**实验设置**：训练两个 policy，唯一区别是观测空间。

- **Policy A (reactive)**：input = 当前 proprioception (~50 dim) + 当前 EE target (6 dim) + last action (18 dim)
- **Policy B (anticipatory)**：上面所有 + future EE trajectory window（8 步 × 6 dim = 48 dim）

网络结构相同：3 层 MLP，hidden = [256, 256, 128]，PPO 训练。任务 T2（详见 §6.1）。Future window length sweep ∈ {0, 4, 8, 16}，3 seeds，共 12 runs。

**通过判据**：tracking error (RMSE position) 改善 ≥ 25% 算显著；velocity profile fidelity (cosine similarity) 改善 ≥ 10% 是 bonus 信号。

**预期 ✓ 时观察到的现象**：base 在 arm 接近 ws 边界**前** ~200ms 就开始移动；training curve 显示 anticipatory 收敛更快、final error 更低。

**失败意味着什么**：anticipatory 故事死，砍 Story B / F。

#### H2：Pointwise reachability/manipulability reward 显著优于无几何 prior 的 baseline

**核心问题**：把"当前 base pose 是否让 arm 能解 IK"作为 dense reward，是否改善 trajectory tracking？这是 PFG (Hou 2025) 思路在 trajectory task 上的复现。

**实验设置**：

- Baseline (no prior)：同 H1 的 Policy A，reward = `r_track = exp(-‖p_ee - p_target‖² / σ²)` + 标准 regularization。
- PFG-style reward：上面 + `r_PFG = w · feasibility(q_arm, q_base, p_target_arm)`，其中 `feasibility = exp(-‖q_arm - q_IK‖²)` if IK 解得出，else 0。IK 用 damped least squares 实时解。
- Reward weight `w` ∈ {0.1, 0.5, 1.0, 2.0}，3 seeds。

**通过判据**：tracking error 改善 ≥ 20% **或** sample efficiency 改善 ≥ 30%（达到 80% final performance 的 step 数减少）。

**预期 ✓ 现象**：base 倾向于站到 IK 可解的 pose；training curve 早期收敛快。

**失败意味着**：几何 prior 故事疑问，倾向 anticipatory-only。

#### H3：Trajectory-level geometric prior (TRT) 显著优于 pointwise 版本 (H2)

**核心问题**：从"每个时刻独立 IK feasibility"升级到"整段 trajectory 的 reachable base tube"，是否带来额外收益？这是 Story C 的核心。

**实验设置**：先做简化版 TRT（只考虑 position reachability，不含 dynamic feasibility），伪代码：

```python
def compute_TRT_simple(EE_traj):
    tube = []
    for t in range(T):
        # 在 EE_traj[t] 周围采样 base pose，只保留 IK 可解的
        candidates = sample_base_poses_around(EE_traj[t], n_samples=200)
        feasible = filter_IK_feasible(candidates, EE_traj[t])
        if t > 0:
            # 时间连续性：t 时刻的 candidate 必须从 t-1 时刻可达
            feasible = filter_dt_reachable(tube[t-1], feasible, dt=0.05)
        tube.append(feasible)
    return tube
```

TRT-based reward：`r_TRT = exp(-d_to_tube² / σ²)`，`d_to_tube` = 当前 base pose 到 tube[t] 中最近 candidate 的距离。

跟 H2 的 pointwise 版本对比，相同 trajectory，相同其他设置。

**通过判据**：dynamic / curved trajectory 上 tracking error 改善 ≥ 15%；tube 计算时间 < 200ms / trajectory（部署可行性）；tube 非空率 > 80%。

**预期 ✓ 现象**：curved trajectory 上 TRT 显著好；straight trajectory 上差不多——这恰好说明 TRT 价值在动态场景。

**失败意味着**：TRT 不值得，退化到 pointwise 即可，故事接近"PFG 增强版"。

#### H4：AMP-from-trajopt 显著提升 EE 精度

**核心问题**：把 OCS2 在简单 setup 下生成的 reference trajectory 当作 AMP discriminator 的 expert，是否让 RL 学到更精确的 arm 控制？

**实验设置**：

- 数据：OCS2 在 fixed-base setting 生成 200 条 EE trajectory（直线 / 圆弧 / 复合曲线），每条 5 秒。
- Baseline：纯 PPO，reward = tracking。
- AMP variant：PPO + AMP discriminator。Discriminator 是 3 层 MLP，input = state-action transition 投影到 (EE_pos_t, EE_pos_{t+1}, EE_vel_t)；AMP reward = `-log(1 - D(transition)) · w_amp`。
- 任务 T1（base 静止，arm tracking），3 seeds。

**通过判据**：静态 base 下 EE pose tracking RMSE 改善 ≥ 30%；训练稳定（discriminator 不 collapse）。

**风险特别提醒**：AMP 训练 known unstable，可能 Day 4 还在调。如果 H4 拖累整个 Phase 0，**Day 3 早可以 abort H4**——这是整个 Phase 0 唯一可单独 kill 的 hypothesis，因为它是 bonus、不影响主线。

**失败意味着**：AMP 路线砍掉，但不影响主线。

### 4.3 决策矩阵：从假设结果到 Final Story

四个 hypothesis 的不同组合直接决定最后写哪个 story。下表是决策表，Day 5 全员到场对照填。

| H1 | H2 | H3 | H4 | Final Story | 一句话 pitch |
|:---:|:---:|:---:|:---:|---|---|
| ✓ | ✓ | ✓ | ✓ | **Story F** ⭐⭐⭐ | Trajectory-aware WBC：commanded future + reachability tube + trajopt-distilled precision |
| ✓ | ✓ | ✓ | ✗ | **Story C+** ⭐⭐ | Trajectory-aware WBC：commanded future + reachability tube（最稳健的强 story） |
| ✓ | ✓ | ✗ | – | **Story B+** ⭐ | Anticipatory WBC + reachability-aware reward（pointwise 几何 prior 版） |
| ✓ | ✗ | – | – | **Story B** | Anticipatory WBC（只靠 future trajectory，故事窄但清晰） |
| ✗ | ✓ | ✓ | – | **Story C-** | Reachability-tube WBC（无 future，故事窄） |
| ✗ | ✓ | ✗ | – | **Story D** | Reward algebra 深化（高 attack 风险，不优先） |
| ✗ | ✗ | – | ✓ | **Story E** | AMP-only（高风险，不优先） |
| ✗ | ✗ | ✗ | ✗ | **MVP** | 改投 ICRA / IROS 或 backup story |

`–` 表示该位置结果不影响 final story 决定。

### 4.4 Day 5 之前的概率估计

凭经验和文献感觉，对每种情景的发生概率有个 rough estimate：

- **最佳**（H1+H2+H3 全 ✓）：约 30%。Story C+ 推进；H4 视情况加为 bonus contribution。
- **次佳**（H1+H2 ✓，H3 边际）：约 30%。Story B+ 或 Story C+，按显著性精确判断。
- **单 ✓**（只 H1 或只 H2）：约 25%。Story B 或 Story C-，scope 缩小但能写。
- **危险**（全部不显著）：约 10%。MVP fallback。
- **H4 单 ✓**：约 5%。**不推荐** Story E，并入更主流故事。

### 4.5 三个最可能的 Final Story 详细描述

#### Story C+（最可能采用）

**Title 雏形**：*Trajectory Reachability Tube for Anticipatory Whole-Body Loco-Manipulation*

**Pitch**：现有 quad-arm WBC 把 base 和 arm 看作两个独立 tracking 目标，依赖 RL 隐式发现协调。我们提出在 trajectory 层面预计算 base 的 reachability tube，并把 commanded future trajectory 显式注入 policy 输入。这两个几何 / 时序 prior 让 base 学会 anticipatory 移动（在 arm 接近 ws 边界**前**就提前调整），实现高速、精确、鲁棒的 trajectory tracking。

**三个 contribution**：

第一，**Trajectory Reachability Tube (TRT)**。把 QuadWBG 的 GORM 从单时刻推广到一段 trajectory，TRT 同时保证：(i) 每时刻 EE reachable，(ii) base 自身 kinematic feasible，(iii) 时间连续——连续时刻 base pose 不跳变。计算时间 < 200ms / trajectory，可在线 re-plan。

第二，**TRT-driven training pipeline**。包含三件事：reward 上 `r_TRT = exp(-d² / σ²)` 鼓励 base 在 tube 内，并用 RFM 思想做 algebraic combination 而非简单线性叠加；curriculum 上按 tube 几何特征 grading（tube 越窄 = base 选择越少 = 越难，从宽到窄渐进训练）；cross-disturbance pre-training 让 loco expert 在 arm 扰动下稳定，反之亦然。

第三，**Manipulability-aware future observation**。Policy 输入 8-step future window，但每个 future step 不只是 raw 6D pose——还包含该 pose 在当前 manipulability ellipsoid 中的归一化位置，信息密度更高。区别于 MLM 的 NAE：他们是 predicted future from history，我们是 commanded future + geometric encoding。

**故事内部 logical chain**：GORM 揭示 base pose 决定 arm capability → 推广到 trajectory 得到 TRT → TRT 同时给出 reward / curriculum / observation 三处信号 → 加上 commanded future 作为 anticipatory 信息 → policy 学到 sequential whole-body coordination。三个 contribution 来自同一个 core insight，这是 reviewer 反"single trick paper"的关键。

#### Story B+（次可能采用）

**Title 雏形**：*Anticipatory Whole-Body Trajectory Tracking with Reachability-Aware Rewards*

**Pitch**：future EE trajectory 作为 policy 输入 + pointwise reachability reward (PFG-style)。两者分别解决"接下来要去哪"和"现在 base 站位对不对"两个问题，组合起来比单独使用任一种都强。

三个 contribution 是：commanded future trajectory observation + manipulability encoding（同 Story C+ 的 contribution 3）；reachability reward integration with RFM-style algebra；cross-disturbance pre-training。

#### Story B（保底）

**Title 雏形**：*Anticipatory Trajectory-Conditioned Whole-Body Loco-Manipulation*

**Pitch**：单 contribution——policy 显式 condition on future trajectory window，dynamic trajectory tracking 显著优于 reactive baseline。配 reward algebra + cross-disturbance training 作为支撑。

**Risk**：单 contribution 故事在 CoRL 边缘，需要非常扎实的实验和清晰 framing。

### 4.6 故事策略的演化

v1.0 直接押注 Story C；v1.1 引入 4 hypothesis 并行验证；v1.2 把每个 hypothesis 和 final story 都具体到可执行的实验细节；v1.3 没改方法论，只是把这套思路用更顺的语言重述了一遍。

核心思想没变：**AI agent 让我们能并行验证，没必要在文献分析时就二选一**。

---

## 5. 文献挖掘清单

| 来源 | 借鉴的设计 | 对应位置 | 优先级 |
|---|---|---|---|
| QuadWBG | GORM: 6-DoF base reachability map | TRT 与 H2/H3 | P0 |
| QuadWBG | 5D locomotion command (vx, vy, yaw, pitch, height) | base command 接口 | P0 |
| RFM | `r = r_pos + r_pos·r_ori` gating | trajectory reward 层级 | P0 |
| RFM | `r* = r + r^M` 小误差 boost | precision 增强 | P1 |
| RFM | Cumulative tracking penalty | trajectory integral reward | P0 |
| RFM | Sigmoid distance phase blending | 改 manipulability blending | P1 |
| WBPT | SE(3) → 3 keypoints | trajectory observation 编码 | P0 |
| WBPT | Locomotion-policy initialization | reset state distribution | P0 |
| MLM | NAE (history → future predict) | H1 baseline B4 | P0 |
| Multi-critic | 3 critic 拆分 | 我们的 critic architecture | P0 |
| Multi-critic | Twist-based EE command | future trajectory 一阶导部分 | P1 |
| PFG | IK feasibility reward | H2 baseline B3 | P0 |
| ODYSSEY | World-frame EE sampling | reference frame 标准 | P0 |
| Dadiotis | Asymmetric actor-critic | critic 看 future + privileged info | P0 |

P0 = Day 12 之前必须实现；P1 = Day 12-18；P2 = bonus，时间允许才做。

---

## 6. 实验设计

实验设计的核心要回答两件事：**实验在测什么（任务）**和**怎么测（baseline + ablation + metric）**。

### 6.1 Task suite

实验需要覆盖三类 trajectory，每类的目的、参数都明确写下来——这是 sweep 和分析的基础。

#### T1：Static base trajectory tracking

**目的**：建立性能下限 + 验证 baseline 不输给 fixed-arm IK。

base 初始静止站立，地面平整。EE trajectory 三种：T1a 是直线（长度 0.4m，从 base frame 的 (0.3, 0, 0.3) 到 (0.3, 0.4, 0.3)，2 秒匀速）；T1b 是圆弧（半径 0.2m，xy 平面 z = 0.3m，圆心 (0.4, 0, 0.3)，3 秒）；T1c 是 T1a 和 T1b 的拼接复合（5 秒）。

预期：几乎所有 baseline 都能完成。这一组主要用来防止"加 prior 反而把 T1 性能压下去"——也就是检验 over-constrain。

#### T2：Coordinated trajectory（核心）

**目的**：验证 TRT / future trajectory prior 的核心价值。EE 轨迹必须超出 base 静止时的 arm reachability，从而强迫 base 移动。

base 初始静止站立。EE trajectory 三种：

- **T2a**：直线，长度 1.5m（远超 arm 单独可达），从 (0.3, -0.5, 0.3) 到 (0.3, 1.0, 0.3)，4 秒。期望行为是 base 沿 y 方向平移。
- **T2b**：圆弧，半径 0.4m，z = 0.3m，圆心在 base 一侧外（base 不动 arm 就进不去），5 秒。期望行为是 base 朝圆心调整 + 部分旋转。
- **T2c**：上下垂直运动，从 (0.4, 0, 0.1) 到 (0.4, 0, 0.8)，长度 0.7m。期望 base 调整 height + pitch。

T2 的关键 metric：anticipatory baseline (B6) 在 T2 上 **base motion 应在 arm 接近 ws 边界前 ~200ms 开始**。这是定性观察的核心信号。

#### T3：Dynamic disturbance

**目的**：验证 sim2real 鲁棒性。在 T2 基础上加扰动。

base trajectory 同 T2，每条 trajectory 测三种扰动：

- **T3-payload**：arm 末端挂 2kg 负载（训练时未见过）
- **T3-terrain**：地面 ±10° 随机斜坡
- **T3-push**：trajectory 中段（2 秒处）给 base 50N 持续 0.5s 的水平推力

### 6.2 Baseline 列表

所有 baseline 共享相同基础：PPO 训练、3 层 MLP actor (hidden = [256, 256, 128])、LR = 3e-4、batch = 4096、训练 30M sim steps、action 是 18 维关节位置 delta（每步限幅 ±0.05 rad）。Reward 共享部分：`r_track = exp(-‖p_ee - p_target‖² / 0.01) + 0.5 · exp(-angle_err / 0.5)` 加上 `r_reg = -0.001 · ‖action‖² - 0.01 · joint_limit_penalty`。

下表只列**与 baseline 相关的差异部分**：

| ID | Baseline | 输入 | Reward 增项 | 备注 |
|---|---|---|---|---|
| **B1** | Vanilla | proprio (50) + current_target (6) + last_action (18) | （无）| 性能下限 |
| **B2** | + future obs | B1 + future window (8 × 6 = 48) | （无）| 测 H1 |
| **B3** | PFG-style | B1 input | `r_PFG = w · IK_feasibility`，w=1.0 | 测 H2，对应 Hou 2025 |
| **B4** | NAE-style | B1 + 训练一个 NAE (2 层 MLP) 从 history TCP 预测 future TCP，predicted future concat | （无）| 测 commanded vs predicted，对应 MLM |
| **B6** | **Our method** | 取决 Day 5 决策。Story C+ 时：B5 input + future window + manipulability encoding | TRT reward + RFM algebra | — |
| **B7** (可选) | MPC | full state | — | OCS2 实时解；真机对比 |

### 6.3 Ablation matrix

主 ablation（写进 main paper）：

| Ablation | 主条件 | 对照 | 期望 |
|---|---|---|---|
| **TRT reward on/off** | B6 含 r_TRT | B6 去掉 r_TRT，仅 r_track + r_reg | 含 TRT 优 |
| **Curriculum on/off** | TRT-graded（先宽后窄） | 一开始全 trajectory | curriculum 收敛快 30%+ |
| **Future window length** | window = 8 | window ∈ {0, 4, 16, 32} | 8 是 sweet spot |
| **Manipulability encoding on/off** | 含 ME | 仅 raw 6D pose | ME 收敛稍快 |
| **Cross-disturbance training on/off** | 含 | 不含 | 含的 T3 性能好 ≥ 20% |

附录 ablation：tube width ∈ {0.05, 0.10, 0.20, 0.40} 扫描；RFM algebra vs linear weighted；reward weight sensitivity (×0.5, ×1, ×2)。

### 6.4 评估指标

主指标——这五个进 main results table：

- **Position tracking RMSE** (cm)：`sqrt(mean(‖p_ee_actual_t - p_ee_target_t‖²))` over trajectory
- **Orientation tracking error** (rad)：`mean(angle_between(R_ee_actual_t, R_ee_target_t))`
- **Velocity profile fidelity**（cosine similarity ∈ [-1, 1]）：`cos_sim(velocity_actual_seq, velocity_target_seq)`
- **Success rate** (%)：fraction of trajectories where final position error < 5cm AND max position error during trajectory < 15cm

次指标进 ablation / appendix：sample efficiency（达到 80% final performance 的训练 step 数）；energy cost；base motion smoothness（base linear jerk RMS）；joint limit hit rate。

真机指标：sim 主指标 + sim2real gap + disturbance recovery time（从扰动施加到 tracking error 恢复 < 5cm 的时间）。

### 6.5 实验数据规模

每个 hypothesis / baseline / ablation 用 3 seeds（time permitting，main result 用 5 seeds）。每个 seed 训 30M sim steps（约 4-6 小时 / GPU）。每个 trained policy 评估用 500 trajectories（200 in-distribution + 200 OOD + 100 disturbed）。真机每 task ≥ 10 trials，至少 1 个 disturbance demo。

---

## 7. 给 Reviewer 的预先答辩

**Attack 1：TRT 只是 GORM 加时间维度，trivial extension。**
defense：TRT 不是简单 stack pointwise GORM——必须同时保证 (i) base 自身 feasibility，(ii) EE reachability，(iii) 时间维度 coherence。这是非平凡的几何 / 优化问题。Evidence：pointwise GORM stacked vs TRT 在 T2b（圆弧）的对比 ablation，期望 ≥ 15% gap。

**Attack 2：Reachability tube 假设 trajectory 已知，部署不实用。**
defense：trajectory 来自 trajopt / planner / teleop / DP rollout，是 modern robot stack 的标准输入（MLM 的 NAE 也假设 trajectory）。Re-planning 在固定频率仍 tractable。Evidence：real-robot demo 用 DP 输出的 trajectory 直接 feed 我们的 controller。

**Attack 3：和 PFG 区别？两者都是几何 prior。**
defense：PFG 是单时刻 binary IK signal；TRT 是 trajectory-level continuous metric。PFG 解决 stationary reaching；TRT 解决 dynamic tracking。Evidence：B3 vs B6 在 T2 上 ablation，期望 B6 显著优于。

**Attack 4：OCS2 已能解，为什么不直接 deploy MPC？**
defense：sim2real（OCS2 假设精确动力学）；real-time（接触切换 QP 不稳）；unmodeled disturbance；payload。Evidence：B7 (MPC) 在 T3-payload 真机失败、我们成功。

**Attack 5：Multi-critic 已经做过类似事。**
defense：multi-critic 解决 reward 冲突（value space 拆分）；TRT 解决 geometric prior 注入（obs + reward 几何化）。两者互补、不竞争。Evidence：B5 vs B6 直接对比。

**Attack 6：Trajectory tracking 是值得做的吗？现有方法 pose reaching 已经够。**
defense：下游（DP / TAMP / teleop）天然 trajectory；pose reaching 反映不出 velocity profile / temporal coherence。Evidence：pose-reaching-trained policy 在 trajectory task 上的失败 case visualization。

**Attack 7：真机实验规模太小。**
defense：每 task ≥ 10 trials；至少 1 个 challenging disturbance；视频 + dataset + checkpoint 开源。预防机制：R 从 Day 1 起 prep 真机，Phase 3 不会临时抱佛脚。

---

## 8. 28 天怎么排

### 8.1 阶段总览

```
Day 1:        Phase 0a — Pipeline 与 Hypothesis spec     (1 day)
Day 2-4:      Phase 0b — 4 hypotheses 并行验证           (3 days)
Day 5:        Phase 0c — Final story 决策                (1 day)
Day 6-12:     Phase 1  — Core implementation + baselines (7 days)
Day 13-18:    Phase 2  — Full ablation                   (6 days)
Day 19-24:    Phase 3  — Real robot                      (6 days)
Day 25-28:    Phase 4  — Writing                         (4 days)
```

整体节奏的逻辑是：先用 5 天的并行实验把"这个故事到底成不成立"问清楚，然后 7 天主线实现 + 6 天 ablation，留出 6 天给真机（这是唯一不能 AI agent 替代的部分），最后 4 天集中写作。

### 8.2 Phase 0a (Day 1)：Pipeline 准备

### 8.3 Phase 0b (Day 2-4)：4 Hypothesis 并行

每个 hypothesis 独立 git branch、独立 GPU group。

| Hypothesis | 时间表 |
|---|---|
| **H1**：Future obs | D2 实现 future obs encoder（concat 与 transformer 两种）+ 启动 12 runs；D3 中期 check 调超参；D4 收尾出图 |
| **H2**：Pointwise reachability | D2 实现 PFG-style reward + 启动 weight sweep 12 runs；D3 中期 check；D4 收尾 |
| **H3**：TRT prototype | D2 简化 TRT 实现 + reward 集成 + 启动 6 runs；D3 中期 check + TRT 计算可行性最终评估；D4 出图 |
| **H4**：AMP from trajopt | D2 OCS2 200 条 ref + AMP discriminator + 启动；**D3 早 GO/NO-GO**——不稳定就 abort；D4 收尾 if 仍在跑 |

**同时进行**：R 全程做真机 prep——D2 硬件 calibration；D3 sim2real pipeline 测试（先用 vanilla policy）；D4 Phase 3 trajectory dataset 与 logging 准备。这是 v1.2 引入的关键改动：**真机不能等到 Phase 3 才动手**。

Day 4 晚上：4 个 hypothesis 的 first-pass result 全部出齐。

### 8.4 Phase 0c (Day 5)：Final Story 决策

按 §4.3 决策矩阵选 final story，填写下面这张表：

| Hypothesis | 结果（数字） | 是否显著 | 信心 (L/M/H) |
|---|---|---|---|
| H1 | tracking RMSE: ___ → ___ cm (___% 改善) | ☐ ✓ ☐ ✗ | ___ |
| H2 | tracking RMSE: ___ → ___ cm (___% 改善) | ☐ ✓ ☐ ✗ | ___ |
| H3 | pointwise vs TRT: ___ → ___ cm (___% 改善) | ☐ ✓ ☐ ✗ | ___ |
| H4 | RL vs RL+AMP: ___ → ___ cm (___% 改善) | ☐ ✓ ☐ ✗ | ___ |

**Final story**：______________



### 8.5 Phase 1 (Day 6-12)：核心实现 + Baseline

下面以最可能的 Story C+ 为例展开。其他 story 路线类似，关键 milestone 不变。

主线就两件事：**B6 (our method) 跑出第一版 result**，所有 baseline (B3/B4/B5) 跑出对照。

### 8.6 Phase 2 (Day 13-18)：Full Ablation

主线：把 §6.3 的 ablation matrix 跑完。

D13-15 是核心 ablation 三天——TRT reward on/off、curriculum on/off、tube width scan、future window length scan、manipulability encoding ablation、reward algebra ablation、cross-disturbance ablation 平行跑。R 这边在做真机扰动 task 准备 + dry run 完整 protocol。

D16 复盘 + 主结果决定。**D17 Internal review checkpoint**（§11.4）。D18 sim 实验 freeze（之后只允许 minor bug fix），开始准备 figure。

### 8.7 Phase 3 (Day 19-24)：真机主战

D23-24 决策驱动的补充 trial + failure case 分析 + 视频精剪 + supplementary master。

### 8.8 Phase 4 (Day 25-28)：写作

D25 第一版 draft 全部出（每人一节）；D26 polish + figure 完整；D27 all-hands review + 交叉读；D28 final polish + submission。



### 9.1 MVP 最小可行版本

如果 Day 5 全部 hypothesis 失败：

- Title：*Whole-Body Quadrupedal Trajectory Tracking with Reachability-Aware Training*
- 至少保留 1 个清晰 contribution（reward algebra 或 cross-disturbance training）
- Sim-only paper 也能写
- 严重情况认真考虑改投



---

## 14. References

| 标识 | 论文 | 用在哪 |
|---|---|---|
| QuadWBG | Wang et al., ICRA 2025 | TRT motivation |
| Multi-critic | Vijayan et al., CoRL 2025 | Critic architecture |
| WBPT | Portela et al., ICRA 2025 | Keypoint, locomotion init |
| MLM | Liu et al., 2025 (arXiv) | Trajectory baseline |
| PFG | Hou et al., 2025 (arXiv) | IK feasibility baseline |
| RFM | Jiang et al., RA-L 2024 | Reward algebra |
| ODYSSEY | Wang et al., AAAI 2026 | World-frame EE sampling |
| Constrained pushing | Dadiotis et al., RA-L 2025 | Asymmetric actor-critic |
| Deep WBC | Fu et al., CoRL 2022 | Classic baseline |
| RoboDuet | Pan et al., 2024 | Staged training reference |

---



---

## 附录 A：TRT 计算伪代码

```python
def compute_TRT_simple(EE_trajectory, robot_model, dt=0.05):
    tube = []
    for t in range(len(EE_trajectory)):
        # 在 EE_target 周围采样 base pose 候选
        candidates = sample_base_poses_around(
            EE_trajectory[t],
            n_samples=200,
            radius_xy=0.5,
            radius_z=0.2,
            yaw_range=π,
            pitch_range=0.3
        )

        # 过滤 IK 不可解的
        feasible = []
        for base_pose in candidates:
            q_arm = damped_least_squares_IK(EE_trajectory[t], base_pose, robot_model)
            if q_arm is not None and within_joint_limits(q_arm):
                feasible.append((base_pose, q_arm))

        # 时间连续性
        if t > 0:
            feasible = filter_dt_reachable(
                tube[t-1], feasible, dt, max_base_velocity=1.0
            )

        tube.append(feasible)

    return tube


def reward_TRT(current_base_pose, tube_at_t, sigma=0.1):
    if len(tube_at_t) == 0:
        return 0.0
    distances = [
        weighted_distance(current_base_pose, candidate[0])
        for candidate in tube_at_t
    ]
    d_min = min(distances)
    return exp(-d_min**2 / sigma**2)
```

完整版（Phase 1 实现）需要补的关键属性：

1. EE reachable
2. Base kinematic / dynamic feasible
3. Temporal coherence
4. Manipulability-graded
5. 可选的 trajectory optimization smoothing

## 附录 B：Reward 公式草案

```python
def compute_total_reward(state, action, target, tube, manipulability):
    # Base tracking reward (RFM-style gating)
    p_err = norm(state.ee_pos - target.ee_pos)
    o_err = angle_between(state.ee_R, target.ee_R)
    r_pos = exp(-p_err**2 / 0.01)
    r_ori = exp(-o_err**2 / 0.5)
    r_track = r_pos + 0.5 * r_pos * r_ori  # ori 只在 pos 好时加权

    # TRT reward
    r_tube = reward_TRT(state.base_pose, tube[t])

    # Anticipation reward: bonus when base lead arm motion
    base_vel = state.base_lin_vel
    arm_vel_predicted = predict_arm_velocity(target.future_window)
    cos_align = cos_sim(base_vel, arm_vel_predicted)
    r_anticipate = max(0, cos_align) * (norm(arm_vel_predicted) > 0.1)

    # Cumulative penalty (RFM-style)
    state.error_integral += p_err * dt
    r_cumulative = -0.05 * state.error_integral

    # Regularization
    r_reg = -0.001 * norm(action)**2 - 0.01 * joint_limit_penalty(state.q)

    return r_track + 0.5 * r_tube + 0.3 * r_anticipate + r_cumulative + r_reg
```
