# Research Roadmap — MLLM Cognitive Probe & Attention-Behavior Temporal Alignment

> 状态: 草案 v1 (2026-05-01),整合 baseline 复现 + pilot pivot + 时序研究全路线。
> 作用: 单一信息源,后续每个 stage 的具体计划 (pilot-plan / phase-1-plan / ...) 都从这份 roadmap 派生。

---

## 0. 研究问题与 falsifiable claim

### Research question

> 在 ego 视角真实驾驶场景中,**MLLM 作为认知探针**提取的 attention dynamics 是否在时间上领先并对齐 ego 行为变化触发,从而可作为驾驶决策的中间认知表示?

### Falsifiable claim

> MLLM probe 的 attention shift events 在 ego brake onset 之前**显著领先**,且 **Proactive 组的 lead time 显著长于 Reactive 组**(超出常规 reaction time 窗口 200-400ms),并且这种领先性与人类 gaze dynamics 在 timing 上一致。

### 区别于 LLada 原论文

- 原论文: 单帧、空间 saliency + 单帧 What/Why
- 本研究: 时序、attention dynamics 与 ego behavior transition 的 temporal alignment

### 区别于传统行为预测

- 传统: 几何/轨迹 + agent state machine
- 本研究: 语义认知探针作为中间表示,语义关注重分配 → 行为切换的结构规律

---

## 1. 研究 scope 边界(已定,不可滑动)

| 维度 | 边界 |
|---|---|
| 研究主体 | **仅 ego 驾驶员**,他车作为交互对象/触发因素,不被建模为决策主体 |
| 行为变化点 | 第一阶段仅 **brake onset**(纵向行为)— 自动从 ego speed 导数提取 |
| Reactive/Proactive 分组 | 自动检测 + BDD-X reason text 半自动 |
| 关注探针 | MLLM-as-probe(LLada baseline 单帧 → 时序版本)+ raw human gaze (BDDA lab gaze) 仅作 sanity reference |
| 研究 falsifiability | Pilot 失败 → 不投入大模型时序改造,降级 framing 为 attention-behavior coupling |

---

## 2. Stage 全景(包含已完成与计划)

| Stage | 目标 | 数据 | 状态 | 时间 |
|---|---|---|---|---|
| **Stage 0** | Baseline anchor | W³DA (4 DS) | ✅ 已完成 (Phase 0/1/1.5) | 已用 ~3 周 |
| **Stage 1** | Pilot — 验证 probe 携带可测 lead time | BDDA + BDD100K (含 BDD-X subset) | 待启动 | 1-2 周 |
| **Stage 2** | Temporal probe 模型构建 | W³DA + BDD100K self-supervised | 等 Stage 1 PASS | 4-6 周 |
| **Stage 3** | Cross-domain & counterfactual | HDD action / DADA risk / nuScenes | 等 Stage 2 完成 | 3-4 周 |
| **Stage 4** | 论文产出与投稿 | 全部 | 等 Stage 3 完成 | 2-3 周 |
| **总计** | | | | **~10-15 周到投稿** |

---

## 3. Stage 0 — Baseline Repro ✅(已完成)

**已完成产物**:
- `ckpts/ATTN-7B-baseline-5090/` — epoch 9 best ckpt,可直接 inference
- `figures/tab1_main_results/per_dataset_transposed_baseline-5090.tex` — 4 DS × 18 指标完整 LaTeX 表
- `paper/limitations-notes.md` — 5 段 footnote 草稿(必加 3,选用 2)
- `experiments.md` — 完整实验流水账

**Stage 0 给后续提供的 anchor**:
1. baseline-5090 ckpt 直接作为 Stage 1 probe 使用,**不需重训**
2. eval pipeline 已稳定,Stage 2/3 任何模型变体都用同口径评估
3. paper LaTeX 表格基础设施 (`scripts/paper/collect_main_results.py`) 已 ready

---

## 4. Stage 1 — Pilot: 验证 lead time 是否值得投资大模型改造

### 1.1 双层结构

**Layer 1 — Probe-gaze timing alignment sanity (BDDA)**
- 输入: BDDA 原始连续 video + lab subject 30Hz gaze(github 公开下载)
- 探针: baseline-5090 inference
- 核心: probe 的 attention shift event timing vs 同段 video 的 lab gaze shift event timing,cross-correlation 与 IoU 显著性
- 目的: 证明 probe 输出是 human-aligned 视觉响应,不是 video-statistical shortcut
- Limitation 必标: BDDA 是 lab subject 看 video,不是 active driving;sanity 验证的是 "video-watching-like attention",非 "active driving attention"

**Layer 2 — Probe-behavior lead time main (BDD100K + VO)**
- 输入: BDD100K 抽样 ~50-100h video subset(BDD-X 子集优先,有 reason 文本)
- Ego motion: ORB-SLAM3 monocular VO 推 per-frame speed/yaw → brake onset / steering event
- Reactive/Proactive 分组: BDD-X reason text + 自动 cue 双源
- 探针: baseline-5090 inference per-frame
- Attention shift 三定义并行: low-level gaze shift / semantic object switch / fixation entropy 二阶导
- 统计: linear mixed-effect model `lag ~ event_type + brake_intensity + (1|driver_id)`,event_type 主效应是核心 contrast
- VO confound 控制: BDD100K IMU subset (~100 events) 双路对照 lag 校准

### 1.2 三层结果决策

| 结果档 | 判定 | 后续行动 |
|---|---|---|
| **强阳性** | Proactive lead time > 800ms,显著超 reactive,三种 attention shift 定义一致 | 进 Stage 2,投入时序 probe 改造 |
| **弱阳性** | 两组都有相关性但 Proactive 提前不显著 | 降级 framing 为 attention-behavior temporal coupling,Stage 2 收窄到只做 single-frame probe paper |
| **失败** | attention shift 滞后或无稳定关系,或仅 low-level gaze shift 显著(说明 visual scanning 不是 cognitive redistribution) | **stop** — 不投入大模型时序改造,reframe 整个研究方向 |

### 1.3 Stage 1 deliverable

- `plans/pilot-attention-behavior-lag.md` — step-by-step 执行计划(等本 roadmap 拍板后写)
- `runs_pilot/<timestamp>/` — VO 输出 + probe inference 缓存 + 事件 CSV
- `reports/pilot-report.pdf` — 2-3 页 pilot 结果报告 + 三层决策建议
- `figures/lag-distributions.{pdf,csv}` — 主图

### 1.4 Stage 1 GPU/CPU 预算

- VO 跑全 video: ~50-100h CPU(单 server 一晚)
- Probe inference: ~3-13h GPU(Layer 1 + Layer 2 合计)
- 分析 + 报告: 2-3 天纯 CPU

---

## 5. Stage 2 — Temporal Probe 模型构建(等 Stage 1 PASS)

### 2.1 核心改造点

| 改造点 | 现状 | 目标 |
|---|---|---|
| 数据 pipeline (utils/dataset.py) | 单帧 sample | video clip (T 帧) sliding window |
| Visual encoder | CLIP single-frame `[B, N, C]` | CLIP frame-by-frame + temporal aggregation `[B, T, N, C]` |
| `[ATTN]` token | 单 token | `[ATTN_t]` sequence 或单 token + temporal context |
| `attn_decoder` | 单帧 cross-attention | 加 temporal self-attention(此处接入 **B1 dispatch hook**) |
| Loss | 单帧 BCE+KL | 加 temporal consistency 正则 |

### 2.2 B1 dispatch hook 在 Stage 2 进入

Stage 2 启动**之前**,把 B1 改动 4 + smoke test 收尾(~半天 CPU + 5min GPU smoke,**不全训**)。这一步:
- 让 `--decoder_type=temporal_xxx` 直接走 dispatch 接入,不重写 plumbing
- 后续 Stage 2 加新 temporal decoder class 只需 ~50 行 + 一个 elif 分支

### 2.3 Stage 2 关键 design 选择(等 Stage 1 后再细化)

- **训练数据**: W³DA 是 sparse,时序训练需要连续帧。两条候选路径:
  - (a) 在 BDD100K 上 self-supervised:用 baseline-5090 inference 得 pseudo attention GT,训 temporal probe with pseudo labels
  - (b) 找 W³DA 源数据集的原始连续视频(BDDA / DADA 公开),用 W³DA sparse 标注做 anchor + 中间帧 unsupervised
  - 选哪条等 Stage 1 完成后讨论
- **架构选择**: temporal cross-attention vs temporal self-attention vs 3D conv vs Video-CLIP 替换基座 — 由 Stage 1 lead time 数值决定哪种粒度足够

### 2.4 Stage 2 deliverable

- 时序 probe ckpt(命名约定 `ckpts/ATTN-7B-temporal-<variant>/`)
- 在 W³DA 上的复现指标(跟 baseline-5090 同口径)
- 在 BDD100K 上的 lead time 数字 vs Stage 1 单帧 probe 数字 — 验证时序版本是否提升 lead time 量级或显著性

### 2.5 Stage 2 决策点

- 时序 probe 是否在 lead time 测量上比单帧 probe **更显著或更长**? 如果不显著,论文降级为 "single-frame MLLM probe is sufficient",contribution 收窄但不致命

---

## 6. Stage 3 — Cross-domain & Counterfactual

### 3.1 Cross-domain 复现

在多个数据集上重复 Stage 2 lead time 实验,验证 robustness:

- **HDD** (Honda):tier-action label 给丰富 behavior taxonomy(brake / lane change / yield / stop) — Stage 1 仅 brake,Stage 3 扩到全 behavior
- **DADA-2000**: accident timing 给"风险变化"的天然 GT,验证 lead time 在 risk-rising 场景下的表现
- **nuScenes / Waymo**: ego pose ground truth(不需 VO 估算),最严格的 lead time 数字

### 3.2 Counterfactual attention intervention

Pilot/Stage 2 是 correlational evidence,Stage 3 加因果证据:
- 训一个简单的 brake prediction model(只用 video 输入,可以是 MoVie / SlowFast 等 SOTA)
- Probe 输出的 attention map mask 掉 (set 到 zero / random)某关键 attention object → brake prediction accuracy 退化多少
- 控制实验:mask 同样大小但**非关键**区域 → prediction 应不退化
- 这给 "认知关注 → 行为预测" 因果链最强证据

### 3.3 Robustness checks(必做)

- Per-driver mixed effect(已在 Stage 1 包含)
- Brake intensity ANCOVA(已在 Stage 1 包含)
- Weather / lighting subgroup(雨天 / 夜间 lead time 是否退化)
- Day-night subgroup
- Reviewer 的 robustness battery 一次过

### 3.4 Stage 3 deliverable

- `figures/lead-time-cross-dataset.pdf` — 4 数据集 lead time 对比 box plot
- `figures/counterfactual-ablation.pdf` — mask 关键 vs 非关键 attention 的 brake prediction 退化对比
- `figures/robustness-subgroups.pdf` — weather / lighting / driver 分组 lead time

---

## 7. Stage 4 — Paper Writing

### 4.1 投稿目标

- 第一选择: ICCV 2027 / CVPR 2027(根据时间窗口)
- 第二选择: T-PAMI / IJCV(期刊更长容忍度)
- 不投: AAAI(autonomous driving 接受度低)

### 4.2 Paper 结构草案

| 段落 | 主要内容 | 数据来源 |
|---|---|---|
| Intro | 几何/轨迹方法 vs 认知中间表示;MLLM 作为可解释 cognitive probe | — |
| Related work | LLada / GazeXplain / driver attention prediction / attention-behavior in cognitive science | — |
| Method | Temporal MLLM probe (Stage 2 architecture) + lead time measurement framework | Stage 2 |
| Experiments §A | Probe-gaze alignment sanity (Layer 1) | Stage 1 |
| Experiments §B | Lead time on BDD100K(Reactive vs Proactive) | Stage 1 + Stage 2 |
| Experiments §C | Cross-dataset robustness | Stage 3 |
| Experiments §D | Counterfactual attention intervention | Stage 3 |
| Limitations | Lab gaze ≠ active driving / VO confound / single-cohort | paper/limitations-notes.md |
| Conclusion | 关注重分配作为驾驶认知中间表示;applications | — |

### 4.3 Stage 4 deliverable

- 主稿 + supplementary (4-page + ~6-page sup)
- 所有图表 reproducible,scripts 在 `scripts/paper/`
- Code release plan(基于本 repo + 数据下载脚本)

---

## 8. Risk Map & 降级路径

| Risk | 检测时机 | 降级路径 |
|---|---|---|
| Pilot 失败(失败档) | Stage 1 末 | stop research direction;论文不投,改方向 |
| Pilot 弱阳性 | Stage 1 末 | Stage 2 收窄,论文降级 framing 为 coupling not triggering |
| BDDA 下载失败 | Stage 1 初 | 用 DReyeVE sparse W³DA 做粗粒度 sanity,显著弱化 Layer 1 |
| BDD100K 抽样不够 brake event | Stage 1 中 | 加 nuScenes 补充,但 nuScenes 没 reason text,Reactive/Proactive 分组退化为纯启发式 |
| ORB-SLAM3 在某些 video 上 fail | Stage 1 中 | 切换 monocular depth 方法 (MonoDepth2),pilot 报告里标 VO failure rate |
| Stage 2 时序训练数据 unsolvable | Stage 2 中 | 论文降级到 "single-frame probe is sufficient",Stage 3 仍可做 |
| Counterfactual prediction model 难训 | Stage 3 中 | 用现成 SlowFast / VideoMAE pre-trained,只 fine-tune brake-classification head |

---

## 9. 跟现有 repo 资产的映射

| Repo asset | Stage 用法 |
|---|---|
| `ckpts/ATTN-7B-baseline-5090/` | Stage 1 probe / Stage 2 init / Stage 3 baseline reference |
| `figures/tab1_main_results/` | Stage 4 paper Tab 1 (LLada baseline reference) |
| `paper/limitations-notes.md` | Stage 4 paper limitations |
| `scripts/paper/collect_main_results.py` | Stage 2/3 任何变体的 eval 自动化 |
| `model/Attn_model.py` (B1 dispatch hook) | Stage 2 时序 decoder 接入点 |
| `experiments.md` | 全 stage 流水账(单一信息源) |
| `plans/B1-implementation-plan.md` | B1 收尾参考(改动 4 + smoke,不全训) |
| `plans/pilot-feasibility-note.md` | Stage 1 起点的 blocker 记录 |

---

## 10. 立即下一步

按时间顺序:

1. **本 roadmap 用户 review + 拍板**(不动手,先确认)
2. **Step 1 修订版**(~1h CPU): 确认 BDDA + BDD100K 公开下载现状,确认 BDD100K 是否含 IMU,跑 ORB-SLAM3 在一段 BDD video 上 sanity
3. **B1 改动 4 + smoke 收尾**(~半天 CPU + 5min GPU): 让 dispatch hook ready,Stage 2 时序 decoder 接入有现成 plumbing
4. **`plans/pilot-attention-behavior-lag.md` 实施计划**(本 roadmap 拍板后写,半天 CPU): step-by-step pilot 代码模块、ANCOVA / mixed-effect 模型规格、判定阈值
5. **Stage 1 启动**(等 pilot plan PASS)

每一步都等用户授权,不擅自启动 GPU 进程。

---

## 11. 决策点清单(用户拍板节奏)

| # | 决策 | 时机 |
|---|---|---|
| D1 | 接受本 roadmap | 现在 |
| D2 | Step 1 (BDDA/BDD100K 可获取性 + ORB-SLAM3 sanity) 启动 | D1 后 |
| D3 | B1 dispatch 收尾(并行) | D1 后 |
| D4 | Pilot plan v0 review | Step 1 PASS 后 |
| D5 | Stage 1 GPU inference 启动 | D4 后,每次 GPU 启动单独授权 |
| D6 | Stage 1 三档结果决策(进 Stage 2 / 降级 / stop) | Pilot report 出来后 |
| D7 | Stage 2 训练数据策略(self-supervised vs raw video) | D6 PASS 后 |
| D8 | 投稿目标确认(ICCV 2027 / CVPR 2027 / 期刊) | Stage 3 中 |

每个 D 节点 Claude 不擅自前进,等用户明确指示。
