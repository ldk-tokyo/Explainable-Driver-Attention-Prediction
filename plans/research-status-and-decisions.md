# Research Status Report & Decision Brief

> 时间: 2026-05-01
> 用途: 给用户做研究方向决策。Self-contained — 不需要读其他文档也能完整理解现状和需要拍板的事。
> 关联文档: [research-roadmap.md](research-roadmap.md) (具体 stage 细节), [pilot-feasibility-note.md](pilot-feasibility-note.md) (DR(eye)VE blocker), [B1-implementation-plan.md](B1-implementation-plan.md) (B1 草案), [../experiments.md](../experiments.md) (实验流水账)

---

## TL;DR(2 段)

**1. 你已经完成 LLada baseline 的高质量复现**(commit 6a290dc, ckpts/ATTN-7B-baseline-5090/)。4 子集 × 18 指标全跑完,Tab1 attn 跟论文在 ✓/⚠ 级别(CC 差 -0.022 ~ +0.004),Tab2 文本已查清"论文 BLEU = BLEU-1"。**这套数字足够给后续研究做对照锚点**,继续重训缩小绝对数字差距投入产出比不高(单卡硬件天花板)。

**2. 你 2026-05-01 提出新研究方向**(MLLM-as-probe + attention-behavior temporal alignment),跟 LLada 的关系发生质变 — 单帧 decoder 加深(B1)对你新研究**几乎无用**,但 B1 的 dispatch hook 对时序 decoder 接入仍有 ~50 行省力价值。Pilot 设计经过 4 轮迭代(DR(eye)VE → 数据 blocker → BDD-X pivot → VO + Layer 1 sanity),最终方案是 **BDDA lab gaze sanity + BDD100K + VO 双层 pilot**。**这个方案需要你今天拍板 5 件事(D1.1–D1.5),才能进 Step 1**。

---

## Part 1: 前因 — 项目从哪儿来,已经做了什么

### 1.1 项目起点

- **目标**: 复现 *Where, What, Why: Towards Explainable Driver Attention Prediction* (ICCV 2025 Highlight, arXiv 2506.23088),作为后续研究的 baseline
- **用户角色**: 研究型 AI 工程师,主攻 attn_decoder 改造方向
- **硬件约束**: 单张 RTX 5090 (32GB Blackwell, sm_120),bf16 only,无 NVLink
- **关键不可变**: effective batch 16 vs 论文 4×A100 batch=160(差 10×,**单卡无法 close**)

### 1.2 已完成阶段

| Phase | 内容 | 时间 | Deliverable |
|---|---|---|---|
| Phase 0 | 环境搭建 + SETUP-FIXES(METEOR Java + paraphrase-en.gz) | 2026-04-23 ~ 04-24 | `.venv/`, `SETUP-FIXES.md`, `requirements.txt` |
| Phase 1 | baseline-5090 训练 (10 ep × 500 step) | 04-24 ~ 04-25 (5h25m) | `ckpts/ATTN-7B-baseline-5090/` (epoch 9 best, 合并完) |
| Phase 1.5 | 4 DS × 18 指标 完整 eval | 04-26 ~ 04-28 (~36h) | `figures/tab1_main_results/` LaTeX 表 |
| (B1 进行中) | decoder cross-attn 加深 3→4 | 04-28 起 | 分支 `decoder/B1-cross-depth4`, 改动 1-3 已 commit |

### 1.3 复现质量盘点

**Attn 指标(论文 Tab 1, key-frame W³DA test)**:
| 场景 | 我们 CC | 论文 CC | 差距 | 状态 |
|---|---|---|---|---|
| Safety-Critical (BDDA) | 0.573 | 0.579 | -0.006 | ✓ |
| Normal (DR+LBW weighted) | 0.587 | 0.583 | +0.004 | ✓ |
| Accident (DADA) | 0.374 | 0.396 | -0.022 | ⚠ |

**Text 指标(论文 Tab 2)**:
- 已确认论文 "BLEU" = BLEU-1(BDDA -0.039 / DADA +0.015 量级吻合)
- Normal (DR+LBW) BLEU_1 反超论文 +0.074 (DReyeVE 主导,与 train_sample_rates 8,5,2,7 有关)
- KLD/NSS 普遍略差(主要源于 effective batch 10× 差距)
- BDDA NSS 76.6% NaN(gazemap threshold=0.7 artifact)

**结论**: 复现质量足够给后续研究做 anchor。**继续重训不能 close 与论文的绝对差距**(硬件天花板),应在论文 limitations 段落诚实标注(`paper/limitations-notes.md` 已起草 5 段 footnote)。

### 1.4 进行中的工作:B1 decoder 改造

- **目标**: cross-attention decoder `num_layers=3→4`(单点),严格 1-变量"加深"消融
- **状态**: 4 处改动里 3 处已 commit(CLI / model_args plumbing / config stash),改动 4 (dispatch) 尚未做
- **未跑**: smoke test / backward-compat check / 全训(80-100h)

---

## Part 2: 转折点 — 2026-05-01 研究方向 pivot

### 2.1 用户新研究方向(原话精简)

> 研究 ego 视角下交通参与者(=ego driver)在潜在交互出现时,**决策相关关注结构如何随时间变化**;假设人类行为变化来源于这种关注结构的重分配;通过弱标注 GT(行为变化点 + 交互开始时刻 + 风险变化),用 **MLLM 作为认知探针**提取随时间变化的关注结构,验证关注变化是否在时间上**领先**并对齐行为变化触发机制,以提供一种新的**驾驶决策中间认知表示**。

### 2.2 这个方向跟 LLada baseline 的关系

| 维度 | 契合度 | 说明 |
|---|---|---|
| MLLM-as-probe 概念 | **强契合** | LLada 论文 §4.2 把 [ATTN] token 描述为高级认知线索载体,What/Why 是从这里读出的语义关注结构 — 这就是你说的"具备语义先验的认知探针" |
| Ego 视角 driver attention | **强契合** | W³DA 4 子集都符合,数据范式一致 |
| 单帧 vs 时序 | **关键 gap** | LLada 是单帧空间 saliency;你需要时序 attention dynamics |
| 行为弱标注 GT | **gap** | LLada 监督是 attention map + What/Why text;你需要 brake onset / interaction onset / risk delta |
| "领先性"分析 | **gap** | LLada 没做时序统计,你需要 cross-correlation / mixed-effect / counterfactual |

### 2.3 B1 在新方向里的角色变化

- **训练数据点(depth=4 vs 3 的单帧 CC)**: 对时序研究**完全无用**
- **Dispatch hook**: 仍有价值 — 后续时序 decoder 接入只需 ~50 行,否则 ~250 行
- **结论**: 跳过 B1 全训(省 80-100h GPU),只完成 dispatch hook(~半天 CPU + 5min GPU smoke)作为时序 decoder 基础设施

### 2.4 Pilot 设计的 4 轮迭代(完整 reasoning chain)

**v0 (我初稿)**: DR(eye)VE + ego CAN,测 raw human gaze 是否领先 brake onset
- 问题: 简单 lead time > 0 容易被"reaction time = 200-400ms"trivial 解释吃掉

**v1 (用户精修)**: Reactive vs Proactive 对照,三种 attention shift 定义并行,mixed-effect model
- 改进: Reactive 组就是 reaction-time control,Proactive lead time > Reactive lead time 是研究真正主张
- 这一版科研严谨度跳升一档(pre-registration 级)

**v2 (DR(eye)VE blocker 发现)**: 检查 dataset/DReyeVE/ 资产 → 发现是 **W³DA processed sparse subset**,不是原始连续视频
- dataset/DReyeVE/ 是 0.6-1.8s 间距 sparse 抽样
- 没有连续 gaze trace,没有 ego CAN
- **0.6s 时间分辨率根本测不到 reaction time 量级**
- 用户表态: 拿不到 DR(eye)VE 完整数据,需要替代方案

**v3 (Pivot A)**: 重新审视 pilot 角色 — 用户研究 final claim 是关于 MLLM probe 的,不是 raw human gaze
- 改用 BDD-X / BDD100K(完全公开,无 access blocker)
- ORB-SLAM3 monocular VO 推 ego motion(摆脱 CAN 依赖)
- BDD-X 自带 reason 文本 → reactive/proactive 半自动分组比 DR(eye)VE 更准
- BDD100K 不在 W³DA training set 里 → zero-shot OOD 天然挡训练 leakage 反驳

**v4 (Layer 1 sanity 补缺)**: 用户问"是否还需要 gaze GT" — 暴露 reviewer 反驳缺口
- Pivot A 直接测 "probe leads behavior",但 probe 可能学到 video-statistical shortcut(brake light 比 brake onset 早 800ms 出现等),跟人类认知无关
- 加 Layer 1: 在 BDDA(公开,lab subject 连续 gaze)上验证 probe attention shift timing 与 lab gaze timing 一致
- Limitation 必标: BDDA gaze 是 lab 看视频,不是 active driving — sanity 验证的是 "video-watching-like attention"

**v4 = 当前方案,等用户 D1 拍板**

---

## Part 3: 目前的状态

### 3.1 已经决定的事(lock,不再变动)

- ✅ Stage 0 baseline 复现完成,作为后续研究 anchor
- ✅ 研究 scope: ego only;brake onset 优先;MLLM 作为 probe
- ✅ Pilot 三层结果决策框架(强 → Stage 2 / 弱 → 降级 framing / 失败 → stop direction)
- ✅ Reactive vs Proactive 对照(不是简单测 lead time > 0)
- ✅ Mixed-effect 统计模型(`lag ~ event_type + brake_intensity + (1|driver_id)`)
- ✅ 三种 attention shift 定义并行(low-level / semantic object / entropy 二阶导)
- ✅ B1 处置: 跳过全训,只 dispatch 收尾

### 3.2 已写好的文档(本仓库内)

| 文档 | 用途 | 状态 |
|---|---|---|
| `CLAUDE.md` | 项目硬约束 + GPU 规则 | ✅ 已更新 (effective batch 80 → 160 修复) |
| `experiments.md` | Phase 0/1/1.5 完整流水账 | ✅ 现成,Stage 1 起继续追加 |
| `SETUP-FIXES.md` | METEOR Java + paraphrase-en.gz | ✅ |
| `paper/limitations-notes.md` | 5 段 footnote 草稿(必加 3 + 选用 2) | ✅ 本会话刚起草 |
| `plans/B1-implementation-plan.md` | B1 草案 v3(改动 1-3 已 commit) | ✅ |
| `plans/pilot-feasibility-note.md` | DR(eye)VE blocker 发现报告 | ✅ 本会话刚起草 |
| `plans/research-roadmap.md` | 5 stages 全路线图 | ✅ 本会话刚起草 |
| `plans/research-status-and-decisions.md` | **本文档** | ✅ |
| `plans/pilot-attention-behavior-lag.md` | Stage 1 step-by-step plan | ⏳ 待写(D1 PASS 后) |

### 3.3 本会话内的代码改动(等 commit)

| 文件 | 改动 | 影响 |
|---|---|---|
| `train_ds.py:424-426` | `Cider:` 字段从 `ciderR` 改成 `cider` 变量(3 段) | log_test.txt 字段值正确,collect_main_results 不受影响 |
| `CLAUDE.md:20` | effective batch 注释 80 → 160 | 文档准确性 |
| `paper/limitations-notes.md` | 新建 | paper 草案基础 |
| `plans/pilot-feasibility-note.md` | 新建 | DR(eye)VE blocker 记录 |
| `plans/research-roadmap.md` | 新建 | 5 stages 路线图 |
| `plans/research-status-and-decisions.md` | 新建(本文件) | 决策清单 |

### 3.4 当前 git 状态

- **Branch**: `decoder/B1-cross-depth4`
- **未 commit 改动**: 上表 6 个文件
- **问题**: 这些改动跟 B1 关系松散,放 B1 分支上有点混。建议处置见 D1.6 决策

### 3.5 GPU 资源使用现状

- 已用 GPU: ~5.5h baseline 训练 + ~36h 4 DS eval = ~41.5h
- B1 全训(若做): +80-100h(**计划放弃**)
- Stage 1 inference 预算: 3-13h(Layer 1 ~3h + Layer 2 ~5-10h)
- Stage 2 时序模型训练(若 D6 PASS): 100-150h(等 D6 拍板)

---

## Part 4: 之后的计划(Stage 1-4)

详细在 [research-roadmap.md](research-roadmap.md),这里给浓缩版。

### 4.1 Stage 1 Pilot (1-2 周)

**Layer 1 — Probe-gaze timing alignment sanity (BDDA)**
- 用 BDDA 公开下载的原始连续 video + lab gaze
- baseline-5090 inference 出 probe attention shift events
- 跟 lab gaze shift events 做 cross-correlation + IoU
- 通过判定: probe 跟 lab gaze 在 timing 上显著相关 → probe 是 human-aligned

**Layer 2 — Probe leads behavior (BDD100K + VO)**
- BDD100K 50-100h video subset(BDD-X 子集优先,有 reason 文本)
- ORB-SLAM3 monocular VO 推 per-frame ego speed/yaw → brake onset
- BDD-X reason 文本做 reactive/proactive 半自动分组
- baseline-5090 inference per-frame attention map
- Mixed-effect model: `lag ~ event_type + brake_intensity + (1|driver_id)`
- VO confound 控制: BDD100K IMU subset 双路对照(if BDD100K 有 IMU)

**三层决策**:
- 强阳性(Proactive lead > 800ms,显著超 Reactive)→ Stage 2
- 弱阳性(两组都相关但 Proactive 不显著领先)→ 降级 framing,paper claim 收窄
- 失败(probe 滞后或仅 low-level shift 显著)→ stop direction

### 4.2 Stage 2 Temporal Probe (4-6 周, D6 PASS 才启动)

- 5 个改造点: 数据 pipeline / visual encoder / [ATTN] token / decoder / loss
- B1 dispatch hook 在此用上
- 训练数据策略 D7 拍板(self-sup on BDD100K vs 找 W³DA 源原始视频)
- Deliverable: 时序 probe ckpt + W³DA 上同口径指标 + BDD100K lead time 升级数字

### 4.3 Stage 3 Cross-domain & Counterfactual (3-4 周)

- HDD / DADA / nuScenes 三集 lead time 复现
- Counterfactual: ablate 关键 attention 看 brake prediction 退化
- Robustness checks(weather/lighting/per-driver subgroup)

### 4.4 Stage 4 Paper Submission (2-3 周)

- 投稿目标: ICCV 2027 / CVPR 2027 或期刊兜底(D8 拍板)
- 主稿 4-page + supplementary
- Method (Stage 2) + 4 段实验 (Layer 1 sanity + lead time + cross-dataset + counterfactual)

### 4.5 总 timeline

- ✅ Stage 0: 已用 ~3 周
- Stage 1-4: ~10-15 周到投稿(假设 Stage 1 PASS;失败则 6-8 周降级 paper)

---

## Part 5: 用户需要做的决策

### 5.1 立即决策 D1(今天拍板,才能启动后续工作)

| # | 决策 | Default 推荐 | 理由 |
|---|---|---|---|
| **D1.1** | 接受 5 stages roadmap 框架? | ✅ 接受 | 本文档 + roadmap 已展开完整 reasoning chain |
| **D1.2** | 接受 Pivot A(MLLM-probe 直接测,不绕道 raw human gaze)? | ✅ 接受 | DR(eye)VE 拿不到是硬约束;Pivot A 跟你 final research claim 更对齐;失去 raw gaze sanity 用 Layer 1 BDDA 补 |
| **D1.3** | 接受 BDDA lab gaze 作为 Layer 1 sanity reference,limitations 标注 "video-watching-like attention" 折中? | ✅ 接受 | 没有更好选项;reviewer 在 pilot 阶段不会要求拿不到的数据 |
| **D1.4** | 接受 timeline 估算 10-15 周到投稿(无 deadline 约束情况下)? | 看你的 deadline | 如果有硬 deadline(投稿截止),要回头压缩 |
| **D1.5** | B1 处置: 跳过全训,只完成 dispatch 收尾(改动 4 + smoke,不烧 80-100h GPU)? | ✅ 接受 | dispatch hook 是时序 decoder 接入基础设施,~半天 CPU 投入,但全训对时序研究无用 |
| **D1.6** | 当前未 commit 的 6 个文件(本会话改动)的处置? | 见下面三选项 | branch 整洁性 |

### 5.2 D1.6 三个 commit 处置选项

- **选 A** (推荐): 在当前 `decoder/B1-cross-depth4` 分支累积 commit,等 B1 dispatch 收尾完毕一起评估是否 merge 回 main。优点: 不动 branch 结构;缺点: 文档改动跟 B1 编码混在一个分支
- **选 B**: 新建 `docs/research-pivot` 分支,把 6 个文件 commit 过去并 merge 回 main。优点: 文档跟代码改动分清;缺点: branch 多
- **选 C**: cherry-pick 文档类改动到 main(Cider bug + CLAUDE.md + paper/limitations + plans/*.md),B1 编码留在分支。优点: main 立即有最新文档;缺点: 操作多

### 5.3 短期决策 D2-D5(D1 PASS 后陆续遇到)

| # | 决策 | 时机 | 行动 |
|---|---|---|---|
| D2 | 启动 Step 1 重新版(BDDA / BDD100K 可获取性确认 + ORB-SLAM3 minimal sanity test) | D1 后立即 | ~1h CPU,我做 |
| D3 | B1 dispatch 收尾(改动 4 + smoke,并行不阻塞) | D1 后立即 | ~半天 CPU + 5min GPU smoke,等 D1 + 单独 GPU 授权 |
| D4 | Pilot plan v0 review(我写,你 review) | Step 1 PASS 后 | ~半天 CPU 我写,你 review |
| D5 | Stage 1 GPU inference 启动(Layer 1 BDDA + Layer 2 BDD100K) | D4 PASS 后 | **每次 GPU 启动单独授权**(CLAUDE.md GPU 规则) |

### 5.4 中长期决策 D6-D8(Stage 进展时遇到)

| # | 决策 | 时机 | 内容 |
|---|---|---|---|
| D6 | Stage 1 三档结果分流 | Pilot report 出来后 | PASS 进 Stage 2 / 弱阳性降级 / 失败 stop |
| D7 | Stage 2 训练数据策略 | D6 PASS 后 | (a) BDD100K self-sup with pseudo labels (b) 找 W³DA 源原始视频 |
| D8 | 投稿目标确认 | Stage 3 中 | ICCV 2027 / CVPR 2027 / T-PAMI / IJCV |

---

## Part 6: Open Questions / Known Unknowns

> 这些问题不阻塞 D1 决策,但 Step 1 / Stage 1 时会一一确认。

1. **BDDA 公开下载现状** — github.com/pascalxia/driver_attention_prediction 仓库是否仍维护;原始连续 gaze 是否打包发布
2. **BDD100K 是否含 IMU** — Yu et al. CVPR 2020 论文提及 GPS+IMU,但实际 release 包是否包含 IMU 元数据
3. **ORB-SLAM3 在 BDD video 上失败率** — 低纹理场景(高速公路 / 隧道 / 夜间)VO 失败概率
4. **Stage 2 时序训练数据来源** — Stage 1 完成后才能定
5. **`app.py` 功能** — README 没说,我也没读;不影响主线
6. **`explanatory=0.1` 确切作用** — train_ds.py 里有但没追;不影响主线
7. **是否有公开 LoRA 预训练 ckpt 跳过 Stage 0** — 已经训完,无需

---

## Part 7: Risk Map(浓缩版,详见 roadmap §8)

| Risk | 概率 | 影响 | 降级 |
|---|---|---|---|
| Pilot 失败 | 中 | 整个研究方向 stop | Reframe 或不投稿 |
| Pilot 弱阳性 | 中 | paper claim 收窄 | "coupling" not "triggering",仍可投 |
| BDDA 下载失败 | 低 | Layer 1 sanity 弱化 | DReyeVE sparse 做粗粒度 sanity |
| BDD100K 抽样 brake event 不够 | 低 | 统计 power 不足 | 加 nuScenes 补充(无 reason text) |
| ORB-SLAM3 失败率高 | 低 | VO confound 大 | 切 MonoDepth2 |
| Stage 2 训练数据 unsolvable | 中 | 论文降级到 single-frame | 仍可投,contribution 收窄 |

---

## 行动召唤

请用户对 **D1.1–D1.6 六个决策**逐一表态。最关键的是 **D1.5(B1 处置)** 和 **D1.6(commit 策略)**:

- D1.5 直接影响今天我下一步做什么 — 跳过全训就能立即并行 D2 + D3
- D1.6 直接影响 git 分支结构 — 跑偏太久不好回头

其他 D1.x 你接受/拒绝/部分接受都行,有保留意见请直说,我据此调整 roadmap。

D1 PASS 后,我立即:
1. 按 D1.6 选项处置当前 6 个未 commit 文件
2. 启动 D2(BDDA / BDD100K 可获取性确认 + ORB-SLAM3 minimal sanity)— ~1h CPU,完全不烧 GPU
3. 按 D1.5 启动 D3(B1 改动 4 + smoke 收尾)— ~半天 CPU,等 GPU smoke 单独授权

GPU 启动 always 等用户明确授权,never 擅自前进。
