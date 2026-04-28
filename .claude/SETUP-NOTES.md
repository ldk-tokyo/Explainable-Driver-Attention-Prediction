
# Claude Code 启动包 · LLada 研究项目 (v3: Progressive Disclosure)

这是为 [Where, What, Why (ICCV 2025 Highlight)](https://arxiv.org/abs/2506.23088) 代码仓库准备的 Claude Code 上下文包。

## 本版本针对
- 硬件: **RTX 5090 单卡** (32GB GDDR7, Blackwell, 无 NVLink, CUDA ≥ 12.8)
- 研究方向: **改造注意力解码器** (`attn_decoder`)
- 分析需求: 可解释性诊断 + 论文产出

## v3 相比 v2 的优化 (Progressive Disclosure)

**核心改进**: SKILL.md 变薄 → references/*.md 承载细节。让 Claude 按需加载,避免长文件细节被忽略。

| 文件 | v2 行数 | v3 行数 | 说明 |
| --- | --- | --- | --- |
| `CLAUDE.md` | 271 | **~90** | 只保留硬约束 + 决策树 |
| `modify-llada/SKILL.md` | 433 | **~180** | 8 种 decoder 拆到 `references/decoders/` |
| `interpretability-analysis/SKILL.md` | 344 | **~110** | 7 种工具拆到 `references/` |
| `paper-writing/SKILL.md` | 367 | **~120** | matplotlib / 图 / 表 / 显著性拆到 `references/` |
| `train-eval-workflow/SKILL.md` | 314 | **~140** | 显存预算拆到 `references/memory-budget.md` |

每个 references/ 文件独立,Claude 只在需要时才 view。

## 包含内容

```
CLAUDE.md                                   # 项目总指令 (精简导航)
.claude/skills/
├── llada-architecture/SKILL.md             # 架构速查
├── w3da-dataset/SKILL.md                   # 数据集结构
├── train-eval-workflow/
│   ├── SKILL.md                            # 5090 训练/评测核心流程
│   └── references/memory-budget.md         # 32GB 显存详解
├── modify-llada/
│   ├── SKILL.md                            # 改造总控 + 方案矩阵
│   └── references/
│       ├── decoders/                       # 8 种 decoder 方案
│       │   ├── interface-contract.md       # 接口契约
│       │   ├── implementation-checklist.md # 5 步标准流程
│       │   ├── B1-deeper-cross-attn.md     # 详细代码
│       │   ├── B3-pyramid.md               # 详细代码
│       │   ├── B7-sam-style.md             # 详细代码
│       │   ├── ab-test-protocol.md         # 统计检验
│       │   └── other-decoders-summary.md   # B2/B4/B5/B6/B8 概要
│       └── patterns/                       # 非 decoder 的其他 pattern
│           ├── pattern-A-new-token.md
│           └── pattern-CDE-combined.md
├── experiment-tracking/SKILL.md            # 实验管理
├── debug-training/SKILL.md                 # 症状索引 + 修复流程
├── interpretability-analysis/
│   ├── SKILL.md                            # 分析工具路由
│   └── references/
│       ├── qualitative-vis.md              # 4×N 对比图
│       ├── error-clustering.md             # CLIP 聚类失败样本
│       ├── attn-probing.md                 # [ATTN] hidden 探针
│       └── advanced-tools.md               # GradCAM/rollout/counterfactual/crossmodal
└── paper-writing/
    ├── SKILL.md                            # 产出路由
    └── references/
        ├── matplotlib-config.md            # 顶会 rc 配置
        ├── figure-qualitative.md           # 定性对比图模板
        ├── table-main-results.md           # booktabs 主表
        └── ablation-efficiency-repro.md    # 消融/效率/复现性
```

**总计 23 个 markdown 文件,约 3,500 行**。但任何单次 Claude Code 对话只会加载 `CLAUDE.md (~90 行) + 命中 skill 的 SKILL.md (~100-200 行)`,约 5-15% 的总量。按需加载 references/ 文件细节。

## 安装

```bash
cd ~/Explainable-Driver-Attention-Prediction  # 或你的 fork 路径
tar xzf llada-claude-setup.tar.gz
git add CLAUDE.md .claude/
git commit -m "docs: Claude Code context for LLada decoder research"
```

启动 Claude Code: `claude`。

## 验证 Claude 正确加载

试探性问题:
1. "帮我跑 smoke test" → 引用 `train-eval-workflow` 的 5 分钟命令
2. "我想把 attn_decoder 换成金字塔结构" → 引用 `modify-llada`,再进 `references/decoders/B3-pyramid.md` 看具体代码
3. "分析一下 baseline vs pyramid 的差异" → `interpretability-analysis` → `error-clustering.md` + `qualitative-vis.md`
4. "帮我做主表" → `paper-writing` → `table-main-results.md`
5. "我显存 OOM" → `debug-training` 症状索引 → `train-eval-workflow/references/memory-budget.md`

## Description 消歧 (v3 重要改进)

v2 的 `interpretability-analysis` 和 `paper-writing` 都提到"可视化",触发容易冲突。v3 明确分工:

- **interpretability-analysis**: "诊断模型行为",产出 **分析 notebook 和中间图**
- **paper-writing**: "产出论文交付物",生成 **最终 PDF / LaTeX 投稿用**

## 研究路线图 (参考)

1. **Phase 1 (1 周)**: 装环境 → 下数据 → smoke → 跑 baseline
2. **Phase 2 (2-3 周)**: B1 depth 扫描 (练手 + 上限探测)
3. **Phase 3 (3-4 周)**: B3 Pyramid + B7 SAM-style 主攻
4. **Phase 4 (2 周)**: 消融 + A/B 显著性检验
5. **Phase 5 (2-3 周)**: 定性图 + 表 + 论文

**总预算 10-12 周**。

## 常见调整

- **要把某个 decoder 方案 (比如 B2/B4) 扩展成完整代码**: 告诉 Claude "把 B2 从概要扩展成 B1 那样的完整实现"
- **换投稿目标**: 告诉 Claude 会议名,matplotlib 配置会微调
- **加一张卡 (data-parallel)**: 要求"补 data-parallel 的 train-eval-workflow"
- **想做时序扩展**: 读 `modify-llada/references/patterns/pattern-CDE-combined.md` (Pattern D)

---

## Project Status as of 2026-04-28

**进度**: baseline 锚点已建立, 即将进入 Phase 1 (decoder 改造主线), 准备阶段未开始改代码。

### 已完成

- **baseline-5090 训练**: 10 epoch, 5000 step, RTX 5090 单卡 ~5h25m, ckpt `ckpts/ATTN-7B-baseline-5090/`
- **Phase 1.5 完整 eval (4 子集)**: BDDA / DReyeVE / LBW / DADA 全部跑完 6 attn + 12 text 指标。详细数字 + 与论文 Table 1/2 对比见 [experiments.md](../experiments.md)。
- **Q3 解决**: 论文 Table 2 的 "BLEU" = **BLEU-1** (不是 BLEU-4)。证据: BDDA 0.405 vs 论文 0.444 (-0.039), DADA 0.391 vs 0.376 (+0.015), 量级吻合; BLEU-4 量级完全不对 (-0.15 ~ -0.27)。
- **main 分支锚点**: `0ef61b4` (上游 W3DA truncation bug fix + TB layout) + `23ec3b3` (Claude Code context + experiments anchor)
- **B1 实施草案**: [plans/B1-implementation-plan.md](../plans/B1-implementation-plan.md) (CPU-only, 等用户 review, 还没改代码)

### 下一步

Phase 1 B1 deeper cross-attention decoder 实验 (草案见 [plans/B1-implementation-plan.md](../plans/B1-implementation-plan.md))。
- 推荐方案 B1c: 加 `--decoder_type / --decoder_depth` CLI + 轻量 dispatch 钩子, ~30 行改动 4 处文件, 不重构 `model/decoders/`
- 等用户 review 草案 → 说"按草案改" → `git checkout -b decoder/B1-cross-depth4` → 实施。

### 已建立的硬规则 (CLAUDE.md 已固化)

1. **GPU 命令必须等 user 明确说"跑"才执行**, 启动后立即声明 PID
2. **NEVER update git config** (用户已设 `ldk950413 <ldk110714@gmail.com>`)
3. **不写 /tmp 调度脚本** (历史教训: dd3d2295 session 写过 `/tmp/launch_eval.sh` for 循环, 已禁用为 `.disabled`)
4. **改 model code 前必须 git checkout -b**, 每步小 commit + smoke test 通过才进全量训练

### Open Questions (在 [experiments.md](../experiments.md), 不堵 Phase 1)

- **Q1** NSS NaN: BDDA 76.6% 样本 NSS=NaN (gazemap < 0.7 阈值二值化全 0), 上游代码同样行为, 论文 supp 无说明 → LaTeX 表加 dagger 注脚
- **Q2** 训练总 step 数: 论文未公布
- **Q3** ✅ BLEU = BLEU-1 已确认
- **Q4** DReyeVE BLEU-1 反超 (我们 0.518 vs 论文 0.436, +0.074): 推测 `train_sample_rates=8,5,2,7` 让 DReyeVE 训练欠权重, test 时反而擅长。Phase 1 后可单独跑 sample_rate 消融
- **Q5** CIDEr-R Normal 巨差 (我们 0.463 vs 论文 0.963): 待核对论文报的是 CIDEr 还是 CIDEr-R

### 工程债 (Phase 1 完成后处理)

- **磁盘 466 GB → 可清 154 GB**: 4 个中间 epoch ckpt (epoch 2/3/4/6, 各 30 GB = 120 GB) + 8 个 dataset zip (34 GB)
- **log_test.txt Cider 字段 bug**: train_ds.py print 把 Cider 字段写成 CiderR 值 (eval log stdout 是真实值, collect script 不受影响), 修一行打印代码

### 复现 Phase 1.5 evaluation (CPU 命令)

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python3.11 \
  scripts/paper/collect_main_results.py \
  --exps "baseline-5090=Baseline (ours repro)" \
  --out figures/tab1_main_results --per-dataset
```
产 [figures/tab1_main_results/per_dataset_transposed_baseline-5090.tex](../figures/tab1_main_results/per_dataset_transposed_baseline-5090.tex), 4 DS × 10 metric 全满。

