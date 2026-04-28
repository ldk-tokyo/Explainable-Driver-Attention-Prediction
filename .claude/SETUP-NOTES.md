
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
