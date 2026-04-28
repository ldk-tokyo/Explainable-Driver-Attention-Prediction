---
name: interpretability-analysis
description: **诊断模型行为**的工具箱:回答"模型为什么这样预测"、"哪类样本失败"、"[ATTN] 到底学到什么"。涵盖错误样本聚类 / linear probe / GradCAM-for-heatmap / attention rollout / 反事实遮挡 / What-Why 文本 vs Where 热力图的跨模态对齐。产出**分析性 notebook 和中间图**,不产出论文最终 PDF(那是 `paper-writing` skill 的事)。用户说"分析行为"、"为什么预测错"、"失败模式"、"探针"、"`[ATTN]` 学到什么"、"模型在看什么"、"错误聚类"、"GradCAM"、"rollout"、"反事实" 时使用。对 decoder 改造研究尤其重要 —— 不同 decoder 的定性差异只能靠这些工具揭示。
---

# 可解释性分析总控

## 何时用

决定要回答哪类"为什么",再读对应 reference:

| 研究问题 | 工具 | Reference |
| --- | --- | --- |
| 某实验失败样本有共性吗? | 错误样本聚类 | `references/error-clustering.md` |
| 两个 decoder 的定性差异? | 定性可视化 4×N 网格 | `references/qualitative-vis.md` |
| `[ATTN]` hidden 到底编码了什么? | Linear probe | `references/attn-probing.md` |
| 预测热力图是被哪些输入像素驱动的? | GradCAM-for-heatmap | `references/gradcam.md` |
| LLM 内部信息流是怎样的? | Attention rollout | `references/attention-rollout.md` |
| 模型依赖图中哪些区域? | 反事实遮挡 | `references/counterfactual.md` |
| What 文本和 Where 热力图是否自洽? | 跨模态对齐 | `references/crossmodal-alignment.md` |

## 工作目录

```
analysis/
├── attention_vis/          # 可视化产出 (每张样本的叠加图)
├── failure_cases/          # 失败样本 topK
├── probing/                # 探针分类器结果 + hidden state dump
├── gradcam/                # GradCAM 叠加
├── crossmodal/             # What-Why 与 Where 的对齐
├── counterfactual/         # 遮挡扰动实验
└── notebooks/              # Jupyter notebook 串起整个分析
```

## 为什么这个 skill 对 decoder 研究特别重要

不同 decoder 在 CC/KLD/SIM 上数字可能接近,但它们"看"的方式可以非常不同。只看指标让论文结论弱;定性分析 + 探针实验让结论强 10 倍。

**典型 insight**(你跑出来可能会发现):
- Baseline decoder 依赖前景物体,Pyramid 更均衡
- SAM-style decoder 对小物体(远处行人)更准,但对大物体偏差
- 某 decoder 的 `[ATTN]` hidden 编码了更多语义,另一个编码更多空间

## 标准工作流

给用户的研究问题对应如下步骤:

### Workflow 1: "为什么我的 decoder-pyramid 比基线好 3% CC?"

1. 读 `references/error-clustering.md` → 对两个实验做 top/bottom 20 样本对比
2. 读 `references/qualitative-vis.md` → 产出 4×N 网格,找"明显不同"的样本
3. 如果有必要,读 `references/attn-probing.md` → 对比两个模型的 `[ATTN]` hidden 语义编码

**产出**: `analysis/notebooks/pyramid_vs_baseline.ipynb` + 一份 markdown 总结

### Workflow 2: "模型在哪些场景下系统性失败?"

1. 读 `references/error-clustering.md` → 用 CLIP 特征聚类 bottom-CC 样本
2. 人工看每个 cluster 的 5 个代表,归纳共性
3. 产出 `analysis/failure_cases/cluster_summary.md`
4. 如果要深入,用 `references/counterfactual.md` 做遮挡验证

### Workflow 3: "`[ATTN]` hidden 学到了什么?"

1. 读 `references/attn-probing.md` → 抽 hidden,linear probe 多个 attribute
2. 对比不同 decoder 下的 hidden 相似度 (canonical correlation)
3. 这是一篇好论文的 "analysis" 章节

## Claude Code 的默认行为

用户问"分析一下这个实验" / "为什么样本 X 预测差"时:

1. 默认先做 **error clustering + qualitative vis**(最便宜,信息量最大)
2. 发现有价值 signal 后,再深入做 probing / gradcam
3. 不要一上来就跑 counterfactual(慢且复杂),除非前两步发现需要
4. 所有分析脚本放到 `scripts/analysis/`,中间结果放到 `analysis/`,notebook 串起来

## 产出交接

分析结果供下游使用:
- `paper-writing` skill 会拿 `analysis/failure_cases/` 做定性图
- `experiment-tracking` skill 的 `experiments.md` 会引用探针结果
- 你的硕士/博士论文可能有"Model Behavior Analysis"章节,直接用这里的产出
