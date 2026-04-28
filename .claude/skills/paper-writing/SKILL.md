---
name: paper-writing
description: **产出论文最终交付物 + 端到端评估报告**: 顶会质量 PDF 图、LaTeX 表、统计检验报告、参数/FLOPs/速度对比;以及"训练完一个 exp,自动跑完整 evaluation 并产出可对标论文 Table 的全套指标 + 定性图 + 失败案例 + LaTeX 片段"的端到端 pipeline。区别于 `interpretability-analysis`(做诊断分析)和 `experiment-tracking` 的 posttrain-report (轻量内部归档),paper-writing 产出可直接进论文的严格格式。用户说"做论文图"、"做表"、"LaTeX"、"booktabs"、"qualitative figure"、"ablation table"、"投稿清单"、"参数量 FLOPs"、"评估报告"、"eval report"、"端到端报告"、"对标论文指标"、"完整 evaluation"、"训练完出投稿物" 时使用。
---

# 论文产出总控

## 何时读哪个 reference

| 任务 | Reference |
| --- | --- |
| **端到端 eval 报告 (一键产投稿物)** | **`references/eval-report-generator.md`** |
| 全局 matplotlib 顶会配置 | `references/matplotlib-config.md` |
| 定性图 (4×N 网格对比) | `references/figure-qualitative.md` |
| 消融折线图 / 柱状图 | `references/figure-ablation.md` |
| 主对比表 (booktabs LaTeX) | `references/table-main-results.md` |
| 消融表 (Z 字形打勾) | `references/table-ablation.md` |
| 成对 t 检验 / Wilcoxon | `references/significance-test.md` |
| 参数量 / FLOPs / 速度测量 | `references/efficiency-measurement.md` |
| 可复现性 checklist | `references/reproducibility.md` |
| Teaser / 架构图提示 | `references/teaser-architecture.md` |

**用户说"训练完了帮我跑完整 eval 出投稿报告" / "做端到端 evaluation report" / "对标论文指标全套" 时,
直接走 `eval-report-generator.md`** —— 它是上面所有子项的编排版,一次性产出。

## 通用规则 (所有图表必须遵守)

1. **所有图用 vectorized PDF** (不是 PNG)
2. **所有表先产 CSV 再转 LaTeX** (不手写)
3. **字体用 Times / serif**, `pdf.fonttype=42` 避免 Type 3 (很多会议要求)
4. **配色优先 Okabe-Ito** (对色盲友好):`#0072B2 #E69F00 #009E73 #D55E00 #CC79A7 #F0E442 #56B4E9`
5. **双栏论文单图宽度**: 单栏 3.3 inch, 跨栏 6.8 inch
6. **figure caption** 独立 `\input{figures/caption_figX.tex}` 不塞在 `.py` 里

## 工作目录

```
figures/
├── fig1_teaser/            # teaser (一般 Inkscape 手画)
├── fig2_architecture/      # 架构图
├── fig3_qualitative/       # 定性对比 (见 figure-qualitative.md)
├── fig4_ablation/          # 消融图 (见 figure-ablation.md)
├── fig5_failure/           # 失败案例
├── tab1_main_results/      # 主表 (见 table-main-results.md)
├── tab2_ablation/          # 消融表
└── tab3_efficiency/        # 效率对比
paper/
├── main.tex
├── sections/
├── figs/                   # 从 figures/ 软链 PDF
└── refs.bib
scripts/paper/              # 生成图表的 Python 脚本
```

## Claude Code 的角色

**做**:
- 按各 reference 的代码模板产图/表
- 从 `runs/*/attn_eval/*/log_test.txt` 批量收集指标
- 跑 `scripts/paper/significance_test.py` 做统计检验
- 输出可直接 `\input{...}` 的 LaTeX snippets

**不做**:
- 写中文 / 英文论文段落(学术写作是人类的事)
- 决定论文 story / 章节组织
- 选择哪个 figure 进 paper / 哪个进 supplementary

除非用户明确要求,否则 Claude 只提供 data-and-figures。

## 给用户的工作流建议

### Phase 1: 数据收集
```bash
# 先跑 scripts/paper/collect_results.py 汇总所有 exp 的指标
python scripts/paper/collect_results.py
# 产出 figures/all_results.csv
```

### Phase 2: 核心图表 (按会议截止日期倒数分配)
- 定性图 (fig3): ~半天
- 主表 (tab1): ~半天
- 消融表 (tab2): ~半天
- 消融折线图 (fig4): ~2 小时
- 效率表 (tab3): ~半天 (含 FLOPs 实测)

### Phase 3: 锦上添花
- Teaser (fig1): 1-2 天 (手画)
- 架构图 (fig2): 1-2 天
- 失败案例图 (fig5): ~2 小时

### Phase 4: 检查
- 跑 `scripts/paper/reproducibility_check.py`
- 对照 `references/reproducibility.md` 逐项打勾

## 典型交互

用户说 "帮我做一个 4 种 decoder 对比的主表":
1. Claude 读 `references/table-main-results.md`
2. 从 `runs/` 下找 baseline-5090, decoder-pyramid-5090, decoder-sam-5090, decoder-mask2former-5090 的 log_test.txt
3. 按模板生成 `figures/tab1_main_results/main.tex`
4. 打印出来让用户 `\input{tab1_main_results/main.tex}`

用户说 "这个消融提升显著吗":
1. Claude 读 `references/significance-test.md`
2. 找两个 exp 的 `attn_metrics_0.csv`
3. 跑成对 t 检验
4. 输出表格 + markdown 解读
