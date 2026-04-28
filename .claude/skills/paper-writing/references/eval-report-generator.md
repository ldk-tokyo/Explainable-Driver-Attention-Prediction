# 端到端 Evaluation 报告生成器

**目标**: 训练完一个 exp 后,**一条命令**产出可投稿质量的完整 evaluation 报告 ——
对标论文 Table 1/2/3 的全套数字、定性图、失败案例、LaTeX 片段、可复现性档案。

## 与 posttrain-pdf-report 的区别 (重要)

| 维度 | posttrain-pdf-report (experiment-tracking) | eval-report-generator (本文件) |
| --- | --- | --- |
| 目的 | 内部归档 / 快速判断 exp 好坏 | 投稿物 / 写论文素材 |
| 触发频率 | 每跑完 exp 都做 | 关键里程碑做一次 |
| 输出 | 单 PDF 自包含 | 多文件 (PDF 图 + LaTeX 表 + CSV 数据 + 报告 markdown) |
| Eval 范围 | 训练时 val 数据 | **完整 test set + 对标 baseline** |
| 严格性 | 信息密度优先 | 顶会格式、显著性检验、参数/FLOPs/速度 |

**简单原则**: 想看自己改动好不好 → posttrain-pdf-report;想发论文 → 本文件。

## §1 报告产出物清单

`reports/<exp_name>/` 下生成:

```
reports/decoder-pyramid-5090/
├── 00_summary.md               # 总览 (人类可读)
├── 01_main_results/
│   ├── attn_metrics.csv        # 6 个 attn 指标 + 全 4 子集分别
│   ├── text_metrics.csv        # BLEU/METEOR/ROUGE/CIDEr-R
│   ├── main_table.tex          # booktabs LaTeX 表 (vs baseline)
│   └── per_dataset.tex         # 按 BDDA/DReyeVE/LBW/DADA 分别
├── 02_significance/
│   ├── paired_ttest.csv        # 逐指标的 t / p / Wilcoxon-p
│   └── significance_report.md  # 人类可读总结
├── 03_qualitative/
│   ├── win_examples.pdf        # 胜出样本 5 张 4-panel
│   ├── loss_examples.pdf       # 失败样本 5 张
│   ├── tied_examples.pdf       # 势均力敌 5 张
│   └── per_dataset_samples.pdf # 每个子集 2 张代表
├── 04_efficiency/
│   ├── efficiency.csv          # 参数量 / FLOPs / 推理速度
│   ├── efficiency_table.tex    # LaTeX 片段
│   └── speed_breakdown.md      # 各模块耗时分解
├── 05_failure_analysis/
│   ├── error_clusters.png      # bottom-K 的 CLIP 聚类
│   ├── cluster_examples/       # 每个 cluster 的代表图
│   └── failure_modes.md        # 人工归纳的失败模式
└── 06_reproducibility/
    ├── env.txt                 # pip freeze
    ├── git_status.txt          # commit hash + diff
    ├── train_command.sh        # 完整训练命令
    ├── eval_command.sh         # 完整评测命令
    └── checklist.md            # 可复现性 checklist
```

## §2 主入口脚本

`scripts/paper/eval_report.py`:

```python
"""
端到端 evaluation 报告生成器。

用法:
    python scripts/paper/eval_report.py \\
        --exp decoder-pyramid-5090 \\
        --baseline baseline-5090 \\
        --ckpt ckpts/ATTN-7B-decoder-pyramid-5090 \\
        --out reports/decoder-pyramid-5090/

依赖前置:
- 已合并 LoRA (见 train-eval-workflow §5)
- 已跑过 baseline 的 eval (有 attn_metrics_0.csv)
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def step_header(name):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, help="experimental run name")
    p.add_argument("--baseline", required=True, help="baseline run name for A/B")
    p.add_argument("--ckpt", required=True, help="merged HF ckpt path of exp")
    p.add_argument("--baseline-ckpt", required=False,
                   help="baseline HF ckpt path (for re-eval if needed)")
    p.add_argument("--out", required=True, help="output report directory")
    p.add_argument("--skip-eval", action="store_true",
                   help="skip running eval (assume already done)")
    p.add_argument("--skip-text-eval", action="store_true",
                   help="skip text metrics (faster, attn-only)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Step 1: 跑完整 eval (如未跑) ----------
    step_header("Step 1: Full evaluation")
    if not args.skip_eval:
        cmd = [
            "deepspeed", "--num_gpus=1", "--master_port=24999", "train_ds.py",
            "--version", args.ckpt,
            "--vision-tower", "./weights/clip-vit-large-patch14",
            "--dataset_dir", "./dataset",
            "--log_base_dir", "./runs",
            "--exp_name", args.exp,
            "--val_dataset", "BDDA||DReyeVE||LBW||DADA",
            "--val_batch_size", "1",
            "--val_samples_num", "5000",
            "--eval_only",
            "--eval_colormap_save",   # 出可视化
        ]
        if not args.skip_text_eval:
            cmd += ["--eval_text", "--eval_text_save"]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print("(skipped, assuming eval already done)")

    # ---------- Step 2: 主结果表 ----------
    step_header("Step 2: Main results table")
    from scripts.paper import collect_main_results
    collect_main_results.run(
        exps=[(args.baseline, "Baseline"), (args.exp, "Ours")],
        out_dir=out_dir / "01_main_results",
        per_dataset=True,
    )

    # ---------- Step 3: 显著性检验 ----------
    step_header("Step 3: Significance test")
    from scripts.paper import significance_test
    significance_test.run(
        baseline_run=f"runs/{args.baseline}",
        ours_run=f"runs/{args.exp}",
        out_dir=out_dir / "02_significance",
    )

    # ---------- Step 4: 定性图 ----------
    step_header("Step 4: Qualitative figures")
    from scripts.paper import qualitative_figures
    qualitative_figures.run(
        baseline_run=f"runs/{args.baseline}",
        ours_run=f"runs/{args.exp}",
        out_dir=out_dir / "03_qualitative",
        n_per_category=5,
    )

    # ---------- Step 5: 效率测量 ----------
    step_header("Step 5: Efficiency measurement")
    from scripts.paper import efficiency_table
    efficiency_table.run(
        ckpts={"Baseline": args.baseline_ckpt or f"./ckpts/ATTN-7B-{args.baseline}",
               "Ours": args.ckpt},
        out_dir=out_dir / "04_efficiency",
    )

    # ---------- Step 6: 失败分析 ----------
    step_header("Step 6: Failure analysis (CLIP clustering)")
    from scripts.paper import failure_analysis
    failure_analysis.run(
        run_dir=f"runs/{args.exp}",
        out_dir=out_dir / "05_failure_analysis",
        n_clusters=5, n_samples=200,
    )

    # ---------- Step 7: 可复现性档案 ----------
    step_header("Step 7: Reproducibility archive")
    from scripts.paper import reproducibility
    reproducibility.run(
        exp_name=args.exp,
        out_dir=out_dir / "06_reproducibility",
    )

    # ---------- Step 8: Summary markdown ----------
    step_header("Step 8: Summary")
    write_summary(out_dir, args.exp, args.baseline)
    print(f"\n✓ Report complete: {out_dir}")
    print(f"  Open {out_dir}/00_summary.md to start.")


def write_summary(out_dir, exp, baseline):
    """汇总 markdown,串起所有产出"""
    summary = f"""# Evaluation Report: `{exp}`

Generated: {datetime.now().isoformat(timespec='seconds')}
Compared against: `{baseline}`

## Sections

1. [Main Results](01_main_results/) — 6 attn + 4 text metrics, per-dataset breakdown, LaTeX table
2. [Significance](02_significance/significance_report.md) — paired t-test + Wilcoxon
3. [Qualitative](03_qualitative/) — wins / losses / tied examples
4. [Efficiency](04_efficiency/) — params / FLOPs / inference speed
5. [Failure Analysis](05_failure_analysis/failure_modes.md) — CLIP-clustered bottom samples
6. [Reproducibility](06_reproducibility/checklist.md) — env, git, commands

## Quick verdict

(Fill this in by hand after reading sections 1-2:)
- [ ] Improvement on CC is statistically significant (p < 0.001)?
- [ ] Improvement on KLD is statistically significant?
- [ ] Wins on at least 3/6 attn metrics?
- [ ] Cost ≤ 2× params, ≤ 1.5× FLOPs?
- [ ] Failure modes acceptable for paper claim?

## TODO before submission

- [ ] Hand-pick 5-8 best qualitative samples from `03_qualitative/`
- [ ] Write 2-3 sentence narrative around clusters in `05_failure_analysis/`
- [ ] Run additional seeds (currently only 1 seed)
- [ ] Verify reproducibility checklist all green
"""
    (out_dir / "00_summary.md").write_text(summary)


if __name__ == "__main__":
    main()
```

## §3 子模块 stub (新建,占位说明)

下面 6 个子模块需要单独实现。它们大部分**逻辑已经在其他 reference 里写好**,这里只是封装成可调用的 `run()` 函数:

### `scripts/paper/collect_main_results.py`

实现见 `paper-writing/references/table-main-results.md` 的代码,封装为:
```python
def run(exps: list, out_dir: Path, per_dataset: bool = False):
    """
    exps: [(run_name, display_name), ...]
    产出: attn_metrics.csv, text_metrics.csv, main_table.tex, per_dataset.tex
    """
    ...
```

### `scripts/paper/significance_test.py`

实现见 `modify-llada/references/decoders/ab-test-protocol.md` 的成对 t 检验脚本,封装为:
```python
def run(baseline_run: str, ours_run: str, out_dir: Path):
    """
    产出: paired_ttest.csv, significance_report.md
    """
    ...
```

### `scripts/paper/qualitative_figures.py`

实现见 `paper-writing/references/figure-qualitative.md` 的 4×N 网格,加自动样本挑选:
```python
def run(baseline_run, ours_run, out_dir, n_per_category=5):
    """
    自动按 cc_delta 挑 wins / losses / tied 各 N 张
    产出: win_examples.pdf, loss_examples.pdf, tied_examples.pdf, per_dataset_samples.pdf
    """
    ...
```

### `scripts/paper/efficiency_table.py`

实现见 `paper-writing/references/ablation-efficiency-repro.md` §4,封装为:
```python
def run(ckpts: dict, out_dir: Path):
    """
    ckpts: {display_name: ckpt_path}
    产出: efficiency.csv, efficiency_table.tex, speed_breakdown.md
    """
    ...
```

### `scripts/paper/failure_analysis.py`

实现见 `interpretability-analysis/references/error-clustering.md`,封装为:
```python
def run(run_dir: str, out_dir: Path, n_clusters=5, n_samples=200):
    """
    产出: error_clusters.png, cluster_examples/cluster_*/, failure_modes.md
    """
    ...
```

### `scripts/paper/reproducibility.py`

新写,简单:
```python
import subprocess
from pathlib import Path

def run(exp_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # pip freeze
    (out_dir / "env.txt").write_text(
        subprocess.check_output(["pip", "freeze"], text=True))

    # git status + commit
    git_info = ""
    try:
        git_info += subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True)
        git_info += "\n--- diff ---\n"
        git_info += subprocess.check_output(
            ["git", "diff", "HEAD"], text=True)
    except subprocess.CalledProcessError:
        git_info = "(not a git repo)"
    (out_dir / "git_status.txt").write_text(git_info)

    # train + eval commands (从 runs/<exp>/ 找,或从 args 接收)
    # ... 用户要在第一次跑时手动保存命令到 runs/<exp>/cmd.sh,这里复制过来

    # checklist
    checklist = """# 可复现性 Checklist

- [ ] 代码 commit 到 (anonymized) repo?
- [ ] requirements.txt / env.txt 完整?
- [ ] 数据集处理脚本公开?
- [ ] 随机种子固定?
- [ ] 至少 2 个种子跑过?
- [ ] 主表所有数字能从 ckpt 重现?
- [ ] 训练命令完整保存?
- [ ] 评测命令完整保存?
- [ ] Hardware 信息已写 (RTX 5090, 32GB)?
- [ ] ckpt 上传到 HF Hub / Zenodo?
"""
    (out_dir / "checklist.md").write_text(checklist)
```

## §4 用法

### 完整流程 (训练完后,首次跑)

```bash
# 假设 baseline-5090 已训完并 eval 过
# 现在 decoder-pyramid-5090 也训完,合并完 LoRA

python scripts/paper/eval_report.py \
    --exp decoder-pyramid-5090 \
    --baseline baseline-5090 \
    --ckpt ./ckpts/ATTN-7B-decoder-pyramid-5090 \
    --baseline-ckpt ./ckpts/ATTN-7B-baseline-5090 \
    --out reports/decoder-pyramid-5090/
```

时长估算(全跑):
- Step 1 完整 eval (含文本): **~6-10 小时** (5000 样本 × 4 子集,文本 metric 慢)
- Step 1 仅 attn: ~1-2 小时
- Step 2-7: 共 ~10 分钟

### 加速选项

```bash
# 跳过文本指标 (CC/KLD 等还在)
... --skip-text-eval

# 已经手动跑过 eval,只想跑后面的分析
... --skip-eval
```

### 多 exp 批量

```bash
for exp in decoder-pyramid-5090 decoder-sam-5090 decoder-mask2former-5090; do
    python scripts/paper/eval_report.py \
        --exp $exp --baseline baseline-5090 \
        --ckpt ./ckpts/ATTN-7B-$exp \
        --out reports/$exp/ \
        --skip-text-eval     # 第一轮快速看,通过门槛的再跑全套
done
```

## §5 报告写作工作流

报告生成后:

### 第 1 天: 自动产出 + 快速 review

```bash
python scripts/paper/eval_report.py ...
# 等 6-10 小时
# 看 reports/<exp>/00_summary.md 顶部的 quick verdict
```

### 第 2 天: 人工细化

1. 读 `02_significance/significance_report.md`,挑出 p < 0.001 的指标作为论文 claim
2. 读 `03_qualitative/`,从自动挑的样本里**手工再选**最有故事的 5-8 张
3. 读 `05_failure_analysis/failure_modes.md`,加上**人类的归纳**(自动聚类只到这一步)
4. 把 `01_main_results/main_table.tex` `\input{...}` 进 `paper/main.tex`

### 第 3 天: 投稿前最后核对

- 跑 `06_reproducibility/checklist.md` 逐项打勾
- 用第二个种子重跑,验证数字 ±0.005 内可重现
- 把所有 LaTeX 文件 review 一遍格式

## §6 Pipeline 图

```
完成训练 → LoRA 合并 → eval_report.py
                            │
            ┌───────────────┴────────────────┐
            ↓                                ↓
   Step 1: Full Eval                   (后台并行)
   - 4 子集 × 5000 样本                Step 2-7
   - attn_metrics_*.csv                - 主表
   - eval_saving/*.jpg                 - 显著性
   - eval_text/*.txt                   - 定性图
                                       - 效率
                                       - 失败分析
                                       - 可复现性
                                              ↓
                                       Step 8: Summary
                                              ↓
                                  reports/<exp>/00_summary.md
```

## §7 进阶: 加自定义子模块

如果你想加新章节(比如 `07_temporal_analysis/` 看模型对时序的处理),
在 `eval_report.py::main` 里追加 step,新建 `scripts/paper/temporal_analysis.py`,
对照已有子模块的接口写 `run()` 函数。

Claude Code 可以一次性帮你扩展,告诉它:
> "在 eval_report.py 里加一个 step 7.5: temporal_consistency,
> 对相邻帧的 pred_sal 算 cosine 相似度,产出 temporal_consistency.csv 和 plot"
