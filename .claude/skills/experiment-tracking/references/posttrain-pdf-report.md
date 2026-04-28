# 训练后批量出 PDF 报告

训练完成后(或训练中途想存档当前状态时),一键生成单 exp 的可视化 PDF。
**用途**: commit 到 git / 发给合作者看 / 写论文时翻阅历史实验。

## §1 与 paper-writing 的区别

| paper-writing | posttrain-pdf-report (这个) |
| --- | --- |
| 投稿用,严格 LaTeX 字号配色 | 内部归档用,信息密度优先 |
| 跨多个 exp 的对比表 | 单 exp 自包含报告 |
| 手工挑代表样本 | 自动 top/bottom-K |
| 一篇论文做一次 | 每个 exp 都做一次 |

## §2 完整脚本

`scripts/posttrain_report.py`:

```python
"""
为单个 exp 生成可视化 PDF 报告。

用法:
    python scripts/posttrain_report.py runs/baseline-5090

产出:
    runs/baseline-5090/report.pdf

包含 4 页:
1. 训练曲线 (loss + 各分量 + lr)
2. 验证曲线 (6 个 attn 指标 + 文本指标 vs epoch)
3. 最佳 epoch 的指标卡片 + 配置 + 显存峰值
4. Top-5 / Bottom-5 样本的 quad 图 (input | gt | pred | diff)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---------- matplotlib 配置 (轻量,不追顶会风) ----------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ---------- TensorBoard 数据加载 ----------
def load_scalars(run_dir):
    """从 run_dir 下所有 event 文件聚合 scalar 数据"""
    event_files = list(run_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return {}
    # 最新的 event 通常包含全部历史
    ea = EventAccumulator(str(max(event_files, key=lambda p: p.stat().st_mtime).parent))
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        evs = ea.Scalars(tag)
        steps = np.array([e.step for e in evs])
        vals = np.array([e.value for e in evs])
        out[tag] = (steps, vals)
    return out


# ---------- Page 1: 训练曲线 ----------
def page_training_curves(scalars, exp_name):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"[{exp_name}] Training Curves", fontsize=12, fontweight="bold")

    # 总 loss
    ax = axes[0, 0]
    if "loss" in scalars:
        s, v = scalars["loss"]
        ax.plot(s, v, "b-", linewidth=1, alpha=0.8)
        ax.set_title("Total Loss")
        ax.set_xlabel("Step")

    # 各分量 loss
    ax = axes[0, 1]
    for tag, color in [("ce_loss", "C0"), ("attn_loss", "C1"),
                        ("ce_what_loss", "C2"), ("ce_why_loss", "C3")]:
        if tag in scalars:
            s, v = scalars[tag]
            ax.plot(s, v, color=color, label=tag, linewidth=1, alpha=0.8)
    ax.set_title("Loss Components")
    ax.set_xlabel("Step")
    ax.legend(fontsize=7)

    # learning rate
    ax = axes[1, 0]
    if "lr" in scalars:
        s, v = scalars["lr"]
        ax.plot(s, v, "g-", linewidth=1)
        ax.set_title("Learning Rate")
        ax.set_xlabel("Step")
        ax.set_yscale("log")

    # GPU 状态 (如果记录了)
    ax = axes[1, 1]
    has_sys = False
    for tag, label in [("system/gpu_mem_gb", "Mem (GB)"),
                        ("system/gpu_util", "Util (%)")]:
        if tag in scalars:
            s, v = scalars[tag]
            ax.plot(s, v, label=label, linewidth=1, alpha=0.8)
            has_sys = True
    if has_sys:
        ax.set_title("System")
        ax.legend(fontsize=7)
    else:
        ax.set_title("(System scalars not logged)")
        ax.text(0.5, 0.5, "Add tb_writer.add_scalar('system/...') in train_ds.py\nto track GPU memory and utilization",
                ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    return fig


# ---------- Page 2: 验证曲线 ----------
def page_val_curves(scalars, exp_name):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"[{exp_name}] Validation Metrics", fontsize=12, fontweight="bold")

    # higher-is-better attn
    ax = axes[0, 0]
    for tag in ["val/cc", "val/sim", "val/auc_b", "val/auc_j"]:
        if tag in scalars:
            s, v = scalars[tag]
            ax.plot(s, v, label=tag.split("/")[-1], linewidth=1.5, marker="o", markersize=3)
    ax.set_title("Attn Metrics (higher better)")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)

    # NSS (different scale)
    ax = axes[0, 1]
    if "val/nss" in scalars:
        s, v = scalars["val/nss"]
        ax.plot(s, v, "C2-o", markersize=3, linewidth=1.5)
        ax.set_title("NSS ↑")
        ax.set_xlabel("Step")

    # KLD (lower is better)
    ax = axes[1, 0]
    if "val/kld" in scalars:
        s, v = scalars["val/kld"]
        ax.plot(s, v, "r-o", markersize=3, linewidth=1.5)
        ax.set_title("KLD ↓ (lower better)")
        ax.set_xlabel("Step")

    # 文本指标
    ax = axes[1, 1]
    has_text = False
    for tag in ["val/bleu_4", "val/meteor", "val/rouge", "val/cider_r"]:
        if tag in scalars:
            s, v = scalars[tag]
            ax.plot(s, v, label=tag.split("/")[-1], linewidth=1.5, marker="s", markersize=3)
            has_text = True
    if has_text:
        ax.set_title("Text Metrics ↑")
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
    else:
        ax.set_title("(Text metrics not evaluated)")
        ax.text(0.5, 0.5, "Run with --eval_text to compute\nBLEU / METEOR / ROUGE / CIDEr-R",
                ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    return fig


# ---------- Page 3: 最终指标卡片 ----------
def page_summary_card(scalars, run_dir, exp_name):
    fig = plt.figure(figsize=(11, 8))
    fig.suptitle(f"[{exp_name}] Final Summary", fontsize=12, fontweight="bold")
    ax = fig.add_subplot(111)
    ax.axis("off")

    # 提取最终指标
    lines = ["", f"Experiment: {exp_name}", f"Run dir:    {run_dir}", "", "─" * 60, ""]

    lines.append("FINAL ATTENTION METRICS")
    for tag in ["val/cc", "val/kld", "val/sim", "val/nss", "val/auc_b", "val/auc_j"]:
        if tag in scalars:
            _, v = scalars[tag]
            short = tag.split("/")[-1].upper()
            lines.append(f"  {short:8s} = {v[-1]:.4f}     (best: {v.max():.4f} at step {int(scalars[tag][0][v.argmax()])})")

    lines.append("")
    lines.append("FINAL TEXT METRICS")
    for tag in ["val/bleu_4", "val/meteor", "val/rouge", "val/cider_r"]:
        if tag in scalars:
            _, v = scalars[tag]
            short = tag.split("/")[-1]
            lines.append(f"  {short:10s} = {v[-1]:.4f}")

    lines.append("")
    lines.append("─" * 60)
    lines.append("FILES IN run_dir")
    if Path(run_dir).exists():
        for f in sorted(Path(run_dir).rglob("meta_log_*.pth"))[:5]:
            lines.append(f"  ckpt: {f.relative_to(run_dir)}")
        log = Path(run_dir).rglob("log_test.txt")
        for f in log:
            lines.append(f"  log:  {f.relative_to(run_dir)}")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, fontfamily="monospace", fontsize=9,
            verticalalignment="top", transform=ax.transAxes)
    return fig


# ---------- Page 4: Top/Bottom-K 样本 ----------
def page_topk_samples(run_dir, exp_name, k=5):
    """从 attn_metrics_0.csv 找 top-K 和 bottom-K 样本,做 quad 图"""
    csvs = list(run_dir.rglob("attn_metrics_*.csv"))
    if not csvs:
        fig = plt.figure(figsize=(11, 8))
        fig.suptitle(f"[{exp_name}] Sample Visualization", fontsize=12, fontweight="bold")
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No attn_metrics_*.csv found.\nRun with --eval_only to generate per-sample metrics.",
                ha="center", va="center", fontsize=11, color="gray")
        return fig

    df = pd.read_csv(max(csvs, key=lambda p: p.stat().st_mtime))
    if "cc" not in df.columns:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "CSV missing 'cc' column", ha="center", va="center")
        return fig

    top = df.nlargest(k, "cc")
    bot = df.nsmallest(k, "cc")
    rows = pd.concat([top, bot])

    fig, axes = plt.subplots(2 * k, 4, figsize=(11, 1.4 * 2 * k))
    fig.suptitle(f"[{exp_name}] Top-{k} (good) vs Bottom-{k} (bad) by CC", fontsize=12, fontweight="bold")

    for idx, (_, row) in enumerate(rows.iterrows()):
        raw_p = Path(row.get("image_id", ""))
        if not raw_p.exists():
            for j in range(4):
                axes[idx, j].axis("off")
            continue

        # 推断 gt / pred 路径(假设 --eval_colormap_save 开了)
        vid_dir = raw_p.parent.parent
        gt_p = vid_dir / "gazemap_frames" / raw_p.name
        # pred 在 vid_dir/eval_saving/<run-name>/<frame>_pred.jpg
        pred_p = next((vid_dir / "eval_saving").rglob(f"{raw_p.stem}_pred.jpg"), None)

        try:
            raw = cv2.cvtColor(cv2.imread(str(raw_p)), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(str(gt_p), 0) if gt_p.exists() else np.zeros(raw.shape[:2], dtype=np.uint8)
            pred = cv2.imread(str(pred_p), 0) if pred_p and pred_p.exists() else np.zeros_like(gt)
            H, W = raw.shape[:2]
            gt = cv2.resize(gt, (W, H))
            pred = cv2.resize(pred, (W, H))
            diff = np.abs(pred.astype(int) - gt.astype(int)).astype(np.uint8)

            def overlay(bg, hm):
                hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
                return cv2.addWeighted(bg, 0.55, hm_color, 0.45, 0)

            axes[idx, 0].imshow(raw)
            axes[idx, 1].imshow(overlay(raw, gt))
            axes[idx, 2].imshow(overlay(raw, pred))
            axes[idx, 3].imshow(overlay(raw, diff))

            axes[idx, 0].set_ylabel(f"cc={row['cc']:.3f}", fontsize=7, rotation=0,
                                     ha="right", va="center", labelpad=18)
        except Exception as e:
            for j in range(4):
                axes[idx, j].axis("off")

        for ax in axes[idx]:
            ax.set_xticks([]); ax.set_yticks([])

    # 列标题
    for j, h in enumerate(["Input", "GT", "Pred", "Diff"]):
        axes[0, j].set_title(h, fontsize=9)

    # 分组标记 (top vs bottom)
    if k > 0:
        axes[0, 0].annotate(f"TOP-{k}", xy=(-0.5, 0.5), xycoords="axes fraction",
                            fontsize=10, fontweight="bold", color="green",
                            ha="right", va="center", rotation=90)
        axes[k, 0].annotate(f"BOT-{k}", xy=(-0.5, 0.5), xycoords="axes fraction",
                            fontsize=10, fontweight="bold", color="red",
                            ha="right", va="center", rotation=90)

    plt.tight_layout()
    return fig


# ---------- 主入口 ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str, help="e.g., runs/baseline-5090")
    p.add_argument("--out", type=str, default=None, help="output PDF path (default: <run_dir>/report.pdf)")
    p.add_argument("--k", type=int, default=5, help="top/bottom K samples")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist")
        sys.exit(1)

    exp_name = run_dir.name
    out_path = Path(args.out) if args.out else run_dir / "report.pdf"

    print(f"Loading TensorBoard scalars from {run_dir} ...")
    scalars = load_scalars(run_dir)
    print(f"  Found {len(scalars)} scalar tags")

    print(f"Generating PDF report → {out_path}")
    with PdfPages(out_path) as pdf:
        pdf.savefig(page_training_curves(scalars, exp_name)); plt.close()
        pdf.savefig(page_val_curves(scalars, exp_name)); plt.close()
        pdf.savefig(page_summary_card(scalars, run_dir, exp_name)); plt.close()
        pdf.savefig(page_topk_samples(run_dir, exp_name, k=args.k)); plt.close()

    print(f"✓ Report saved: {out_path}")


if __name__ == "__main__":
    main()
```

## §3 用法

```bash
# 训练完后(或中途任何时候)
python scripts/posttrain_report.py runs/baseline-5090
# 产出 runs/baseline-5090/report.pdf

# 自定义输出路径和样本数
python scripts/posttrain_report.py runs/baseline-5090 \
    --out reports/baseline-2026-04-15.pdf --k 10
```

## §4 集成到训练脚本 (可选)

让训练**完成后自动**生成报告:

```bash
# 在你的 launch 脚本里
deepspeed --num_gpus=1 ... --exp_name=baseline-5090 && \
    python scripts/posttrain_report.py runs/baseline-5090 && \
    git add runs/baseline-5090/report.pdf && \
    git commit -m "results: baseline-5090 report"
```

或写到 `Makefile`:
```make
report-%:
	python scripts/posttrain_report.py runs/$* --out reports/$*.pdf
```

用法: `make report-baseline-5090`

## §5 PDF 版本管理

`reports/` 目录建议加到 git:
```bash
mkdir -p reports
echo "*.pdf" >> .gitattributes  # 如果用 git-lfs
git add reports/
```

每个 exp 的 PDF 是 ~2-5 MB,跑 30 个实验大概 100MB,git 直接管理也 OK。

## §6 扩展思路

如果想要更丰富的 PDF,可以加:
- **每个子数据集的指标分别画**(BDDA / DReyeVE / LBW / DADA)
- **参数量 + FLOPs 比基线的差异**
- **失败样本的 cluster 分析**(读 `interpretability-analysis/references/error-clustering.md`)
- **训练命令 + git commit hash + Python 环境**(完全可复现性记录)

需要扩展时告诉 Claude: "在 posttrain_report.py 里加 X 章节",对照本文件结构改。
