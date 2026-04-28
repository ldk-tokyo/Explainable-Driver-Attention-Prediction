"""
为单个 exp 生成可视化 PDF 报告。

用法:
    python scripts/posttrain_report.py runs/baseline-5090
    python scripts/posttrain_report.py runs/baseline-5090 --out reports/baseline-5090.pdf

包含 4 页:
1. 训练曲线 (loss + 各分量 + lr)
2. 验证曲线 (6 个 attn 指标 + 文本指标 vs step)
3. 最终指标卡片 + 文件清单
4. Top-K / Bottom-K 样本 quad 图 (input | gt | pred | diff)
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


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def load_scalars(run_dir):
    event_files = list(run_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return {}
    ea = EventAccumulator(str(max(event_files, key=lambda p: p.stat().st_mtime).parent),
                          size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        evs = ea.Scalars(tag)
        steps = np.array([e.step for e in evs])
        vals = np.array([e.value for e in evs])
        out[tag] = (steps, vals)
    return out


def page_training_curves(scalars, exp_name):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"[{exp_name}] Training Curves", fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    if "train/loss" in scalars:
        s, v = scalars["train/loss"]
        ax.plot(s, v, "b-", linewidth=1, alpha=0.8)
        ax.set_title("Total Loss (train)")
        ax.set_xlabel("Step")

    ax = axes[0, 1]
    for tag, color, label in [("train/ce_loss", "C0", "ce_loss"),
                              ("train/attn_loss", "C1", "attn_loss"),
                              ("train/ce_what_losses", "C2", "ce_what"),
                              ("train/ce_why_losses", "C3", "ce_why")]:
        if tag in scalars:
            s, v = scalars[tag]
            ax.plot(s, v, color=color, label=label, linewidth=1, alpha=0.8)
    ax.set_title("Loss Components (train)")
    ax.set_xlabel("Step")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    if "train/lr" in scalars:
        s, v = scalars["train/lr"]
        ax.plot(s, v, "g-", linewidth=1)
        ax.set_title("Learning Rate")
        ax.set_xlabel("Step")
        ax.set_yscale("log")

    ax = axes[1, 1]
    has_sys = False
    for tag, label in [("metrics/total_secs_per_batch", "total s/batch"),
                        ("metrics/data_secs_per_batch", "data s/batch")]:
        if tag in scalars:
            s, v = scalars[tag]
            ax.plot(s, v, label=label, linewidth=1, alpha=0.8)
            has_sys = True
    if has_sys:
        ax.set_title("Throughput (sec/batch)")
        ax.set_xlabel("Step")
        ax.legend(fontsize=7)
    else:
        ax.set_title("(Throughput scalars not logged)")
        ax.axis("off")

    plt.tight_layout()
    return fig


def page_val_curves(scalars, exp_name):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"[{exp_name}] Validation Metrics", fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    for tag in ["val/cc", "val/sim", "val/auc_b", "val/auc_j"]:
        if tag in scalars:
            s, v = scalars[tag]
            ax.plot(s, v, label=tag.split("/")[-1], linewidth=1.5, marker="o", markersize=3)
    ax.set_title("Attn Metrics (higher better)")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    if "val/nss" in scalars:
        s, v = scalars["val/nss"]
        ax.plot(s, v, "C2-o", markersize=3, linewidth=1.5)
        ax.set_title("NSS (higher better)")
        ax.set_xlabel("Step")

    ax = axes[1, 0]
    if "val/kld" in scalars:
        s, v = scalars["val/kld"]
        ax.plot(s, v, "r-o", markersize=3, linewidth=1.5)
        ax.set_title("KLD (lower better)")
        ax.set_xlabel("Step")

    ax = axes[1, 1]
    has_text = False
    for tag in ["val/bleu_4", "val/meteor", "val/rouge", "val/ciderR"]:
        if tag in scalars:
            s, v = scalars[tag]
            ax.plot(s, v, label=tag.split("/")[-1], linewidth=1.5, marker="s", markersize=3)
            has_text = True
    if has_text:
        ax.set_title("Text Metrics (higher better)")
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
    else:
        ax.set_title("(Text metrics not evaluated)")
        ax.text(0.5, 0.5, "Run with --eval_text to compute\nBLEU / METEOR / ROUGE / CIDEr-R",
                ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    return fig


def page_summary_card(scalars, run_dir, exp_name):
    fig = plt.figure(figsize=(11, 8))
    fig.suptitle(f"[{exp_name}] Final Summary", fontsize=12, fontweight="bold")
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = ["", f"Experiment: {exp_name}", f"Run dir:    {run_dir}", "", "-" * 60, ""]

    lines.append("FINAL ATTENTION METRICS")
    for tag in ["val/cc", "val/kld", "val/sim", "val/nss", "val/auc_b", "val/auc_j"]:
        if tag in scalars:
            s, v = scalars[tag]
            best_op = np.argmin if "kld" in tag else np.argmax
            best_idx = int(best_op(v))
            short = tag.split("/")[-1].upper()
            lines.append(f"  {short:8s} = {v[-1]:.4f}     (best: {v[best_idx]:.4f} at step {int(s[best_idx])})")

    lines.append("")
    lines.append("FINAL TEXT METRICS")
    for tag in ["val/bleu_4", "val/meteor", "val/rouge", "val/ciderR"]:
        if tag in scalars:
            _, v = scalars[tag]
            short = tag.split("/")[-1]
            lines.append(f"  {short:10s} = {v[-1]:.4f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("FILES IN run_dir")
    if Path(run_dir).exists():
        for f in sorted(Path(run_dir).rglob("meta_log_*.pth"))[:10]:
            lines.append(f"  ckpt: {f.relative_to(run_dir)}")
        for f in Path(run_dir).rglob("log_test.txt"):
            lines.append(f"  log:  {f.relative_to(run_dir)}")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, fontfamily="monospace", fontsize=9,
            verticalalignment="top", transform=ax.transAxes)
    return fig


def page_topk_samples(run_dir, exp_name, k=5):
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
        ax.text(0.5, 0.5, f"CSV missing 'cc' column. Available: {list(df.columns)}",
                ha="center", va="center", fontsize=9)
        return fig

    top = df.nlargest(k, "cc")
    bot = df.nsmallest(k, "cc")
    rows = pd.concat([top, bot])

    fig, axes = plt.subplots(2 * k, 4, figsize=(11, 1.4 * 2 * k))
    fig.suptitle(f"[{exp_name}] Top-{k} (good) vs Bottom-{k} (bad) by CC", fontsize=12, fontweight="bold")

    for idx, (_, row) in enumerate(rows.iterrows()):
        raw_p = Path(str(row.get("image_id", "")))
        if not raw_p.exists():
            for j in range(4):
                axes[idx, j].axis("off")
            continue

        vid_dir = raw_p.parent.parent
        gt_p = vid_dir / "gazemap_frames" / raw_p.name
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
        except Exception:
            for j in range(4):
                axes[idx, j].axis("off")

        for ax in axes[idx]:
            ax.set_xticks([]); ax.set_yticks([])

    for j, h in enumerate(["Input", "GT", "Pred", "Diff"]):
        axes[0, j].set_title(h, fontsize=9)

    if k > 0:
        axes[0, 0].annotate(f"TOP-{k}", xy=(-0.5, 0.5), xycoords="axes fraction",
                            fontsize=10, fontweight="bold", color="green",
                            ha="right", va="center", rotation=90)
        axes[k, 0].annotate(f"BOT-{k}", xy=(-0.5, 0.5), xycoords="axes fraction",
                            fontsize=10, fontweight="bold", color="red",
                            ha="right", va="center", rotation=90)

    plt.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str, help="e.g., runs/baseline-5090")
    p.add_argument("--out", type=str, default=None,
                   help="output PDF path (default: <run_dir>/report.pdf)")
    p.add_argument("--k", type=int, default=5, help="top/bottom K samples")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist")
        sys.exit(1)

    exp_name = run_dir.name
    out_path = Path(args.out) if args.out else run_dir / "report.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading TensorBoard scalars from {run_dir} ...")
    scalars = load_scalars(run_dir)
    print(f"  Found {len(scalars)} scalar tags: {sorted(scalars.keys())[:10]}{'...' if len(scalars) > 10 else ''}")

    print(f"Generating PDF report -> {out_path}")
    with PdfPages(out_path) as pdf:
        pdf.savefig(page_training_curves(scalars, exp_name)); plt.close()
        pdf.savefig(page_val_curves(scalars, exp_name)); plt.close()
        pdf.savefig(page_summary_card(scalars, run_dir, exp_name)); plt.close()
        pdf.savefig(page_topk_samples(run_dir, exp_name, k=args.k)); plt.close()

    print(f"OK Report saved: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
