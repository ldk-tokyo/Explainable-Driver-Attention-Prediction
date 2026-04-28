# Matplotlib 顶会级 rc 配置

所有论文图脚本开头 `from scripts.paper.plot_config import *` 继承这套配置。

## `scripts/paper/plot_config.py`

```python
"""
ICCV/CVPR/NeurIPS 风格的 matplotlib 配置。
- 字体: Times Roman, 8-9pt
- 颜色: Okabe-Ito (色盲友好)
- PDF: Type 42 (会议要求)
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- 全局 rc ----------
mpl.rcParams.update({
    # 字体
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

    # PDF 导出
    "pdf.fonttype": 42,      # 必须: 会议不接受 Type 3
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    # 线宽
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,

    # 输出
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.format": "pdf",

    # 边框/网格
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,

    # LaTeX (如果装了)
    # "text.usetex": True,   # 开了渲染慢,但数学公式漂亮
    # "text.latex.preamble": r"\usepackage{times}",
})

# ---------- 配色 ----------
# Okabe-Ito: 对色盲友好, 顶会论文推荐
COLORS_OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # red/vermillion
    "#CC79A7",  # pink/magenta
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#000000",  # black
]

# 给不同 method 固定颜色,整个论文一致
METHOD_COLOR = {
    "baseline":       "#000000",
    "cross_attn":     "#0072B2",
    "pyramid":        "#E69F00",
    "mask2former":    "#009E73",
    "sam_style":      "#D55E00",
    "detr_queries":   "#CC79A7",
    "unet":           "#56B4E9",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


# ---------- 尺寸辅助 ----------
INCH_PER_CM = 0.3937
SINGLE_COL = 3.3    # inch, 双栏论文单栏宽
DOUBLE_COL = 6.8    # inch, 跨栏宽

def fig_single(height_ratio=0.66):
    """单栏图, 默认 4:3"""
    return plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * height_ratio))

def fig_double(height_ratio=0.4):
    """跨栏图 (适合宽矮图表)"""
    return plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * height_ratio))


# ---------- 常用样式 ----------
def style_ax(ax, xlabel=None, ylabel=None, title=None, grid=True):
    """统一样式"""
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title)
    if grid:   ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(direction="in", length=3)


if __name__ == "__main__":
    # 配色 sanity check
    fig, ax = fig_single()
    for i, c in enumerate(COLORS_OKABE_ITO):
        ax.bar(i, 1, color=c)
    plt.savefig("/tmp/okabe_ito_check.pdf")
    print("Saved /tmp/okabe_ito_check.pdf")
```

## 使用示例

```python
# scripts/paper/fig4_ablation_depth.py
import sys
sys.path.append("scripts/paper")
from plot_config import (fig_single, style_ax, METHOD_COLOR, MARKERS)
import pandas as pd

df = pd.read_csv("figures/all_results.csv")
depths = [2, 4, 6, 8]
cc_by_depth = [df[df.exp == f"decoder-cross-depth{d}-5090"]["CC"].values[0] for d in depths]

fig, ax = fig_single()
ax.plot(depths, cc_by_depth, "o-",
        color=METHOD_COLOR["cross_attn"], label="Cross-Attn")
style_ax(ax, xlabel="Decoder depth", ylabel="CC ↑",
         title="Effect of decoder depth on Where accuracy")
ax.legend()
plt.savefig("figures/fig4_ablation/depth_ablation.pdf")
```

## 验证 Type 42

```bash
# 检查 PDF 没嵌入 Type 3 字体
pdffonts figures/fig3_qualitative/main.pdf
# 应该只看到 TrueType 或 Type 1, 不要看到 Type 3
```

如果看到 Type 3,说明某个字体(可能 matplotlib 的数学公式渲染)没用 42。解决:
```python
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["mathtext.default"] = "regular"
```
