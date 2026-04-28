# 定性对比图 (Qualitative Figure)

Decoder 研究的核心产出之一 —— 让审稿人**肉眼看到**你方法的优势。

## 推荐格式: 4×N 网格

行 = 样本(5-8 个代表性),列 = [Input | GT | Baseline | Ours-v1 | Ours-v2 | ...]

## 样本挑选原则

**公允 > 好看**. 必须包含:
- 2-3 张 **你方法明显胜出** 的
- 1-2 张 **势均力敌** 的
- 1 张 **你方法输给基线** 的 (审稿人欣赏诚实)

操作:
```python
# 从 significance_test 的数据里挑
merged = baseline_df.merge(ours_df, on="image_id", suffixes=("_base", "_ours"))
merged["cc_delta"] = merged["cc_ours"] - merged["cc_base"]

wins   = merged.nlargest(3, "cc_delta")
losses = merged.nsmallest(1, "cc_delta")
tied   = merged.iloc[(merged["cc_delta"].abs()).argsort()[:2]]
selected = pd.concat([wins, tied, losses])
```

## 完整代码

```python
# scripts/paper/fig3_qualitative.py
import sys
sys.path.append("scripts/paper")
from plot_config import *

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========== 配置 ==========
SAMPLES = [
    # (raw_path, gt_path, preds_dict)
    # preds_dict: {display_name: pred_jpg_path}
    ("dataset/BDDA/test/0001/raw_frames/000050.jpg",
     "dataset/BDDA/test/0001/gazemap_frames/000050.jpg",
     {
        "Baseline":   "dataset/BDDA/test/0001/eval_saving/baseline-5090/000050_pred.jpg",
        "Pyramid":    "dataset/BDDA/test/0001/eval_saving/decoder-pyramid-5090/000050_pred.jpg",
        "SAM-style":  "dataset/BDDA/test/0001/eval_saving/decoder-sam-5090/000050_pred.jpg",
     }),
    # ... 6-8 张
]

COL_HEADERS = ["Input", "GT", "Baseline", "Pyramid", "SAM-style"]

# ========== 绘制 ==========
def overlay(img_rgb, heatmap):
    hm_rs = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    hm_color = cv2.applyColorMap(hm_rs, cv2.COLORMAP_JET)
    hm_color_rgb = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 0.55, hm_color_rgb, 0.45, 0)


N_ROW = len(SAMPLES)
N_COL = len(COL_HEADERS)

fig, axes = plt.subplots(
    N_ROW, N_COL,
    figsize=(2.0 * N_COL, 1.5 * N_ROW)
)

for i, (raw_p, gt_p, preds) in enumerate(SAMPLES):
    raw = cv2.cvtColor(cv2.imread(raw_p), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(gt_p, 0)

    # 列 0: 原图
    axes[i, 0].imshow(raw)

    # 列 1: GT
    axes[i, 1].imshow(overlay(raw, gt))

    # 列 2+: 各方法 pred
    for j, method in enumerate(COL_HEADERS[2:]):
        pred = cv2.imread(preds[method], 0)
        axes[i, 2 + j].imshow(overlay(raw, pred))

    # 行首标注 (可选: 数据集 / 场景类型)
    dataset = raw_p.split("/")[1]
    axes[i, 0].set_ylabel(dataset, fontsize=8, rotation=0,
                          ha="right", va="center", labelpad=10)

    # 关闭 ticks
    for ax in axes[i]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

# 列标题只在第一行
for j, h in enumerate(COL_HEADERS):
    axes[0, j].set_title(h, fontsize=9, pad=4)

plt.subplots_adjust(wspace=0.03, hspace=0.06)
Path("figures/fig3_qualitative").mkdir(parents=True, exist_ok=True)
plt.savefig("figures/fig3_qualitative/decoder_comparison.pdf")
print("Saved figures/fig3_qualitative/decoder_comparison.pdf")
```

## LaTeX 使用

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/decoder_comparison.pdf}
    \caption{
        Qualitative comparison of decoders on W$^3$DA test set.
        Heatmaps are overlaid on input frames. Our Pyramid decoder
        captures small objects (distant pedestrians) more precisely
        than the baseline, while SAM-style shows more concentrated
        attention on safety-critical regions.
        See supplementary for more examples.
    }
    \label{fig:qualitative}
\end{figure*}
```

## 加标注 / 箭头

想在特定样本上画箭头指出差异:

```python
# 在 imshow 后加
axes[1, 3].annotate("",  # pyramid 列
    xy=(100, 200),       # 箭头尖
    xytext=(150, 250),   # 箭头尾
    arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))
```

## 失败案例 figure (fig5)

同样模板,但样本选 `losses` 和 `both_fail`。Caption 要诚实:
> Both our method and the baseline struggle with nighttime low-light
> scenes (row 1) and crowded intersections (row 3). We leave these
> as open problems for future work.

## 可视化调试

先输出 PNG 快速看:
```python
plt.savefig("/tmp/quick_vis.png", dpi=100)
```
满意后再切 PDF。
