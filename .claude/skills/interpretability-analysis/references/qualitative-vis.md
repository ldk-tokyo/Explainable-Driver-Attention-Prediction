# 定性可视化: 预测热力图对比

最基础也最常用的分析手段。每次新 decoder 跑完都要做。

## 产出什么

### A. 单实验 4 联图 (原图 | GT | 预测 | 差异)

```python
# scripts/analysis/quad_vis.py
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


def overlay(img, heatmap):
    """把 heatmap 叠加到 RGB 图上"""
    hm_rs = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    hm_color = cv2.applyColorMap(hm_rs, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.55, hm_color, 0.45, 0)


def make_quad(raw_path, gt_path, pred_path, save_path):
    """生成 4 联对比: 原图 | GT | 预测 | |pred-gt| 差异"""
    raw = cv2.imread(str(raw_path))
    gt = cv2.imread(str(gt_path), 0)
    pred = cv2.imread(str(pred_path), 0)

    H, W = raw.shape[:2]
    gt = cv2.resize(gt, (W, H))
    pred = cv2.resize(pred, (W, H))
    diff = np.abs(pred.astype(int) - gt.astype(int)).astype(np.uint8)

    panel = np.concatenate([
        raw,
        overlay(raw, gt),
        overlay(raw, pred),
        overlay(raw, diff),
    ], axis=1)
    cv2.imwrite(str(save_path), panel)


if __name__ == "__main__":
    exp = "decoder-pyramid-5090"
    df = pd.read_csv(f"runs/{exp}/attn_eval/<ts>/attn_metrics_0.csv")

    # 做 4 组: top / bottom / random / 各 datset 中位数
    for label, selector in [
        ("top20_cc", df.nlargest(20, "cc")),
        ("bottom20_cc", df.nsmallest(20, "cc")),
        ("random20", df.sample(20, random_state=42)),
    ]:
        out_dir = Path(f"analysis/attention_vis/{exp}/{label}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for _, row in selector.iterrows():
            raw = Path(row["image_id"])
            vid_dir = raw.parent.parent   # 从 raw_frames/ 回到 video 目录
            gt = vid_dir / "gazemap_frames" / raw.name
            # 假设 --eval_colormap_save 已开,pred 在 eval_saving 下
            pred_name = raw.stem + "_pred.jpg"
            pred = vid_dir / "eval_saving" / exp / pred_name
            if not pred.exists():
                continue
            save = out_dir / f"{raw.stem}_cc{row['cc']:.3f}.jpg"
            make_quad(raw, gt, pred, save)
```

### B. 多实验 4×N 网格 (对比 decoder)

```python
# scripts/analysis/multi_decoder_grid.py
import cv2
import numpy as np
from pathlib import Path


def make_grid(samples, decoders, save_path):
    """
    samples: list of (raw_path, gt_path, {decoder_name: pred_path})
    decoders: list of decoder names (顺序=列顺序)

    输出网格: 行=样本, 列=[Input | GT | decoder1 pred | decoder2 pred | ...]
    """
    rows = []
    for raw_p, gt_p, preds in samples:
        raw = cv2.imread(str(raw_p))
        gt = cv2.imread(str(gt_p), 0)
        H, W = raw.shape[:2]
        gt = cv2.resize(gt, (W, H))

        # 每行: input | gt overlay | 各 decoder overlay
        row_imgs = [raw, overlay(raw, gt)]
        for d in decoders:
            if d in preds and Path(preds[d]).exists():
                pred = cv2.imread(str(preds[d]), 0)
                row_imgs.append(overlay(raw, pred))
            else:
                row_imgs.append(np.zeros_like(raw))  # 占位
        rows.append(np.concatenate(row_imgs, axis=1))

    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(save_path), grid)


# 使用
samples = [
    # 挑 5-8 张代表样本
    ("dataset/BDDA/test/0001/raw_frames/000050.jpg",
     "dataset/BDDA/test/0001/gazemap_frames/000050.jpg",
     {"baseline": "dataset/BDDA/test/0001/eval_saving/baseline-5090/000050_pred.jpg",
      "pyramid":  "dataset/BDDA/test/0001/eval_saving/decoder-pyramid-5090/000050_pred.jpg",
      "sam":      "dataset/BDDA/test/0001/eval_saving/decoder-sam-5090/000050_pred.jpg"}),
    # ...
]
make_grid(samples,
          ["baseline", "pyramid", "sam"],
          "analysis/attention_vis/compare_decoders.jpg")
```

## 挑选样本的策略

### 策略 1: 指标差异最大
```python
merged = base_df.merge(ours_df, on="image_id", suffixes=("_base", "_ours"))
merged["cc_delta"] = merged["cc_ours"] - merged["cc_base"]

# 我方大胜的 5 张
big_wins = merged.nlargest(5, "cc_delta")
# 我方大败的 5 张
big_losses = merged.nsmallest(5, "cc_delta")
# 公平对比的 5 张 (两者都差)
both_bad = merged[(merged["cc_base"] < 0.5) & (merged["cc_ours"] < 0.5)].head(5)
```

### 策略 2: 按子数据集
```python
for dataset in ["BDDA", "DReyeVE", "LBW", "DADA"]:
    subset = df[df["image_id"].str.contains(f"/{dataset}/")]
    # 每个子数据集 top/bottom 5
    ...
```

**论文定性图的样本挑选原则**: 必须有 "我方胜" "势均力敌" "我方败" 三类,**审稿人欣赏诚实**。

## 配色注意

- 原图: 保持 BGR 不做 color map
- Heatmap: `cv2.COLORMAP_JET` 是标准,但对色盲不友好;可选 `COLORMAP_VIRIDIS`
- 差异图: 可以用 `COLORMAP_HOT` 强调大误差

## 批量生成索引 HTML

```python
# 生成一个 index.html 能一次看所有对比图
from pathlib import Path
imgs = sorted(Path("analysis/attention_vis").rglob("*.jpg"))
with open("analysis/attention_vis/index.html", "w") as f:
    f.write("<html><body>\n")
    for img in imgs:
        rel = img.relative_to("analysis/attention_vis")
        f.write(f'<div><h3>{rel}</h3><img src="{rel}" style="max-width:100%"/></div>\n')
    f.write("</body></html>\n")
```

打开 `analysis/attention_vis/index.html` 在浏览器里快速翻阅所有样本。
