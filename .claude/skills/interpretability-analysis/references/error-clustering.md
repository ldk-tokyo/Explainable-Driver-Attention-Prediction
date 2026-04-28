# 错误样本聚类

发现"模型在 X 类场景系统性失败",是论文讨论章节的重要素材。

## Step 1: 按指标找失败样本

```python
import pandas as pd
df = pd.read_csv("runs/<exp>/attn_eval/<ts>/attn_metrics_0.csv")

# 按 CC 分 5 个等级
df["cc_bucket"] = pd.qcut(df["cc"], q=5, labels=["very_bad","bad","mid","good","very_good"])
print(df.groupby("cc_bucket").size())

# 最差 200 张
worst_200 = df.nsmallest(200, "cc").copy()
worst_200.to_csv("analysis/failure_cases/worst_200_raw.csv", index=False)
```

## Step 2: 用 CLIP 特征聚类

```python
# scripts/analysis/cluster_failures.py
import torch
import clip
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from pathlib import Path

device = "cuda"
model, preprocess = clip.load("ViT-L/14", device=device)

worst = pd.read_csv("analysis/failure_cases/worst_200_raw.csv")

features = []
for _, row in worst.iterrows():
    img = preprocess(Image.open(row["image_id"])).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img).float()
    features.append(feat.cpu().numpy()[0])

features = np.stack(features)

# KMeans
N_CLUSTERS = 5
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit(features)
worst["cluster"] = km.labels_
worst.to_csv("analysis/failure_cases/worst_200_clustered.csv", index=False)

# 每个 cluster 的代表样本
import shutil
for c in range(N_CLUSTERS):
    c_dir = Path(f"analysis/failure_cases/cluster_{c}")
    c_dir.mkdir(parents=True, exist_ok=True)
    reps = worst[worst["cluster"] == c].head(10)
    for _, row in reps.iterrows():
        src = Path(row["image_id"])
        shutil.copy(src, c_dir / src.name)
    print(f"Cluster {c}: {len(reps)} samples, saved to {c_dir}")
```

## Step 3: 人工 inspect

打开每个 `cluster_N/` 目录,肉眼扫 10 张图,填这张表:

| cluster | 图像特征 | 推测失败原因 | 可改进方向 |
| --- | --- | --- | --- |
| 0 | 夜间低光 + 多光源 | 视觉 tower 在低光上训练不足 | 夜间数据增广 |
| 1 | 多行人复杂路口 | GT 热力图非常分散,模型偏保守 | 改 attn loss 到 Focal |
| 2 | 高速,单一道路 | 过于简单反而没触发 LLM 推理 | 数据不平衡 |
| 3 | 雨天模糊 | 视觉特征质量差 | 图像去模糊预处理 |
| 4 | 事故瞬间 | DADA 标注 GT 本身可能有歧义 | 增加 DADA 采样率 |

填完存到 `analysis/failure_cases/cluster_summary.md`。

## 进阶: 用 GPT-4V 自动描述 cluster

如果手动看太累,可以:
```python
import openai
# 对每个 cluster 的代表图,让 GPT-4V 描述共性
# 生成 cluster caption
```

省力,但不如人眼对驾驶场景敏感。

## 对比 decoder 的 failure pattern

更有意思的分析: **两个 decoder 的失败 cluster 是否不同**?

```python
base_worst = base_df.nsmallest(200, "cc")
ours_worst = ours_df.nsmallest(200, "cc")

# 交集 / 差集
both_fail = set(base_worst["image_id"]) & set(ours_worst["image_id"])
only_base_fail = set(base_worst["image_id"]) - set(ours_worst["image_id"])
only_ours_fail = set(ours_worst["image_id"]) - set(base_worst["image_id"])

print(f"Both fail: {len(both_fail)}")
print(f"Only baseline fails (our improvement): {len(only_base_fail)}")
print(f"Only ours fails (regression): {len(only_ours_fail)}")
```

**论文 story**:
- `only_base_fail` 展示你的 decoder "解锁"的能力
- `only_ours_fail` 展示代价(要诚实报告)
- `both_fail` 是 open problem,可做 future work

## 关联到定量

聚完类后,算每个 cluster 的平均 CC:

```python
import pandas as pd
cluster_stats = worst.groupby("cluster").agg(
    n=("cc", "count"),
    cc_mean=("cc", "mean"),
    kld_mean=("kld", "mean"),
    sim_mean=("sim", "mean"),
)
cluster_stats.to_csv("analysis/failure_cases/cluster_stats.csv")
```

这表格可以直接进论文附录。
