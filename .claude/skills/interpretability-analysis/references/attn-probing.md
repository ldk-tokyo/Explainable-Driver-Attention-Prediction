# `[ATTN]` Hidden State 探针分析

**研究问题**: `[ATTN]` token 的 hidden state 是否编码了语义信息(不只是空间位置)?

## Step 1: 抽取 hidden state

```python
# scripts/analysis/extract_attn_hidden.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

from model.Attn_model import AttnForCausalLM

EXP = "decoder-pyramid-5090"
CKPT = f"./ckpts/ATTN-7B-{EXP}"

model = AttnForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(CKPT)
attn_idx = tokenizer("[ATTN]", add_special_tokens=False).input_ids[0]

# 从 val_loader 取 ~1000 样本
hiddens, metas = [], []
for sample in val_loader_limited(1000):
    sample = {k: v.cuda() if isinstance(v, torch.Tensor) else v
              for k, v in sample.items()}
    with torch.no_grad():
        outputs = model.forward(**sample, output_hidden_states=True)
    last_hidden = outputs["hidden_states"][-1]   # [B, L, D]
    mask = (sample["input_ids"] == attn_idx)
    for b in range(last_hidden.size(0)):
        h = last_hidden[b][mask[b]][0]           # 第一个 [ATTN]
        hiddens.append(h.float().cpu().numpy())
        metas.append({
            "image_path": sample["image_paths"][b],
            "gt_text": sample["answers_list"][b],
        })

np.save(f"analysis/probing/{EXP}_hidden.npy", np.stack(hiddens))
pd.DataFrame(metas).to_csv(f"analysis/probing/{EXP}_meta.csv", index=False)
```

## Step 2: 准备 probe 标签

你需要一些 attribute 来 probe。方案:

### Option A: 人工标 200 样本
```python
# analysis/probing/hand_label.csv
# columns: image_path, scene_type (highway/urban/intersection/accident), weather (clear/rain/night), ...
```

### Option B: 规则生成
从 `image_path` 抽信息:
```python
df["dataset"] = df["image_path"].str.extract(r"/(BDDA|DReyeVE|LBW|DADA)/")
# DADA = accident scene
# BDDA = safety-critical
# DReyeVE = daily driving
# LBW = intersections
```

### Option C: GT 文本的关键词
```python
df["has_pedestrian"] = df["gt_text"].str.contains("pedestrian|pedestr", case=False).astype(int)
df["has_vehicle"]    = df["gt_text"].str.contains("vehicle|car|truck", case=False).astype(int)
df["is_braking"]     = df["gt_text"].str.contains("brak|slow", case=False).astype(int)
```

## Step 3: Linear Probe

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = np.load(f"analysis/probing/{EXP}_hidden.npy")
meta = pd.read_csv(f"analysis/probing/{EXP}_meta.csv")

for attr in ["dataset", "has_pedestrian", "has_vehicle", "is_braking"]:
    y = meta[attr]
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="f1_macro")
    print(f"{attr:20s}: F1 = {scores.mean():.3f} ± {scores.std():.3f}")
```

典型解读:
- `has_pedestrian` F1 = 0.85: `[ATTN]` 强烈编码"有行人"这一语义
- `dataset` F1 = 0.92: 不同数据集能被明显区分,说明 domain-specific
- 随机 baseline: 2 类 0.5, 4 类 0.25

## Step 4: 对比两个 decoder 的 hidden

```python
h1 = np.load("analysis/probing/baseline_hidden.npy")
h2 = np.load("analysis/probing/pyramid_hidden.npy")

# 对每个 attribute 分别 probe,看 F1 差异
for attr in ATTRS:
    y = meta[attr]
    score_1 = cross_val_score(LogisticRegression(max_iter=1000), h1, y, cv=5).mean()
    score_2 = cross_val_score(LogisticRegression(max_iter=1000), h2, y, cv=5).mean()
    print(f"{attr}: baseline={score_1:.3f}, pyramid={score_2:.3f}, diff={score_2-score_1:+.3f}")
```

**典型 insight**: 如果 pyramid 的 `has_pedestrian` probe F1 显著高于 baseline,说明新 decoder 的训练信号推着 `[ATTN]` 编码更多具体物体语义。这是很漂亮的论文论点。

## CKA 相似度 (两个模型 hidden 的关系)

```python
from sklearn.metrics.pairwise import cosine_similarity

def linear_CKA(X, Y):
    """Linear Centered Kernel Alignment"""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    return np.linalg.norm(XtY, "fro")**2 / (np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro"))

cka = linear_CKA(h1, h2)
print(f"CKA similarity baseline↔pyramid: {cka:.3f}")
# 接近 1: 两个模型学到相似表征
# 接近 0: 完全不同的表征结构
```

## 可视化 (t-SNE / UMAP)

```python
import umap
import matplotlib.pyplot as plt

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
emb = reducer.fit_transform(X)   # [N, 2]

fig, ax = plt.subplots(figsize=(6, 5))
for ds in meta["dataset"].unique():
    mask = (meta["dataset"] == ds)
    ax.scatter(emb[mask, 0], emb[mask, 1], label=ds, s=5, alpha=0.6)
ax.legend()
plt.savefig("analysis/probing/umap_by_dataset.pdf")
```

看不同子数据集的 hidden 在低维是否可分 — 可分说明 `[ATTN]` 知道自己在哪个 domain。
