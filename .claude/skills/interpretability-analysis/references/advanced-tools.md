# 进阶可解释性工具

GradCAM / Attention Rollout / 反事实遮挡 / 跨模态对齐 — 按需使用。

---

# §1 GradCAM-for-Heatmap

回答: "预测热力图在位置 (x, y) 的高亮,是由输入图的哪些像素驱动?"

```python
# scripts/analysis/gradcam.py
import torch
import cv2
import numpy as np
from model.Attn_model import AttnForCausalLM

model = AttnForCausalLM.from_pretrained("./ckpts/ATTN-7B-<exp>").cuda().eval()

def gradcam_for_saliency(model, img_tensor, prompt_tensor, top_k_percentile=0.95):
    """
    img_tensor: [1, 3, H, W], 需 requires_grad=True
    prompt_tensor: [1, L] (input_ids)
    """
    img_tensor = img_tensor.clone().requires_grad_(True)

    # 前向
    output = model.evaluate(img_tensor, prompt_tensor, ...)  # 具体 API 看 chat.py
    pred_sal = output.pred_sal   # [1, 1, 256, 256]

    # 选 top 5% 像素作 "目标分数"
    thresh = pred_sal.quantile(top_k_percentile)
    score = (pred_sal * (pred_sal > thresh).float()).sum()

    # 反传
    score.backward()
    grad = img_tensor.grad.abs().sum(dim=1, keepdim=True)   # [1, 1, H, W]
    return grad.detach().cpu().numpy()


# 可视化
grad_map = gradcam_for_saliency(model, img, prompt)
# normalize + 上色 + 叠加原图
```

**使用场景**: 对比两个 decoder 在同一样本上的"决策来源" — 可能发现一个依赖前景物体、另一个依赖道路边缘 (训练 artifact)。

---

# §2 Attention Rollout (LLM 侧信息流)

把 LLaVA 每层 attention 累乘,得到从 `[ATTN]` 到视觉 token 的总注意力权重。

```python
# scripts/analysis/attention_rollout.py
import torch

def rollout(attentions, head_fusion="mean"):
    """
    attentions: list of [B, heads, L, L], 每层一个
    返回: [B, L, L]
    """
    if head_fusion == "mean":
        attns = [a.mean(dim=1) for a in attentions]
    elif head_fusion == "max":
        attns = [a.max(dim=1)[0] for a in attentions]

    # 加残差 + normalize
    eye = torch.eye(attns[0].size(-1), device=attns[0].device)
    attns = [a + eye for a in attns]
    attns = [a / a.sum(dim=-1, keepdim=True) for a in attns]

    # 累乘
    joint = attns[0]
    for a in attns[1:]:
        joint = a @ joint
    return joint


# 跑
with torch.no_grad():
    outputs = model.forward(..., output_attentions=True)
rolled = rollout(outputs.attentions)   # [B, L, L]

# 定位 [ATTN] token 的 row,取视觉 token 列
attn_pos = (input_ids == attn_idx).nonzero()[0, 1].item()
vis_start, vis_end = ...  # 视觉 token 在序列里的范围 (通常是 <im_start> 到 <im_end> 之间)
attn_to_vis = rolled[:, attn_pos, vis_start:vis_end]   # [B, N_vis]

# reshape 回 [H, W] 可视化
H = W = int(attn_to_vis.size(1) ** 0.5)
vis_map = attn_to_vis.reshape(B, H, W)
```

**输出**: LLM 在产生 `[ATTN]` 之前"注意了图像的哪些区域"。常常比最终热力图更粗但揭示 LLM 内部的 reasoning。

---

# §3 反事实遮挡 (Causal Test)

问: **遮住图像某区域,预测会怎么变?**

```python
# scripts/analysis/counterfactual.py
import numpy as np
from itertools import product

def occlusion_importance(img, predict_fn, patch_size=64, stride=32):
    """
    img: np.ndarray, [H, W, 3]
    predict_fn: callable, img -> saliency map [256, 256]
    返回: importance map [H, W],每个位置的值 = 遮住那里时预测变化量
    """
    H, W = img.shape[:2]
    base_pred = predict_fn(img)
    importance = np.zeros((H, W))

    for y, x in product(range(0, H - patch_size, stride),
                        range(0, W - patch_size, stride)):
        occluded = img.copy()
        occluded[y:y+patch_size, x:x+patch_size] = 128   # 灰色填充

        new_pred = predict_fn(occluded)
        delta = np.abs(new_pred - base_pred).sum()
        importance[y:y+patch_size, x:x+patch_size] += delta

    return importance
```

**应用**:
- 验证 "模型是否真依赖前方车辆" → 遮前车看 prediction 是否崩
- 排查训练 artifact → 发现模型强烈依赖天空区域就有问题

**慢**: 一张图要跑 ~100 次 forward,只对少量关键样本做(~10-20 张)。

---

# §4 What-Why 文本 ↔ Where 热力图 对齐度

回答: "模型说 'a pedestrian on the left' 时,热力图真的在左侧吗?"

```python
# scripts/analysis/crossmodal_alignment.py
import clip
import torch

model_clip, preprocess = clip.load("ViT-L/14", device="cuda")

def text_region_score(img_rgb, pred_sal, phrase):
    """
    img_rgb: [H, W, 3]
    pred_sal: [H, W], [0, 1]
    phrase: str, 比如 "a pedestrian on the left"

    返回: 对齐分数 (CLIP cosine sim)
    """
    # 用 pred_sal 掩盖图像(保留高亮区域)
    mask = pred_sal[..., None]    # [H, W, 1]
    masked = (img_rgb.astype(np.float32) * mask).astype(np.uint8)

    img_t = preprocess(Image.fromarray(masked)).unsqueeze(0).cuda()
    text_t = clip.tokenize([phrase]).cuda()

    with torch.no_grad():
        img_feat = model_clip.encode_image(img_t)
        text_feat = model_clip.encode_text(text_t)
    return torch.cosine_similarity(img_feat, text_feat).item()


# 对每个样本,抽 What 部分的名词短语,算对齐分数
# 看高 CC 和低 CC 样本的对齐分数差异
```

**论文 insight**: 如果新 decoder 让 Where 和 What 更一致 (alignment 从 0.18 → 0.25),这是"可解释性提升"的直接证据。

---

# §5 结合使用模板

```python
# analysis/notebooks/full_analysis.ipynb
# 1. 错误聚类 (error-clustering.md)
# 2. 从 bottom cluster 挑 10 张
# 3. 对这 10 张做:
#    - Quad vis (qualitative-vis.md)
#    - GradCAM (§1)
#    - Counterfactual (§3)
# 4. 把所有图并排,人工看共性
# 5. 写 analysis_report.md 给 paper-writing 接力
```

## 何时用哪个

| 工具 | 开销 | 信息量 | 建议 |
| --- | --- | --- | --- |
| Qualitative vis | 低 | 高 | 每个实验都做 |
| Error clustering | 低 | 高 | 每个实验都做 |
| Linear probe | 中 | 中 | 关键对比实验做 |
| GradCAM | 中 | 中 | 解释特定失败样本 |
| Attention rollout | 中 | 低-中 | 想说 "LLM 内部看哪里" 时 |
| Counterfactual | 高 | 高 | 关键 10 张样本 |
| Crossmodal alignment | 中 | 高 | 论文核心 claim 是可解释性时 |
