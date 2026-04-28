# Pattern C: 修改 Loss 组合

## 适用场景

- 原 KL-based attn loss 可能不是最优,试 BCE / Focal / Tversky
- 想加 GT 注视点的 NSS 作为辅助监督
- 想加 contrastive loss 对 What/Why 的语义对齐

## 改动清单

### 1. CLI 参数
```python
# train_ds.py
parser.add_argument(
    "--attn_loss_type", default="kl",
    choices=["kl", "bce", "focal", "tversky", "combined"]
)
parser.add_argument("--focal_gamma", default=2.0, type=float)
parser.add_argument("--tversky_alpha", default=0.7, type=float)
parser.add_argument("--tversky_beta", default=0.3, type=float)
```

### 2. `model/Attn_model.py::forward` 替换 loss 计算
```python
def _compute_attn_loss(self, pred_sal, gt_sal):
    if self.attn_loss_type == "kl":
        return kl_divergence_loss(pred_sal, gt_sal)
    elif self.attn_loss_type == "bce":
        return F.binary_cross_entropy(pred_sal, gt_sal)
    elif self.attn_loss_type == "focal":
        return focal_loss(pred_sal, gt_sal, gamma=self.focal_gamma)
    elif self.attn_loss_type == "tversky":
        return tversky_loss(pred_sal, gt_sal,
                            alpha=self.tversky_alpha, beta=self.tversky_beta)
    elif self.attn_loss_type == "combined":
        return (kl_divergence_loss(pred_sal, gt_sal) * 0.5 +
                focal_loss(pred_sal, gt_sal) * 0.5)
```

### 3. Unit test (推荐)
```python
# tests/test_losses.py
import torch

def test_focal_loss_not_nan():
    pred = torch.rand(2, 1, 256, 256)
    gt   = torch.rand(2, 1, 256, 256)
    loss = focal_loss(pred, gt, gamma=2.0)
    assert not torch.isnan(loss)
    assert 0 < loss.item() < 100

def test_tversky_loss_bounds():
    # 完全正确时 loss ≈ 0
    gt = torch.rand(2, 1, 256, 256)
    loss = tversky_loss(gt, gt)
    assert loss.item() < 0.01
```

### 4. Logging
新 loss 加到 TensorBoard + AverageMeter。

## Focal / Tversky 实现

```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Binary focal loss, 抑制易样本"""
    pred = pred.clamp(1e-7, 1 - 1e-7)
    pt = target * pred + (1 - target) * (1 - pred)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    return -(alpha_t * (1 - pt).pow(gamma) * pt.log()).mean()

def tversky_loss(pred, target, alpha=0.7, beta=0.3, eps=1e-6):
    """Tversky = 不对称 Dice, alpha 大惩罚 FN, beta 大惩罚 FP"""
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    fp = (pred * (1 - target)).sum()
    return 1 - (tp + eps) / (tp + alpha * fn + beta * fp + eps)
```

---

# Pattern D: 时序 / 多帧输入

## 最稳路径: 晚期融合 (3 帧)

每帧独立过 CLIP + LLM,最后融合 `[ATTN]` 的 hidden state。

### 改动要点

**`utils/dataset.py::__getitem__`**:
- 返回 `images_clip: [T, 3, 224, 224]`, `images: [T, 3, 1024, 1024]`
- 文本 prompt 只对应"中间帧"的 gazemap

**`collate_fn`**:
- stack 成 `[B, T, C, H, W]`

**`model/Attn_model.py::forward`**:
```python
B, T = images_clip.shape[:2]
images_clip_flat = images_clip.reshape(B*T, 3, 224, 224)
visual_feats = self.vision_tower(images_clip_flat)   # [B*T, N, C]

# 每个时间步独立走 LLM 取 [ATTN] hidden
# ... (需要把 input_ids 复制 T 份或只对中间帧做)

# 融合 T 个 [ATTN] hidden
attn_hidden_t = ...                        # [B, T, hidden]
attn_hidden_fused = self.temporal_agg(attn_hidden_t)   # [B, hidden]

# 之后走原 decoder
```

### 5090 上的可行性

- T=3 时显存约 3× (CLIP forward 3 次) + LLM 基础 = **可能 OOM**
- 建议先 `--image_size=512` + `batch=1` + `grad_accum=16`
- 可能必须开 ZeRO-3 + CPU offload

### Lazy-fusion 替代方案

只过一次 LLM(只给中间帧),但在 CLIP 阶段融合多帧:
- 3 帧分别过 CLIP → 3×visual_feats
- concat 或 attention 融合 → 单一 visual_feats
- 余下和原流程一样

更省显存,但时序感知弱于正规 late fusion。

### 光流作为额外通道 (备选)

离线算光流存盘:
```bash
python scripts/compute_flow.py --dataset_dir ./dataset
# 在每个 video 目录下生成 flow_frames/
```

`__getitem__` 返回 `{"flow": flow_tensor}`,在 model 里可选作为额外输入。

---

# Pattern E: 换视觉 Backbone

## 风险告知

**LLaVA 的 mm_projector 是按 CLIP-ViT-L/14 的特定输出(16×16×1024)训练的**。换 backbone 后:
- 如果输出 dim 不同 → mm_projector 无法加载
- 如果 token 数不同 → attn_decoder 内的 reshape (N → H×W) 要改
- 必须重新训练(或至少微调)mm_projector

## 候选 backbone

| Backbone | 输出 | 优势 | 劣势 |
| --- | --- | --- | --- |
| SigLIP | [256, 1024] | 语义对齐更好 | 需重训 mm_projector |
| DINOv2 | [256, 1024] (各 size 可选) | 自监督强表征 | 没有语言对齐 |
| EVA-CLIP | [256, 1024] | 比 CLIP 略强 | 需权重下载 |
| SAM encoder | [256, 1024] → [64, 64, 256] 转换 | 分割先验 | 接口完全不同 |

## 建议流程

1. **先检查 LLaVA 原仓库是否已有替代 backbone 的支持**
   - 如有 SigLIP/DINO 的 vision_tower 类,直接复用
2. **最小改动: 换 CLIP weight path**
   - 如果新 backbone 输出格式一样,只改 `--vision-tower` 指向
3. **mm_projector 微调**
   - 冻结 LLM + 新 backbone,只训练 mm_projector (可能 1-2 epoch 够)
   - 然后再跑 LLada 的训练流程

## 我的建议

换 backbone 是**高风险高工作量**的改动。**除非你的论文 story 就是关于 backbone 的**(如"哪种视觉表征最适合驾驶注视预测"),否则优先级低于 decoder 改造。
