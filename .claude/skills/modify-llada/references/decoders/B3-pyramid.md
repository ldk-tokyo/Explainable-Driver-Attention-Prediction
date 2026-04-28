# B3: Pyramid (FPN-style) Decoder

**多尺度融合**,是改善"细节定位"能力的经典路子。对驾驶注视场景尤其合适 —— 小物体(远处行人)和大物体(迎面车辆)都要被关注。

## 动机

原 decoder 从单尺度(16×16 token grid,但 token 本身是 LLaVA mm_projector 之后的 4096 维特征)上采样到 256×256,空间细节容易丢。Pyramid decoder 在多个 resolution level 融合信息 + 用 `[ATTN]` embedding 做 FiLM 调制,让文本侧的"where/what/why"推理能在每一级都参与空间细化。

## 显存预算

~ +4GB。5090 上 `batch=1, image_size=1024` 勉强可跑,稳妥建议 `image_size=512`。

## 接口 (对齐真实代码)

**读 `interface-contract.md` 先**。关键点:

- `forward(visual_features, llm_hidden_state)` —— 签名**顺序与原 `AttentionDecoder` 一致**
- `visual_features`: `[B, 256, 4096]`(token 序列,C=4096,**不是 1024**)
- `llm_hidden_state`: `[B, 1024]`(已经 text_hidden_fcs 投影过)
- 输出 `[B, 1, 256, 256]`,sigmoid 后
- decoder 外还有 `GaussianBlur(11, 2) + Resize`,保持默认不改

## 完整代码

```python
# model/decoders/pyramid.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttnDecoderBase


class PyramidDecoder(AttnDecoderBase):
    """
    FPN-style decoder: 从 [ATTN] embedding 驱动,在多个空间尺度上
    融合 visual features 并渐进上采样。

    接口对齐 Attn_model.py::model_forward 的真实调用:
        pred_sal = attn_decoder(output_image_features, pred_embeddings)
        #                       [B, 256, 4096]         [B, 1024]

    设计:
    - 输入 visual_features [B, 256, 4096] 来自 LLaVA mm_projector 之后
    - visual_proj: Linear(4096 → hidden_dim=1024),对齐原 AttentionDecoder 的做法
    - reshape 成 [B, 1024, 16, 16] 作为"最深层 feature"
    - 4 个上采样阶段:16→32→64→128→256
    - 每阶段: conv → GN → GELU → (conv → GN → GELU) → upsample
    - [ATTN] embedding (llm_hidden_state) 通过 FiLM-style 调制注入每一层
    """
    def __init__(self,
                 visual_dim=4096,
                 hidden_dim=1024,
                 out_size=256,
                 depth=4,
                 channels=(512, 256, 128, 64),
                 use_film=True,
                 **kwargs):
        super().__init__(visual_dim=visual_dim, hidden_dim=hidden_dim, out_size=out_size)
        assert len(channels) == depth, f"channels 长度 {len(channels)} ≠ depth {depth}"

        self.use_film = use_film

        # 对齐原 AttentionDecoder: 先把 4096 视觉 token 降到 hidden_dim
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)

        # Stage blocks: 每个 stage 上采样 2 倍
        self.stages = nn.ModuleList()
        in_c = hidden_dim
        for out_c in channels:
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.GroupNorm(8, out_c),
                nn.GELU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.GroupNorm(8, out_c),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ))
            in_c = out_c

        # FiLM 调制: llm_hidden_state → 每个 stage 的 (gamma, beta)
        if use_film:
            self.film = nn.ModuleList([
                nn.Linear(hidden_dim, out_c * 2)
                for out_c in channels
            ])

        # 最终 head,以 sigmoid 收尾(对齐原 decoder 的 readout)
        self.head = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1] // 2, 3, padding=1),
            nn.GroupNorm(4, channels[-1] // 2),
            nn.GELU(),
            nn.Conv2d(channels[-1] // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, visual_features, llm_hidden_state):
        """
        Args:
            visual_features:  [B, 256, 4096]  (LLaVA mm_projector 之后)
            llm_hidden_state: [B, 1024]       (text_hidden_fcs 投影后,[ATTN] 位置)

        Returns:
            pred_sal: [B, 1, 256, 256] 范围 [0, 1]
        """
        # 1. 视觉侧降维 [B, N, 4096] → [B, N, 1024]
        x = self.visual_proj(visual_features)

        # 2. reshape 成 [B, C, H, W]
        B, N, C = x.shape
        H = W = int(N ** 0.5)                  # CLIP ViT-L/14: 256 tokens → 16×16
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        # 3. 逐 stage 上采样 + FiLM
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.use_film:
                gamma_beta = self.film[i](llm_hidden_state)   # [B, 2*out_c]
                gamma, beta = gamma_beta.chunk(2, dim=-1)
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)     # [B, C, 1, 1]
                beta  = beta.unsqueeze(-1).unsqueeze(-1)
                x = x * (1 + gamma) + beta

        # 4. head + 尺寸兜底
        out = self.head(x)
        if out.shape[-1] != self.out_size:
            out = F.interpolate(out, size=(self.out_size, self.out_size),
                                mode='bilinear', align_corners=False)

        return out
```

## 与原 `AttentionDecoder` 的差异点(论文可以写)

| 维度 | 原 AttentionDecoder | PyramidDecoder |
| --- | --- | --- |
| 视觉-文本融合 | 3 层 Cross-Attention(query=hidden, kv=visual) | FiLM 调制(每级 γ, β 来自 hidden) |
| 上采样路径 | 4 × `Decoder_ConvBlock`(conv + BN + upsample) | 4 × 双卷积块 + GN + upsample |
| Norm | BatchNorm2d | GroupNorm |
| 参数量 | ~5M | ~20M |
| 归纳偏置 | 全局注意力先融合 → 纯上采样 | 每级都重新注入文本信号 |

## 实验命令

```bash
deepspeed --num_gpus=1 --master_port=24999 train_ds.py \
  --version="./weights/LLaVA-7B-Lightening-v1-1" \
  --vision-tower="./weights/clip-vit-large-patch14" \
  --dataset_dir="./dataset" --log_base_dir="./runs" \
  --dataset="BDDA||DReyeVE||LBW||DADA" --train_sample_rates="8,5,2,7" \
  --exp_name="decoder-pyramid-d4-5090" \
  --batch_size=1 --grad_accumulation_steps=16 \
  --precision=bf16 --epochs=10 --steps_per_epoch=500 \
  --image_size=512 \
  --decoder_type=pyramid --decoder_depth=4
```

## 变体扫描

| 变体 | 改动 | 命令差 |
| --- | --- | --- |
| 深度 3 vs 4 vs 5 | `depth=3/4/5`, channels 相应改 | `--decoder_depth=3` etc. |
| 带/不带 FiLM | `use_film=False` | 加 `--decoder_extra_config='{"use_film": false}'` |
| 通道数减半 | `(256, 128, 64, 32)` | 加 `--decoder_extra_config='{"channels": [256, 128, 64, 32]}'` |
| 用 BN2d 复现原行为 | 替换 GN → BN2d | 新加 `--decoder_extra_config='{"norm": "bn"}'`(需代码支持) |

FiLM ablation 通常是最能写进论文的(是否需要文本调制视觉)。

## 外部后处理的注意

`model_forward` 在 decoder 输出后还做 `GaussianBlur(11, σ=2) + Resize`。PyramidDecoder 内部已经有多级上采样,本身比原 decoder 平滑,再过一次 blur 可能**过柔**。

**第一次 A/B 不改外部后处理**,保持与基线同一条件。如果 pyramid 的 CC 不升反降,可以单独开消融:关掉外部 blur(改 `model_forward`),比较差异。

## 预期结果

- CC 相比基线 +0.03~0.05
- KLD 下降 0.08~0.12
- **小物体(行人/远车)检测明显变好** —— 可视化对比最能看出来
- 参数量比 B1 稍多(~20M)但比 B4 (~35M) 少
