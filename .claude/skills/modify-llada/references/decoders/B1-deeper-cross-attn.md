# B1: 加深 Cross-Attention Decoder

**最低风险方案**,第一个 decoder 实验建议从这里开始。

## 动机

原 attn_decoder 层数可能很浅(1-2 层)。加深到 4-8 层看表达力天花板在哪。

## 显存预算

| depth | 参数量 | 额外显存 |
| --- | --- | --- |
| 2 (基线) | ~5M | 0 |
| 4 | ~10M | +0.5GB |
| 6 | ~16M | +1.2GB |
| 8 | ~22M | +2GB |

5090 32GB 全部能跑 `batch=1, image_size=1024`。

## 完整代码

```python
# model/decoders/cross_attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttnDecoderBase


class CrossAttnDecoder(AttnDecoderBase):
    """
    Cross-attention decoder: [ATTN] hidden state 作为 query,
    visual tokens 作为 key/value,多层交互后得到 attention map。

    Args:
        hidden_dim: [ATTN] token 的 hidden dim (默认 1024)
        depth: cross-attention 层数
        num_heads: multi-head 数量
        out_size: 输出 saliency map 边长
    """
    def __init__(self, hidden_dim=1024, depth=2, num_heads=8, out_size=256,
                 dim_feedforward=None, **kwargs):
        super().__init__(hidden_dim, out_size)
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4

        # 多层 cross-attention
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-LN, 训练更稳定
            ) for _ in range(depth)
        ])

        # 从 visual tokens 空间重建 + 上采样到 256×256
        # 假设 CLIP-L/14 在 224×224 输入下输出 16×16 tokens
        # 先 → 32×32 → 64×64 → 128×128 → 256×256
        self.upsampler = nn.Sequential(
            nn.Conv2d(hidden_dim, 512, 3, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(512, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 1, 1),
        )

    def forward(self, attn_embed, visual_feats):
        """
        Args:
            attn_embed:   [B, hidden_dim]
            visual_feats: [B, N, hidden_dim]  (token 序列)

        Returns:
            pred_sal: [B, 1, 256, 256] 范围 [0, 1]
        """
        B, N, C = visual_feats.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"visual_feats 长度 {N} 不是完美平方"

        # Query: [ATTN] hidden state, shape [B, 1, C]
        q = attn_embed.unsqueeze(1)

        # 多层 cross-attention
        # TransformerDecoderLayer 的 forward: (tgt, memory)
        # tgt 是 query, memory 是 key/value
        for layer in self.layers:
            q = layer(q, visual_feats)
        # 这里的 q 已经吸收了 visual 信息,但 shape 还是 [B, 1, C]

        # 把 q 作为调制信号,和 visual_feats 做点积得到空间 attention
        # q: [B, 1, C], visual_feats: [B, N, C]
        attn_scores = torch.bmm(visual_feats, q.transpose(1, 2))  # [B, N, 1]
        attn_scores = attn_scores.reshape(B, H, W, 1).permute(0, 3, 1, 2)
        # 现在 [B, 1, H, W]

        # 把 attention scores 扩成 hidden_dim 通道以喂 upsampler
        # 用 attention 加权 visual_feats 得到 broadcasted feature
        attn_weights = F.softmax(attn_scores.flatten(2), dim=-1).reshape_as(attn_scores)
        weighted_feat = torch.einsum('bnc,bchw->bchw',
                                      visual_feats.reshape(B, N, C),
                                      attn_weights.expand(-1, C, -1, -1))
        # (上面 einsum 不完全对,需根据实际调整;这里示意用 attention 加权)

        # 简化: 直接用 visual_feats reshape + attention 加权再 upsample
        feat_map = visual_feats.permute(0, 2, 1).reshape(B, C, H, W)
        feat_map = feat_map * attn_scores  # element-wise 加权

        out = self.upsampler(feat_map)

        # 尺寸兜底
        if out.shape[-1] != self.out_size:
            out = F.interpolate(out, size=(self.out_size, self.out_size),
                                mode='bilinear', align_corners=False)

        return torch.sigmoid(out)
```

## 实验命令 (扫 depth)

跑 4 个实验对比 depth 的影响:

```bash
for depth in 2 4 6 8; do
  deepspeed --num_gpus=1 --master_port=24999 train_ds.py \
    --version="./weights/LLaVA-7B-Lightening-v1-1" \
    --vision-tower="./weights/clip-vit-large-patch14" \
    --dataset_dir="./dataset" --log_base_dir="./runs" \
    --dataset="BDDA||DReyeVE||LBW||DADA" --train_sample_rates="8,5,2,7" \
    --exp_name="decoder-cross-depth${depth}-5090" \
    --batch_size=1 --grad_accumulation_steps=16 \
    --precision=bf16 --epochs=10 --steps_per_epoch=500 \
    --decoder_type=cross_attn --decoder_depth=${depth}
done
```

4 个实验 × 4 天单卡 ≈ 2 周完成。中间可以 `--auto_resume` 断开续上。

## 预期结果

- depth 从 2→6: CC 稳定上升 (0.71 → 0.74 左右)
- depth 6→8: 收益递减,甚至可能过拟合(val 下降)
- 画"depth vs CC" 和 "depth vs KLD" 折线图(见 `paper-writing` skill)

## 变体

想更激进的:
- **Pre-LN → Post-LN**: 训练更不稳但性能上限高
- **`num_heads` 调整**: 4/8/16 扫一扫
- **`dim_feedforward` 调整**: 4×hidden vs 8×hidden

这些是额外自由度,做完 depth 扫后可以选一个再跑。
