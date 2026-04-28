# B7: SAM-style Prompt Decoder

借用 **Segment Anything (SAM)** 的预训练 mask decoder。把 `[ATTN]` 的 hidden state 当作 single-point prompt 的 embedding,SAM 的 decoder 本来就是为"从 prompt 生成 mask"设计的,**与 LLada 任务结构高度同构**。

## 动机

- SAM 预训练于 SA-1B (1.1 B mask),拥有强大的分割先验
- SAM 的 MaskDecoder 是现成的、经充分验证的 prompt-conditioned decoder
- 可冻结 / 部分微调,训练参数量小

## 安装依赖

```bash
pip install segment-anything
# 下载 SAM ViT-H 权重(不需要 ViT encoder,只要 decoder 部分)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./weights/sam_vit_h.pth
```

## 显存预算

+2GB (SAM MaskDecoder 本身轻量),5090 可直接 `batch=1, image_size=1024`。

## 完整代码

```python
# model/decoders/sam_style.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttnDecoderBase


class SAMStyleDecoder(AttnDecoderBase):
    """
    基于 SAM MaskDecoder 架构。

    把 [ATTN] hidden state 当作 single sparse prompt embedding,
    visual_feats reshape 到 SAM 需要的格式后走 SAM decoder。

    可选: 从预训练 SAM ckpt 加载 decoder 权重做 warm-start。
    """
    def __init__(self, hidden_dim=1024, out_size=256,
                 sam_ckpt_path=None, freeze_sam=False, **kwargs):
        super().__init__(hidden_dim, out_size)

        from segment_anything.modeling.mask_decoder import MaskDecoder
        from segment_anything.modeling.transformer import TwoWayTransformer

        # SAM 内部 embedding_dim = 256
        self.sam_embed_dim = 256

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=1,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=self.sam_embed_dim,
        )

        # 加载 SAM 预训练权重(只取 mask_decoder 部分)
        if sam_ckpt_path is not None:
            self._load_sam_weights(sam_ckpt_path)

        if freeze_sam:
            for p in self.mask_decoder.parameters():
                p.requires_grad = False

        # [ATTN] hidden_dim → SAM prompt dim
        self.prompt_proj = nn.Linear(hidden_dim, self.sam_embed_dim)

        # visual_feats → SAM image embedding format [B, 256, 64, 64]
        # SAM 期望 image embedding 是 64×64 空间,256 通道
        self.visual_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, self.sam_embed_dim, 1),
            nn.GroupNorm(8, self.sam_embed_dim),
        )

        # 固定位置编码 (64×64)
        self.register_buffer(
            "image_pe",
            self._build_positional_encoding(64, self.sam_embed_dim),
        )

    def _load_sam_weights(self, ckpt_path):
        """加载 SAM 预训练权重的 mask_decoder 部分"""
        state = torch.load(ckpt_path, map_location="cpu")
        mask_decoder_state = {
            k.replace("mask_decoder.", ""): v
            for k, v in state.items()
            if k.startswith("mask_decoder.")
        }
        missing, unexpected = self.mask_decoder.load_state_dict(
            mask_decoder_state, strict=False
        )
        print(f"[SAMStyleDecoder] Loaded SAM decoder weights. "
              f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def _build_positional_encoding(self, size, dim):
        """SAM 风格的 sinusoidal PE"""
        # 简化版: 直接用 learnable 或 sinusoidal
        # SAM 原生用 random Fourier,这里用简化 sinusoidal
        h = w = size
        y, x = torch.meshgrid(
            torch.arange(h).float(), torch.arange(w).float(), indexing="ij"
        )
        pos = torch.stack([x, y], dim=-1) / size  # [h, w, 2]

        # 经过 Fourier 投影得到 [h, w, dim]
        freqs = torch.randn(2, dim // 2) * 2 * 3.14159
        pos_enc = pos @ freqs   # [h, w, dim/2]
        pe = torch.cat([pos_enc.sin(), pos_enc.cos()], dim=-1)  # [h, w, dim]
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # [1, dim, h, w]
        return pe

    def forward(self, attn_embed, visual_feats):
        """
        Args:
            attn_embed:   [B, hidden_dim]
            visual_feats: [B, N, hidden_dim] 或 [B, C, H, W]

        Returns:
            pred_sal: [B, 1, 256, 256] 范围 [0, 1]
        """
        # 1. 视觉特征: 统一 [B, C, H, W] 并投影
        if visual_feats.dim() == 3:
            B, N, C = visual_feats.shape
            H = W = int(N ** 0.5)
            x = visual_feats.permute(0, 2, 1).reshape(B, C, H, W)
        else:
            x = visual_feats
            B = x.shape[0]

        # CLIP 出来是 16×16,SAM 期望 64×64,先上采样
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        image_emb = self.visual_proj(x)  # [B, 256, 64, 64]

        # 2. [ATTN] 作为 sparse prompt
        sparse_emb = self.prompt_proj(attn_embed).unsqueeze(1)  # [B, 1, 256]

        # 3. 无 dense prompt
        dense_emb = torch.zeros(B, self.sam_embed_dim, 64, 64,
                                 device=image_emb.device, dtype=image_emb.dtype)

        # 4. SAM decoder
        low_res_masks, iou_pred = self.mask_decoder(
            image_embeddings=image_emb,
            image_pe=self.image_pe.expand(B, -1, -1, -1),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        # low_res_masks: [B, 1, 256, 256]

        if low_res_masks.shape[-1] != self.out_size:
            low_res_masks = F.interpolate(
                low_res_masks, size=(self.out_size, self.out_size),
                mode='bilinear', align_corners=False
            )

        return torch.sigmoid(low_res_masks)
```

## 训练策略

**两种玩法**:

### A. 冻结 SAM decoder 只训练投影层
- 极省参数(只训 `prompt_proj` 和 `visual_proj`)
- 训练快
- 验证 "SAM 先验有多有用"

```bash
--decoder_type=sam_style \
--decoder_extra_config='{"sam_ckpt_path": "./weights/sam_vit_h.pth", "freeze_sam": true}'
```

### B. 全量微调 SAM decoder
- 性能更好
- 参数量大
- 训练更慢

```bash
--decoder_type=sam_style \
--decoder_extra_config='{"sam_ckpt_path": "./weights/sam_vit_h.pth", "freeze_sam": false}'
```

## A vs B 对比本身就是一篇论文的章节

"SAM decoder frozen (only adapter trainable)" vs "SAM decoder fully fine-tuned" vs "SAM decoder from scratch (no pretrain)" —— 三个变体跑完,你就能讲清楚"SAM 预训练先验在驾驶注视任务上的迁移价值"。

## 预期结果

- A 变体: CC ~ +0.03,但文本指标几乎不动
- B 变体: CC ~ +0.06, 参数量涨但 effective
- 关键 insight: **SAM 的 prompt-to-mask 归纳偏置非常适合 [ATTN]-to-heatmap 任务**

## 潜在问题

- SAM 期望输入 1024×1024,内部 patch embed 到 64×64 (16× 下采样);你这里跳过 ViT 直接给 64×64 image_emb,分布可能和 SAM 预训练时不一致。如果 freeze_sam=True 效果不好,这是最可能的原因
- 解决方案: 加一个 "alignment layer" 让 `visual_proj` 的输出统计量靠近 SAM ViT 输出(可选 task,有时间再做)
