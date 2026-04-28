# 其他 Decoder 方案概要 (B2, B4, B5, B6, B8)

这些方案优先级低于 B1/B3/B7。记录下来备查;真正要实现前需要补充细节。

---

## B2: DETR-style Learned Queries

**核心思想**: 用 K 个 learnable query token(比如 K=8),让每个 query 关注不同的注视区域(前方 / 左右 / 交通灯等)。最后聚合成一张 heatmap。

**伪代码骨架**:
```python
class DETRQueryDecoder(AttnDecoderBase):
    def __init__(self, hidden_dim, num_queries=8, depth=6, ...):
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)  # 把 [ATTN] embed 加到 queries
        self.transformer = nn.TransformerDecoder(...)  # learned queries × visual tokens
        # 输出: K 张 attention map,加权(学到的权重)得到最终 map
```

**优点**: 可解释(每个 query 可以可视化看"关注什么")
**缺点**: 参数多 (+3GB), 训练比较吃技巧 (DETR 原论文说训练慢)

---

## B4: Mask2Former-Style

**核心思想**: 用 Mask2Former 的 masked attention —— 每层 cross-attention 时,把注意力限制在上一层预测的 mask 区域内。迭代 refine。

**实现依赖**: `mmcv` + `mmdet`,安装相对重。

**显存**: +5GB,5090 上必须 `image_size=512`。

**潜在价值**: 对 "模型多次 refine 自己的注视区域" 这一生物学启发有对应,论文故事好讲。

**代码参考**: https://github.com/facebookresearch/Mask2Former 的 `mask2former/modeling/transformer_decoder/`

---

## B5: UNet-style Upsampling with Skip Connections

**核心思想**: 把 CLIP 不同层(浅层、中层、深层)的 feature 作为 skip connection,UNet 结构上采样。

**为什么**: CLIP 浅层含有更多纹理/边缘信息,深层含语义。Saliency prediction 两者都需要。

**问题**: LLaVA 目前只用 CLIP 最后一层。要改 LLaVA 的 vision_tower 接口以暴露中间层。这算 **半个 Pattern E (换 backbone)**,工作量大。

---

## B6: Diffusion-based Head

**核心思想**: 用 few-step DDPM/DDIM 作为 decoder,把 [ATTN] embedding 作为 conditioning,从噪声逐步 denoise 出 heatmap。

**为什么**: 生成式先验,理论上能产更"自然"的 heatmap 分布(不再是 point estimate 而是 distribution)。

**坑**:
- 训练慢(每样本多次 forward)
- 推理慢(多步 denoise)
- 对 KL/CC 等 deterministic 指标不一定友好 —— 这些指标假设单一 prediction
- 论文故事好讲(生成式),但不一定赢基线

**只有在非常有时间 + 想写 novelty 强的 paper 才考虑**。

---

## B8: Hungarian-matching Structured Output

**核心思想**: 不输出 dense heatmap,而是输出一组 gaze 注视点 `(x, y, weight) × K`,用 Hungarian matching 和 GT 注视点对齐,再用高斯生成 heatmap 作为可视化。

**为什么**: 结构化输出,比 dense prediction 在稀疏注视点数据集上(如 DReyeVE)可能更准。

**挑战**:
- W³DA 现有 GT 是 heatmap 不是点集,要先"反推"出点(找 local maxima)
- 训练信号从 heatmap 变成点集,loss 重新设计(Hungarian + MSE)
- 失败模式多(点数 K 选错会崩)

**适合作为 side project / future work**,不是主攻方案。

---

# 如何进一步细化

任何一个方案你决定要深入做,通知 Claude Code:
> "把 B2 从概要扩展成完整实现,写到 references/decoders/B2-detr-queries.md"

Claude 会参照 B1/B3/B7 的格式(动机 + 显存预算 + 完整代码 + 实验命令 + 预期结果)产出详细版。
