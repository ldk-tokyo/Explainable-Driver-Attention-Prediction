---
name: modify-llada
description: 在 LLada 基础上做研究型改动的总控 skill,**核心场景是改造/替换注意力解码器 (attn_decoder)**。先读本 SKILL.md 定位要改哪类,然后按指引 view 对应 `references/` 子文件获取详细代码。用户说"改 decoder"、"换解码器"、"pyramid decoder"、"Transformer decoder"、"SAM decoder"、"改模型"、"加 token"、"改 loss"、"接视频"、"换 CLIP"、"做消融"、"扩展模型" 时必须使用。不先读这个直接改 `model/Attn_model.py` 几乎必炸(5 处代码位置耦合)。
---

# 改造 LLada 总控

## 心法 (牢记)

LLada 的每个改动可能同时影响 **5 个位置**:
1. `model/Attn_model.py` — 模型结构
2. `train_ds.py` — tokenizer、参数解冻、LoRA 排除列表
3. `utils/dataset.py` — 数据 pipeline 输出字段
4. loss / 指标聚合 + TensorBoard logging
5. ckpt 命名规则

漏改一处要么炸,要么安静训出废点。先读 `llada-architecture` skill 理解数据流,再动手。

## 改动类型分派

| 想做什么 | Pattern | 阅读路径 |
| --- | --- | --- |
| **替换/改造 attn_decoder**(主攻) | **B** | 见下 §B |
| 加新 special token (如 `[RISK]`) | A | `references/patterns/pattern-A-new-token.md` |
| 修改 loss 组合 | C | `references/patterns/pattern-C-loss.md` |
| 接视频/多帧输入 | D | `references/patterns/pattern-D-temporal.md` |
| 换视觉 backbone | E | `references/patterns/pattern-E-backbone.md` |

## §B Decoder 改造 (核心场景)

### B.0 候选方案矩阵 (先选 2-3 个深做)

| 编号 | 方案 | 归纳偏置 | 显存 | 预期 CC 提升 | 难度 | 详细文件 |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | 加深原 cross-attention (2→6 层) | 表达力↑ | +2GB | +0.02~0.05 | 低 | `references/decoders/B1-deeper-cross-attn.md` |
| B2 | DETR 风 learned queries | 多 query 并行 | +3GB | +0.03 | 中 | `references/decoders/B2-detr-queries.md` |
| B3 | FPN / Pyramid Decoder | 多尺度融合 | +4GB | +0.05 | 中 | `references/decoders/B3-pyramid.md` |
| B4 | Mask2Former 风格 | 显式 mask 引导 | +5GB | +0.05~0.08 | 高 | `references/decoders/B4-mask2former.md` |
| B5 | UNet 上采样路径 | 空间细节↑ | +3GB | +0.02 | 中 | `references/decoders/B5-unet.md` |
| B6 | Diffusion-based head | 生成式多样性 | +6GB,推理慢 | 不确定 | 高 | `references/decoders/B6-diffusion.md` |
| B7 | SAM 风 prompt decoder | 分割大模型先验 | +2GB | +0.03~0.06 | 中 | `references/decoders/B7-sam-style.md` |
| B8 | Hungarian-matching 输出 | 结构化输出 | +2GB | 不确定 | 高 | `references/decoders/B8-hungarian.md` |

**新手入门路径**: 先 B1(最低风险,练手),再 B3 或 B7(论文故事最强),可选 B4/B6(高风险高收益)。

### B.1 所有 decoder 必须满足的接口契约

**在写任何新 decoder 前,先读 `references/decoders/interface-contract.md`**。违反契约会让 6 个指标函数算出怪数字但不报错(极其阴险)。

核心三条:
- 输入: `(attn_embed: [B, hidden], visual_feats: [B, N, C] or [B, C, H, W])`
- 输出: `pred_sal: [B, 1, 256, 256]`,范围 `[0, 1]`(与原 decoder 一致,原 decoder 有没有 sigmoid 要读代码确认)
- 新模块命名必须含 `attn_decoder`(会被 LoRA 排除、被参数解冻循环覆盖)

### B.2 改造的 5 步标准流程

```
1. 在 model/decoders/ 下写新模块 (继承 AttnDecoderBase)
2. train_ds.py 加 --decoder_type / --decoder_depth 参数
3. Attn_model.py::initialize_attn_modules 里按 decoder_type dispatch
4. 确认 find_linear_layers 排除列表含 "attn_decoder" (原本就在)
5. Smoke test → pilot epoch (2h) → full train (4 天)
```

**完整 diff 级别的 step-by-step 见 `references/decoders/implementation-checklist.md`**。

### B.3 显存预算 (5090 32GB 内怎么分)

原 decoder 加到 6 层 +2GB;改成 Pyramid +4GB;加 Mask2Former +5GB。基线已占 26-30GB,所以:
- B1 depth=6: 仍可 batch=1 image=1024
- B3/B4: 必须降 image=512 或开 ZeRO-3 offload

**详见 `train-eval-workflow` skill 的显存预算章节**。

### B.4 A/B 对比实验协议

为了写论文,每个 decoder 变体要跑:
- 控制组 `baseline-5090` + 实验组 `decoder-<type>-5090`
- 完全相同的测试集 (`val_sample_rates` 不变)
- 固定种子 (`torch.manual_seed(42)`,原代码未设,**建议加**)
- 逐样本指标(`attn_metrics_0.csv`)做成对 t 检验

**详细协议 + 统计检验代码见 `references/decoders/ab-test-protocol.md`**。

### B.5 最常见 3 个 bug

1. **visual_feats 形式不对**: token 序列 `[B, N, C]` vs feature map `[B, C, H, W]` 用错
2. **sigmoid 与否与原 decoder 不一致** → KL/CC 被破坏
3. **batch=1 时用了 BN** → 必 NaN,改 GN/LN

## 其他 Pattern (简要,详见 references)

- **Pattern A (加新 token)**: 典型用例是加 `[RISK]` 输出风险分。涉及 tokenizer resize + 新 head + 数据标注 + 新 loss。→ `references/patterns/pattern-A-new-token.md`
- **Pattern C (改 loss)**: Focal/Tversky/Contrastive 替代原 KL。→ `references/patterns/pattern-C-loss.md`
- **Pattern D (时序)**: 单帧 → 3 帧融合,5090 上要降 batch/image_size。→ `references/patterns/pattern-D-temporal.md`
- **Pattern E (换 backbone)**: CLIP → SigLIP/DINOv2/SAM encoder,需重训 mm_projector。→ `references/patterns/pattern-E-backbone.md`

## 通用安全网 (任何改动后必验证)

1. Smoke test 跑完 2 步 (`train-eval-workflow` §5)
2. `model.print_trainable_parameters()` 数字合理
3. 一个 batch 前向,检查 `output_dict` 每字段 shape/dtype
4. 跑 100 步 loss 不爆炸
5. pilot epoch val 指标不比基线差 10×

## 不要做的事

- 同时改两个以上 Pattern(出 bug 无法定位)
- 改 `model/llava/`(除非你维护 LLaVA fork)
- 在没 GT 监督的 task 上期望模型自己学会
- `master_port=24999` 下同时跑多个 DeepSpeed 进程
