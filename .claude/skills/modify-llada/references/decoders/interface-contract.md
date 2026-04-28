# Decoder 接口契约

所有新 attn_decoder 必须满足的约束。违反会让指标算错但不报错,**极其阴险**。

> 本文档以 `model/Attn_model.py` 中的真实调用为准(见 `model_forward` 约 L513、`AttnMetaModel.initialize_attn_modules` 约 L361、`AttentionDecoder` 约 L730)。曾有旧版本的契约与真实代码不符(签名顺序、通道数),以本文件为唯一来源。

## 基类

```python
# model/decoders/base.py
import torch.nn as nn

class AttnDecoderBase(nn.Module):
    """所有注意力解码器的抽象基类。签名顺序对齐 Attn_model.py::model_forward 的真实调用。"""
    def __init__(self, visual_dim: int = 4096, hidden_dim: int = 1024, out_size: int = 256):
        super().__init__()
        self.visual_dim = visual_dim    # LLaVA mm_projector 之后的 token 通道数
        self.hidden_dim = hidden_dim    # text_hidden_fcs 投影后维度(见下方 "⚠️ hidden_dim 的来源")
        self.out_size = out_size

    def forward(self, visual_features, llm_hidden_state):
        raise NotImplementedError
```

## 调用位置 (为什么签名必须这样)

`model/Attn_model.py::AttnForCausalLM.model_forward` 约 L513:

```python
# output_image_features 来自 super().forward() 的 output.image_features
#   → LLaVA mm_projector 之后的视觉 token,shape [B, 256, 4096]
# pred_embeddings = text_hidden_fcs(last_hidden)[attn_token_mask]
#   → 取 [ATTN] 位置的投影后 hidden,shape [B, 1024]
pred_sal = self.model.attn_decoder(output_image_features, pred_embeddings)
#                                  ^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
#                                  第 1 参(positional)    第 2 参(positional)
```

**参数顺序不能改**。`model_forward` 里有 3 处这样的调用(训练、validate、evaluate),全部是 `(visual_features, llm_hidden_state)`。

## 输入约定

```python
visual_features: torch.Tensor  # [B, N, C] = [B, 256, 4096]
  # - token 序列(不是 feature map),来自 LLaVA mm_projector 之后
  # - C = config.hidden_size = 4096 (LLM hidden dim),不是 CLIP ViT 的 1024
  # - N = 256,对应 16×16 空间格子 (CLIP ViT-L/14 patch grid)
  # - 如果 decoder 内部需要 [B, C, H, W],自己 reshape:
  #       B, N, C = visual_features.shape
  #       H = W = int(N ** 0.5)                # 16
  #       x = visual_features.permute(0, 2, 1).reshape(B, C, H, W)

llm_hidden_state: torch.Tensor  # [B, hidden_dim] = [B, 1024]
  # - 来自 text_hidden_fcs(last_hidden)[attn_token_mask]
  # - text_hidden_fcs 已经把 4096 投到 1024,所以这里 hidden_dim=1024
  # - 通常每个样本恰好 1 个 [ATTN] token,mask 取出来后自然是 [B, 1024]
```

### ⚠️ hidden_dim 的来源(别被 `config.out_dim` 误导)

`text_hidden_fcs` 的输出维度是 **`AttnMetaModel.initialize_attn_modules` 里硬编码的 `1024`**(`Attn_model.py` 约 L369-L374 的局部变量 `out_dim = 1024`),**不**读 `config.out_dim`。虽然 `AttnMetaModel.__init__` 会把 `config.out_dim = kwargs["out_dim"]` 存下来,但这个字段在 `initialize_attn_modules` 里被忽略了。

含义:
- 改 `args.out_dim` / `config.out_dim` **不会**改变 `text_hidden_fcs` 的输出维度,也不会改变喂给 decoder 的 `llm_hidden_state` 的维度 —— 永远是 1024
- 想改这个维度,必须直接改 `initialize_attn_modules` 里的硬编码 `1024`(以及 decoder 的 `hidden_dim` 默认值)
- `Attn_model.py` 里几处注释写着 `# 256`(如 L370 的 `# 256`、L510/L513 的 `(2, 256)`)是早期版本的残留,真实值是 1024,不要被注释骗

**通道数很重要**: 如果你把第一个 Conv 写成 `Conv2d(1024, ...)`(照搬 ViT 裸特征的假设),会立即 shape mismatch。真实 C=4096,需要先降维:

```python
self.visual_proj = nn.Linear(4096, hidden_dim)   # 仿照原 AttentionDecoder
# 或:
self.visual_proj = nn.Conv2d(4096, hidden_dim, 1)  # reshape 成 [B,C,H,W] 后用
```

## 输出约定

```python
pred_sal: torch.Tensor  # [B, 1, 256, 256]
                        # 数值范围: [0, 1] (sigmoid 后)
                        # dtype: 跟输入一致 (bf16 / fp32)
```

**原 `AttentionDecoder` 的 readout 最后一层是 `nn.Sigmoid()`**,所以输出已经是概率。新 decoder 默认也 sigmoid 收尾,保持一致。

随后 `model_forward` 对 `pred_sal` 和 `gt_sal` 的 shape 有硬断言:

```python
assert pred_sal.shape == gt_sal.shape   # gt_sal: [B, 1, 256, 256]
```

## Decoder 外的后处理(必须知道)

`model_forward` 在 decoder 返回后,**还会再做两步**,才计算 loss:

```python
pred_sal = self.model.attn_decoder(output_image_features, pred_embeddings)
assert pred_sal.shape == gt_sal.shape

blur_func = transforms.GaussianBlur(11, 2)       # kernel=11, σ=2
pred_sal = blur_func(pred_sal)
pred_sal = transforms.Resize(gt_sal.shape[-2:])(pred_sal)   # 对齐 GT 尺寸
```

含义:
- 原 decoder 输出尖锐 → 外部 Gaussian 模糊一下,匹配 GT 热力图的软分布
- 如果 GT 尺寸不是 256,Resize 会兜底
- **新 decoder 如果内部已做了多尺度上采样 / 大核卷积 / 本身很平滑,再过一次 Gaussian 可能"过柔"**,CC 反而下降。第一次 A/B 时**保持这两步不动**,避免同时改两个变量;消融再测开/关

如果要改掉这两步,改的是 `model_forward`(不是你的 decoder 内部)。

## 输出尺寸 256

`args.map_size=256` 决定 GT 热力图尺寸(`utils/dataset.py` 生成),pred 必须匹配。原 decoder 从 16×16 做 4 次 2× 上采样:16→32→64→128→256,正好。如果你的 decoder 内部到不了 256,在 decoder 里 interpolate 到 256(别依赖外部 Resize,因为它会改原图尺寸,语义不一样):

```python
if out.shape[-1] != self.out_size:
    out = F.interpolate(out, size=(self.out_size, self.out_size),
                        mode='bilinear', align_corners=False)
```

## 命名约定

新 decoder 类本身可以叫任何名字 (`PyramidDecoder`, `SAMDecoder`...),但 **在 `AttnMetaModel.initialize_attn_modules` 里赋值时必须叫 `self.attn_decoder`**:

```python
# ✅ 正确 —— 在 AttnMetaModel.initialize_attn_modules 里
self.attn_decoder = PyramidDecoder(...)

# ❌ 错误 - 参数名不含 "attn_decoder",会被 LoRA 覆盖、不会进解冻列表
self.pyramid_decoder = PyramidDecoder(...)
```

原因:
- `train_ds.py::find_linear_layers` 的排除列表硬编码 `"attn_decoder"`
- `train_ds.py` 的参数解冻循环硬编码 `"attn_decoder"` 判断

**访问路径注意**: 虽然属性挂在 `AttnMetaModel` 上,但外部从 `AttnForCausalLM` 访问时路径是 `self.model.attn_decoder`(多一层 `self.model`),因为 `AttnForCausalLM` 里有 `self.model = AttnModel(config, **kwargs)`。见 `model_forward` 的调用。

## Norm 层 / batch=1

5090 单卡 `batch=1`。关于 BN:

- `BatchNorm2d`(**2d**): **可用**,原 `Decoder_ConvBlock` 就用了 BN2d 且能正常训。因为 BN2d 在 `(N, H, W)` 维上统计,H×W 提供了足够"样本",即使 N=1 也不退化。
- `BatchNorm1d`: **不能用**(只有 N 一个维度统计,batch=1 直接退化为 0)。
- `LayerNorm` / `GroupNorm`: 安全,推荐用在 Transformer 内部或替代 BN2d 做更稳的选择。

**默认推荐 `GroupNorm(8, C)`**(参数独立于 batch,收敛更稳),但如果要精准复现原行为,可以用 BN2d。

## bf16 下的小坑

原 `Decoder_ConvBlock.forward` 里 upsample 前手动 cast 到 fp32:

```python
dtype = x.dtype
if dtype == torch.bfloat16:
    x = x.to(torch.float32)
x = self.upsample_layer(x)
if dtype == torch.bfloat16:
    x = x.to(torch.bfloat16)
```

原因:某些版本的 `UpsamplingBilinear2d` 对 bf16 支持不好。新 decoder 用 `F.interpolate(mode='bilinear')` 一般没问题,但若用老的 `nn.UpsamplingBilinear2d` 层,照抄这段 cast 更稳。

## Checklist (写完新 decoder 后对照)

- [ ] 继承 `AttnDecoderBase`
- [ ] `forward(visual_features, llm_hidden_state)` **顺序正确**(不是 `(attn_embed, visual_feats)`)
- [ ] 处理输入 `visual_features` 的 C=4096(不是 1024)—— 内置 `visual_proj` 或 pointwise conv
- [ ] 输出 `[B, 1, 256, 256]`,`[0, 1]` 范围(sigmoid)
- [ ] 没有 BN**1d**(BN2d 可用,但 GN 更稳)
- [ ] 在 `AttnMetaModel.initialize_attn_modules` 里赋值给 `self.attn_decoder`(外部访问走 `self.model.attn_decoder`)
- [ ] 参数量 `sum(p.numel() for p in decoder.parameters())` 符合预算
- [ ] 理解 decoder **外**还有 `GaussianBlur(11, 2) + Resize`,第一次 A/B 不动它
- [ ] Smoke test pass (见 `train-eval-workflow` §5)
