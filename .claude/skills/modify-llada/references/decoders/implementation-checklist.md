# 添加新 decoder 的 5 步实施清单

照这个顺序做,每步结束后验证再进下一步。

## Step 1. 新建 decoder 模块

```
model/
├── Attn_model.py
├── decoders/              ← 新建
│   ├── __init__.py
│   ├── base.py            ← AttnDecoderBase (见 interface-contract.md)
│   ├── cross_attn.py      ← 可选: 把原 decoder 从 Attn_model.py 剥离
│   ├── pyramid.py         ← 你的新方案
│   └── ...
```

`model/decoders/__init__.py`:
```python
from .cross_attn import CrossAttnDecoder
from .pyramid import PyramidDecoder
from .sam_style import SAMStyleDecoder
# ... 所有 decoder 都 export 出来
```

## Step 2. CLI 参数

在 `train_ds.py::parse_args` 加:

```python
parser.add_argument(
    "--decoder_type", default="cross_attn",
    choices=["cross_attn", "pyramid", "mask2former", "sam_style", "detr", "unet"],
    help="注意力解码器类型"
)
parser.add_argument(
    "--decoder_depth", default=2, type=int,
    help="decoder 层数(对 B1/B2/B3 适用)"
)
parser.add_argument(
    "--decoder_extra_config", default="", type=str,
    help="JSON 字符串,传给特定 decoder 的额外参数"
)
```

把这些参数传进 `model_args`:

```python
model_args = {
    ...
    "decoder_type": args.decoder_type,
    "decoder_depth": args.decoder_depth,
    "decoder_extra_config": args.decoder_extra_config,
}
```

## Step 3. Dispatch 逻辑

**关键: dispatch 在 `AttnMetaModel.initialize_attn_modules`,不在 `AttnForCausalLM`。**

真实模块层级是:
- `AttnForCausalLM(LlavaLlamaForCausalLM)`  —— 外壳,`self.model = AttnModel(...)`
- `AttnModel(AttnMetaModel, LlavaLlamaModel)` —— 多继承
- `AttnMetaModel`                            —— **`initialize_attn_modules` 定义在这里**,创建 `self.attn_decoder`

外部调用路径是 `self.model.attn_decoder(...)`(注意多一层 `.model`)。

`AttnMetaModel.__init__` 走的是"两遍初始化"模式: 第一次把配置塞到 `config` 上,第二次(`hasattr(config, "train_attn_decoder")` 为真时)才调 `initialize_attn_modules` 建模块。所以把新字段也塞到 config 上最自然。

### 3a. 在 `AttnMetaModel.__init__` 里 stash 新字段

```python
class AttnMetaModel:
    def __init__(self, config, **kwargs):
        super(AttnMetaModel, self).__init__(config)
        self.config = config
        if not hasattr(self.config, "train_attn_decoder"):
            self.config.train_attn_decoder = kwargs["train_attn_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            # ── 新增 ──
            self.config.decoder_type = kwargs.get("decoder_type", "cross_attn")
            self.config.decoder_depth = kwargs.get("decoder_depth", 2)
            self.config.decoder_extra_config = kwargs.get("decoder_extra_config", "")
            # ────────
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_attn_modules(self.config)
```

### 3b. 改 `AttnMetaModel.initialize_attn_modules` 做 dispatch

```python
def initialize_attn_modules(self, config):
    import json
    from model.decoders import (
        CrossAttnDecoder, PyramidDecoder, SAMStyleDecoder,
        # ...
    )

    decoder_cls_map = {
        "cross_attn": CrossAttnDecoder,        # 原 AttentionDecoder 剥离后的封装
        "pyramid":    PyramidDecoder,
        "sam_style":  SAMStyleDecoder,
        # ...
    }

    decoder_type = getattr(config, "decoder_type", "cross_attn")
    depth        = getattr(config, "decoder_depth", 2)
    extra_json   = getattr(config, "decoder_extra_config", "")
    extra        = json.loads(extra_json) if extra_json else {}

    if decoder_type not in decoder_cls_map:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")

    # 所有 decoder 都接受 visual_dim / hidden_dim / out_size / depth 四个标准参数;
    # 其它特殊参数走 extra_config (JSON) —— 见 interface-contract.md
    self.attn_decoder = decoder_cls_map[decoder_type](
        visual_dim=config.hidden_size,    # LLaVA LLM hidden = 4096
        hidden_dim=config.out_dim,        # text_hidden_fcs 之后 = 1024
        out_size=256,
        depth=depth,
        **extra,
    )

    if config.train_attn_decoder:
        self.attn_decoder.train()
        for param in self.attn_decoder.parameters():
            param.requires_grad = True

    # text_hidden_fcs 保持原样 (ModuleList + Sequential,不是单个 Linear)
    in_dim = config.hidden_size   # 4096
    out_dim = config.out_dim      # 1024
    text_fc = [
        nn.Linear(in_dim, in_dim),
        nn.ReLU(inplace=True),
        nn.Linear(in_dim, out_dim),
        nn.Dropout(0.0),
    ]
    self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
    self.text_hidden_fcs.train()
    for param in self.text_hidden_fcs.parameters():
        param.requires_grad = True
```

### 3c. 在 `train_ds.py` 把 CLI 传进 `model_args`

```python
model_args = {
    ...
    "train_attn_decoder": True,
    "out_dim": args.out_dim,
    # ── 新增 ──
    "decoder_type": args.decoder_type,
    "decoder_depth": args.decoder_depth,
    "decoder_extra_config": args.decoder_extra_config,
}
# 这些 kwargs 会流到 AttnForCausalLM → AttnModel → AttnMetaModel
```

## Step 4. 验证参数冻结策略

`train_ds.py` 的参数解冻循环(大约第 160 行):

```python
for n, p in model.named_parameters():
    if any(x in n for x in ["lm_head", "embed_tokens",
                             "attn_decoder", "text_hidden_fcs"]):
        p.requires_grad = True
```

**因为你的新 decoder 赋值给 `self.attn_decoder`,所以参数名会含 `attn_decoder`,自动进入可训练列表。**

确认 LoRA 排除列表(`find_linear_layers`,约第 140 行):
```python
and all(x not in name for x in [
    "attn_decoder",    # ← 必须在
    "vision_tower",
    "mm_projector",
    "text_hidden_fcs",
])
```

## Step 5. 验证流程

### 5a. print 参数量

```bash
# 临时在 train_ds.py main() 开头加:
# model.print_trainable_parameters()
# exit()
# 跑看看:
deepspeed --num_gpus=1 ... --decoder_type=pyramid --epochs=0 2>&1 | grep trainable
```

预期 output:
```
trainable params: 25,000,000 || all params: 7,100,000,000 || trainable%: 0.35
```
参数量应该比原 decoder (~5M) 大一点(新 decoder 更重)。如果 = 0 或 = 全部,说明 Step 3 或 Step 4 有错。

### 5b. Smoke test

```bash
deepspeed --num_gpus=1 --master_port=26000 train_ds.py \
  --version="./weights/LLaVA-7B-Lightening-v1-1" \
  --vision-tower="./weights/clip-vit-large-patch14" \
  --dataset_dir="./dataset" --log_base_dir="./runs_smoke" \
  --dataset="BDDA" --train_sample_rates="1" \
  --val_dataset="BDDA" --val_sample_rates="1" \
  --exp_name="smoke_pyramid_$(date +%s)" \
  --epochs=1 --steps_per_epoch=2 \
  --batch_size=1 --grad_accumulation_steps=1 \
  --val_samples_num=4 --val_batch_size=1 \
  --precision=bf16 \
  --decoder_type=pyramid --decoder_depth=4
```

通过条件:
- 出现 2 次 `Epoch: [0]` 训练日志
- validate 最后打印 `cc`, `kld` 等非 NaN 数值
- `pred_sal` shape 确认 `[1, 1, 256, 256]`(需在 forward 里加 print)
- 注意 `decoder_depth` 必须与 decoder 默认 `channels` 长度一致 (PyramidDecoder 默认 `channels=(512,256,128,64)` → depth=4),否则 assert 炸

### 5c. Pilot epoch (2 小时)

```bash
# 类似 smoke 但跑完整 epoch
--epochs=2 --steps_per_epoch=100 --val_samples_num=500
```

看 loss 曲线正常下降,val 指标不比基线低 10×。

### 5d. Full train (~4 天)

```bash
deepspeed --num_gpus=1 --master_port=24999 train_ds.py \
  ... (正式超参) ... \
  --decoder_type=pyramid --decoder_depth=3 \
  --exp_name="decoder-pyramid-depth3-5090"
```

## 常见失误

| 症状 | 修复 |
| --- | --- |
| `RuntimeError: Expected all tensors to be on the same device` | 新 decoder 里手动 `.cuda()` 了某个 tensor;应该相信 DeepSpeed |
| `trainable%: 100.00` | Step 4 没配好,所有参数都 grad=True;LoRA 没生效 |
| `trainable params: 0` | 新 decoder 名字不含 `attn_decoder`;见 Step 3 |
| Smoke 时 `pred_sal.shape` 不是 `[1, 1, 256, 256]` | Decoder 输出尺寸没对齐,见 interface-contract.md |
| Loss NaN 一开始就是 | BN 层(batch=1);改 GN |
| Loss NaN 几百步后 | 学习率太高 for 新 decoder;降到 `--lr=1e-4` 试试 |
