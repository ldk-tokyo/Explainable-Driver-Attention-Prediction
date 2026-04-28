# 5090 32GB 显存预算详解

心算你的改动会不会 OOM,以及 OOM 了怎么降。

## §1 基线配置下各项占用

| 项 | 估算 | 说明 |
| --- | --- | --- |
| LLaVA-7B bf16 权重 | ~14 GB | 固定 |
| CLIP-L/14 bf16 | ~0.6 GB | 固定,冻结 |
| attn_decoder + text_hidden_fcs | ~0.5-1.5 GB | **改造时主要变这里** |
| LoRA params + 梯度 | ~0.1 GB | LoRA 开销小 |
| AdamW optimizer state (fp32) for LoRA + decoder | ~2-4 GB | 每个可训练参数存 momentum + variance |
| activations (image_size=1024, batch=1) | ~6-10 GB | gradient checkpointing 下 |
| DeepSpeed buffers (reduce_bucket etc.) | ~1 GB | 可调 |
| **总计** | **~26-30 GB** | 刚好够 |

## §2 改 decoder 的显存含义

| 改动 | 新增参数量 | 显存涨幅 |
| --- | --- | --- |
| Cross-attn depth 2 → 6 | +11M | +1.5 GB |
| 换 Pyramid decoder (depth=4) | +17M | +3-4 GB (含 activation) |
| 换 Mask2Former | +30M | +5 GB |
| 换 SAM-style (freeze SAM) | +3M | +1 GB |
| 换 SAM-style (unfreeze SAM) | +30M | +4 GB |
| 加 UNet skip connections | +15M | +3 GB |

**先估算,不要让 OOM 浪费 2 小时训练启动时间**。

## §3 OOM 降级三连击 (完整配置)

### 降级 1: 降图像尺寸

```bash
--image_size=512        # 从 1024 → 512
```
- 省 ~6GB
- attn_decoder 输出分辨率减半 (可能需同步改 GT 的 map_size?需验证)
- 指标可能掉 0.01-0.02

### 降级 2: ZeRO-3

编辑 `train_ds.py::main` 的 `ds_config`:

```python
ds_config = {
    "train_micro_batch_size_per_gpu": args.batch_size,
    "gradient_accumulation_steps": args.grad_accumulation_steps,
    "optimizer": {...},
    "scheduler": {...},
    "bf16": {"enabled": args.precision == "bf16"},
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 3,                            # ← 改 2 → 3
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
    },
}
```
- 省 ~2-3 GB
- 慢 ~15%
- 单卡 ZeRO-3 的 optimizer state 切不了(只有 1 rank),但 param 切分仍省显存

### 降级 3: CPU offload

```python
"zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": True},
    "offload_param":     {"device": "cpu", "pin_memory": True},
    # ... 其他 ZeRO-3 配置
},
```
- 省 ~10+ GB
- 慢 2-3× (CPU↔GPU 传输瓶颈)
- 系统内存要 ≥32GB, 最好 64GB

## §4 gradient checkpointing 检查

原 LLada 代码应该已经开了 gradient checkpointing (LLaVA base 默认开)。验证:

```python
# train_ds.py 里找 model.gradient_checkpointing_enable() 或类似
# 没开就手动加:
model.gradient_checkpointing_enable()
```

开了之后显存明显降但训练慢 ~20%。**单卡 5090 必须开**。

## §5 批次降级

如果上面都试了还 OOM:

```bash
--grad_accumulation_steps=32    # 从 16 翻倍
```
effective batch 仍是 16,但每 step 显存不变,迭代次数翻倍 → 训练时间翻倍。这是"最后挽救手段"。

## §6 推理 / 评测显存

评测时 `--val_batch_size=1` 足够;不会 OOM。如果 OOM 说明 decoder 改造有严重问题(比如某个 buffer 没 release)。

`chat.py` 单图推理可能有 `deepspeed.init_inference` 开销,约 3-5 GB 额外。

## §7 监控

```bash
# 训练时另一个终端
watch -n 2 "nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu,power.draw --format=csv"
```

关键观察:
- Memory 稳定在 28-30 GB → 正常
- 缓慢上涨到 32 GB → 有 memory leak(可能每 step 留了 intermediate tensor 没 del)
- 突然 OOM → step 1 batch 过大,或新 decoder 内部某个 op 生成巨大 intermediate

## §8 如果真的需要更大 batch

几个选项:
- **租 cloud**: Spheron / Runpod 的 A100 80GB $1-2/小时, 跑关键实验
- **两张 5090 data-parallel**: 无 NVLink 但 PCIe data-parallel 可行, effective batch 翻倍
- **降精度到 fp8**: 只对推理合适,训练不稳

**但对 decoder 研究, effective batch 16 已经能看出显著提升**,不需要追求 80。
