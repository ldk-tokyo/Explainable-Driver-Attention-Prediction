---
name: train-eval-workflow
description: LLada 在 **RTX 5090 单卡 (32GB Blackwell)** 上的训练 / 评测 / LoRA 合并 / smoke test / 长时训练保命措施。包含 DeepSpeed 显存紧张三连击 (image_size → ZeRO-3 → CPU offload) 和 Blackwell 专项 (`replace_with_kernel_inject`)。用户说"跑训练"、"跑评测"、"怎么 eval"、"显存不够"、"OOM"、"DeepSpeed 配"、"ckpt 合并"、"resume"、"smoke test"、"chat.py"、"推理"、"batch 多大"、"grad accum" 时必用。不先 smoke 就直接全量训练是这个项目最大的时间浪费源。显存预算细算见 `references/memory-budget.md`。
---

# 训练 / 评测 / 推理工作流

## 核心原则

1. **任何代码改动后先跑 smoke test,再全量训练** (§5)
2. **显存预算按 32GB 算**,不按 A100 80GB。详见 `references/memory-budget.md`
3. **`--exp_name` 必须有语义**,禁止 `test`, `tmp`, `exp1`

## §1 基线训练 (5090 单卡)

```bash
deepspeed --num_gpus=1 --master_port=24999 train_ds.py \
  --version="./weights/LLaVA-7B-Lightening-v1-1" \
  --vision-tower="./weights/clip-vit-large-patch14" \
  --dataset_dir="./dataset" --log_base_dir="./runs" \
  --dataset="BDDA||DReyeVE||LBW||DADA" --train_sample_rates="8,5,2,7" \
  --exp_name="baseline-5090" \
  --batch_size=1 --grad_accumulation_steps=16 \
  --image_size=1024 --precision=bf16 \
  --epochs=10 --steps_per_epoch=500 --lr=0.0003
```

- Effective batch = 16(原论文 80,指标略降但单卡可接受)
- 预计显存: 26-30 GB (ZeRO-2 默认)
- 预计时长: ~80-100 小时

## §2 显存紧张三连击

按优先级降:

1. **降图像尺寸**: 加 `--image_size=512` → 省 ~6GB,attn_decoder 输出分辨率减半
2. **开 ZeRO-3**: 编辑 `train_ds.py::main` 的 `ds_config`, `"stage": 3` → 省 ~2-3GB,慢 ~15%
3. **开 CPU offload**: `"offload_optimizer": {"device": "cpu", ...}` 和 `"offload_param"` → 省 ~10GB, 慢 2-3×

完整配置代码见 `references/memory-budget.md`。

## §3 Smoke Test (改动后必跑,5 分钟内)

```bash
deepspeed --num_gpus=1 --master_port=26000 train_ds.py \
  --version="./weights/LLaVA-7B-Lightening-v1-1" \
  --vision-tower="./weights/clip-vit-large-patch14" \
  --dataset_dir="./dataset" --log_base_dir="./runs_smoke" \
  --dataset="BDDA" --train_sample_rates="1" \
  --val_dataset="BDDA" --val_sample_rates="1" \
  --exp_name="smoke_$(date +%s)" \
  --epochs=1 --steps_per_epoch=2 \
  --batch_size=1 --grad_accumulation_steps=1 \
  --val_samples_num=4 --val_batch_size=1 --precision=bf16
```

通过标志:
- 2 次 `Epoch: [0]` 日志
- val 最后打印 `cc`, `kld` 等非 NaN 数
- 改 decoder 后额外确认 `pred_sal.shape == [1, 1, 256, 256]`

## §4 评测

```bash
# 仅热力图指标 (快)
deepspeed --num_gpus=1 ... --eval_only

# + 文本指标 (慢 10×)
deepspeed --num_gpus=1 ... --eval_only --eval_text --eval_text_save

# 带可视化 (改 decoder 必开)
deepspeed --num_gpus=1 ... --eval_only --eval_colormap_save
```

注意: `--version` 必须指向 `./ckpts/ATTN-7B-<exp>` (合并后的 HF 格式),不是原始 `./runs/<exp>/ckpt_model`。

## §5 LoRA 合并

```bash
# 1. ZeRO ckpt → fp32 权重
cd ./runs/baseline-5090/ckpt_model
python zero_to_fp32.py . ../pytorch_model.bin
cd -

# 2. LoRA 合并 (CPU)
CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py \
  --version="./weights/LLaVA-7B-Lightening-v1-1" \
  --vision-tower="./weights/clip-vit-large-patch14" \
  --weight="./runs/baseline-5090/pytorch_model.bin" \
  --save_path="./ckpts/ATTN-7B-baseline-5090"
```

## §6 Blackwell 专项注意

### 6.1 CUDA / PyTorch 版本
```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
# 必须 (12, 0) = sm_120 = Blackwell 正确识别
# 如果报 NotImplementedError: sm_120 → 升级 PyTorch 到 2.5+
```

### 6.2 DeepSpeed kernel inject

`chat.py` 里 `deepspeed.init_inference(replace_with_kernel_inject=True)` 在早期 Blackwell 支持不全的 DeepSpeed 版本会 crash。若报 kernel not found:

```python
# 改为 False 或直接不用 deepspeed.init_inference
model_engine = deepspeed.init_inference(
    model=model, dtype=torch.bfloat16,
    replace_with_kernel_inject=False,   # ← 改这里
)
```

### 6.3 精度

**强制 bf16**,fp16 在 Blackwell 上训 LLM 不稳。FP8 推理可选,但 LLada 未集成,优先级低。

## §7 长时训练保命

```bash
# tmux 防 SSH 断开
tmux new -s llada_train
deepspeed --num_gpus=1 ... 2>&1 | tee train.log
# Ctrl+B D 退出

# 监控三件套 (开三个终端)
tail -f train.log
watch -n 5 "nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,memory.total --format=csv"
tensorboard --logdir=./runs --port=6006
```

**温度保护**: GPU > 85°C 持续上涨时降功耗:
```bash
sudo nvidia-smi -pl 450    # 从 575W 降到 450W, 性能 -5%, 温度 -8°C
```

## §8 恢复训练

```bash
# 自动找最新 ckpt
deepspeed --num_gpus=1 ... --auto_resume

# 指定 ckpt
deepspeed --num_gpus=1 ... --resume="./runs/baseline-5090/ckpt_model"
```

## §9 评测完 checklist

1. [ ] `log_test.txt` 里的 attn 指标跟基线同量级,不是 NaN
2. [ ] 开 `--eval_text` 时 BLEU-4 > 0.05
3. [ ] `attn_metrics_0.csv` 行数 == 测试集样本数
4. [ ] 抽 2-3 张 `val_vis/epoch_*/pred_*.jpg` 肉眼确认
5. [ ] 结果写进 `experiments.md`
