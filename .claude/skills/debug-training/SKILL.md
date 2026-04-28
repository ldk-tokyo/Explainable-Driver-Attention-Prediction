---
name: debug-training
description: 训练 / 评测 / 推理遇到的常见错误诊断与修复:OOM、NaN loss、hang 卡死、shape mismatch、tokenizer size mismatch、DeepSpeed kernel 错、LoRA 合并失败、METEOR 全 0、LLada `[ATTN]` 输出异常、`pred_sal is None`、Blackwell/CUDA 不兼容。用户说"报错"、"训不动"、"OOM"、"NaN"、"炸了"、"崩溃"、"hang"、"卡住"、"结果全 0"、"指标不对"、"为什么 loss 不降" 时使用。先按症状索引定位,再进对应修复流程。
---

# 训练报错诊断手册

## 快速症状索引

| 症状 | 最可能原因 | 第一步操作 |
| --- | --- | --- |
| `CUDA out of memory` | batch/image_size/decoder 太大 | 见 §1 |
| `RuntimeError: NotImplementedError: sm_120` | PyTorch 不支持 Blackwell | 见 §2 |
| DeepSpeed `replace_with_kernel_inject` 错 | DeepSpeed 版本对 Blackwell 不全 | 见 §2 |
| `loss = NaN` 从一开始 | BN 层 (batch=1) / 数值爆炸 | 见 §3 |
| `loss = NaN` 几百步后 | 学习率过大 / KL 分母为 0 | 见 §3 |
| `tokenizer size mismatch` | 加 token 后没 `resize_token_embeddings` | 见 §4 |
| `pred_sal is None` 验证时 | 模型没生成 `[ATTN]` 或 decoder 接口不匹配 | 见 §5 |
| `METEOR: 0.0` 所有样本 | Java 版本不对 | `sudo apt install openjdk-8-jdk` |
| LoRA 合并后指标比训练时差 10× | ckpt 路径错 / 没 zero_to_fp32 | 见 §6 |
| 训练 hang 无输出 | `master_port` 被占 / DeepSpeed 通信卡 | 见 §7 |
| 所有 trainable params 都解冻了 | 参数冻结 loop 写错 | 见 §8 |
| GPU 温度 > 85°C 持续上涨 | 5090 散热跟不上 | `sudo nvidia-smi -pl 450` |

---

## §1 OOM

详见 `train-eval-workflow/references/memory-budget.md` 的降级三连击。

### 快速定位

```bash
# 训练启动时开显存 profile
CUDA_LAUNCH_BLOCKING=1 deepspeed --num_gpus=1 ... 2>&1 | tee oom_debug.log
# 看 OOM 在第几步, 是 forward 还是 backward
```

### 常见 OOM 触发点

| 位置 | 原因 | 修复 |
| --- | --- | --- |
| step 1 forward | batch/image_size 直接太大 | 降 batch / image_size |
| step 1 backward | activation 太大 | 开 gradient_checkpointing |
| step 100 附近 | memory leak (某个中间 tensor 没 del) | 查改动的 decoder 有没有保留 reference |
| validation 时 | val_batch_size 太大,或 attn_metrics 累积 | `--val_batch_size=1` |

### 新 decoder 的 OOM

改 decoder 后 OOM 首要怀疑:
1. **多 head attention 的 `num_heads` 太大**: 每个 head 有独立的 Q/K/V projection
2. **FFN 的 `dim_feedforward` 太大**: 默认 4× hidden,可降到 2×
3. **没用 gradient checkpointing**: 在 decoder 里手动加

```python
# 在新 decoder 内部对 transformer layer 加 ckpt
from torch.utils.checkpoint import checkpoint

def forward(self, attn_embed, visual_feats):
    # ...
    for layer in self.layers:
        q = checkpoint(layer, q, visual_feats, use_reentrant=False)
    # ...
```

---

## §2 Blackwell / CUDA / PyTorch 不兼容

```bash
# 诊断
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
# 理想输出: 2.5.1+cu128 12.8 (12, 0)
```

### 常见症状与修复

| 报错 | 原因 | 修复 |
| --- | --- | --- |
| `NotImplementedError: CUDA capability sm_120` | PyTorch < 2.5 | `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu128` |
| `CUDA driver version is insufficient` | 驱动 < 570 | 升级 NVIDIA 驱动 |
| `CUDA error: invalid device function` | CUDA Toolkit 不匹配 | 装 CUDA 12.8 |
| DeepSpeed `replace_with_kernel_inject` crash | DeepSpeed 版本对 Blackwell 不全 | 升级 DeepSpeed,或设 `False` |
| nccl init fail | 单卡不需要 nccl | 用 `--num_gpus=1` 不会触发 |

### DeepSpeed kernel inject fallback

`chat.py` 里:
```python
# 原
model_engine = deepspeed.init_inference(
    model=model, dtype=torch.half,
    replace_with_kernel_inject=True, replace_method="auto",
)

# 改为
model_engine = deepspeed.init_inference(
    model=model, dtype=torch.bfloat16,   # 注意 Blackwell 用 bf16
    replace_with_kernel_inject=False,
)
# 或直接跳过 DeepSpeed inference:
model.cuda().eval()
# 后面用 model 而不是 model_engine
```

---

## §3 NaN / 数值不稳

### NaN 从第一步

**最常见: BatchNorm + batch=1**

```python
# 在新 decoder 里搜 BN
grep -n "BatchNorm" model/decoders/*.py
```

改成:
```python
# 原: nn.BatchNorm2d(C)
# 改: nn.GroupNorm(8, C)   # 8 组,根据 C 可调
```

**其他可能**:
- `torch.log(0)` → 输入给 log 的 tensor 没 clamp
- `sqrt(负数)` → 方差计算出负
- 初始化权重过大 → `nn.init.kaiming_normal_`

### NaN 在几百步后

```bash
# 查 TensorBoard 里 loss 曲线找 NaN 出现点
# 记下附近的 step
# 降学习率 3× 重试
--lr=1e-4    # 从 3e-4 降
```

**KL divergence 的分母为 0** 是 LLada 的经典坑。查:
```python
# model/Attn_model.py 的 KLDivergence
eps = 1e-7    # 确认有 epsilon
```

### 把 bf16 临时切 fp32 定位

```bash
# 训练 50 步 fp32 看 loss 正常不
--precision=fp32 --epochs=1 --steps_per_epoch=50
```
如果 fp32 下 loss 正常,bf16 下 NaN,说明是精度问题,某个 op 对 bf16 不友好。查:
- 大值 softmax (concurrent large scores)
- 分母很小的除法

---

## §4 Tokenizer / Special Token

### `size mismatch for lm_head.weight`

加新 token 后:
```python
# train_ds.py
num_added = tokenizer.add_tokens("[NEW_TOKEN]")
model.resize_token_embeddings(len(tokenizer))  # ← 必须!

# LoRA 配置后也要:
model.get_input_embeddings().weight.requires_grad = True
```

### `[ATTN]` token id 不是 32003

有些情况 LLaVA 的 tokenizer 会变。动态获取:
```python
args.attn_token_idx = tokenizer("[ATTN]", add_special_tokens=False).input_ids[0]
```
而不是硬编码 32003。

---

## §5 `pred_sal is None` 验证时

**诊断流程**:

```python
# 临时在 Attn_model.py::forward 末尾加 debug print
print(f"[DEBUG] input_ids has [ATTN]? {(input_ids == self.attn_token_idx).any()}")
print(f"[DEBUG] generated ids: {generated_ids.shape}")
print(f"[DEBUG] generated tokens sample: {tokenizer.decode(generated_ids[0])[:100]}")
```

### 可能原因

1. **模型没训够步数,还没学会吐 `[ATTN]`**: smoke test 2 步确实太少
2. **prompt 模板坏了, 没引导 `[ATTN]`**: 查 dataset 产出的 `conversations`
3. **新 decoder 签名不对,forward 里 dispatch 进错分支**: 加 print
4. **generate 的 max_new_tokens 太小,还没生成到 `[ATTN]` 就停了**: 增大

---

## §6 LoRA 合并问题

### 合并后指标炸烂

按标准流程:
```bash
cd ./runs/<exp>/ckpt_model
python zero_to_fp32.py . ../pytorch_model.bin   # ← 不能省
cd -
```

常见错:
- 忘了 `zero_to_fp32.py`,直接把 ZeRO 切分 ckpt 当 .bin 用 → 权重错
- 合并时 `--version` 指错基座(要用原始 LLaVA,不是别的 ckpt)
- `CUDA_VISIBLE_DEVICES=""` 确保 CPU,否则 OOM

### 检查合并后 ckpt 完整

```python
import torch
state = torch.load("./ckpts/ATTN-7B-<exp>/pytorch_model-00001-of-00003.bin", map_location="cpu")
print(f"Keys: {len(state)}")
print(f"Has attn_decoder? {any('attn_decoder' in k for k in state)}")
print(f"Has text_hidden_fcs? {any('text_hidden_fcs' in k for k in state)}")
```
应该都有。如果 `attn_decoder` 没了,说明合并脚本没覆盖到你的新 decoder 名字。

---

## §7 Hang

### 症状: 启动后无任何日志

```bash
# 强制 kill 并换端口
pkill -9 -f "deepspeed"
deepspeed --num_gpus=1 --master_port=25000 ...   # 换端口
```

### 症状: 训练中途 hang

可能 data loader 死锁:
```bash
# 降 num_workers 到 0 排查
--workers=0
```

或 nccl 通信超时(5090 单卡不会触发)。

---

## §8 参数冻结策略错

### 症状: `trainable params: 7000000000` (全部解冻)

查 `train_ds.py` 的参数解冻循环:
```python
for n, p in model.named_parameters():
    if any(x in n for x in ["lm_head", "embed_tokens", "attn_decoder", "text_hidden_fcs"]):
        p.requires_grad = True
    else:
        p.requires_grad = False   # ← 默认 False,只解冻关心的
```

**易错点**: 没写 `else: p.requires_grad = False`,所有参数保持默认 True。

### 症状: `trainable params: 0`

LoRA 没正确套上。查:
```python
# print LoRA target 模块
for n, p in model.named_parameters():
    if p.requires_grad:
        print(n, p.shape)
# 应该看到一堆 lora_A / lora_B, 还有 attn_decoder / text_hidden_fcs
```

---

## 通用诊断流程

遇到新报错,按这个顺序:

1. **复现在 smoke test 上**(变量最少)
2. **二分查找**: 回到上一个 commit 看是否正常;逐步应用改动直到复现
3. **打 print debug**: 不要依赖 pdb(DeepSpeed 兼容性差)
4. **查 `runs/<exp>/log_test.txt`**: 很多 silent failure 会在这里留痕
5. **搜原仓库 issue + LLaVA issue**

## 兜底: 求助

开 issue 前先收集:
- 完整错误栈
- `torch.__version__`, `torch.cuda.get_device_capability()`
- 完整训练命令
- `nvidia-smi` 输出
- 最近的 commit hash (`git log -1 --oneline`)
