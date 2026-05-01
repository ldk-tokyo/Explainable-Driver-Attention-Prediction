# B1: Deeper Cross-Attention Decoder — 实施草案 (v3)

> 状态: **修订草案 v3 (2026-04-28),等用户最终 review。不要按此动代码,等用户说"按草案改"。**
> 基础锚点: baseline-5090 (commit 6a290dc + ckpts/ATTN-7B-baseline-5090/)
> v3 修订: §3 改动 3 加显式 default + 注释 / 新增 §6.5 backward-compat check
> v2 修订: 加 smoke 3 assert / CC 期望放宽 / 只 depth=4 单点 / dispatch NotImplementedError / Phase 1 决策树

---

## 0. 重要发现:B1 reference 与真实代码偏差很大

读了 `model/Attn_model.py:730-783` 真实代码后,跟 [.claude/skills/modify-llada/references/decoders/B1-deeper-cross-attn.md](../.claude/skills/modify-llada/references/decoders/B1-deeper-cross-attn.md) 对比:

| 项 | B1 reference 假设 | **真实 baseline** | 影响 |
|---|---|---|---|
| 默认层数 | "可能很浅(1-2 层)" | `num_layers=3` (L731) | reference 说"加深"目标 depth 4-8,真实是 **3 → 4** (单点) |
| 跨注意力层实现 | `nn.TransformerDecoderLayer`(Pre-LN, GELU, dropout) | 自定义 `CrossAttentionLayer`(**Post-LN, ReLU**, 无 dropout, MLP=4×) (L707-727) | reference 的"完整代码"实际是**改 architecture**,不只是加深 |
| Visual feature 投影 | reference 没明说 | `nn.Linear(4096, 1024)` 显式降维 (L735) | 跟 interface-contract 一致 ✓ |
| Query × Visual 融合 | `bmm + element-wise * attn_scores` | **TILE+ADD**: `query.repeat(1, 256, 1) + visual_features` (L772-773) | 完全不同的融合机制 |
| Conv decoder 通道 | 512→256→128→64→1 + GroupNorm + GELU | 1024→512→256→64→32→1 + **BN2d** + ReLU + bf16 cast hack (L744-754) | reference 用 GN,真实用 BN(BN2d 在 H×W 上有足够样本所以 batch=1 也能跑) |
| 输出 sigmoid | reference 写了 `torch.sigmoid(out)` | `nn.Sigmoid()` 内置在 readout 里 (L752) | 一致 ✓ |
| 文件位置 | reference 提议新建 `model/decoders/{base,cross_attn,__init__}.py` | 真实是 `model/Attn_model.py` 内联定义 | 重构 vs 不重构是大决策 |
| CLI 参数 | reference 说加 `--decoder_type / --decoder_depth / --decoder_extra_config` | 现在没有任何 decoder-related CLI | 需要新增 |

**结论**: B1 reference 描述的"完整代码"实际是 **重构 + 替换 architecture**,不是单纯"加深"。如果照搬,**B1 与 baseline 不严格可比**(同时改 num_layers + Pre/Post-LN + GELU/ReLU + 融合方式 + 通道数 + Norm 类型)。

---

## 1. B1 目标重新定义

**原则**: 严格"加深"实验 = 只改 1 个变量(num_layers),其他全等于 baseline。

**B1 目标**: 把 `AttentionDecoder` 的 `num_layers` 从 3 改成 **4**(单点),其他全部沿用 baseline-5090 的 architecture。这样 B1 vs baseline 的差距能 100% 归因到"层数"。

**为什么不扫 6/8**:depth=3 是上游已调过的 default(不是随便选的),加 1 层是最小、最可解释的改动。如果 depth=4 都没有显著差异,后续重点应该转去 **decoder 异质对比** (B3 Pyramid / B7 SAM-style),而不是继续扫同一 architecture 的层数。深扫层数是收益递减的工作。

**不在 B1 范围内**(留给后续实验):
- B1' (Pre-LN ablation): 把 Post-LN 改 Pre-LN 看训练稳定性
- B1'' (activation ablation): ReLU → GELU
- B1''' (融合机制): TILE+ADD → bmm-attention-weighted
- B3 (Pyramid decoder): 完全不同的 architecture
- B7 (SAM-style): 同上

---

## 2. 实施方案 — B1c (MINIMAL + 轻量 dispatch 钩子)

### 三个候选

| 方案 | 改动量 | 严格可比 | 后续 B3/B7 复用度 | 评分 |
|---|---|---|---|---|
| B1a 直接改 `AttentionDecoder` 默认值 | ~5 行 | ✓ | ✗(后续要全推倒) | 短视 |
| B1b 按 reference 重构 `model/decoders/` | ~250 行 + 新建 4 个文件 | ✗(同时改 architecture) | ✓ | 风险大 |
| **B1c 最小 plumbing + 轻量 dispatch 钩子** ⭐ | ~30 行 + 0 个新文件 | ✓ | ✓(后续 B3 加 elif 分支即可) | **采用** |

### B1c 核心思想

- 加 `--decoder_depth=N` CLI(B1 立即用)
- 加 `--decoder_type="cross_attn"` CLI(占位字符串,**不用 argparse choices= 限制**,B3 时直接传 `--decoder_type=pyramid`,代码侧用 `NotImplementedError` 兜底)
- `AttentionDecoder.__init__` 已经有 `num_layers` 参数(L731),只需 plumbing
- `AttnMetaModel.initialize_attn_modules` 加 1 个 if/elif 分支
- **不新建 `model/decoders/` 文件夹**(B3 来时再决定是否重构)

---

## 3. 改动清单(code-level,不是伪代码)

**4 处文件改动,~30 行新增,~0 行删除。**

### 改动 1: `train_ds.py:90-104` 加 CLI 参数

**Before** (L98-104):
```python
    parser.add_argument("--out_dim", default=1024, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_attn_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
```

**After** (在 `--out_dim` 之后,`--resume` 之前插入 2 行,**不加 `choices=`**):
```python
    parser.add_argument("--out_dim", default=1024, type=int)
    parser.add_argument("--decoder_type", default="cross_attn", type=str,
                        help="Attention decoder type. Currently only 'cross_attn' is implemented; "
                             "future variants ('pyramid', 'sam_style', ...) will be added incrementally.")
    parser.add_argument("--decoder_depth", default=3, type=int,
                        help="Number of cross-attention layers in attn_decoder. Baseline=3, B1=4.")
    parser.add_argument("--resume", default="", type=str)
    ...
```

注意:
- `default=3` 跟 baseline 一致,任何不传 `--decoder_depth` 的旧命令行为不变
- `--decoder_type` **不用 argparse choices=**,允许任意字符串。dispatch 在模型代码里用 `NotImplementedError` 拒绝未实现项 → 错误信息更明确,且不需要每加一个 decoder 同步改 argparse

### 改动 2: `train_ds.py:165-175` 把新参数传进 `model_args`

**Before** (L165-175):
```python
    model_args = {
        "train_attn_decoder": args.train_attn_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        ...
        "use_mm_start_end": args.use_mm_start_end,
    }
```

**After** (在 `out_dim` 后插 2 行):
```python
    model_args = {
        "train_attn_decoder": args.train_attn_decoder,
        "out_dim": args.out_dim,
        "decoder_type": args.decoder_type,        # 新增
        "decoder_depth": args.decoder_depth,      # 新增
        "ce_loss_weight": args.ce_loss_weight,
        ...
    }
```

### 改动 3: `model/Attn_model.py:344-360` AttnMetaModel.\_\_init\_\_ stash 新字段

**Before** (L344-359):
```python
class AttnMetaModel:
    def __init__(self, config, **kwargs):
        super(AttnMetaModel, self).__init__(config)
        self.config = config
        if not hasattr(self.config, "train_attn_decoder"):
            self.config.train_attn_decoder = kwargs["train_attn_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_attn_modules(self.config)
```

**After** (在 `out_dim` stash 后加 2 行,**显式标注 default 必须跟 baseline 一致**):
```python
class AttnMetaModel:
    def __init__(self, config, **kwargs):
        super(AttnMetaModel, self).__init__(config)
        self.config = config
        if not hasattr(self.config, "train_attn_decoder"):
            self.config.train_attn_decoder = kwargs["train_attn_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            # ── B1 新增:decoder dispatch 配置 ──
            # CRITICAL: default 值 ("cross_attn", 3) 必须与 baseline-5090 训练时的
            # AttentionDecoder() 默认架构一致(num_layers=3, embed_dim=1024,
            # num_heads=8, conv decoder 1024→512→256→64→32→1)。
            # 否则:
            #   - 加载 baseline ckpt 但不传 --decoder_depth → 老 ckpt 在 depth=3 训的,
            #     如果 default 改成别的值,加载时会建出不同层数的 decoder → 静默架构错配
            #   - shape 对得上时 ckpt 能 load,但权重错位 → 数字漂移
            #   - shape 对不上时直接 size mismatch error
            # 增加新 decoder 类型时,这两个 default 永远不动。default 只描述 baseline。
            self.config.decoder_type = kwargs.get("decoder_type", "cross_attn")
            self.config.decoder_depth = kwargs.get("decoder_depth", 3)
            # ──────────────────────────────────
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_attn_modules(self.config)
```

注意:
- `kwargs.get("decoder_depth", 3)`:**第一次创建模型时**(`hasattr(config, "train_attn_decoder")` 为 False),从 `kwargs`(来自 [train_ds.py](train_ds.py) 的 `model_args`)取值,缺省 3
- `getattr(config, "decoder_depth", 3)` 在 [§3 改动 4 的 `initialize_attn_modules`](#改动-4-modelattn_modelpy361-380-initialize_attn_modules-dispatch-notimplementederror-fallback) 里读 config 也用 default=3 兜底,**两处 default 必须保持同步**
- 加载老 ckpt(commit 6a290dc 之前训的 baseline-5090)时,config 里没 `decoder_type` / `decoder_depth` 字段,两处 `getattr`/`kwargs.get` 都自动兜底到 baseline 值 → 加载行为不变 ✓

### 改动 4: `model/Attn_model.py:361-380` initialize_attn_modules dispatch (NotImplementedError fallback)

**Before** (L361-380):
```python
    def initialize_attn_modules(self, config):
        # attention prediction decode
        self.attn_decoder = AttentionDecoder()
        if config.train_attn_decoder:
            self.attn_decoder.train()
            for param in self.attn_decoder.parameters():
                param.requires_grad = True
        # Projection layer
        in_dim = config.hidden_size     # 4096
        out_dim = 1024        # 256
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

**After** (改 1 行 `self.attn_decoder = AttentionDecoder()` 为 dispatch,用 `NotImplementedError`):
```python
    def initialize_attn_modules(self, config):
        # attention prediction decoder (dispatch by decoder_type)
        decoder_type = getattr(config, "decoder_type", "cross_attn")
        decoder_depth = getattr(config, "decoder_depth", 3)
        if decoder_type == "cross_attn":
            self.attn_decoder = AttentionDecoder(num_layers=decoder_depth)
        # elif decoder_type == "pyramid":     # B3 实验时启用
        #     self.attn_decoder = PyramidDecoder(...)
        # elif decoder_type == "sam_style":   # B7 实验时启用
        #     self.attn_decoder = SAMStyleDecoder(...)
        else:
            raise NotImplementedError(
                f"decoder_type='{decoder_type}' is not implemented yet. "
                f"Currently supported: ['cross_attn']. "
                f"To add a new decoder, register a class in this file and add an "
                f"elif branch here. See plans/B1-implementation-plan.md."
            )
        if config.train_attn_decoder:
            self.attn_decoder.train()
            for param in self.attn_decoder.parameters():
                param.requires_grad = True
        # Projection layer (unchanged)
        ...
```

`AttentionDecoder.__init__` 真实签名 (L731):
```python
def __init__(self, embed_dim=1024, num_heads=8, num_layers=3, output_layer=6):
```
传 `num_layers=decoder_depth` 即可,其他参数沿用 default。

为什么用 `NotImplementedError` 而不是 `ValueError`:
- 语义更准确(代码框架支持,只是某些分支还没实现)
- 错误信息直接告知 reader 怎么扩展 → self-documenting
- 不依赖 argparse `choices=` 维护清单(单一信息源原则)

---

## 4. 参数冻结自动验证(无需新加,but 必须确认)

`train_ds.py:209-229` 的 `find_linear_layers` LoRA 排除列表已含 `"attn_decoder"`(L219),所以即使我们改 num_layers,新加的 cross_attention_layers.{N}.attention.in_proj_weight 等 Linear 层名都包含 `attn_decoder` 前缀,**自动被排除**。

`train_ds.py:250-258` 的解冻循环判断也是 `"attn_decoder" in n`,**自动包括**新加的层。

**无需改动这两处**。

---

## 5. 显存预算

| depth | cross_attn 层数 | attn_decoder 参数量(估算) | 额外显存 vs baseline |
|---|---|---|---|
| 3 (baseline) | 3 | ~47M (3×12.4M cross + 6M conv + 4M visual_proj) | 0 |
| **4 (B1)** | **4** | **~59M** | **+0.4 GB** |
| 6 | 6 | ~84M | +1.2 GB(本次 B1 不跑) |
| 8 | 8 | ~109M | +2.0 GB(本次 B1 不跑) |

baseline-5090 显存 ~17 GB(从 nvidia-smi 观察),32 GB 显存对 depth=4 绰绰有余。`batch=1, image_size=1024` 不变。

每层估算 ≈ MultiheadAttention(4×1024² ≈ 4M) + 2 LayerNorm(2K) + MLP(2×1024×4096 ≈ 8.4M) ≈ **12.4M / 层**。

---

## 6. Smoke test 命令 + 3 个必查 assert

### Smoke 命令

跟 baseline smoke 同模板,加 `--decoder_depth=4`:

```bash
deepspeed --num_gpus=1 --master_port=26000 train_ds.py \
  --version="./weights/LLaVA-7B-Lightening-v1-1" \
  --vision-tower="./weights/clip-vit-large-patch14" \
  --dataset_dir="./dataset" --log_base_dir="./runs_smoke" \
  --dataset="BDDA" --train_sample_rates="1" \
  --val_dataset="BDDA" --val_sample_rates="1" \
  --exp_name="smoke_B1_depth4_$(date +%s)" \
  --epochs=1 --steps_per_epoch=2 \
  --batch_size=1 --grad_accumulation_steps=1 \
  --val_samples_num=4 --val_batch_size=1 \
  --precision=bf16 \
  --decoder_type=cross_attn --decoder_depth=4
```

预计耗时 5 分钟以内。

### 3 个必查 assert(临时插到 [train_ds.py](train_ds.py) 内)

在 `train_ds.py::main()` 里 `model.get_model().initialize_attn_modules(...)` 调用之后(约 L195 之后)、训练循环开始之前,临时插入以下 sanity check 代码块。**smoke 通过后立即删除这段(保持 train_ds.py 干净)**:

```python
# ── B1 SMOKE-TEST SANITY CHECK (TEMPORARY, REMOVE AFTER VERIFICATION) ─────
if args.local_rank == 0:
    # Assert 1: cross_attention_layers 数量正确(dispatch + plumbing 工作)
    n_layers = len(model.get_model().attn_decoder.cross_attention_layers)
    assert n_layers == args.decoder_depth, (
        f"[B1 SMOKE FAIL] Expected {args.decoder_depth} cross_attention_layers, "
        f"got {n_layers}. Plumbing path: CLI → model_args → AttnMetaModel.config "
        f"→ initialize_attn_modules → AttentionDecoder(num_layers=...)"
    )
    print(f"[B1 SMOKE OK] cross_attention_layers count: {n_layers}")

    # Assert 2: attn_decoder 参数量在合理区间
    # baseline (depth=3): ~47M  →  B1 (depth=4): ~59M
    # 加 1 层 ≈ +12.4M, 区间放宽到 ±3M 容忍 estimate 误差
    decoder_params = sum(p.numel() for p in model.get_model().attn_decoder.parameters())
    expected_min = 47_000_000 + 9_000_000   # depth=4 至少加 9M
    expected_max = 47_000_000 + 16_000_000  # depth=4 至多加 16M
    assert expected_min <= decoder_params <= expected_max, (
        f"[B1 SMOKE FAIL] attn_decoder params {decoder_params:,} out of expected "
        f"range [{expected_min:,}, {expected_max:,}] for depth=4"
    )
    print(f"[B1 SMOKE OK] attn_decoder params: {decoder_params:,} "
          f"(baseline ~47M, B1 expected ~59M)")
# ────────────────────────────────────────────────────────────────────────
```

并在 `model_forward` 第一次 forward 后立即检查(可加在 [train_ds.py](train_ds.py) 训练循环 first-step 之后,smoke 完毕删除):

```python
# Assert 3: pred_sal shape + attn_loss finite
# (output_dict 已经从 model_forward 返回, 走的是 train.py 的 train() loop)
# 在 train() 里第一次 model() 调用后:
assert output_dict["pred_sal"].shape == (1, 1, 256, 256), (
    f"[B1 SMOKE FAIL] pred_sal shape is {output_dict['pred_sal'].shape}, "
    f"expected (1, 1, 256, 256)"
)
attn_loss_val = output_dict["attn_loss"].item()
assert torch.isfinite(output_dict["attn_loss"]), (
    f"[B1 SMOKE FAIL] attn_loss is not finite: {attn_loss_val}"
)
print(f"[B1 SMOKE OK] pred_sal shape OK, attn_loss = {attn_loss_val:.4f}")
```

实际插入位置等到执行时由用户指定(可能 train_ds.py 训练循环 in [train_ds.py](train_ds.py) 的具体位置,需读相关行号),但**三个 assert 的内容和阈值是固定的**:

1. **Assert 1**: `len(cross_attention_layers) == decoder_depth`(验证 dispatch + plumbing)
2. **Assert 2**: `attn_decoder` 总参数量 `∈ [56M, 63M]`(验证模块大小符合估算)
3. **Assert 3**: `pred_sal.shape == (1, 1, 256, 256)` 且 `attn_loss` 有限(验证 forward 路径)

### Smoke 通过的判定

- 看到 3 行 `[B1 SMOKE OK]` 输出
- 出现 2 次 `Epoch: [0]` 训练日志
- validate 最后打印 `cc`, `kld` 等非 NaN 数值
- Process exits successfully

---

## 6.5. Backward-compatibility check(smoke 通过后必跑,全训前最后一道闸)

**目标**: 用 baseline-5090 ckpt + 新代码(默认参数,不传 `--decoder_depth`)跑 BDDA 100 样本 eval,跟 baseline 锚点的对应 100 样本对比,确认**新代码加载老 ckpt 行为完全等价于旧代码**。这是 §3 改动 3 default 一致性的运行时验证。

### 命令

```bash
deepspeed --num_gpus=1 --master_port=24999 train_ds.py \
  --version=./ckpts/ATTN-7B-baseline-5090 \
  --vision-tower=./weights/clip-vit-large-patch14 \
  --dataset_dir=./dataset --log_base_dir=./runs_eval \
  --val_dataset=BDDA --val_sample_rates=1 \
  --exp_name=baseline-5090-bcompat-check \
  --batch_size=1 --val_batch_size=1 --precision=bf16 \
  --epochs=1 --steps_per_epoch=1 \
  --val_samples_num=100 \
  --eval_only \
  2>&1 | tee runs_eval/bcompat-check.log
```

**关键**: **不传 `--decoder_depth` 也不传 `--decoder_type`**。这测试新代码"啥都不传"的默认行为,对应 §3 改动 3 的 default 兜底路径。如果 default 设错(比如改成 4),加载老 ckpt 会得到不同架构 → 这一步直接抓出来。

### 时长

~10 分钟(BDDA 100 样本 × ~5.5 s/it ≈ 9 min + 模型加载 1 min)。tmux 起 detached 即可,不需要 attach 看进度。

### 验证(eval 跑完后,纯 CPU)

新 eval 的 sample-level CSV 会写到 `runs_eval/baseline-5090-bcompat-check/text_eval/<timestamp>/attn_metrics_0.csv`(100 行 + header)。锚点参考用 baseline-5090-eval-BDDA(2026-04-26 01:47 完成)的 CSV 前 100 行,这是同一 ckpt 在 baseline 代码下的输出。

```python
# bcompat 验证脚本 (CPU only, 跑完 eval 后手动跑)
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python3.11 -c "
import pandas as pd, os
new_dir = 'runs_eval/baseline-5090-bcompat-check/text_eval/'
new = pd.read_csv(new_dir + os.listdir(new_dir)[0] + '/attn_metrics_0.csv').head(100)
old = pd.read_csv('runs_eval/baseline-5090-eval-BDDA/text_eval/2026_04_26_01_47/attn_metrics_0.csv').head(100)
assert len(new) == len(old) == 100, f'sample count mismatch: new={len(new)} old={len(old)}'
print(f'compared {len(new)} samples')
max_diff_overall = 0
for c in ['cc','kld','sim','nss','auc_b','auc_j']:
    diff = (new[c] - old[c]).abs().dropna().max()
    flag = 'PASS' if diff < 1e-5 else ('INVESTIGATE' if diff < 1e-3 else 'ABORT')
    print(f'{c.upper():6s}: max abs diff = {diff:.2e}  [{flag}]')
    max_diff_overall = max(max_diff_overall, diff)
print()
if max_diff_overall < 1e-5:
    print('==> bcompat PASS, can proceed to §7 full training')
elif max_diff_overall < 1e-3:
    print('==> bcompat INVESTIGATE: small drift, check rng/dropout/eval mode')
else:
    print('==> bcompat ABORT: significant drift, default mismatch detected')
"
```

### 判定阈值

| `max abs diff` | 判定 | 行动 |
|---|---|---|
| `< 1e-5` | ✅ PASS | 浮点噪声范围,新代码加载老 ckpt 等价于旧代码,可进 §7 全训 |
| `1e-5 ~ 1e-3` | ⚠ INVESTIGATE | 小漂移,可能是 rng / dropout 状态 / eval mode 差异。停一停查清楚再说,**不能直接进全训** |
| `> 1e-3` | ❌ ABORT | 显著漂移 → 默认值或 dispatch 路径有 bug,**禁止进全量训练**。回 §3 review 改动是否正确,可能漏改一处 |

### 为什么这一步不能省

- §6 smoke 用的是新 architecture (depth=4),改了的部分确实工作 ✓
- 但 default 路径(不传任何 `--decoder_*` flag)是否依然兼容老 ckpt,smoke 没测 —— **这正是 backward compat 的核心**
- 如果将来有人 resume baseline-5090 但忘传 flag,这一步是唯一防线

§6.5 完成判定为 PASS 后才进 §7。

---

## 7. 全量训练命令(smoke + bcompat 都通过后)

跟 baseline-5090 训练命令**完全相同**,只换 `exp_name` 和加 `--decoder_depth`:

```bash
deepspeed --num_gpus=1 --master_port=24999 train_ds.py \
  --version=./weights/LLaVA-7B-Lightening-v1-1 \
  --vision-tower=./weights/clip-vit-large-patch14 \
  --dataset_dir=./dataset --log_base_dir=./runs \
  --dataset=BDDA||DReyeVE||LBW||DADA --train_sample_rates=8,5,2,7 \
  --val_dataset=BDDA --val_sample_rates=1 \
  --exp_name=decoder-B1-cross-depth4-5090 \
  --epochs=10 --steps_per_epoch=500 \
  --batch_size=1 --grad_accumulation_steps=16 --image_size=1024 \
  --precision=bf16 --val_samples_num=200 --val_batch_size=1 --lr=0.0003 \
  --decoder_type=cross_attn --decoder_depth=4
```

**单点跑** —— 不并行,不串其他 depth。完成后看效果再决定下一步(见 §10 决策树)。

---

## 8. 预期对比基线 baseline-5090

baseline-5090 epoch 9 (2026-04-25 05:15) BDDA val:
- KLD=1.183, CC=0.607, SIM=0.443, NSS=5.694, AUC_B=0.923, AUC_J=0.955

**B1-depth4 期望**(修订为放宽区间):
- **CC: -0.005 ~ +0.025**(depth=3 是上游已调过的 default,加 1 层未必单调好;允许微降是合理预期)
- KLD: -0.10 ~ +0.05
- 训练时长跟 baseline 类似(80~100h)—— 加深不显著影响 LLM 主体计算
- 显存: +0.4 GB(在 32 GB 内)

**判定标准**:
- ✅ **强阳性**: CC ≥ +0.015 → "deeper" 真有帮助,可考虑后续 depth=6/8 扫描
- 🟢 **弱阳性**: CC ∈ [+0.005, +0.015] → 略好,但收益小,**优先转 B3/B7 异质对比**
- 🟡 **中性**: CC ∈ [-0.005, +0.005] → 没差异,**论文里写 "depth 不是关键变量",转 B3/B7**
- 🔴 **微降**: CC ∈ [-0.025, -0.005] → 加深反而过拟合或训练不稳,**论文写 "B1 confirms depth=3 is near optimal for this architecture"**,转 B3/B7
- ❌ **强阴性 / NaN**: 训不起来 → 看 §11 失败回退

---

## 9. 实施流程(等用户授权后)

1. `git checkout -b decoder/B1-cross-depth4`
2. 改动 1+2(train_ds.py CLI + model_args),`git diff` 看,小 commit:`Add --decoder_type / --decoder_depth CLI args`
3. 改动 3(AttnMetaModel.__init__ stash + 注释),`git diff` 看,小 commit:`Stash decoder_type / decoder_depth on config`
4. 改动 4(initialize_attn_modules dispatch with NotImplementedError),`git diff` 看,小 commit:`Dispatch attn_decoder by decoder_type`
5. 临时插入 3 个 smoke assert(temporary)
6. **Smoke test (§6) — 等用户授权后跑** (5 min,新 architecture 验证)
7. 通过 → 删 smoke assert,小 commit:`Remove B1 smoke-test asserts after verification`
8. **§6.5 Backward-compat check — 等用户授权后跑** (10 min,baseline ckpt + default 参数,验证向后兼容)
9. bcompat 跑完用 CSV 验证脚本对比,判定 PASS / INVESTIGATE / ABORT
10. PASS → 起 depth=4 全量训练(deepspeed 命令在 §7,**再次**等用户明确说"跑")
11. 训练时长 80-100h,完成后跑 4 子集 eval(参考 Phase 1.5 流程,4 个单 DS deepspeed 命令依次起,**每次等用户授权**)
12. 跑 [collect_main_results.py](scripts/paper/collect_main_results.py) 出 B1 vs baseline 对比表
13. 按 §8 判定标准走决策树(见 §14)

每步小 commit + smoke + bcompat 双闸 = 失败时 `git reset --hard <ref>` 干净回退。

---

## 10. 不在 B1 范围内的事(防 scope creep)

- ❌ 不动 [utils/dataset.py](utils/dataset.py) 数据 pipeline
- ❌ 不动 [model/llava/](model/llava/) LLaVA 基座
- ❌ 不动 loss 组合
- ❌ 不动 [train_ds.py](train_ds.py) 的 ckpt 命名规则
- ❌ 不动 GaussianBlur(11, 2) + Resize 后处理(decoder 外)
- ❌ 不并行做 B3/B7
- ❌ 不重训 baseline(锚点已固化在 commit 6a290dc)
- ❌ 不调 `train_sample_rates`(留给 Q4 单独实验)
- ❌ **本次 B1 不扫 depth=6/8**(只 depth=4 单点)

---

## 11. 失败回退方案

| 失败模式 | 诊断 | 修复 |
|---|---|---|
| smoke 启动崩 (import 错) | 改动 3/4 拼写错 | `git checkout -- model/Attn_model.py` 撤回,重做 |
| smoke 跑通但 Assert 1 fail | `decoder_depth` 没传到 `AttentionDecoder` | 检查 model_args 是否真的进了 config(改动 2/3 任一处错位) |
| smoke 跑通但 Assert 2 fail | 参数量异常(可能 visual_proj 重复创建) | 看 `print_trainable_parameters` 看是否双倍 attn_decoder |
| smoke `Assert 3` shape 不对 | 不应该(num_layers 不影响输出 shape) | 看 forward 跟 baseline 有没有别的 diff |
| 全训 NaN | LR 太高 for 深层 | `--lr=1e-4` 试试(原 3e-4) |
| 全训 OOM (不该发生 at depth=4) | 别的进程在 | 检查 GPU 占用 |
| 全训 hang | 分布式不一致 | 看是否有 if/else 分支跨 rank 行为不同 |

**Branch 隔离**: 在 `decoder/B1-cross-depth4` 分支上做。如果完全失败:
```bash
git checkout main
git branch -D decoder/B1-cross-depth4
```
不影响 main 的 baseline 锚点。

---

## 12. 后续扩展(B3/B7 怎么演进)

B1 完成后,B3 (Pyramid) 时:
1. 在 `model/Attn_model.py` 加 `class PyramidDecoder(nn.Module): ...`(或新建 `model/decoders/pyramid.py`,届时再决定是否重构)
2. 在 `initialize_attn_modules` dispatch 加 `elif decoder_type == "pyramid": ...`
3. 把 §3 改动 4 的注释行(`# elif decoder_type == "pyramid": ...`)取消注释填实
4. 其他 plumbing 不变;CLI 不用改(`--decoder_type=pyramid` 直接通过)

**B1c 的轻量 dispatch 钩子让 B3 改动量降到 ~50 行**(不是 200+)。

---

## 13. Phase 1 后续路径预览

(B1 完成后视情况调整,这里仅为参考路线图)

```
B1 (depth=4)
   │
   ├── 强阳性 (CC ≥ +0.015) ──→ 可选: 扫 depth=6/8 进一步探边界
   │                              然后转 B3 (主线)
   │
   └── 弱/中性/微降 (其余) ────→ 直接转 B3 Pyramid (主线)
                                    │
                                    └── B3 完成后转 B7 SAM-style
                                         │
                                         └── 进 paper-writing skill 出投稿物
```

预算: B3 = 1 周(参考 .claude/skills/modify-llada/references/decoders/B3-pyramid.md),B7 = 1 周(B7-sam-style.md),投稿物 = 2 周。**总 4-6 周到投稿草稿**(B1 这周 + 上面 3 周)。

---

## 14. Phase 1 决策树

每个分叉点用户决策,Claude 不擅自行动。

```
                       ┌─ 草案 v2 review ────┐
                       │   (本文档)           │
                       └────────┬─────────────┘
                                │
                  ┌─────────────┼─────────────┐
                  │             │             │
              [A 接受]      [B 修订]      [C 拒绝]
                  │             │             │
                  ▼             ▼             ▼
             实施 §3-§4    回到 §3-§4    其他方向
             小 commit     调整草案        (B3? Q4?)
                  │
                  ▼
            插入 §6 临时 assert
                  │
                  ▼
         [启动 smoke test? 等用户授权]
                  │
            ┌─────┴─────┐
          通过           失败
            │             │
            ▼             ▼
       删 assert      §11 回退
       小 commit      或修代码再 smoke
            │
            ▼
   [§6.5 bcompat check? 等用户授权]
   (baseline ckpt + 默认参数, 100 样本, ~10min)
            │
   ┌────────┼────────┬─────────┐
 PASS   INVESTIGATE  ABORT
(<1e-5) (1e-5~1e-3) (>1e-3)
   │         │         │
   ▼         ▼         ▼
 进全训   查清楚    回 §3
         再决定    review
                   修 default
            │
            ▼
   [启动全量训练? 等用户授权]
            │
       80-100h
            │
            ▼
        训练完成
            │
            ▼
   [起 4 子集 eval queue?
    每个 DS 独立等用户授权]
            │
            ▼
   collect_main_results.py
            │
            ▼
   B1-depth4 vs baseline 对比 (§8 标准)
            │
   ┌────────┼────────┬──────────┐
   ▼        ▼        ▼          ▼
强阳性    弱阳性   中性/微降   强阴性
(≥+0.015) (+0.005~  (-0.005~   (NaN/
          +0.015)   +0.005 / 训练崩)
                    -0.025~
                    -0.005)
   │        │        │          │
   ▼        ▼        ▼          ▼
可选 depth  转 B3   转 B3       §11 排查
6/8 扫描   主线     主线        是否真的是
否则转 B3                       架构问题, 再
主线                            决定继续/转向
```

**决策点都在用户手里**,Claude 在每个分叉点等明确指示。

---

## Reviewer 关注点 (v3 修订状态)

| # | 问题 | v1 | v2 修订 | v3 修订 |
|---|---|---|---|---|
| 1 | B1c 方案选对了吗? | 推荐 | 保留 | 保留 |
| 2 | `--decoder_type` default 占位 | 推荐 | 去 argparse choices=,用 NotImplementedError ✓ | 保留 |
| 3 | smoke test 方案够不够? | 6 项 | 加 3 个具体 assert ✓ | 保留 |
| 4 | depth 扫 4/6/8 还是单点? | 扫 | 只 depth=4 单点 ✓ | 保留 |
| 5 | 预期 +0.01~0.03 CC 对吗? | +0.01~0.03 | -0.005 ~ +0.025(允许微降) ✓ | 保留 |
| 6 | 失败回退机制充分吗? | 6 行表 | 加 Assert 1/2 fail 诊断 | **加 bcompat 三档判定** ✓ |
| 7 | 后续 B3/B7 复用度? | B1c 钩子 | §12 + §13 路径预览 | 保留 |
| 8 | default 一致性如何防错? | 未涉及 | 未涉及 | **§3 改动 3 显式注释 + §6.5 运行时 bcompat check 双闸** ✓ |

**v3 新增**:
- §3 改动 3 加 6 行注释解释 default 必须跟 baseline 一致(否则加载老 ckpt 架构错配)
- §6.5 Backward-compat check (~10 min eval,baseline ckpt + default 参数,与锚点 CSV 前 100 样本对比,<1e-5 PASS / 1e-5~1e-3 INVESTIGATE / >1e-3 ABORT)
- §9 实施流程加 §6.5 步骤,从 11 步扩到 13 步
- §14 决策树加 bcompat 分叉点
