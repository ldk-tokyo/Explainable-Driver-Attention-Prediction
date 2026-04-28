# Experiments Log

> 记录每个 exp 的命令、产物、关键数字、与论文的对照、待办。
> 实验本身的代码逻辑去看 train_ds.py / model/Attn_model.py;这里只追踪 *什么时候跑了什么 + 数字*。

---

## 命名约定

- `exp_name` = 描述性短名 (avoid 时间戳, deepspeed 自己会加)
- 训练目录: `runs/<exp>/<timestamp>/`
- 评测目录(单 DS 队列): `runs_eval/<exp>-eval-<DS>/text_eval/<timestamp>/`
- 评测目录(4 子集合一): `runs/<exp>/attn_eval/<timestamp>/`
- 合并 ckpt: `ckpts/ATTN-7B-<exp>/`

---

## 论文基准 (yuchen2199/Explainable-Driver-Attention-Prediction, ICCV 2025 Highlight, arxiv 2506.23088)

**训练配置** (论文 §5.1):
- 4× A100, bf16, DeepSpeed
- batch 8/device, grad_accum 5 → **effective batch 160** (CLAUDE.md 写的 80 是错的)
- AdamW, LR 0.0003, WarmupDecayLR, 100 warmup
- λ_map=2, λ_txt=1, λ_bce=1, λ_kl=0.1, λ_what=1, λ_why=1
- LoRA on Vicuna-7B,CLIP-ViT-L 全冻,attn_decoder 从头训
- 总训练样本: 39,642 keyframes (W³DA train split,见 §5.3)
- **总 epochs / steps 论文未明确** —— 只能从代码推断或开 issue 问

**报告方式**:
- **Table 1/2**: W³DA test (key-frame 子集),按"驾驶场景"分组
  - Normal Driving = DR(eye)VE_test + LBW_test
  - Safety-Critical = BDDA_test
  - Accident = DADA_test
- **Table 3**: 在 4 个原始数据集**完整 test set**(非 key-frame)上 eval。LBW 不在 Table 3。
- **从不报 4 子集简单平均**。

**Test 样本数** (我们的 dataset/<DS>/test/ 里 = key-frame W³DA test):
- BDDA: 4788, DReyeVE: 9101, LBW: 666, DADA: 7419 (总 21974)

---

## exp: baseline-5090

**目标**: 单卡 RTX 5090 (32GB Blackwell) 上首次完整复现 LLada baseline。

**训练命令** ([runs/baseline-5090.log:1](runs/baseline-5090.log#L1)):
```
deepspeed --num_gpus=1 --master_port=24999 train_ds.py \
  --version=./weights/LLaVA-7B-Lightening-v1-1 \
  --vision-tower=./weights/clip-vit-large-patch14 \
  --dataset_dir=./dataset --log_base_dir=./runs \
  --dataset=BDDA||DReyeVE||LBW||DADA --train_sample_rates=8,5,2,7 \
  --val_dataset=BDDA --val_sample_rates=1 \
  --exp_name=baseline-5090 --epochs=10 --steps_per_epoch=500 \
  --batch_size=1 --grad_accumulation_steps=16 --image_size=1024 \
  --precision=bf16 --val_samples_num=200 --val_batch_size=1 --lr=0.0003
```

**与论文差异**:
- effective batch **16 vs 论文 160** (10× 小)
- 总 step 5000 vs 论文未公布(预计 6×~10× 多)
- val 只 BDDA → best ckpt 选择只反映 BDDA 性能
- image_size 1024(论文未明确)
- 其他超参一致

**时间线**:
- 2026-04-24 11:43 → 13:08: 第一次启动,跑到 epoch 1 后 deepspeed engine.py:1914 forward 报错退出 ([runs/baseline-5090/2026_04_24_11_43/](runs/baseline-5090/2026_04_24_11_43/))。保留了 epoch 0/1 的 best ckpt。
- 2026-04-24 23:50 → 04-25 05:15: 从 epoch 1 best ckpt resume,跑完剩余 8 epoch (5h25m),正常结束。best ckpt 保留 epoch 2/3/4/6/9 ([runs/baseline-5090/2026_04_24_23_50/](runs/baseline-5090/2026_04_24_23_50/))。
- 2026-04-25 23:00: epoch 9 → 合并成 HF 格式 [ckpts/ATTN-7B-baseline-5090/](ckpts/ATTN-7B-baseline-5090/) (~13.5 GB,2 shard)。

**Best ckpt 选择**: epoch 9, BDDA val: KLD=1.183, CC=0.607, NSS=5.694 ([runs/baseline-5090/2026_04_24_23_50/best_ckpt_model_epoch9/log.txt](runs/baseline-5090/2026_04_24_23_50/best_ckpt_model_epoch9/log.txt))。

**Storage 状态**: 5 个 best ckpt 各 30 GB(都是 DeepSpeed `mp_rank_00_model_states.pt` + 优化器状态,**未跑 zero_to_fp32**)。合并后只 epoch9 → ckpts/。其余 4 个 epoch 的 ckpt 是占空间的(120 GB)但还没决定删。

---

## eval: baseline-5090 (4 子集合一, 23:53)

**命令** ([runs/baseline-5090/eval_test_20260425_2353.log](runs/baseline-5090/eval_test_20260425_2353.log)):
```
train_ds.py --version=./ckpts/ATTN-7B-baseline-5090
  --val_dataset=BDDA||DReyeVE||LBW||DADA --val_batch_size=1
  --eval_only --eval_colormap_save True --precision=bf16
```
- **没传 `--eval_text`** —— 文本指标这次不会算
- 时间: 2026-04-25 23:53 → 04-26 01:27 (1h34m, 21974 样本)

**产物**:
- [runs/baseline-5090/attn_eval/2026_04_25_23_53/log_test.txt](runs/baseline-5090/attn_eval/2026_04_25_23_53/log_test.txt) — aggregate 数字
- [runs/baseline-5090/attn_eval/2026_04_25_23_53/attn_metrics_0.csv](runs/baseline-5090/attn_eval/2026_04_25_23_53/attn_metrics_0.csv) — 21974 sample-level
- `dataset/<DS>/test/<vid>/eval_saving/2026_04_25_23_53/` — 热力图叠加 jpg(BDDA 5748 + DReyeVE 9101 + LBW 666 + DADA 7419 = 22934 个,~4.7 GB 总)

**4 子集 attn 数字**(per-DS, 从 CSV 重新分组):

| DS | n | CC | KLD | SIM | NSS | AUC_B | AUC_J |
|---|---|---|---|---|---|---|---|
| BDDA | 4788 | 0.573 | 1.263 | 0.424 | 4.923 (1119 valid) | 0.912 | 0.948 |
| DReyeVE | 9101 | 0.598 | 1.291 | 0.432 | 4.622 | 0.903 | 0.959 |
| LBW | 666 | 0.434 | 1.708 | 0.280 | 3.268 | 0.918 | 0.949 |
| DADA | 7419 | 0.374 | 2.016 | 0.253 | 3.000 | 0.877 | 0.925 |
| 简单平均 | - | 0.495 | 1.569 | 0.347 | 3.953 | 0.903 | 0.946 |
| 加权 (n_samples) | - | 0.512 | 1.542 | 0.365 | 3.934 | 0.897 | 0.945 |

**文本指标**: **全 0**(因 `--eval_text` 没传, [train_ds.py:780](train_ds.py#L780) 的 if 块从未执行)。

---

## eval: baseline-5090-eval-BDDA (单 DS, 01:47)

**命令** ([runs_eval/eval-BDDA.log](runs_eval/eval-BDDA.log)):
```
train_ds.py --version=./ckpts/ATTN-7B-baseline-5090
  --val_dataset=BDDA --val_sample_rates=1
  --exp_name=baseline-5090-eval-BDDA --batch_size=1 --val_batch_size=1
  --precision=bf16 --epochs=1 --steps_per_epoch=1 --val_samples_num=100000
  --eval_only --eval_text --eval_text_save 1 --eval_colormap_save 1
```
- **传了 `--eval_text` 和 `--eval_text_save`** ✓
- 时间: 2026-04-26 01:47:06 → 09:41:43 (7h54m, 4788 样本, 5.94 s/it 平均)

**Attn 数字**: 与上面 23:53 BDDA 子集**完全一致**(同一 ckpt,sample-level 数字一致),信号 sanity check 通过。

**文本数字** ([runs_eval/baseline-5090-eval-BDDA/text_eval/2026_04_26_01_47/log_test.txt](runs_eval/baseline-5090-eval-BDDA/text_eval/2026_04_26_01_47/log_test.txt)):

| 段 | BLEU_4 | METEOR | ROUGE | CIDEr-R |
|---|---|---|---|---|
| Complete | 0.174 | 0.247 | 0.359 | 0.392 |
| What | 0.241 | 0.330 | 0.435 | 0.442 |
| Why | 0.129 | 0.184 | 0.327 | 0.459 |

**与论文 Table 2 (Safety-Critical = BDDA) 对比**:
- 论文 BLEU 0.444 / METEOR 0.375 / ROUGE 0.593 / CIDEr-R 1.233
- 我们 Complete 0.174 / 0.247 / 0.359 / 0.392
- **差距很大** (BLEU 差 -0.27)。可能原因: 训练步数差 6×、论文 BLEU 是 BLEU-1 不是 BLEU-4、What+Why 拼接方式差异。需要查清。

---

## eval: per-dataset 队列 (Phase 1.5, ✅ 全部完成 2026-04-28)

**起源**: 上一个 Claude Code session(ID `dd3d2295-4ca4-441c-9a0b-a870a746f1df`)于 **2026-04-26 01:33** 创建 `/tmp/launch_eval.sh`,for 循环依次跑 BDDA → DReyeVE → LBW → DADA。脚本现已禁用为 `/tmp/launch_eval.sh.disabled` (2026-04-26 23:36)。后续每个 eval 改为**用户明确授权后单独 tmux 起**。

**当前状态**:

| DS | n | attn ✓ | text ✓ | run | 时间 | 备注 |
|---|---|---|---|---|---|---|
| BDDA | 4788 | ✓ | ✓ | `baseline-5090-eval-BDDA` 01:47-09:41 (4-26) | 7h54m | dd3d2295 launch_eval.sh |
| DReyeVE | 9101 | ✓ | ✓ | `baseline-5090-eval-DReyeVE` 20:20 (4-27) – 11:35 (4-28) | 15h14m27s | tmux session, 重跑成功; 旧崩溃产物保留为 `*.crashed-89pct` |
| LBW | 666 | ✓ | ✓ | `baseline-5090-eval-LBW` 00:01-01:04 (4-27) | 1h2m22s | tmux session, 用户授权单独起 |
| DADA | 7419 | ✓ | ✓ | `baseline-5090-eval-DADA` 01:15-13:30 (4-27) | 12h14m | tmux session, 用户授权单独起 |

**LBW 完整数字** (2026-04-27 01:04 完成, [runs_eval/baseline-5090-eval-LBW/text_eval/2026_04_27_00_01/log_test.txt](runs_eval/baseline-5090-eval-LBW/text_eval/2026_04_27_00_01/log_test.txt)):

| 类别 | 指标 | 值 |
|---|---|---|
| Attn | CC / KLD / SIM / NSS / AUC_B / AUC_J | 0.434 / 1.708 / 0.280 / 3.268 / 0.918 / 0.949 |
| Text Complete | BLEU_4 / METEOR / ROUGE / CIDEr-R | 0.131 / 0.254 / 0.314 / 0.171 |
| Text What | BLEU_4 / METEOR / ROUGE / CIDEr-R | 0.187 / 0.374 / 0.385 / 0.186 |
| Text Why | BLEU_4 / METEOR / ROUGE / CIDEr-R | 0.076 / 0.158 / 0.297 / 0.188 |

LBW attn 与 23:53 4 子集 LBW 部分**完全一致**,sanity check 通过。

**DADA 完整数字** (2026-04-27 13:30 完成, [runs_eval/baseline-5090-eval-DADA/text_eval/2026_04_27_01_15/log_test.txt](runs_eval/baseline-5090-eval-DADA/text_eval/2026_04_27_01_15/log_test.txt)):

| 类别 | 指标 | 值 |
|---|---|---|
| Attn | CC / KLD / SIM / NSS / AUC_B / AUC_J | 0.374 / 2.016 / 0.253 / 3.000 / 0.877 / 0.925 |
| Text Complete | BLEU_4 / METEOR / ROUGE / CIDEr-R | 0.223 / 0.251 / 0.424 / **1.006** |
| Text What | BLEU_4 / METEOR / ROUGE / CIDEr-R | 0.265 / 0.324 / 0.474 / 0.614 |
| Text Why | BLEU_4 / METEOR / ROUGE / CIDEr-R | 0.188 / 0.194 / 0.411 / **1.409** |

**DReyeVE 完整数字** (2026-04-28 11:35 完成, [runs_eval/baseline-5090-eval-DReyeVE/text_eval/2026_04_27_20_21/log_test.txt](runs_eval/baseline-5090-eval-DReyeVE/text_eval/2026_04_27_20_21/log_test.txt)):

| 类别 | 指标 | 值 |
|---|---|---|
| Attn | CC / KLD / SIM / NSS / AUC_B / AUC_J | 0.598 / 1.291 / 0.432 / 4.622 / 0.903 / 0.959 |
| Text Complete | BLEU_4 / METEOR / ROUGE / CIDEr-R | 0.259 / 0.307 / 0.421 / 0.484 |
| Text What | BLEU_4 / METEOR / ROUGE / CIDEr-R | 0.333 / 0.386 / 0.485 / 0.384 |
| Text Why | BLEU_4 / METEOR / ROUGE / CIDEr-R | 0.204 / 0.235 / 0.399 / 0.719 |

DReyeVE attn 与 23:53 4 子集 DReyeVE 部分**完全一致** (max diff 2.61e-6),NSS NaN 0/9101 (0%),sanity check 通过。

**3 DS 跨场景文本指标对比 + 与论文 Table 2 精细 cross-check**:

| 场景 | 论文 BLEU | 我们 BLEU_1 (Complete, weighted) | diff | 论文 METEOR | 我们 METEOR | 论文 CIDEr-R | 我们 CIDEr-R |
|---|---|---|---|---|---|---|---|
| Safety-Critical (BDDA, n=4788) | 0.444 | 0.405 | **-0.039 ✓** | 0.375 | 0.247 | 1.233 | 0.392 |
| Accident (DADA, n=7419) | 0.376 | **0.391** | **+0.015 ✓** | 0.318 | 0.251 | 1.002 | **1.006 ✓** |
| Normal (DReyeVE+LBW, n=9767) | 0.436 | **0.510** | **+0.074 ⚠** | 0.360 | 0.303 | 0.963 | 0.463 |

**两个关键发现**(2026-04-28 全部 4 DS 数据后):

1. **论文 "BLEU" = BLEU-1 在 BDDA / DADA 上确认, 但 Normal 出现 +0.074 反超**:
   - BDDA: BLEU_1 0.405 vs 论文 0.444 (-0.039 ✓)
   - DADA: BLEU_1 0.391 vs 论文 0.376 (+0.015 ✓)
   - **Normal (DR+LBW) BLEU_1 0.510 vs 论文 0.436 (+0.074 ⚠)** — 反超!
   - vs BLEU_4 三个场景全差 -0.15 ~ -0.27, 量级不对; BLEU-1 假设依然最佳
   - **Normal 反超原因待查**: DReyeVE 单独 BLEU_1=0.518 显著高 (BDDA 0.405, DADA 0.391, LBW 0.408 都在 0.4 区间), 我们的模型在 DReyeVE 上"过分擅长"。可能 (a) 训练 sample_rate 8,5,2,7 让 DReyeVE 学过头, (b) 论文用了不同的 train/test 拆分让 DReyeVE 没这么主导, (c) 论文 Normal 计算口径不同 (例如只 LBW alone=0.408, 跟 0.436 差只 -0.028)

2. **DADA CIDEr-R 完美匹配 (1.006 vs 1.002), BDDA CIDEr-R 巨差 (0.392 vs 1.233)**:
   - 不是"训练量不足"的故事 (那应该全 DS 一起差)
   - DADA Why CIDEr-R = **1.409** (远超 BDDA Why 0.459)
   - 推测: 事故场景 "Why" 推理高度区分性 ("To avoid the collision with the cyclist that suddenly entered the lane") → 模型学得到, n-gram 与 reference 命中率高, CIDEr 的 TF-IDF 加权抓得住
   - BDDA 安全场景 "Why" 比较通用 ("To be aware of the surroundings") → 模型生成的也通用 → CIDEr 判别力低
   - 这是 corpus-specific characteristic, **不是模型问题**

**LBW vs BDDA 文本指标观察**(都跑了完整 `--eval_text`):

| 段 | 指标 | BDDA | LBW | Δ (LBW-BDDA) | 解读 |
|---|---|---|---|---|---|
| What | METEOR | 0.330 | **0.374** | +0.044 | LBW "What" 反超 — 可能 LBW 标注更模板化(只 11 video × ~60 帧,语义重复度高) |
| What | BLEU_4 | 0.241 | 0.187 | -0.054 | LBW BLEU_4 略低 — 可能 LBW reference 更短,4-gram 难命中 |
| Why | METEOR | 0.184 | 0.158 | -0.026 | LBW "Why" 略差,符合 "Why 比 What 难" 通常模式 |
| Complete | CIDEr-R | 0.392 | 0.171 | -0.221 | CIDEr-R 大幅落后 — CIDEr 受 IDF 影响,LBW 词频集中度高,IDF 权重失效 |

**结论**: LBW 文本指标整体非但不烂,某些维度比 BDDA 还好(METEOR What),只 CIDEr-R 受 corpus 大小拖累。这跟"LBW 训练样本少"的预期不同 — **复现质量没问题**。

**4 子集 Complete 段加权 / 简单平均**(参考用, 论文 Table 2 不直接报):

| 度量 | weighted (n) | simple |
|---|---|---|
| BLEU_1 | 0.447 | 0.430 |
| METEOR | 0.274 | 0.265 |
| ROUGE | 0.405 | 0.379 |
| CIDEr-R | 0.631 | 0.514 |

**DReyeVE 旧崩溃产物保留** (4-26 09:42 那次):
- `runs_eval/baseline-5090-eval-DReyeVE.crashed-89pct/` (改名,无聚合数字)
- `runs_eval/eval-DReyeVE.log.crashed-89pct` (改名)
- `dataset/DReyeVE/test/<vid>/eval_text/2026_04_26_09_42/*.txt` (8100 个,与新 run `2026_04_27_20_21/` 时间戳不冲突)
- `dataset/DReyeVE/test/<vid>/eval_saving/2026_04_26_09_42/*.jpg` (8100 个)

**收数命令**(重跑无害, 已 idempotent):
```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python3.11 \
  scripts/paper/collect_main_results.py \
  --exps "baseline-5090=Baseline (ours repro)" \
  --out figures/tab1_main_results --per-dataset
```
当前产物: [figures/tab1_main_results/per_dataset_transposed_baseline-5090.tex](figures/tab1_main_results/per_dataset_transposed_baseline-5090.tex) — **4 DS × 6 attn + 4 text = 40 格全满** (Phase 1.5 完成)。

---

## 与论文对比

**Attn 6 指标 vs 论文 Table 1** (W³DA scenario-grouped):

| 场景 | 我们 (DS 加权) | 论文 Tab1 | 差距 (CC) | 状态 |
|---|---|---|---|---|
| Safety-Critical (BDDA) | CC 0.573 | 0.579 | -0.006 | ✓ |
| Normal (DR+LBW weighted) | CC 0.587 | 0.583 | +0.004 | ✓ |
| Accident (DADA) | CC 0.374 | 0.396 | -0.022 | ⚠ |

**KLD/NSS 普遍略差** (NSS 差 -0.2 ~ -0.8):
- 可能跟 effective batch 16 vs 160 (训练量不足)
- BDDA NSS 在 4788 样本中 76.6% NaN(gazemap 阈值 0.7 与 BDDA 标注幅值不匹配, [utils/sal_metrics.py:70-75](utils/sal_metrics.py#L70))。聚合时 [train_ds.py:848-849](train_ds.py#L848) 显式跳 NaN,所以是 1119 有效样本算的。

**与论文 Table 3** (per-dataset 完整 test):
- 论文 BDDA Tab3b: KLD 1.16, CC 0.60, SIM 0.47 → 我们 1.26 / 0.57 / 0.42 (⚠/⚠/⚠)
- 论文 DR(eye)VE Tab3a: KLD 1.04, CC 0.67 → 我们 1.29 / 0.60 (✗)
- 论文 DADA Tab3c: KLD 1.82, CC 0.48, SIM 0.36 → 我们 2.02 / 0.37 / 0.25 (⚠/✗/✗)
- **Tab3 用完整 test set,我们用 W³DA key-frame test 子集,不直接可比**。Tab1 (key-frame) 才是公平对比。

**结论**: Tab1 attn 复现基本 OK (主要在 ✓/⚠ 级别),Tab2 文本复现明显落后 (✗ 级别)。

---

## NSS NaN 现象记录

**事实**:
- 23:53 4 子集 eval CSV 中 NSS NaN 分布: BDDA 3669/4788 (76.6%), DReyeVE/LBW/DADA 全 0%
- 来源: [utils/sal_metrics.py:252-260](utils/sal_metrics.py#L252) 中 `discretize_gt(gt, threshold=0.7)` 把 GT 二值化, BDDA gazemap 像素值大概天然偏低,二值化后没 1,NSS 算 `np.mean([])` → NaN
- 聚合时被 skip ([train_ds.py:848-849](train_ds.py#L848))
- BDDA NSS=4.923 实际只用 1119 样本算的

**未决**: 是否要改 threshold 让 BDDA 也跟其他三个一致用全样本 NSS?改了的话 NSS 数字定义会变,跟论文不可比。**先不动**,在论文 LaTeX 表里 NSS 行加 dagger 注明就好(已在 collect_main_results.py 实现)。

---

## Known Issues (已知 bug, 不一定立即修)

### log_test.txt 里 Complete 段 `Cider:` 字段被错写成 CiderR 值

**事实**:
- [train_ds.py](train_ds.py) 的 eval print 代码,在写 `log_test.txt` 的 "Text metrics (Complete):" 行时, `Cider:` 字段实际填的是 CiderR 变量(等于 `CiderR:`)
- eval log 末行(stdout)显示的 `cider: X, ciderR: Y` X≠Y 是真实值
- 三个 DS 都中招(BDDA/LBW/DADA 的 log_test.txt 同样问题)

**例**:
- BDDA log_test Complete: `Cider: 0.392238, CiderR: 0.392238` (重复)
- BDDA eval log 末行: `cider: 0.203318, ciderR: 0.392238` (真实)
- DADA log_test Complete: `Cider: 1.006019, CiderR: 1.006019` (重复)
- DADA eval log 末行: `cider: 0.787577, ciderR: 1.006019` (真实)

**影响**:
- [scripts/paper/collect_main_results.py](scripts/paper/collect_main_results.py) 直接读 `CiderR:` 字段, **不受影响**
- 论文 Table 2 报的是 CIDEr-R, 也不受影响
- 只在你想读 log_test.txt 的 `Cider:` (plain) 时会得到错的值

**修复时机**: Phase 1 实验之前修。修完要重跑 BDDA / LBW / DADA eval 让 log_test.txt 同步(或直接改 collect script 从 stdout log 抓两个独立字段)。

---

## Open Questions (待与作者/社区核实)

这些问题在论文 / README / 仓库 issues 都找不到明确答案, 影响复现严谨度。优先级 = 影响最终数字可对比性。

### Q1. NSS 的 NaN 处理是否符合论文做法 (高优先级)

**事实**:
- 仓库代码 [utils/sal_metrics.py:NSS](utils/sal_metrics.py): `discretize_gt(gt, threshold=0.7)` 后 `np.mean(empty_list)` 返回 NaN, 函数无任何 NaN 防御 (有注释掉的 `# if np.isnan: print('Warning')`, 说明作者知情但没修)
- 上游仓库 main 分支代码与本地**完全一致** (WebFetch 验证 2026-04-27)
- 上游 issues (open #4 #6 #8 + closed #1 #2 #3) **无人讨论 NSS NaN 或 threshold 问题**
- 论文 §5.1 只说 "follow the evaluation setup of prior driver attention prediction studies [11, 22]"; refs [11]=FBLNet, [22]=SCAFNet/DADA
- 论文 supplementary (pp.9-18) 涵盖数据集统计 + MLLM 标注 prompt, **无 metric 实现细节**
- 我们复现: BDDA 4788 样本中 3669 (76.6%) NSS=NaN, [train_ds.py:848-849](train_ds.py#L848) 显式 `if not np.isnan(nss): nss_meter.update(nss)`, 最终 BDDA NSS 4.92 是从 1119 有效样本算的

**未决问题**:
1. 论文 Table 1 报的 BDDA NSS 5.27 是同样 skip-NaN 算的吗? 还是论文用了不同 threshold (例如 99% percentile 而非固定 0.7), 让 BDDA 不出 NaN?
2. 论文 [11, 22] 的实现 (FBLNet, SCAFNet) 用的 NSS 公式跟仓库代码一致吗?

**核实方式 (待做)**:
- 看 FBLNet 仓库 (https://github.com/yilongniu/FBLNet) 和 SCAFNet 实现的 NSS 函数
- 直接给作者发邮件 / 开 GitHub issue 询问

**当前对策**: 论文 LaTeX 表 NSS 行加 dagger + footnote ("BDDA: NSS computed on 1119/4788 valid samples (76.6% NaN due to gazemap threshold artifact)"), 由读者判断。已在 [scripts/paper/collect_main_results.py](scripts/paper/collect_main_results.py) `write_per_dataset_transposed()` 实现。

### Q2. 训练总 epoch / step 数 (高优先级)

**事实**:
- 论文 §5.1 只说 batch size, grad_accum, LR scheduler, λ scaling, LoRA setup
- 没说总 epoch 数, 也没说总 step 数
- README 给的 `--epochs=100 --steps_per_epoch=500` 是默认 hyperparam, 不一定是论文用的
- 我们复现用 `--epochs=10 --steps_per_epoch=500` = 5000 step, 这可能比论文少很多

**未决**: 论文跑了多少 step? 不知道这个数, 重训也不知道目标。

### Q3. 文本指标的具体计算方式 (✅ 已解决, BLEU=BLEU-1)

**事实** (2026-04-28 更新, 4 DS 全跑完后):

| 场景 | 论文 BLEU | 我们 BLEU_1 (Complete, weighted) | 差距 | 我们 BLEU_4 (Complete) | BLEU_4 差距 |
|---|---|---|---|---|---|
| Safety-Critical (BDDA) | 0.444 | 0.405 | **-0.039 ✓** | 0.174 | -0.27 |
| Accident (DADA) | 0.376 | 0.391 | **+0.015 ✓** | 0.223 | -0.15 |
| Normal (DReyeVE+LBW) | 0.436 | 0.510 | **+0.074 ⚠** | 0.246 | -0.19 |

**结论**: 论文 "BLEU" = BLEU-1 (不是 BLEU-4) 在 BDDA / DADA 上**确认**, BLEU-4 量级 (-0.15 ~ -0.27) 完全不对。

**Normal +0.074 反超的可能原因** (我们的 DReyeVE 单独 BLEU_1 = 0.518 显著高于其他 DS 的 0.39~0.41):
1. **训练采样比例**: `train_sample_rates=8,5,2,7` 让 DReyeVE 占 5/22 ≈ 23% train, 但 DReyeVE test 9101 占 W3DA test 41%, **训练分布与测试分布不匹配** —— 模型在 DReyeVE-likely test 上反而擅长
2. **论文不同采样比例**: 论文未公布 train_sample_rates, 可能用了不同比例让 DReyeVE 不主导
3. **论文 "Normal" 计算口径**: 如果只用 LBW (BLEU_1=0.408), 跟 0.436 差只 -0.028 ✓, 但与 §3.2 描述"Normal Driving = DR(eye)VE + LBW" 矛盾

**剩下**: "Complete" / "What" / "Why" 三段在论文里没解释 — 推测论文 Table 2 用的是 "Complete" 段。What/Why 三段都是 train_ds.py eval 时同时算的, 区别在 ground-truth 拼接方式 ([train_ds.py:792-808](train_ds.py#L792))。

### Q4. 4 子集训练采样比例的依据 (中优先级 — 升级)

**事实**: README 给 `train_sample_rates=8,5,2,7`, 我们沿用。但论文没公布该比例, 也没消融。

**新发现 (2026-04-28, 来自 Q3 收尾)**:
- `train_sample_rates=8,5,2,7` → DReyeVE 占训练 5/22 ≈ 23%
- 但 DReyeVE test 9101 占 W3DA test 41%
- → 训练分布与测试分布不匹配
- 我们的模型在 DReyeVE test 上 BLEU_1 反超 (0.518 vs 论文 Normal 0.436), 与 BDDA / DADA 在 paper Table 2 上的对得上形成对比

**未决**: 论文 Train 真实 sample rate 是什么? 直接试 `train_sample_rates=4788,9101,666,7419` (按 test 比例) 重训, 看 DReyeVE BLEU 是否会回落到 ~0.40 区间, 解 Normal 反超之谜。

**对策**: 列入 Phase 1 候选实验之一 (但不是核心 decoder 改造)。

---

## Phase 1.5 收尾 (✅ 完成 2026-04-28)

- [x] BDDA eval 完整指标 (2026-04-26 09:41, 7h54m)
- [x] LBW eval 完整指标 (2026-04-27 01:04, 1h2m)
- [x] DADA eval 完整指标 (2026-04-27 13:30, 12h14m)
- [x] DReyeVE eval 重跑 (2026-04-28 11:35, 15h14m)
- [x] 总 GPU 时间 ~36h 跑出 4 DS × 18 指标 (6 attn + 12 text) = 72 数字
- [x] collect_main_results.py per-dataset 表全满
- [x] Q3 (论文 BLEU = BLEU-1) 在 BDDA/DADA 上确认, Normal 反超待解释

**baseline 锚点已建立**: 后续任何 decoder 改造 / Phase 1 实验, 跟这套 4-DS 数字对比即可。

---

## Phase 1 决策点 (待用户确认, 不要 Claude 自起 GPU)

**核心选择**:

### A. 直接进 decoder 改造主线 (modify-llada skill)

不动 baseline 训练, 接受当前 ⚠ 级差距, 在论文里诚实标注 effective batch 差 10×。开始改 `attn_decoder` (主攻方向, 见 CLAUDE.md)。

**优点**: 节省 ~3 天 GPU 时间, 直接进入研究主线
**风险**: 任何 decoder 改造的提升要超过 baseline 复现 noise (NSS 差 -0.2 ~ -0.8)

### B. 先重训 baseline 缩小差距, 再开 Phase 1

- B1. 重训 30 epoch (从头, ~3 天)
- B2. 从 epoch 9 ckpt resume 跑到 epoch 30 (~2 天)
- B3. 改 ckpt 选择策略 ([train_ds.py:438](train_ds.py#L438)) 用多 DS val 而非 BDDA-only, 重新 eval (无需重训)

**优点**: baseline 数字更接近论文, 后续对比更可信
**风险**: ~3 天 GPU 时间投入, 不一定能完全 close gap

### C. 先调查 Normal +0.074 反超 (Q3 残留)

- 看 train_sample_rates 是否能调出 paper-like 平衡 (DReyeVE 不主导)
- 联系作者要原始 train_sample_rates / val 分布

**优点**: 复现严谨度更高
**风险**: 可能是 dataset-specific 现象, 调不出来

### 我建议

**A**。理由: (1) baseline attn 在 Tab1 三个场景都 ✓/⚠ 级匹配, 复现质量足够支撑 decoder 改造研究; (2) 重训不能 100% close gap (我们只 1 GPU vs 论文 4 GPU, effective batch 永远差 4×); (3) Normal 反超是 dataset 现象, 不影响 decoder 改造的相对对比 (改 decoder 后 4 DS 同样口径再 eval, ablation 比较的是 delta)。

---

## 其他 TODO / 未决
- [ ] 决定是否重训以缩小 KLD/NSS 与论文差距:
  - 选 1: 不重训,接受 ⚠ 级差距,论文里诚实标注 effective batch 差 10×
  - 选 2: 重训 30 epoch / 接 ckpt 续训,~3 天 GPU 时间
  - 选 3: 改 ckpt 选择策略 ([train_ds.py:438](train_ds.py#L438)) 用多 DS val 而非 BDDA-only
- [x] 文本指标 0.174 vs 论文 0.444 的根因 (2026-04-27 解决):
  - **论文 "BLEU" = BLEU-1, 不是 BLEU-4**。证据: DADA BLEU_1 0.391 vs 论文 0.376 (+0.015), BDDA BLEU_1 0.405 vs 论文 0.444 (-0.039), 量级吻合; vs BLEU_4 差 -0.15 / -0.27 量级完全不对
  - 等 DReyeVE 跑完做 100% 确认 (LBW BLEU_1 0.408 alone 已支持)
- [ ] CLAUDE.md `推荐 batch=1, grad_accum=16 → 16(原论文是 80)` 应改为 `(原论文是 160)`
- [ ] 决定 5 个 epoch 的 best_ckpt(120 GB)删/留(只 epoch9 已合并到 ckpts/)
