# Paper Limitations / Footnote 草稿

> 后续投稿在 baseline 表 / Methodology / Limitations 段落用的 footnote 模板。
> 复现锚点: `baseline-5090` (commit 6a290dc, ckpts/ATTN-7B-baseline-5090/)。
> 数据来自 [experiments.md](../experiments.md) Phase 1.5 (✅ 2026-04-28 完成)。

---

## Footnote 1 — Effective batch 与原论文差距 (必加)

**用在**: baseline 表标题脚注,或 Implementation Details 段落末。

> Our reproduced LLada baseline is trained on a single RTX 5090 (32 GB) with
> `batch_size=1, grad_accumulation_steps=16`, yielding an effective batch of 16.
> The original paper reports training on 4×A100 with `batch=8, grad_accum=5`
> (effective batch 160, ~10× larger). All other hyperparameters (LR=3e-4,
> WarmupDecayLR with 100 warmup steps, λ scaling, LoRA setup, frozen CLIP-ViT-L,
> from-scratch attention decoder) follow the original paper. Absolute metric
> gaps (KLD/NSS) primarily reflect this batch-size difference; cross-decoder
> ablations within our codebase use this single anchor consistently.

---

## Footnote 2 — NSS NaN on BDDA (必加,Q1)

**用在**: 主结果表 NSS 列加 dagger (†),footnote 解释。已在 [scripts/paper/collect_main_results.py](../scripts/paper/collect_main_results.py) 实现。

> NSS is computed via `discretize_gt(gt, threshold=0.7)` followed by mean over
> fixated pixels (sal_metrics.py). On BDDA, gazemap pixel intensities tend to
> peak below this threshold, producing empty fixation masks for 3669/4788
> (76.6%) test samples and yielding NaN; aggregation skips NaN entries
> (train_ds.py L848-849), so reported BDDA NSS is computed over 1119 valid
> samples. DR(eye)VE / LBW / DADA show 0% NaN under the same threshold.
> We retain the original threshold for direct comparability with prior reports
> rather than adjusting per-dataset.

---

## Footnote 3 — Paper "BLEU" = BLEU-1 (Methodology / Tab 2 caption)

**用在**: 文本指标表 caption,或 metrics 段落。Q3 已解决。

> The original paper reports a single "BLEU" score; we empirically determined
> this corresponds to BLEU-1 rather than BLEU-4. Evidence: under matched
> evaluation, our BLEU-1 on Safety-Critical (BDDA) is 0.405 vs paper 0.444
> (Δ=-0.039) and on Accident (DADA) is 0.391 vs paper 0.376 (Δ=+0.015), while
> BLEU-4 differs by -0.15 to -0.27 across all scenarios — a magnitude
> inconsistent with the paper's reported range. We therefore report BLEU-1 in
> all main tables, with BLEU-2/3/4 in supplementary.

---

## Footnote 4 — train_sample_rates 与 Normal BLEU 反超 (Q4, 选用)

**用在**: 如果 reviewer 追问 Normal scenario 数字,或 ablation 表里出现 sample_rate 比较。

> Following the public README, we adopt `train_sample_rates=8,5,2,7` for
> {BDDA, DR(eye)VE, LBW, DADA}, which the original paper does not explicitly
> publish. Under this ratio our model attains BLEU-1=0.518 on DR(eye)VE
> (a 9101-sample test split forming ~41% of W³DA test) — substantially above
> other splits and 0.074 above the paper's reported "Normal" BLEU. We attribute
> this to a train/test sample-distribution mismatch (DR(eye)VE = 23% of train
> mixture but 41% of evaluated test); a sample_rate sensitivity study is
> deferred to future work as it does not affect cross-decoder relative
> comparisons.

---

## Footnote 5 — Epoch / step count not specified (Q2, 选用)

**用在**: Implementation Details 段落,reproducibility 声明。

> The original paper specifies the optimizer, LR scheduler, and per-step batch
> configuration but does not report the total number of training epochs or
> steps. We trained for 10 epochs × 500 steps = 5000 optimizer updates, with
> best checkpoint selected on BDDA validation (epoch 9). All decoder ablations
> in this work use identical schedule for fair comparison.

---

## 用法提示

- **必加**: 1, 2, 3 — 这三个直接影响数字解读,reviewer 会问
- **选用**: 4, 5 — 看 reviewer 是否追问;不主动暴露也合理
- 写成正式 LaTeX 时 escape `_` `&` 等,以及把 Δ 改成 `$\Delta$`
- 数字最终从 [figures/tab1_main_results/](../figures/tab1_main_results/) per-DS LaTeX 引,这里的数字是 2026-04-28 锚点,后续若改 baseline 需要同步
