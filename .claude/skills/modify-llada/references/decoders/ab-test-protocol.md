# A/B 对比实验协议

为了让论文结果站得住脚(审稿人问"这点提升显著吗?"时有答案),每个 decoder 变体都要按这个协议做。

## 基本要求

| 项 | 规定 |
| --- | --- |
| 控制组 | `baseline-5090`(原 CrossAttnDecoder, 原超参) |
| 实验组 | `decoder-<type>-5090`(**只改 decoder**,其他完全一致) |
| 测试集 | 固定,4 个子数据集全用,`--val_sample_rates` 不变 |
| 样本数 | `--val_samples_num=5000` 以上(小样本 p-value 不可靠) |
| 随机种子 | 固定 `42`,同时跑 `0, 42, 123` 三个种子(至少关键对比) |
| 指标 | 全部 6 个 attn + 3×5 个文本 + 参数量 + FLOPs + 推理速度 |

## 固定随机种子的改动

原代码似乎没显式设种子。建议在 `train_ds.py::main` 开头加:

```python
import random, numpy as np, torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 在 parse_args 之后
set_seed(getattr(args, 'seed', 42))
```

加 CLI 参数:
```python
parser.add_argument("--seed", default=42, type=int)
```

**注意**: 完全复现仍然需要 `torch.backends.cudnn.deterministic=True` 和 `benchmark=False`,但这会慢 ~20%。**建议只在最终消融运行时开 deterministic**,调试时关着。

## 成对 t 检验脚本

```python
# scripts/significance_test.py
"""
成对 t 检验 (paired t-test) + Wilcoxon signed-rank test.
对两个实验的 attn_metrics_0.csv 做逐样本对比。
"""
import argparse
import pandas as pd
from scipy import stats


def compare(baseline_csv, ours_csv, metrics=("cc", "kld", "sim", "nss", "auc_b", "auc_j")):
    base = pd.read_csv(baseline_csv)
    ours = pd.read_csv(ours_csv)

    # 按 image_id 对齐 (逐样本成对)
    merged = base.merge(ours, on="image_id", suffixes=("_base", "_ours"))
    n = len(merged)
    print(f"Paired samples: {n}")
    print(f"{'Metric':8s} | {'Base':>8s} | {'Ours':>8s} | {'Δ':>8s} | {'t':>8s} | {'p (t)':>10s} | {'p (Wilcox)':>12s}")
    print("-" * 80)

    for m in metrics:
        b = merged[f"{m}_base"]
        o = merged[f"{m}_ours"]
        delta = o.mean() - b.mean()
        t, p_t = stats.ttest_rel(o, b)
        w, p_w = stats.wilcoxon(o, b)
        sig = "***" if p_t < 0.001 else "**" if p_t < 0.01 else "*" if p_t < 0.05 else ""
        print(f"{m.upper():8s} | {b.mean():8.4f} | {o.mean():8.4f} | {delta:+8.4f} | "
              f"{t:8.3f} | {p_t:10.3e} | {p_w:12.3e} {sig}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="baseline attn_metrics_0.csv")
    p.add_argument("--ours", required=True, help="ours attn_metrics_0.csv")
    args = p.parse_args()
    compare(args.baseline, args.ours)
```

使用:
```bash
python scripts/significance_test.py \
  --baseline runs/baseline-5090/attn_eval/<ts>/attn_metrics_0.csv \
  --ours     runs/decoder-pyramid-5090/attn_eval/<ts>/attn_metrics_0.csv
```

典型输出:
```
Metric   |     Base |     Ours |        Δ |        t |      p (t) |   p (Wilcox)
--------------------------------------------------------------------------------
CC       |   0.7134 |   0.7421 |  +0.0287 |   12.345 |   3.21e-34 |     2.45e-30 ***
KLD      |   1.2345 |   1.1023 |  -0.1322 |  -10.234 |   1.23e-24 |     5.67e-22 ***
...
```

## 多种子方差报告

至少对关键对比(baseline vs final method)跑 3 个种子:

```bash
for seed in 0 42 123; do
  deepspeed --num_gpus=1 ... \
    --seed=$seed \
    --exp_name="decoder-pyramid-s${seed}-5090"
done
```

然后报 `mean ± std`:

```python
# scripts/multi_seed_stats.py
import pandas as pd
import re, glob
from pathlib import Path

def parse_log(path):
    text = Path(path).read_text()
    result = {}
    for m in re.finditer(r"(CC|KLD|SIM|NSS|AUC_B|AUC_J):\s*([\d.]+)", text):
        result[m.group(1)] = float(m.group(2))
    return result

runs = {}
for seed in [0, 42, 123]:
    logs = sorted(glob.glob(f"runs/decoder-pyramid-s{seed}-5090/attn_eval/*/log_test.txt"))
    runs[seed] = parse_log(logs[-1])

df = pd.DataFrame(runs).T
print(df.agg(["mean", "std"]).round(4))
```

## 报告方式 (论文表格)

```
                   |  CC ↑   |  KLD ↓  | ...
Baseline (n=3)     | 0.713±0.004 | 1.234±0.008 | ...
Ours Pyramid (n=3) | 0.742±0.005 | 1.102±0.007 | ...
```

表脚注:
> All improvements over the baseline are statistically significant at p < 0.001
> (paired t-test on N=5000 test samples, 3 random seeds per method).

## Pitfalls

- **N 太小**: `val_samples_num<1000` 时 p-value 不可靠
- **测试集泄漏**: 同一视频不同帧都在 train/test 里 → 过度乐观的指标。检查 W³DA 的 train/test split 是否按 video 切分(应该是,但要确认)
- **多重检验**: 6 个 attn 指标做 6 次 t 检验,需要 Bonferroni 校正:`p_adj = p * 6`。或报告前明说用的是未校正
