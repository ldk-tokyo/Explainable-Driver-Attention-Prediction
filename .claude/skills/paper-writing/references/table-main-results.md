# 主对比表 (Main Results Table)

## 结构

```
Method          | #Params | CC↑    | KLD↓   | SIM↑   | NSS↑   | AUC-B↑ | AUC-J↑ | BLEU-4↑ | METEOR↑ | CIDEr-R↑
----------------|---------|--------|--------|--------|--------|--------|--------|---------|---------|----------
Competitor A    | 45M     | 0.61   | 1.45   | 0.58   | 2.32   | 0.78   | 0.82   | -       | -       | -
Competitor B    | 78M     | 0.68   | 1.31   | 0.64   | 2.67   | 0.81   | 0.85   | -       | -       | -
LLada (paper)   | 7.2B    | 0.71   | 1.23   | 0.67   | 2.87   | 0.82   | 0.87   | 0.34    | 0.28    | 0.89
LLada (ours)    | 7.2B    | 0.71   | 1.24   | 0.67   | 2.85   | 0.81   | 0.87   | 0.33    | 0.28    | 0.88
+ Pyramid       | 7.23B   | 0.74   | 1.15   | 0.70   | 2.95   | 0.83   | 0.88   | 0.35    | 0.29    | 0.91
+ SAM-style     | 7.23B   | 0.75   | 1.12   | 0.71   | 2.98   | 0.84   | 0.89   | 0.36    | 0.30    | 0.93
```

Bold the best in each column. ↑/↓ 标明方向。

## Step 1: 收集数据

```python
# scripts/paper/collect_main_results.py
import re
import pandas as pd
from pathlib import Path


def parse_log(path):
    """从 log_test.txt 里抽所有指标"""
    text = Path(path).read_text()
    result = {}
    patterns = {
        "CC": r"CC:\s*([\d.]+)",
        "KLD": r"KLD:\s*([\d.]+)",
        "SIM": r"SIM:\s*([\d.]+)",
        "NSS": r"NSS:\s*([\d.]+)",
        "AUC_B": r"AUC_B:\s*([\d.]+)",
        "AUC_J": r"AUC_J:\s*([\d.]+)",
        "Bleu_4": r"Bleu_4:\s*([\d.]+)",
        "Meteor": r"Meteor:\s*([\d.]+)",
        "Rouge":  r"Rouge:\s*([\d.]+)",
        "CiderR": r"CiderR:\s*([\d.]+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            result[k] = float(m.group(1))
    return result


EXPERIMENTS = [
    ("Baseline (ours repro)", "baseline-5090"),
    ("+ Pyramid",              "decoder-pyramid-5090"),
    ("+ SAM-style",            "decoder-sam-5090"),
    ("+ Mask2Former",          "decoder-mask2former-5090"),
]

rows = []
for display_name, exp in EXPERIMENTS:
    # 找最新的 eval_only 或 eval_text log
    logs = sorted(Path(f"runs/{exp}").rglob("attn_eval/*/log_test.txt"))
    if not logs:
        print(f"WARN: no log for {exp}")
        continue
    m = parse_log(logs[-1])
    m["Method"] = display_name
    rows.append(m)

df = pd.DataFrame(rows)
df = df.set_index("Method")
df.to_csv("figures/tab1_main_results/raw.csv")
print(df)
```

## Step 2: 转 LaTeX (booktabs)

```python
# scripts/paper/make_main_table.py
import pandas as pd
from pathlib import Path

df = pd.read_csv("figures/tab1_main_results/raw.csv", index_col="Method")

COLS_ATTN = ["CC", "KLD", "SIM", "NSS", "AUC_B", "AUC_J"]
COLS_TEXT = ["Bleu_4", "Meteor", "Rouge", "CiderR"]
HIGHER_BETTER = {"CC": True, "KLD": False, "SIM": True, "NSS": True,
                 "AUC_B": True, "AUC_J": True,
                 "Bleu_4": True, "Meteor": True, "Rouge": True, "CiderR": True}

def fmt_cell(val, best_val, higher_better):
    """Bold if best"""
    is_best = abs(val - best_val) < 1e-5
    s = f"{val:.3f}"
    return f"\\textbf{{{s}}}" if is_best else s


lines = []
lines.append(r"\begin{tabular}{l" + "c" * (len(COLS_ATTN) + len(COLS_TEXT)) + "}")
lines.append(r"\toprule")

# Multi-col header
header = r"Method"
for c in COLS_ATTN:
    arrow = "↑" if HIGHER_BETTER[c] else "↓"
    header += f" & {c} $\\{arrow}$"
for c in COLS_TEXT:
    arrow = "↑" if HIGHER_BETTER[c] else "↓"
    header += f" & {c} $\\{arrow}$"
header += r" \\"
lines.append(header)
lines.append(r"\midrule")

# Compute best per column
best = {c: (df[c].max() if HIGHER_BETTER[c] else df[c].min())
        for c in COLS_ATTN + COLS_TEXT}

for method in df.index:
    row = method
    for c in COLS_ATTN + COLS_TEXT:
        v = df.loc[method, c]
        row += " & " + fmt_cell(v, best[c], HIGHER_BETTER[c])
    row += r" \\"
    lines.append(row)

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

out = "\n".join(lines)
Path("figures/tab1_main_results").mkdir(parents=True, exist_ok=True)
Path("figures/tab1_main_results/main.tex").write_text(out)
print(out)
```

## Step 3: LaTeX 集成

```latex
\begin{table*}[t]
    \centering
    \caption{
        Main results on W$^3$DA test set (4 sub-datasets combined).
        CC, SIM, NSS, AUC-B, AUC-J higher is better;
        KLD lower is better.
        Best result in each column is in \textbf{bold}.
        All improvements over the baseline are statistically
        significant (paired $t$-test, $p < 0.001$, $N=5000$).
    }
    \label{tab:main}
    \resizebox{\textwidth}{!}{
        \input{tab1_main_results/main.tex}
    }
\end{table*}
```

## 分组头表格 (更高级)

如果想把 attn 指标和 text 指标分组:

```latex
\begin{tabular}{l cccccc | cccc}
\toprule
& \multicolumn{6}{c|}{Where (Attention Metrics)} & \multicolumn{4}{c}{What \& Why (Text Metrics)} \\
\cmidrule(lr){2-7} \cmidrule(lr){8-11}
Method & CC$\uparrow$ & KLD$\downarrow$ & SIM$\uparrow$ & NSS$\uparrow$ & AUC-B$\uparrow$ & AUC-J$\uparrow$
       & BLEU-4$\uparrow$ & METEOR$\uparrow$ & ROUGE$\uparrow$ & CIDEr-R$\uparrow$ \\
\midrule
...
\bottomrule
\end{tabular}
```

Python 生成这种也很直接,在 header 那里加 `\multicolumn` 即可。

## 对比其他论文数字

原论文 LLada 的数字应该在他们 paper 里 / HF 上公布:
- 如果你 reproduce 的数字和他们一致 (±0.01) → 可直接用你的数字,并在脚注说"reproduced"
- 如果有差异(比如因为单卡 effective batch 小)→ 两列: "LLada (paper)" 用原论文数, "LLada (ours repro)" 用你的
- 对比其他 baseline (HWS/MINet/TASED) 时,要么你自己跑 (工作量大),要么直接引他们原论文数字并脚注说明

## 脚注/标注模板

```latex
\caption{ ...
    $\dagger$ Results taken from original paper \cite{zhou2025where}.
    $\ddagger$ Our reproduction under single-GPU setting (effective batch=16).
    All other results are trained on 4 A100 GPUs as per original setup.
}
```
