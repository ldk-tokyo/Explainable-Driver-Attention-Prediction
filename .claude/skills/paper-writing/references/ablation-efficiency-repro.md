# 消融 / 效率 / 复现性 / 统计检验 合集

比 fig3 / tab1 次要但必备的四类产出。

---

# §1 消融折线图 (ablation figure)

例: decoder depth 2→8 的 CC 变化。

```python
# scripts/paper/fig4_depth_ablation.py
import sys
sys.path.append("scripts/paper")
from plot_config import *
import pandas as pd

df = pd.read_csv("figures/all_results.csv")

depths = [2, 4, 6, 8]
cc  = [df[df.exp == f"decoder-cross-depth{d}-5090"]["CC"].values[0]  for d in depths]
kld = [df[df.exp == f"decoder-cross-depth{d}-5090"]["KLD"].values[0] for d in depths]

fig, ax1 = fig_single(height_ratio=0.7)
l1, = ax1.plot(depths, cc,  "o-", color="#0072B2", label="CC ↑", markersize=5)
ax1.set_xlabel("Decoder depth")
ax1.set_ylabel("CC ↑", color="#0072B2")
ax1.tick_params(axis="y", labelcolor="#0072B2")
ax1.grid(alpha=0.3, linestyle="--")

ax2 = ax1.twinx()
l2, = ax2.plot(depths, kld, "s--", color="#D55E00", label="KLD ↓", markersize=5)
ax2.set_ylabel("KLD ↓", color="#D55E00")
ax2.tick_params(axis="y", labelcolor="#D55E00")

ax1.legend(handles=[l1, l2], loc="center right", frameon=True)

plt.tight_layout()
plt.savefig("figures/fig4_ablation/depth.pdf")
```

---

# §2 消融表 (组件加减)

```latex
% figures/tab2_ablation/ablation.tex
\begin{tabular}{cccc|cccc}
\toprule
CrossAttn & FPN & Mask & Prompt
 & CC $\uparrow$ & KLD $\downarrow$ & SIM $\uparrow$ & BLEU-4 $\uparrow$ \\
\midrule
\checkmark &            &            &            & 0.71 & 1.23 & 0.67 & 0.34 \\
\checkmark & \checkmark &            &            & 0.74 & 1.15 & 0.70 & 0.35 \\
\checkmark & \checkmark & \checkmark &            & 0.76 & 1.10 & 0.72 & 0.36 \\
\checkmark & \checkmark & \checkmark & \checkmark & \textbf{0.77} & \textbf{1.08} & \textbf{0.73} & \textbf{0.37} \\
\bottomrule
\end{tabular}
```

Python 自动生成版见 `table-main-results.md` 的类似模板。

---

# §3 统计显著性

参见 `modify-llada/references/decoders/ab-test-protocol.md` §成对 t 检验脚本。

关键输出格式:

```
Metric   |     Base |     Ours |        Δ |        t |      p (t) |   p (Wilcox)
--------------------------------------------------------------------------------
CC       |   0.7134 |   0.7421 |  +0.0287 |   12.345 |   3.21e-34 |     2.45e-30 ***
```

## 论文脚注模板

```latex
\footnotesize
$\dagger$ All improvements over the baseline are statistically significant
at $p < 0.001$ (paired two-sided $t$-test, $N = 5000$ matched samples
across 4 sub-datasets). For KLD, lower is better so we test the
opposite direction. Wilcoxon signed-rank test yields consistent
conclusions in all cases.
```

---

# §4 效率对比表 (tab3)

```python
# scripts/paper/tab3_efficiency.py
import torch
import time
from fvcore.nn import FlopCountAnalysis

from model.Attn_model import AttnForCausalLM

def measure_decoder(ckpt_path):
    model = AttnForCausalLM.from_pretrained(ckpt_path,
                                             torch_dtype=torch.bfloat16).cuda().eval()

    # 参数量 (只数 decoder + fcs, 不包括 LLM)
    decoder_params = sum(p.numel() for n, p in model.named_parameters()
                         if "attn_decoder" in n or "text_hidden_fcs" in n)

    # FLOPs: 只跑 decoder 部分
    dummy_attn = torch.randn(1, 1024, dtype=torch.bfloat16).cuda()
    dummy_vis  = torch.randn(1, 256, 1024, dtype=torch.bfloat16).cuda()
    flops = FlopCountAnalysis(model.attn_decoder, (dummy_attn, dummy_vis))
    decoder_gflops = flops.total() / 1e9

    # 推理速度 (full forward, batch=1)
    sample = build_dummy_sample()  # 自己构造,含完整 input_ids + images
    # warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(**sample)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            _ = model(**sample)
        torch.cuda.synchronize()
        ms_per_sample = (time.time() - t0) * 10  # / 100 * 1000

    return {
        "decoder_params_M": decoder_params / 1e6,
        "decoder_gflops": decoder_gflops,
        "full_ms_per_sample": ms_per_sample,
    }


for exp in ["baseline-5090", "decoder-pyramid-5090", "decoder-sam-5090"]:
    r = measure_decoder(f"./ckpts/ATTN-7B-{exp}")
    print(f"{exp}: {r}")
```

对应 LaTeX:

```latex
\begin{tabular}{l c c c}
\toprule
Method & Decoder params (M) & Decoder GFLOPs & Full inference (ms) \\
\midrule
Baseline     & 5.2  & 0.8  & 312 \\
Pyramid      & 22.1 & 3.4  & 348 \\
SAM-style    & 8.7  & 1.9  & 325 \\
Mask2Former  & 34.5 & 5.7  & 372 \\
\bottomrule
\end{tabular}
```

Caption:
> Efficiency comparison on RTX 5090. Only decoder module counted
> for params/FLOPs. Full inference includes LLM + CLIP + decoder
> at image resolution $1024{\times}1024$.

---

# §5 可复现性 Checklist

投稿前对照:

- [ ] 代码匿名公开 (投稿期 anonymous repo)
- [ ] README 有 one-liner reproduce: `python main.py --config configs/ours.yaml`
- [ ] 随机种子固定 (`torch.manual_seed`, `np.random.seed`, `random.seed`)
- [ ] `requirements.txt` 固定到严格版本
- [ ] 附录列所有超参 (lr, batch, epochs, weight decay, warmup, scheduler, seed)
- [ ] 数据集处理脚本完整公开
- [ ] 附录 Hardware section (本项目: RTX 5090 单卡 32GB)
- [ ] 附录含 training curve (loss + val metric vs epoch)
- [ ] 至少 2 个种子的 mean±std (`experiments.md` 里记录过的)
- [ ] 模型 ckpt 上传 (HF Hub / Zenodo)
- [ ] 主要消融至少能从 ckpt reproduce eval 数字

## 通用陷阱

1. **Single-GPU 复现与 4-GPU 原论文 效果不同**: 在论文里**明确说**你用单卡,effective batch 减小,这不是 bug 而是硬件约束
2. **CUDA 随机性**: 即使 seed 固定,非 deterministic cuDNN 会有 ±0.005 波动,报 mean±std 即可
3. **数据下载成本**: W³DA 数据集较大,在 README 里提供 torrent / magnet / 分片镜像以防 HF 拉不下来
4. **环境冻结**: 用 `pip freeze > requirements.freeze.txt` 存完全可复现的环境,不是原 requirements.txt

---

# §6 Teaser / 架构图

这俩**不适合纯 Python**,一般流程:

1. Python 产出中间 PDF 元素(比如一个好看的热力图叠加示例)
2. Inkscape 打开 PDF 元素,手工加箭头、公式、文字
3. 存成 SVG + 导出 PDF 给论文用

模板图参考:
- 大部分 ICCV 论文的 teaser 是 "1 张方法概览图 + 1 行标语"
- 架构图通常包含: 输入 → 各模块 (用不同颜色框) → 输出,箭头清晰标向

想 Claude 帮你生成 SVG 模板:
```
告诉我 teaser 要包含的元素,我可以产一个 Inkscape 起点 SVG。
```
