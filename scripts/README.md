# scripts/

项目辅助脚本与使用速查。每个脚本对应下面一个小节，包含：用途、依赖、常用命令、产物说明。

> 约定：所有 Python 一律走 `.venv/bin/python`，不用系统 `python3`。

---

## live_dashboard.py — 终端训练监控

**场景**：SSH 远程盯训练进度，不想开浏览器跑 TensorBoard。

**依赖**（已装在 `.venv`）：`rich`, `tensorboard`, `pynvml`

### 单 exp 监控

```bash
tmux new -s monitor
.venv/bin/python scripts/live_dashboard.py runs/baseline-5090
# Ctrl+B D 离开 tmux，dashboard 继续在后台
# tmux attach -t monitor 再回来
```

### 多 exp 并排对比

```bash
.venv/bin/python scripts/live_dashboard.py runs/baseline-5090 runs/decoder-pyramid-5090
```

### 调刷新频率

```bash
.venv/bin/python scripts/live_dashboard.py runs/baseline-5090 --refresh 30   # 省 CPU
.venv/bin/python scripts/live_dashboard.py runs/baseline-5090 --refresh 1    # 调试
```

### 界面元素

- **RTX 5090 panel**：显存、GPU%、温度、功耗
  - 温度：绿 <75°C / 黄 75–85°C / 红 >85°C（>85°C 立刻降功耗或开窗）
  - 显存：绿 <90% / 黄 90–95% / 红 >95%
- **exp panel**：当前 step 的 loss、6 个 attn val 指标、文本指标
  - 数字颜色：绿 = higher is better（cc/sim/nss/auc），红 = lower is better（kld）
  - **panel 边框 = 健康信号**：
    - 绿（<2 min）训练在推进
    - 黄（2–10 min）可能在 validate / 存 ckpt
    - 红（>10 min）可能 hang 了，去查 `runs/<exp>.log`

### 何时不用它

- 想看完整 loss 曲线、做平滑或缩放 → 用 TensorBoard
- 训练已结束、要归档 → 用训练后 PDF 报告（见 `experiment-tracking` skill 的 `posttrain-pdf-report.md`）

---

## posttrain_report.py — 训练后 PDF 归档报告

**场景**：训练（或 eval）跑完，要一份可 commit 到 git 的 PDF 快照，以后翻历史实验、发给合作者看。

**依赖**（已装在 `.venv`）：`matplotlib`, `tensorboard`, `pandas`, `opencv-python`

### 生成

```bash
.venv/bin/python scripts/posttrain_report.py runs/baseline-5090 \
    --out reports/baseline-5090.pdf
```

不传 `--out` 则默认写到 `runs/<exp>/report.pdf`，但 `runs/` 被 ckpt 占满不适合 commit；**统一放 `reports/` 再入 git**。

### PDF 4 页内容

1. **Training Curves**：total loss / 各分量 loss / lr / throughput (sec/batch)
2. **Validation Metrics**：6 个 attn 指标（cc/sim/nss/auc 一组，kld 单独，nss 单独）+ 4 个文本指标（bleu_4/meteor/rouge/ciderR）
3. **Final Summary**：最终值 + 历史最佳 + 对应 step + ckpt 文件清单
4. **Top-K vs Bottom-K**：按 cc 排序的好/坏样本 quad 图（input | gt | pred | diff），需要 `attn_metrics_*.csv` 且 `eval_saving/` 里有 pred 图，否则显示占位

### 跳过 / 自定义样本数

```bash
# top/bottom 各 10 张
.venv/bin/python scripts/posttrain_report.py runs/baseline-5090 --k 10
```

### TensorBoard tag 命名约定（脚本依赖）

脚本读这些 tag，新增训练分支时保持一致：
- 训练：`train/loss`, `train/ce_loss`, `train/attn_loss`, `train/ce_what_losses`, `train/ce_why_losses`, `train/lr`
- 验证 attn：`val/cc`, `val/kld`, `val/sim`, `val/nss`, `val/auc_b`, `val/auc_j`
- 验证文本：`val/bleu_4`, `val/meteor`, `val/rouge`, `val/ciderR`
- 吞吐：`metrics/total_secs_per_batch`, `metrics/data_secs_per_batch`

如改了 tag 名，记得同步改 `scripts/posttrain_report.py` 和 `scripts/live_dashboard.py`。

### 何时不用它

- 想做投稿图表（要严格字号配色、对比多 exp） → 走 `paper-writing` skill
- 训练中实时看 → 用 `live_dashboard.py` 或 TensorBoard

---

<!-- 新脚本按此模板追加：用途 / 依赖 / 命令 / 产物 / 何时不用 -->

