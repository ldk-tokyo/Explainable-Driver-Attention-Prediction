---
name: experiment-tracking
description: 实验的**组织、汇总与实时监控** —— exp_name 命名约定、`experiments.md` 维护、从多个 `log_test.txt` / `attn_metrics_0.csv` 汇总对比 CSV;以及训练**实时监控** (TensorBoard 增强 / 终端 live dashboard / 训练后 PDF 报告)。用户说"怎么命名 exp"、"对比多个 run"、"哪个 ckpt 最好"、"实时看训练进展"、"训练曲线"、"TensorBoard"、"loss 曲线"、"指标可视化"、"训练中查看"、"live monitor"、"训练完出报告" 时用。**产内部 CSV / TensorBoard / 训练快照 PDF,不产论文最终图表**(那是 paper-writing)。
---

# 实验追踪与管理约定

## 为什么这个很重要

一周内你会跑 20+ 次训练。没有严格约定的话,两周后你面对一堆 `runs/test1/` `runs/exp_final/` `runs/new/`,不知道哪个是基线、哪个加了什么,最终论文表格里的数 reproduce 不出来。

## 一、目录约定

```
runs/
├── <exp_name>/
│   ├── <timestamp>/                  # 每次 python 启动会生成一个
│   │   ├── events.out.tfevents...    # TensorBoard
│   │   ├── log_test.txt              # 如果是 eval_only 生成
│   │   ├── attn_metrics_<rank>.csv   # 逐样本 attn 指标
│   │   ├── train_vis/                # 训练期间的 pred_*.jpg, gt_*.jpg
│   │   └── val_vis/
│   ├── ckpt_model/                   # DeepSpeed ckpt(所有 rank 的)
│   ├── best_ckpt_model_epoch<N>/     # 该 epoch 的 best(可能多个,看 is_best 逻辑)
│   ├── pytorch_model.bin             # zero_to_fp32 转出来的
│   └── meta_log_*.pth                # 指标元信息
experiments.md                         # 每个 exp 的设计动机与结果,手工维护
```

**约定**:
- 一个 `exp_name` = 一个研究变量 = 一次 clear 对比
- 同一个 `exp_name` 可以重跑多次,时间戳区分
- 不要在同一个 `exp_name` 下修改不同超参 —— 会覆盖日志

## 二、命名约定

**格式**: `{baseline_ref}-{change_type}-{short_desc}`

**推荐示例**:
```
Attn-7b-baseline-reproduce             # 原论文复现
Attn-7b-ablation-no-attn-loss          # 消融: 去掉 attention loss
Attn-7b-ablation-no-why                # 消融: 去掉 Why
Attn-7b-decoder-pyramid                # 换成 pyramid decoder
Attn-7b-backbone-siglip                # 换 backbone
Attn-7b-temporal-3frame                # 加时序
Attn-7b-data-bdda-only                 # 只用一个子数据集
Attn-7b-loss-focal                     # loss 改 focal
Attn-7b-token-add-risk                 # 加 [RISK] token
```

**禁止**:
- `test`, `test1`, `new`, `final`, `real_final`
- 日期作主标识(目录时间戳已经有了,不需要 `0423_exp`)
- 过长到超出终端宽度 80 字符

## 三、`experiments.md` 模板

在仓库根建一个 `experiments.md`,一次实验一个 section:

```markdown
## Attn-7b-ablation-no-attn-loss

- **日期**: 2026-04-24
- **分支**: `abl/no-attn-loss` @ abc1234
- **动机**: 验证 attention loss 是否必要。原论文 attn_loss_weight=2,本次设为 0
- **改动**:
    - `train_ds.py`: `--attn_loss_weight=0.0`
    - 无代码改动
- **命令**: 见下
- **资源**: 1× A100 80G, ~18h
- **结果(test split 全部 4 子数据集)**:
    | 指标 | baseline | 本次 | 差异 |
    |---|---|---|---|
    | CC  | 0.71 | 0.48 | -0.23 |
    | KLD | 1.23 | 2.89 | +1.66 |
    | SIM | 0.67 | 0.44 | -0.23 |
    | BLEU-4 (full) | 0.34 | 0.35 | +0.01 |
- **结论**: 热力图严重退化,文本无变化。attention loss 是 where 分支的主要监督源,符合预期。

- **命令**:
    ```bash
    deepspeed ... --attn_loss_weight=0.0 --exp_name="Attn-7b-ablation-no-attn-loss"
    ```
```

## 四、配置版本化 (最容易忽略但超重要)

`train_ds.py` 的所有 args 在训练启动时会打印到 stdout(见 `main()` 开头),但**没有持久化**。必须手动补一道:

**方案 A(推荐)**: 训练启动时 dump args 到 ckpt 目录
在 `train_ds.py::main` 合适位置加(或作为 Claude 要完成的第一个任务):
```python
import json
if args.local_rank == 0:
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
```

**方案 B(必须)**: `git` 分支 / commit
**每次**启动长训前:
```bash
git add -A && git commit -m "Freeze for experiment Attn-7b-xxx"
git tag exp/Attn-7b-xxx
git push && git push --tags
```
别嫌烦,关键时刻救命。

## 五、结果对比工具

### 快速看 TensorBoard 对比多个实验

```bash
tensorboard --logdir_spec \
  baseline:./runs/Attn-7b-baseline-reproduce,\
focal:./runs/Attn-7b-loss-focal,\
noattn:./runs/Attn-7b-ablation-no-attn-loss \
  --host=0.0.0.0 --port=6006
```

每条曲线会自动带 prefix,方便肉眼对比。

### 汇总 eval 结果到一张表

写一个 `scripts/collect_results.py`(Claude 要么找这个文件要么帮你写一个):

```python
import os, re, pandas as pd

RUNS = "./runs"
rows = []
for exp in os.listdir(RUNS):
    for ts in sorted(os.listdir(os.path.join(RUNS, exp))):
        log = os.path.join(RUNS, exp, ts, "log_test.txt")
        if not os.path.exists(log):
            continue
        txt = open(log).read()
        row = {"exp": exp, "timestamp": ts}
        for m in re.finditer(r"(\w+):\s*([\d.]+)", txt):
            row[m.group(1)] = float(m.group(2))
        rows.append(row)
pd.DataFrame(rows).to_csv("all_results.csv", index=False)
```

## 六、哪个 ckpt 算"最好"

训练里的 `is_best` 条件是**任意一个**指标 > 历史最好。所以 `best_ckpt_model_epoch*` 可能有多个,对应不同指标的最好。

**你要的是什么"最好"?**:
- 论文主表: 一般是 CC + KLD 的组合最好
- 某个特定下游: 对应那个指标最好

决策:
```bash
# 列出所有 meta_log pth 文件,按名字排序看指标
ls -1 runs/<exp_name>/meta_log_*.pth | sort
# 选你要的那个对应的 best_ckpt_model_epoch<N>
```

文件名里 `tl`=total loss, `wtl`=what loss, `wyl`=why loss, `al`=attn loss, `sim`/`cc`/`kld`/`nss`/`aucb`/`aucj` 是 6 个 attn 指标。

## 七、长期维护

- 磁盘紧张时,**保留** ckpt + log,**删掉** TensorBoard 的事件文件(用 `find . -name "events.out.tfevents*" -mtime +30 -delete`)
- 非 best ckpt 训练完就删,`ckpt_model/` 每个目录都是十几 GB
- `eval_saving/`(大量 jpg)和 `eval_text/`(大量 txt)也是磁盘杀手,整理完结果就清

## 八、给 Claude Code 的具体提示

当用户让你**启动一次实验**,你要:
1. 问清楚: `exp_name` 叫啥?动机是什么?对比基线是哪个?
2. 确认 `git status` 干净,建议 commit
3. 写好完整命令并告知预计时长和显存
4. 跑完后主动更新 `experiments.md`

当用户让你**对比实验结果**,你要:
1. 用 `ls runs/` 看所有 exp
2. 对每个找到 `log_test.txt` 或最新 `meta_log_*.pth`
3. 生成对比 markdown 表格
4. 指出显著差异并给假设解释(但要说明"这是猜测,不是验证")

---

## 五、实时监控 / 训练可视化 (按需选模式)

训练 80+ 小时,你需要随时看进展。三种模式各有适用场景,**按用户实际处境选**:

| 用户处境 | 推荐模式 | Reference |
| --- | --- | --- |
| 本机 GUI 可用,想看交互式曲线 | TensorBoard 增强配置 | `references/tensorboard-setup.md` |
| SSH 远程 / 不想开浏览器 / 想要 live ASCII | 终端 live dashboard | `references/terminal-dashboard.md` |
| 训练完想生成可归档的 PDF 快照 | 训练后批量出 PDF | `references/posttrain-pdf-report.md` |
| 想要全部 | 三个都装,按场景切 | 三个 reference 都读 |

### 决策树

用户说什么 → 用哪个:

- "**训练时实时看曲线**" / "TensorBoard 怎么开" → `tensorboard-setup.md`
- "**ssh 远程看不了 web**" / "终端里看" / "live dashboard" → `terminal-dashboard.md`
- "**训练完了出个总结**" / "归档 / commit 到 git" / "训练快照 PDF" → `posttrain-pdf-report.md`
- "**给我看现在 baseline-5090 跑得怎样**":
  1. 先 `ls runs/baseline-5090/` 确认有日志
  2. 默认建议 TensorBoard,如果用户在 SSH 切换到 terminal-dashboard

### 三种模式的核心差异

- **TensorBoard 增强**: 启动后浏览器看,适合**训练过程中**频繁查看,有交互(放缩/对比/平滑)
- **终端 dashboard**: 单终端窗口持续显示当前 step / loss / 各指标,适合**长时盯防**
- **PDF 快照**: 训练结束后一键产 PDF,适合**事后归档 + 实验对比时翻阅**

注意: 三种产物**互不替代** —— TensorBoard 是临时交互, dashboard 是 live 状态, PDF 是永久归档。建议三者并存。

