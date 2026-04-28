# TensorBoard 增强配置

LLada 默认已经写 TensorBoard,但 layout 不友好,多 exp 对比也没分组。这里给增强版。

## §1 启动

最简单:
```bash
tensorboard --logdir=./runs --host=0.0.0.0 --port=6006
# 浏览器开 http://localhost:6006
```

加 reload 间隔(每 30 秒刷新一次,默认 5 分钟太慢):
```bash
tensorboard --logdir=./runs --port=6006 --reload_interval=30
```

多机器 / 远程访问:
```bash
# 服务器
tensorboard --logdir=./runs --host=0.0.0.0 --port=6006

# 本地 SSH 端口转发
ssh -L 16006:localhost:6006 user@server
# 本地浏览器开 http://localhost:16006
```

## §2 自定义 Layout (强烈推荐)

默认 TensorBoard 把所有 scalar 平铺,你会被淹没。LLada 至少有这些 scalar:
- `loss`, `ce_loss`, `attn_loss`, `ce_what_loss`, `ce_why_loss`(训练)
- `val/cc`, `val/kld`, `val/sim`, `val/nss`, `val/auc_b`, `val/auc_j`
- `val/bleu_4`, `val/meteor`, `val/rouge`, `val/cider_r`(开 `--eval_text` 时)

用 **Custom Scalars**(TensorBoard 自带功能)分组。

### 一次性配置脚本

在训练**开始前**或任何时候跑:

```python
# scripts/setup_tensorboard_layout.py
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto import summary_pb2
import os, time

LAYOUT = layout_pb2.Layout(category=[
    layout_pb2.Category(
        title="Training Loss (核心)",
        chart=[
            layout_pb2.Chart(title="Total Loss",
                multiline=layout_pb2.MultilineChartContent(tag=[r"^loss$"])),
            layout_pb2.Chart(title="Loss Components",
                multiline=layout_pb2.MultilineChartContent(
                    tag=[r"ce_loss", r"attn_loss", r"ce_what_loss", r"ce_why_loss"])),
        ]),
    layout_pb2.Category(
        title="Val · 6 Attention Metrics",
        chart=[
            layout_pb2.Chart(title="Higher is Better (CC/SIM/NSS/AUC)",
                multiline=layout_pb2.MultilineChartContent(
                    tag=[r"val/cc", r"val/sim", r"val/nss", r"val/auc_b", r"val/auc_j"])),
            layout_pb2.Chart(title="Lower is Better (KLD)",
                multiline=layout_pb2.MultilineChartContent(tag=[r"val/kld"])),
        ]),
    layout_pb2.Category(
        title="Val · Text Metrics (What + Why)",
        chart=[
            layout_pb2.Chart(title="BLEU-4 / METEOR / ROUGE / CIDEr-R",
                multiline=layout_pb2.MultilineChartContent(
                    tag=[r"val/bleu_4", r"val/meteor", r"val/rouge", r"val/cider_r"])),
        ]),
    layout_pb2.Category(
        title="System",
        chart=[
            layout_pb2.Chart(title="Learning Rate",
                multiline=layout_pb2.MultilineChartContent(tag=[r"lr"])),
        ]),
])

# 写到 runs/ 顶层(全局生效,所有 exp 共用 layout)
os.makedirs("runs", exist_ok=True)
writer = EventFileWriter("runs", flush_secs=1)
summary = summary_pb2.Summary(value=[
    summary_pb2.Summary.Value(
        tag="custom_scalars__config__",
        metadata=summary_pb2.SummaryMetadata(
            plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name="custom_scalars"),
        ),
        tensor=summary_pb2.TensorProto(
            dtype=4, tensor_shape=summary_pb2.TensorShapeProto(),
            string_val=[LAYOUT.SerializeToString()],
        ),
    ),
])
event = summary_pb2.Event(wall_time=time.time(), step=0, summary=summary)
writer.add_event(event)
writer.close()
print("✓ TensorBoard custom layout written. Open Custom Scalars tab.")
```

跑一次:
```bash
python scripts/setup_tensorboard_layout.py
```

刷新 TensorBoard,在顶部找 **CUSTOM SCALARS** tab,你会看到分组好的 dashboard。

## §3 多实验对比 (smoothing + 选择性显示)

**核心痛点**: 跑了 5 个 decoder 变体,TensorBoard 默认全部叠加,曲线乱。

技巧:
1. **左下角 Runs 列表** 用 regex 过滤:
   - 只看 baseline + 当前 exp: `baseline-5090|decoder-pyramid-5090`
   - 只看 depth 扫描: `decoder-cross-depth\d+`
2. **Smoothing**: 默认 0.6,**改成 0.0** 看原始曲线(看噪声),改成 **0.95** 看趋势
3. **X 轴切换**: STEP(全局可比)→ RELATIVE(各 run 从 0 开始)→ WALL(实际时间)

## §4 (可选) 项目代码改动: 加更多 scalar

如果你想看 **GPU 利用率 / 显存占用** 也进 TensorBoard,在 `train_ds.py` 训练循环里加:

```python
import torch

if global_step % 10 == 0:
    tb_writer.add_scalar("system/gpu_mem_gb",
        torch.cuda.max_memory_allocated() / 1e9, global_step)
    tb_writer.add_scalar("system/gpu_util",
        torch.cuda.utilization(), global_step)
```

(`torch.cuda.utilization()` 需要 NVML,可能需要 `pip install pynvml`)

## §5 看 attn 可视化在 TensorBoard

LLada 训练时已经在 `train_vis/` 存 jpg。让它们也进 TensorBoard:

```python
# train_ds.py 里训练循环加
if global_step % 100 == 0 and rank == 0:
    pred_img = cv2.imread(latest_pred_jpg)  # BGR
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    tb_writer.add_image("train_vis/pred", pred_img, global_step, dataformats="HWC")
    # 同样加 gt
```

之后 TensorBoard **IMAGES** tab 可看预测随训练的演变,非常直观。

## §6 常见问题

| 症状 | 修复 |
| --- | --- |
| TensorBoard 启动后空白 | `--logdir` 路径错,要指 `runs/` 不是 `runs/baseline-5090/` |
| 新 step 不出现 | reload_interval 太长, 加 `--reload_interval=30` |
| Custom layout 不显示 | layout 文件没写到 logdir,重跑 `setup_tensorboard_layout.py` |
| 端口 6006 被占 | `--port=6007` 或 `lsof -ti:6006 | xargs kill` |
| SSH 端口转发慢 | 改用 sshfs 挂载 `runs/` 本地跑 TensorBoard |
