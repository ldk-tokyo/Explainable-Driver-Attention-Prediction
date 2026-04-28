# 终端 Live Dashboard

不开浏览器,直接在终端里持续显示训练状态。适合 SSH 远程 / 不想切窗口 / 喜欢 TUI 的场景。

## §1 依赖

```bash
pip install rich tensorboard
# rich: 终端富文本库
# tensorboard: 用来读 event 文件
```

## §2 完整脚本

放到 `scripts/live_dashboard.py`:

```python
"""
LLada 训练 live dashboard (终端版)。

用法:
    python scripts/live_dashboard.py runs/baseline-5090
    python scripts/live_dashboard.py runs/baseline-5090 runs/decoder-pyramid-5090

功能:
- 持续读 TensorBoard event 文件
- 显示当前 step / loss / 6 个 attn 指标 / 文本指标
- GPU 显存 + 温度 + 功耗
- 多 exp 并排对比 (传多个 run 路径)

按 Ctrl+C 退出。
"""
import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# 读 TensorBoard event 文件
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# GPU 监控 (可选)
try:
    import pynvml
    pynvml.nvmlInit()
    HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


# 关心的 scalar tag (按显示顺序)
LOSS_TAGS = ["loss", "ce_loss", "attn_loss", "ce_what_loss", "ce_why_loss"]
ATTN_TAGS = ["val/cc", "val/kld", "val/sim", "val/nss", "val/auc_b", "val/auc_j"]
TEXT_TAGS = ["val/bleu_4", "val/meteor", "val/rouge", "val/cider_r"]


def get_latest_scalar(ea: EventAccumulator, tag: str):
    """取该 tag 最新的 (step, value);取不到返回 None"""
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    if not events:
        return None
    last = events[-1]
    return (last.step, last.value)


def make_exp_table(run_dir: Path) -> Panel:
    """构造单 exp 的指标 panel"""
    # 找最新的 event 文件
    event_files = list(run_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return Panel(Text(f"No TensorBoard events yet under {run_dir}", style="yellow"),
                     title=str(run_dir.name), border_style="yellow")

    latest_event = max(event_files, key=lambda p: p.stat().st_mtime)
    ea = EventAccumulator(str(latest_event.parent))
    ea.Reload()

    # 表格
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Metric", style="bold cyan")
    table.add_column("Step", justify="right", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("[bold]LOSS[/bold]", "", "")
    for tag in LOSS_TAGS:
        v = get_latest_scalar(ea, tag)
        if v is not None:
            step, val = v
            table.add_row(f"  {tag}", f"{step}", f"{val:.4f}")

    table.add_row("", "", "")
    table.add_row("[bold]ATTN VAL[/bold] (cc/sim/nss↑ kld↓)", "", "")
    for tag in ATTN_TAGS:
        v = get_latest_scalar(ea, tag)
        if v is not None:
            step, val = v
            short = tag.split("/")[-1]
            color = "green" if "kld" not in tag else "red"
            table.add_row(f"  {short}", f"{step}", f"[{color}]{val:.4f}[/{color}]")

    has_text = any(get_latest_scalar(ea, t) is not None for t in TEXT_TAGS)
    if has_text:
        table.add_row("", "", "")
        table.add_row("[bold]TEXT VAL[/bold]", "", "")
        for tag in TEXT_TAGS:
            v = get_latest_scalar(ea, tag)
            if v is not None:
                step, val = v
                short = tag.split("/")[-1]
                table.add_row(f"  {short}", f"{step}", f"{val:.4f}")

    # 文件 mtime 作为 "上次更新"
    age = time.time() - latest_event.stat().st_mtime
    age_s = f"{int(age)}s ago" if age < 60 else f"{int(age // 60)}m ago"
    title = f"{run_dir.name}  (last update: {age_s})"
    border = "green" if age < 120 else "yellow" if age < 600 else "red"

    return Panel(table, title=title, border_style=border)


def make_gpu_panel() -> Panel:
    if not GPU_AVAILABLE:
        return Panel(Text("pynvml not available; pip install pynvml",
                          style="dim"), title="GPU")

    info = pynvml.nvmlDeviceGetMemoryInfo(HANDLE)
    util = pynvml.nvmlDeviceGetUtilizationRates(HANDLE)
    temp = pynvml.nvmlDeviceGetTemperature(HANDLE, pynvml.NVML_TEMPERATURE_GPU)
    power = pynvml.nvmlDeviceGetPowerUsage(HANDLE) / 1000.0

    used_gb = info.used / 1e9
    total_gb = info.total / 1e9
    mem_pct = info.used / info.total * 100

    temp_color = "green" if temp < 75 else "yellow" if temp < 85 else "red"
    mem_color = "green" if mem_pct < 90 else "yellow" if mem_pct < 95 else "red"

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Field", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Mem",   f"[{mem_color}]{used_gb:.1f} / {total_gb:.1f} GB ({mem_pct:.0f}%)[/{mem_color}]")
    table.add_row("GPU%",  f"{util.gpu}%")
    table.add_row("Temp",  f"[{temp_color}]{temp}°C[/{temp_color}]")
    table.add_row("Power", f"{power:.0f} W / 575 W")

    return Panel(table, title="RTX 5090", border_style="cyan")


def build_layout(run_dirs):
    """主 layout: GPU panel 顶部 + 各 exp panel 底部"""
    layout = Layout()
    layout.split_column(
        Layout(name="gpu", size=8),
        Layout(name="experiments"),
    )
    layout["gpu"].update(make_gpu_panel())

    # 多 exp 并排
    if len(run_dirs) == 1:
        layout["experiments"].update(make_exp_table(run_dirs[0]))
    else:
        sub_layouts = [Layout(name=f"exp_{i}") for i in range(len(run_dirs))]
        layout["experiments"].split_row(*sub_layouts)
        for sl, rd in zip(sub_layouts, run_dirs):
            sl.update(make_exp_table(rd))

    return layout


def main():
    parser = argparse.ArgumentParser(description="LLada Live Dashboard")
    parser.add_argument("run_dirs", nargs="+", help="One or more runs/<exp_name>/")
    parser.add_argument("--refresh", type=float, default=5.0,
                        help="Refresh interval in seconds (default 5)")
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    for rd in run_dirs:
        if not rd.exists():
            print(f"WARN: {rd} does not exist yet (will retry)")

    console = Console()
    with Live(build_layout(run_dirs), refresh_per_second=1, screen=True,
              console=console) as live:
        try:
            while True:
                time.sleep(args.refresh)
                live.update(build_layout(run_dirs))
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped.[/yellow]")


if __name__ == "__main__":
    main()
```

## §3 用法

### 单 exp 监控

```bash
python scripts/live_dashboard.py runs/baseline-5090
```

显示:
```
┌─ RTX 5090 ──────────────────────────────────┐
│ Mem    28.3 / 31.4 GB (90%)                 │
│ GPU%   97%                                   │
│ Temp   78°C                                  │
│ Power  548 W / 575 W                         │
└──────────────────────────────────────────────┘
┌─ baseline-5090  (last update: 12s ago) ─────┐
│ LOSS                                         │
│   loss              4500    1.2345          │
│   ce_loss           4500    0.8765          │
│   attn_loss         4500    0.3210          │
│   ...                                        │
│ ATTN VAL (cc/sim/nss↑ kld↓)                 │
│   cc                4000    0.7123          │
│   kld               4000    1.2345 (red)    │
│   sim               4000    0.6789          │
│   ...                                        │
└──────────────────────────────────────────────┘
```

### 多 exp 并排对比

```bash
python scripts/live_dashboard.py runs/baseline-5090 runs/decoder-pyramid-5090
```

两列并排,可以看哪个 exp 在哪个指标领先。

### 自定义刷新

```bash
# 慢一点 (省 CPU)
python scripts/live_dashboard.py runs/baseline-5090 --refresh 30

# 快一点 (调试时)
python scripts/live_dashboard.py runs/baseline-5090 --refresh 1
```

### 后台运行 (TUI 不适合 nohup,建议 tmux)

```bash
tmux new -s monitor
python scripts/live_dashboard.py runs/baseline-5090
# Ctrl+B D 退出 tmux,dashboard 继续
tmux attach -t monitor   # 回来看
```

## §4 颜色含义

- **绿色 panel border**: 数据 < 2 分钟新,训练正常推进
- **黄色 panel border**: 数据 2-10 分钟旧,可能在 validate 或 ckpt 保存
- **红色 panel border**: 数据 > 10 分钟旧,可能 hang 了 (赶快查 train.log)
- **绿色数字**: higher is better 指标 (cc/sim/nss/auc)
- **红色数字**: lower is better 指标 (kld)
- **GPU 温度**: 绿 (<75°C) → 黄 (75-85°C) → 红 (>85°C, 立刻降功耗或开窗)

## §5 vs TensorBoard 对比

| 维度 | TensorBoard | 终端 dashboard |
| --- | --- | --- |
| 启动 | 要开浏览器 | 一行命令 |
| 历史曲线 | 完整可缩放 | 只看最新值 |
| GPU 状态 | 需手动加 scalar | 直接显示 |
| 远程 | 要端口转发 | SSH 直接看 |
| 资源占用 | 中等 | 极轻 |
| 多 run 对比 | 强 (regex/smooth) | 简单并排 |

**实际推荐**: 训练第一周用 TensorBoard 探索 layout,熟悉指标范围;之后日常盯防用终端 dashboard,要画细节图时再开 TensorBoard。
