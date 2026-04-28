"""
LLada 训练 live dashboard (终端版)。

用法:
    python scripts/live_dashboard.py runs/baseline-5090
    python scripts/live_dashboard.py runs/baseline-5090 runs/decoder-pyramid-5090

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

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    import pynvml
    pynvml.nvmlInit()
    HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


LOSS_TAGS = ["train/loss", "train/ce_loss", "train/attn_loss",
             "train/ce_what_losses", "train/ce_why_losses"]
ATTN_TAGS = ["val/cc", "val/kld", "val/sim", "val/nss", "val/auc_b", "val/auc_j"]
TEXT_TAGS = ["val/bleu_4", "val/meteor", "val/rouge", "val/ciderR"]


def get_latest_scalar(ea: EventAccumulator, tag: str):
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    if not events:
        return None
    last = events[-1]
    return (last.step, last.value)


def make_exp_table(run_dir: Path) -> Panel:
    event_files = list(run_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return Panel(Text(f"No TensorBoard events yet under {run_dir}", style="yellow"),
                     title=str(run_dir.name), border_style="yellow")

    latest_event = max(event_files, key=lambda p: p.stat().st_mtime)
    ea = EventAccumulator(str(latest_event.parent))
    ea.Reload()

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
    layout = Layout()
    layout.split_column(
        Layout(name="gpu", size=8),
        Layout(name="experiments"),
    )
    layout["gpu"].update(make_gpu_panel())

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
