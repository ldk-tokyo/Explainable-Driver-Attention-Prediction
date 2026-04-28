"""
主对比表收集器: 从一组 exp 抽 6 attn + 4 text 指标,产出 CSV + booktabs LaTeX.

数据源:
- log_test.txt        : aggregate 数字 (跨 4 子集合一)
- attn_metrics_*.csv  : sample-level,可按 image_id 路径切子集做 per-dataset 表

CLI:
    .venv/bin/python scripts/paper/collect_main_results.py \\
        --exps "baseline-5090=Baseline" \\
        --out figures/tab1_main_results

    多 exp:
    .venv/bin/python scripts/paper/collect_main_results.py \\
        --exps "baseline-5090=Baseline" "decoder-pyramid-5090=+Pyramid" \\
        --out figures/tab1_main_results --per-dataset

模块调用:
    from scripts.paper.collect_main_results import run
    run(exps=[("baseline-5090", "Baseline")],
        out_dir=Path("figures/tab1_main_results"),
        per_dataset=True)

产出:
    raw.csv                              # 主表数据 (method × metric, 4 子集合一)
    main_table.tex                       # booktabs LaTeX, attn 6 列 + text 4 列
    per_dataset.csv                      # (method, dataset) × 6 attn (legacy)
    per_dataset.tex                      # method × dataset 多行 LaTeX (legacy)
    per_dataset_full_<exp>.csv           # 每 method 一份, 行=DS+Mean, 列=10 指标+meta
    per_dataset_transposed_<exp>.tex     # 行=metric, 列=BDDA/DReyeVE/LBW/DADA/Mean
                                         # NSS 行带 dagger + footnote (BDDA NaN 比例)

数据源优先级 (per-dataset 模式):
    1. runs_eval/<exp>-eval-<DS>/text_eval/<latest>/  (单 DS, 含文本指标)
    2. runs/<exp>/attn_eval/<latest>/                 (4 子集合一, 仅 attn, fallback)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# ---------- 配置 ----------

ATTN_METRICS = ["CC", "KLD", "SIM", "NSS", "AUC_B", "AUC_J"]
TEXT_METRICS = ["Bleu_4", "Meteor", "Rouge", "CiderR"]
HIGHER_BETTER = {
    "CC": True, "KLD": False, "SIM": True, "NSS": True,
    "AUC_B": True, "AUC_J": True,
    "Bleu_4": True, "Meteor": True, "Rouge": True, "CiderR": True,
}
DATASETS = ["BDDA", "DReyeVE", "LBW", "DADA"]

# CSV 里是小写,LaTeX 里要大写;统一在外部用大写,parse 时映射
CSV_TO_DISPLAY = {"cc": "CC", "kld": "KLD", "sim": "SIM",
                   "nss": "NSS", "auc_b": "AUC_B", "auc_j": "AUC_J"}


# ---------- 解析 ----------

def find_latest_eval_dir(exp: str) -> Path | None:
    """找 runs/<exp>/attn_eval/<timestamp>/ 中最新的目录."""
    base = Path(f"runs/{exp}")
    if not base.exists():
        return None
    cands = [d for d in base.rglob("attn_eval/*/") if d.is_dir()]
    if not cands:
        # 兼容: 训练 val 也产 attn_metrics_<rank>.csv,放在 runs/<exp>/<timestamp>/
        cands = [d for d in base.rglob("*/") if (d / "attn_metrics_0.csv").exists()
                 and "attn_eval" not in str(d)]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def parse_log_test(path: Path) -> dict[str, float]:
    """从 log_test.txt 抽 6 attn + 4 text(complete 段). 缺失返回空 dict.

    train_ds.py:415-419 的格式:
        Attn metrics:
        CC: 0.x, KLD: 0.x, SIM: 0.x, NSS: 0.x, AUC_B: 0.x, AUC_J: 0.x
        Text metrics (Complete):
        Bleu_4: 0.x, Bleu_3: 0.x, ..., Meteor: 0.x, Rouge: 0.x, Cider: 0.x, CiderR: 0.x
        Text metrics (What part): ...
        Text metrics (Why part):  ...

    我们只抽 (Complete) 段;What/Why 备份用,这里不放主表.
    """
    if not path.exists():
        return {}
    text = path.read_text()

    # 把 "Text metrics (Complete):" 之后到下一段之前的文本截出来
    complete_block = ""
    m_blk = re.search(
        r"Text metrics \(Complete\):\s*\n(.+?)(?=\n(?:Text metrics|\Z))",
        text, re.DOTALL,
    )
    if m_blk:
        complete_block = m_blk.group(1)

    # Attn 在所有段之前,直接全文抓即可(只会匹配第一组)
    out: dict[str, float] = {}
    for key in ATTN_METRICS:
        m = re.search(rf"\b{re.escape(key)}:\s*([0-9.eE+-]+)", text)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass

    # Text 限定在 Complete block 内
    text_keys = {"Bleu_4": "Bleu_4", "Meteor": "Meteor",
                 "Rouge": "Rouge", "CiderR": "CiderR"}
    for display, key in text_keys.items():
        m = re.search(rf"\b{re.escape(key)}:\s*([0-9.eE+-]+)", complete_block)
        if m:
            try:
                out[display] = float(m.group(1))
            except ValueError:
                pass
    return out


def load_attn_csv(eval_dir: Path) -> pd.DataFrame:
    """读 eval 目录下所有 rank 的 attn_metrics_*.csv 拼起来,加 dataset 列."""
    csv_paths = sorted(eval_dir.glob("attn_metrics_*.csv"))
    if not csv_paths:
        return pd.DataFrame()
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)

    # 从 image_id 路径抽子集名: ./dataset/<DATASET>/test/...
    def infer_ds(p: str) -> str:
        m = re.search(r"/dataset/([^/]+)/", str(p))
        return m.group(1) if m else "UNKNOWN"

    if "image_id" in df.columns:
        df["dataset"] = df["image_id"].apply(infer_ds)
    return df


def aggregate_overall(df: pd.DataFrame) -> dict[str, float]:
    """全样本 mean (NaN skip)."""
    out = {}
    for csv_col, disp in CSV_TO_DISPLAY.items():
        if csv_col in df.columns:
            out[disp] = float(df[csv_col].mean(skipna=True))
    return out


def aggregate_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """按 dataset 列分组 mean. 返回 DataFrame, index=dataset, columns=ATTN_METRICS."""
    if "dataset" not in df.columns:
        return pd.DataFrame()
    grp_cols = [c for c in CSV_TO_DISPLAY if c in df.columns]
    out = df.groupby("dataset")[grp_cols].mean()
    out = out.rename(columns=CSV_TO_DISPLAY)
    # 排序: BDDA / DReyeVE / LBW / DADA, 缺的跳过
    keep = [d for d in DATASETS if d in out.index]
    return out.loc[keep] if keep else out


def find_per_dataset_eval_dirs(exp: str) -> dict[str, Path]:
    """找 runs_eval/<exp>-eval-<DS>/text_eval/<latest>/ —— per-dataset eval 队列产物.

    要求 log_test.txt 存在(=eval 跑完). 没跑完的 DS 不返回, 由调用方 fallback 到 4 子集 CSV.
    """
    out: dict[str, Path] = {}
    for ds in DATASETS:
        base = Path(f"runs_eval/{exp}-eval-{ds}/text_eval")
        if not base.exists():
            continue
        cands = [d for d in base.iterdir() if d.is_dir() and (d / "log_test.txt").exists()]
        if not cands:
            continue
        out[ds] = max(cands, key=lambda p: p.stat().st_mtime)
    return out


def collect_method_per_dataset(exp: str, fallback_dir: Path | None) -> pd.DataFrame:
    """单 method 的 per-DS 指标. 行=DS, 列=ATTN+TEXT+(n_total,n_nss_valid).

    数据源优先级:
    1. runs_eval/<exp>-eval-<DS>/text_eval/<latest>/ (单 DS, 含文本指标)
    2. fallback_dir 的 CSV 按 dataset 列分组 (4 子集合一, 仅 attn)
    """
    rows: dict[str, dict] = {}
    per_ds_dirs = find_per_dataset_eval_dirs(exp)

    for ds, d in per_ds_dirs.items():
        log_metrics = parse_log_test(d / "log_test.txt")
        df = load_attn_csv(d)
        attn = aggregate_overall(df) if not df.empty else {}
        # log_test 的 attn 是 NaN-skip-mean, CSV 重算应一致; 都给, attn 优先 CSV
        row = {**log_metrics, **attn}
        if not df.empty:
            row["n_total"] = int(len(df))
            row["n_nss_valid"] = (
                int(df["nss"].notna().sum()) if "nss" in df.columns else None
            )
        rows[ds] = row

    # 缺失的 DS 走 fallback (4 子集 CSV 分组)
    if fallback_dir is not None:
        df_all = load_attn_csv(fallback_dir)
        if not df_all.empty and "dataset" in df_all.columns:
            for ds in DATASETS:
                if ds in rows or ds not in df_all["dataset"].unique():
                    continue
                df_ds = df_all[df_all["dataset"] == ds]
                attn = aggregate_overall(df_ds)
                row = dict(attn)
                row["n_total"] = int(len(df_ds))
                row["n_nss_valid"] = (
                    int(df_ds["nss"].notna().sum()) if "nss" in df_ds.columns else None
                )
                rows[ds] = row

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(rows, orient="index")
    keep = [d for d in DATASETS if d in df.index]
    return df.loc[keep]


def add_mean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """追加两个伪行: 'Mean (weighted)' 和 'Mean (simple)'.

    Weighted by n_total, 但 NSS 用 n_nss_valid (因 NaN 在聚合时已被 skip).
    简单平均忽略 NaN.
    """
    if df.empty:
        return df
    out = df.copy()
    metric_cols = [c for c in ATTN_METRICS + TEXT_METRICS if c in out.columns]
    n_total = out["n_total"] if "n_total" in out.columns else None
    n_nss = out["n_nss_valid"] if "n_nss_valid" in out.columns else n_total

    weighted: dict = {}
    simple: dict = {}
    for c in metric_cols:
        vals = out[c].dropna()
        if vals.empty:
            weighted[c] = simple[c] = None
            continue
        simple[c] = float(vals.mean())
        w = (n_nss if c == "NSS" else n_total)
        if w is None:
            weighted[c] = simple[c]
            continue
        w_sub = w.reindex(vals.index)
        if w_sub.isna().all() or float(w_sub.fillna(0).sum()) == 0:
            weighted[c] = simple[c]
        else:
            weighted[c] = float((vals * w_sub.fillna(0)).sum() / w_sub.fillna(0).sum())

    out.loc["Mean (weighted)"] = weighted
    out.loc["Mean (simple)"] = simple
    return out


# ---------- LaTeX 生成 ----------

def _fmt_cell(val, best_val, higher_better, prec=3):
    if val is None or pd.isna(val):
        return "--"
    is_best = (best_val is not None and not pd.isna(best_val)
               and abs(val - best_val) < 10 ** (-prec - 1))
    s = f"{val:.{prec}f}"
    return rf"\textbf{{{s}}}" if is_best else s


def _arrow(metric: str) -> str:
    return r"$\uparrow$" if HIGHER_BETTER[metric] else r"$\downarrow$"


def _tex_metric_name(m: str) -> str:
    return m.replace("_", r"\_")


def write_main_table(df: pd.DataFrame, out_path: Path):
    """单行 = 一个 method, 列 = ATTN + TEXT."""
    cols = [c for c in ATTN_METRICS + TEXT_METRICS if c in df.columns]

    best = {}
    for c in cols:
        valid = df[c].dropna()
        if len(valid):
            best[c] = valid.max() if HIGHER_BETTER[c] else valid.min()
        else:
            best[c] = None

    n_attn = sum(1 for c in cols if c in ATTN_METRICS)
    n_text = sum(1 for c in cols if c in TEXT_METRICS)

    lines = []
    if n_text > 0:
        lines.append(r"\begin{tabular}{l " + "c " * n_attn + "| " + "c " * n_text + "}")
    else:
        lines.append(r"\begin{tabular}{l " + "c " * n_attn + "}")
    lines.append(r"\toprule")

    # 分组 header (只在有 text 时给)
    if n_text > 0:
        grp = " & " + rf"\multicolumn{{{n_attn}}}{{c|}}{{Where (Attn)}}"
        grp += " & " + rf"\multicolumn{{{n_text}}}{{c}}{{What \& Why (Text)}}" + r" \\"
        lines.append(grp)
        # cmidrule
        lines.append(rf"\cmidrule(lr){{2-{1 + n_attn}}} \cmidrule(lr){{{2 + n_attn}-{1 + n_attn + n_text}}}")

    header = "Method"
    for c in cols:
        header += f" & {_tex_metric_name(c)}{_arrow(c)}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for method in df.index:
        row = method
        for c in cols:
            row += " & " + _fmt_cell(df.loc[method, c], best[c], HIGHER_BETTER[c])
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path.write_text("\n".join(lines) + "\n")


def write_per_dataset_transposed(method: str, df: pd.DataFrame, out_path: Path,
                                  nss_note: str = ""):
    """转置 per-dataset 表: 行=metric, 列=DS + Mean(weighted) + Mean(simple).

    df: index 已经包含 DS 行 + 'Mean (weighted)' + 'Mean (simple)' 行,
        columns 含 ATTN_METRICS + TEXT_METRICS (+ meta).
    nss_note: 若非空, 加在 NSS 标签上的 footnote 引用 + 注释行.
    最佳粗体只在 DS 列之间对比, 不算 Mean 列.
    """
    if df.empty:
        return
    metric_rows = [c for c in ATTN_METRICS + TEXT_METRICS if c in df.columns]
    ds_cols = [d for d in DATASETS if d in df.index]
    mean_cols = [c for c in ("Mean (weighted)", "Mean (simple)") if c in df.index]
    col_order = ds_cols + mean_cols
    if not metric_rows or not col_order:
        return

    # 各 metric 行内, 最佳值只在 DS 列之间比较
    best: dict[str, float | None] = {}
    for m in metric_rows:
        sub = df.loc[ds_cols, m].dropna() if ds_cols else pd.Series(dtype=float)
        best[m] = (sub.max() if HIGHER_BETTER[m] else sub.min()) if len(sub) else None

    n_ds = len(ds_cols)
    n_mean = len(mean_cols)
    lines: list[str] = []
    lines.append(rf"% Method: {method}")
    if nss_note:
        lines.append(rf"% NSS note: {nss_note}")
    spec = "l " + "c " * n_ds + ("| " + "c " * n_mean if n_mean else "")
    lines.append(rf"\begin{{tabular}}{{{spec.strip()}}}")
    lines.append(r"\toprule")
    header = "Metric"
    for c in col_order:
        header += f" & {c}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for m in metric_rows:
        label = _tex_metric_name(m) + _arrow(m)
        if m == "NSS" and nss_note:
            label += r"$^{\dagger}$"
        row = label
        for c in col_order:
            v = df.loc[c, m] if (c in df.index and m in df.columns) else None
            b = best[m] if c in DATASETS else None
            row += " & " + _fmt_cell(v, b, HIGHER_BETTER[m])
        row += r" \\"
        lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if nss_note:
        lines.append(rf"% \multicolumn{{{1+n_ds+n_mean}}}{{l}}{{$^{{\dagger}}$ {nss_note}}}")
    out_path.write_text("\n".join(lines) + "\n")


def write_per_dataset_table(per_ds: dict[str, pd.DataFrame], out_path: Path):
    """每行 (method, dataset),只 attn 6 列.

    per_ds: {display_name: per-dataset DataFrame}
    """
    rows = []
    for method, df_ds in per_ds.items():
        for ds in df_ds.index:
            r = {"Method": method, "Dataset": ds}
            for c in ATTN_METRICS:
                r[c] = df_ds.loc[ds, c] if c in df_ds.columns else None
            rows.append(r)
    if not rows:
        return
    big = pd.DataFrame(rows).set_index(["Method", "Dataset"])

    # 对每 (Dataset, metric) 找列内最佳; 不同 dataset 间不互比
    best = {}
    for ds in big.index.get_level_values("Dataset").unique():
        for c in ATTN_METRICS:
            sub = big.xs(ds, level="Dataset")[c].dropna()
            if len(sub):
                best[(ds, c)] = sub.max() if HIGHER_BETTER[c] else sub.min()

    cols = ATTN_METRICS
    lines = [r"\begin{tabular}{l l " + "c " * len(cols) + "}", r"\toprule"]
    header = "Method & Dataset"
    for c in cols:
        header += f" & {_tex_metric_name(c)}{_arrow(c)}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    last_method = None
    for (method, ds), _ in big.iterrows():
        method_cell = method if method != last_method else ""
        last_method = method
        row = f"{method_cell} & {ds}"
        for c in cols:
            v = big.loc[(method, ds), c]
            b = best.get((ds, c))
            row += " & " + _fmt_cell(v, b, HIGHER_BETTER[c])
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n")


# ---------- 主流程 ----------

def run(exps: list[tuple[str, str]], out_dir: Path, per_dataset: bool = False):
    """exps: [(run_name, display_name), ...]"""
    out_dir.mkdir(parents=True, exist_ok=True)

    main_rows = []
    per_ds_tables: dict[str, pd.DataFrame] = {}

    for run_name, display in exps:
        eval_dir = find_latest_eval_dir(run_name)
        if eval_dir is None:
            print(f"[WARN] no eval dir under runs/{run_name}/, skipping")
            continue
        print(f"[{display}] using {eval_dir}")

        # CSV: attn aggregates (overall + per-dataset)
        df_csv = load_attn_csv(eval_dir)
        attn_overall = aggregate_overall(df_csv) if not df_csv.empty else {}

        # log_test.txt: text metrics (CSV 没有), 顺便交叉验证 attn
        log_metrics = parse_log_test(eval_dir / "log_test.txt")

        # attn 优先用 CSV (sample-level mean, 可信);log 的当 sanity check
        merged = {**log_metrics, **attn_overall}

        # CSV 0 / NaN 的情况(text 全 0 表示没跑 --eval_text): 仍写入,LaTeX 显示原值
        row = {"Method": display, **{k: merged.get(k) for k in ATTN_METRICS + TEXT_METRICS}}
        main_rows.append(row)

        # per-dataset (legacy: 仅 attn, 走 4 子集 CSV 分组)
        if per_dataset and not df_csv.empty:
            per_ds_tables[display] = aggregate_per_dataset(df_csv)

        # per-dataset (新: 含文本指标 + Mean 列, 优先用 runs_eval/<exp>-eval-<DS>/ 队列产物)
        if per_dataset:
            df_pd = collect_method_per_dataset(run_name, eval_dir)
            if not df_pd.empty:
                df_pd_full = add_mean_columns(df_pd)
                df_pd_full.to_csv(
                    out_dir / f"per_dataset_full_{run_name}.csv",
                    float_format="%.6f",
                )
                # NSS NaN 注记 —— 任何 DS 出现 valid<total 都列出来
                notes = []
                if "n_total" in df_pd.columns and "n_nss_valid" in df_pd.columns:
                    for ds in df_pd.index:
                        n_t = df_pd.loc[ds, "n_total"]
                        n_v = df_pd.loc[ds, "n_nss_valid"]
                        if pd.notna(n_t) and pd.notna(n_v) and n_v < n_t:
                            pct = 100 * (n_t - n_v) / n_t
                            notes.append(
                                f"{ds}: NSS computed on {int(n_v)}/{int(n_t)} "
                                f"valid samples ({pct:.1f}% NaN due to gazemap "
                                f"threshold artifact)"
                            )
                nss_note = "; ".join(notes)
                write_per_dataset_transposed(
                    display, df_pd_full,
                    out_dir / f"per_dataset_transposed_{run_name}.tex",
                    nss_note,
                )
                print(f"[OK] {out_dir}/per_dataset_full_{run_name}.csv")
                print(f"[OK] {out_dir}/per_dataset_transposed_{run_name}.tex")

    if not main_rows:
        print("[ERROR] no rows collected, nothing written")
        return

    df_main = pd.DataFrame(main_rows).set_index("Method")
    df_main.to_csv(out_dir / "raw.csv", float_format="%.6f")
    write_main_table(df_main, out_dir / "main_table.tex")
    print(f"[OK] {out_dir}/raw.csv")
    print(f"[OK] {out_dir}/main_table.tex")
    print()
    print(df_main.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "  --  "))

    if per_dataset and per_ds_tables:
        # 拼一个 per-dataset CSV
        big = pd.concat(
            {m: t for m, t in per_ds_tables.items()}, names=["Method", "Dataset"]
        )
        big.to_csv(out_dir / "per_dataset.csv", float_format="%.6f")
        write_per_dataset_table(per_ds_tables, out_dir / "per_dataset.tex")
        print(f"[OK] {out_dir}/per_dataset.csv")
        print(f"[OK] {out_dir}/per_dataset.tex")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--exps", nargs="+", required=True,
                   help='Each: "run_name=Display Name"')
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--per-dataset", action="store_true")
    args = p.parse_args()

    exps = []
    for spec in args.exps:
        if "=" not in spec:
            raise SystemExit(f"--exps spec must be 'run=display', got: {spec!r}")
        run_name, display = spec.split("=", 1)
        exps.append((run_name.strip(), display.strip()))

    run(exps=exps, out_dir=args.out, per_dataset=args.per_dataset)


if __name__ == "__main__":
    cli()
