#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate all Adaptive-alpha CLIP experiment logs into CSV tables and PNGs.

Input is the output structure of adaptive_clip_cifar10.py:
runs/<experiment>/summary.json, epochs.csv, steps.csv

Usage:
    python clip_results_report.py --root runs --out reports/all_six
Dependencies:
    pip install numpy matplotlib

No model loading or retraining is needed.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_COLS = [
    "run", "method", "group", "seed", "model", "pretrained",
    "batch_size", "lr", "epochs_configured", "train_blocks",
    "fixed_alpha", "tau", "warmup_steps",
    "best_epoch", "best_acc_i2t", "best_error_count",
    "best_error_rate_pp", "best_map_t2i", "best_map_gap_pp",
    "best_score", "init_acc_i2t", "init_map_t2i",
    "last_epoch", "last_acc_i2t", "last_map_t2i",
    "delta_acc_from_pretrain_pp", "delta_map_from_pretrain_pp",
    "alpha_start", "alpha_end", "alpha_mean", "alpha_std",
    "alpha_min", "alpha_max", "total_train_seconds",
    "train_steps", "source",
]
EPOCH_COLS = [
    "run", "group", "seed", "epoch",
    "train_i2t", "train_t2i", "train_total",
    "alpha_mean", "alpha_min", "alpha_max",
    "val_acc_i2t", "val_map_t2i", "val_score", "seconds",
    "val_error_count", "val_map_gap_pp",
]
STEP_COLS = [
    "run", "group", "seed", "step", "epoch",
    "loss_i2t", "loss_t2i", "loss_total", "alpha",
]
PAIRED_COLS = [
    "seed", "baseline_run", "candidate_run", "candidate_group",
    "baseline_best_acc", "candidate_best_acc",
    "delta_acc_pp", "delta_i2t_error_count",
    "baseline_best_map", "candidate_best_map",
    "delta_map_pp", "comparable_hyperparams",
]
GROUP_COLS = [
    "group", "n_runs", "seeds",
    "best_acc_mean", "best_acc_std",
    "best_error_count_mean", "best_error_count_std",
    "best_map_mean", "best_map_std",
    "best_map_gap_pp_mean", "best_map_gap_pp_std",
    "best_score_mean", "best_score_std",
    "alpha_mean_across_runs", "train_seconds_mean",
]


def to_float(value, default=math.nan):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def to_int(value, default=None):
    value = to_float(value)
    if not math.isfinite(value):
        return default
    return int(value)


def as_cell(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, float):
        return round(value, 9)
    return value


def write_csv(filepath, rows, fields):
    with filepath.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: as_cell(row.get(k)) for k in fields})


def read_csv(filepath):
    if not filepath.is_file():
        return []
    with filepath.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number_or_nan(row, key):
    return to_float(row.get(key))


def metric(data, key):
    return to_float((data or {}).get(key))


def valid(values):
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def clean_group(mode, alpha, tau, folder):
    if mode == "fixed":
        return f"fixed_a{alpha:.2f}" if math.isfinite(alpha) else "fixed_unknown"
    if mode == "adaptive":
        return f"adaptive_t{tau:.2f}" if math.isfinite(tau) else "adaptive_unknown"
    return mode or folder


def collect(root, val_size):
    paths = sorted(root.rglob("summary.json"))
    if not paths:
        raise FileNotFoundError(f"No summary.json under: {root.resolve()}")
    summaries, epochs_all, steps_all, warnings = [], [], [], []
    runs = []
    for summary_path in paths:
        folder = summary_path.parent
        try:
            info = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as error:
            warnings.append(f"Skipped invalid JSON {summary_path}: {error}")
            continue
        if "best_validation" not in info:
            warnings.append(f"Skipped {summary_path}: missing best_validation")
            continue

        try:
            run = str(folder.relative_to(root)).replace("\\", "/")
        except ValueError:
            run = folder.name
        settings = info.get("settings") or {}
        method = str(info.get("method") or settings.get("mode") or "unknown")
        seed = to_int(info.get("seed", settings.get("seed")), default=-1)
        alpha = to_float(info.get("alpha_fixed", settings.get("alpha")))
        tau = to_float(info.get("tau", settings.get("tau")))
        group = clean_group(method, alpha, tau, folder.name)

        raw_epochs = read_csv(folder / "epochs.csv")
        raw_steps = read_csv(folder / "steps.csv")
        if not raw_epochs:
            warnings.append(f"{run}: missing or empty epochs.csv; epoch plots incomplete")
        if not raw_steps:
            warnings.append(f"{run}: missing or empty steps.csv; alpha/loss plots incomplete")

        epoch_data = []
        for e in raw_epochs:
            row = {
                "run": run, "group": group, "seed": seed,
                "epoch": to_int(e.get("epoch")),
                "train_i2t": number_or_nan(e, "train_i2t"),
                "train_t2i": number_or_nan(e, "train_t2i"),
                "train_total": number_or_nan(e, "train_total"),
                "alpha_mean": number_or_nan(e, "alpha_mean"),
                "alpha_min": number_or_nan(e, "alpha_min"),
                "alpha_max": number_or_nan(e, "alpha_max"),
                "val_acc_i2t": number_or_nan(e, "val_acc_i2t"),
                "val_map_t2i": number_or_nan(e, "val_map_t2i"),
                "val_score": number_or_nan(e, "val_score"),
                "seconds": number_or_nan(e, "seconds"),
            }
            a, m = row["val_acc_i2t"], row["val_map_t2i"]
            row["val_error_count"] = round((1 - a) * val_size) if math.isfinite(a) else math.nan
            row["val_map_gap_pp"] = 100 * (1 - m) if math.isfinite(m) else math.nan
            epoch_data.append(row)
            epochs_all.append(row)

        step_data = []
        for s in raw_steps:
            row = {
                "run": run, "group": group, "seed": seed,
                "step": to_int(s.get("step")),
                "epoch": to_int(s.get("epoch")),
                "loss_i2t": number_or_nan(s, "loss_i2t"),
                "loss_t2i": number_or_nan(s, "loss_t2i"),
                "loss_total": number_or_nan(s, "loss_total"),
                "alpha": number_or_nan(s, "alpha"),
            }
            step_data.append(row)
            steps_all.append(row)

        best = info.get("best_validation") or {}
        first = info.get("pretrained_validation") or {}
        final = epoch_data[-1] if epoch_data else None
        best_acc, best_map = metric(best, "acc_i2t"), metric(best, "map_t2i")
        init_acc, init_map = metric(first, "acc_i2t"), metric(first, "map_t2i")
        alphas = valid([s["alpha"] for s in step_data])
        durations = valid([e["seconds"] for e in epoch_data if e["epoch"] and e["epoch"] > 0])
        srow = {
            "run": run, "method": method, "group": group, "seed": seed,
            "model": settings.get("model", ""),
            "pretrained": settings.get("pretrained", ""),
            "batch_size": settings.get("batch_size", ""),
            "lr": settings.get("lr", ""),
            "epochs_configured": settings.get("epochs", ""),
            "train_blocks": settings.get("train_blocks", ""),
            "fixed_alpha": alpha if method == "fixed" else math.nan,
            "tau": tau if method == "adaptive" else math.nan,
            "warmup_steps": settings.get("warmup_steps", ""),
            "best_epoch": to_int(info.get("best_epoch")),
            "best_acc_i2t": best_acc,
            "best_error_count": round((1-best_acc)*val_size) if math.isfinite(best_acc) else math.nan,
            "best_error_rate_pp": 100*(1-best_acc) if math.isfinite(best_acc) else math.nan,
            "best_map_t2i": best_map,
            "best_map_gap_pp": 100*(1-best_map) if math.isfinite(best_map) else math.nan,
            "best_score": metric(best, "score"),
            "init_acc_i2t": init_acc,
            "init_map_t2i": init_map,
            "last_epoch": final["epoch"] if final else None,
            "last_acc_i2t": final["val_acc_i2t"] if final else math.nan,
            "last_map_t2i": final["val_map_t2i"] if final else math.nan,
            "delta_acc_from_pretrain_pp": 100*(best_acc-init_acc)
                if math.isfinite(best_acc) and math.isfinite(init_acc) else math.nan,
            "delta_map_from_pretrain_pp": 100*(best_map-init_map)
                if math.isfinite(best_map) and math.isfinite(init_map) else math.nan,
            "alpha_start": alphas[0] if alphas else math.nan,
            "alpha_end": alphas[-1] if alphas else math.nan,
            "alpha_mean": statistics.mean(alphas) if alphas else math.nan,
            "alpha_std": statistics.stdev(alphas) if len(alphas)>1 else 0.0 if alphas else math.nan,
            "alpha_min": min(alphas) if alphas else math.nan,
            "alpha_max": max(alphas) if alphas else math.nan,
            "total_train_seconds": sum(durations) if durations else math.nan,
            "train_steps": len(step_data),
            "source": str(summary_path),
        }
        summaries.append(srow)
        runs.append({"id": run, "group": group, "seed": seed,
                     "summary": srow, "settings": settings,
                     "epochs": epoch_data, "steps": step_data})

    summaries.sort(key=lambda x: (-x["best_score"] if math.isfinite(x["best_score"]) else math.inf,
                                  x["run"]))
    return summaries, epochs_all, steps_all, runs, warnings


def group_table(summaries):
    groups = defaultdict(list)
    for s in summaries:
        groups[s["group"]].append(s)
    output = []
    for group, entries in sorted(groups.items()):
        def avg_sd(field):
            nums = valid([e[field] for e in entries])
            if not nums:
                return math.nan, math.nan
            return statistics.mean(nums), (statistics.stdev(nums) if len(nums) > 1 else math.nan)
        a_m, a_s = avg_sd("best_acc_i2t")
        e_m, e_s = avg_sd("best_error_count")
        m_m, m_s = avg_sd("best_map_t2i")
        gap_m, gap_s = avg_sd("best_map_gap_pp")
        score_m, score_s = avg_sd("best_score")
        alpha_m, _ = avg_sd("alpha_mean")
        duration_m, _ = avg_sd("total_train_seconds")
        output.append({
            "group": group, "n_runs": len(entries),
            "seeds": ";".join(str(e["seed"]) for e in entries),
            "best_acc_mean": a_m, "best_acc_std": a_s,
            "best_error_count_mean": e_m, "best_error_count_std": e_s,
            "best_map_mean": m_m, "best_map_std": m_s,
            "best_map_gap_pp_mean": gap_m, "best_map_gap_pp_std": gap_s,
            "best_score_mean": score_m, "best_score_std": score_s,
            "alpha_mean_across_runs": alpha_m,
            "train_seconds_mean": duration_m,
        })
    return output


def paired_baseline_table(runs, warnings):
    baselines = defaultdict(list)
    for r in runs:
        s = r["summary"]
        if s["method"] == "fixed" and math.isfinite(s["fixed_alpha"]) and abs(s["fixed_alpha"] - 0.5) < 1e-7:
            baselines[s["seed"]].append(r)

    result = []
    for r in runs:
        cand = r["summary"]
        if cand["method"] == "fixed" and abs(cand["fixed_alpha"] - 0.5) < 1e-7:
            continue
        options = baselines.get(cand["seed"], [])
        if not options:
            warnings.append(f"{r['id']}: no fixed-alpha=0.5 baseline with seed={cand['seed']}")
            continue
        if len(options) > 1:
            warnings.append(f"{r['id']}: multiple baselines at seed {cand['seed']}; first used")
        base = options[0]["summary"]
        bs = options[0]["settings"]
        cs = r["settings"]
        settings_equal = all(str(bs.get(k)) == str(cs.get(k)) for k in
                             ["model", "pretrained", "batch_size", "lr", "epochs", "train_blocks"])
        if not settings_equal:
            warnings.append(f"{r['id']}: baseline training hyperparameters differ; deltas not controlled")
        result.append({
            "seed": cand["seed"],
            "baseline_run": base["run"], "candidate_run": cand["run"],
            "candidate_group": cand["group"],
            "baseline_best_acc": base["best_acc_i2t"],
            "candidate_best_acc": cand["best_acc_i2t"],
            "delta_acc_pp": 100*(cand["best_acc_i2t"] - base["best_acc_i2t"]),
            "delta_i2t_error_count": cand["best_error_count"] - base["best_error_count"],
            "baseline_best_map": base["best_map_t2i"],
            "candidate_best_map": cand["best_map_t2i"],
            "delta_map_pp": 100*(cand["best_map_t2i"] - base["best_map_t2i"]),
            "comparable_hyperparams": settings_equal,
        })
    return result


def save_chart(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def labels_of(rows):
    return [s["run"].split("/")[-1] for s in rows]


def plot_best_bar(summaries, key, title, ylabel, target):
    rows = [s for s in summaries if math.isfinite(s[key])]
    if not rows:
        return
    labels = labels_of(rows)
    values = [s[key] for s in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows)*1.3), 5.0))
    x = np.arange(len(values))
    bars = ax.bar(x, values)
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    for bar, value in zip(bars, values):
        annotation = f"{int(value)}" if key == "best_error_count" else f"{value:.3f}"
        ax.annotate(annotation, (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save_chart(fig, target)


def plot_epoch_lines(runs, value_key, title, ylabel, target, steps=False):
    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    n = 0
    for r in runs:
        source = r["steps"] if steps else r["epochs"]
        coords = []
        for entry in source:
            x = entry.get("step") if steps else entry.get("epoch")
            y = entry.get(value_key, math.nan)
            if x is not None and math.isfinite(y):
                coords.append((x, y))
        if not coords:
            continue
        coords.sort()
        x, y = np.asarray(coords).T
        if steps and len(x) > 900:
            # Plot chunk means to keep long histories readable.
            chunk = max(1, math.ceil(len(x) / 900))
            x = np.array([np.mean(x[i:i+chunk]) for i in range(0, len(x), chunk)])
            y = np.array([np.mean(y[i:i+chunk]) for i in range(0, len(y), chunk)])
        ax.plot(x, y, marker=None if steps else "o", linewidth=1.4,
                label=r["id"].split("/")[-1], alpha=0.9)
        n += 1
    if n == 0:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel("Training step" if steps else "Epoch")
    ax.set_ylabel(ylabel)
    if value_key == "alpha":
        ax.axhline(0.5, linewidth=0.9, linestyle="--")
        ax.set_ylim(0, 1)
    if not steps:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    save_chart(fig, target)


def plot_paired(paired, target):
    data = [p for p in paired if p["comparable_hyperparams"]]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(9.5, max(4, len(data)*0.55 + 2)))
    labels = [f"{p['candidate_run'].split('/')[-1]} (s{p['seed']})" for p in data]
    positions = np.arange(len(data))
    a = [p["delta_acc_pp"] for p in data]
    m = [p["delta_map_pp"] for p in data]
    ax.scatter(a, positions-0.13, marker="o", label="I2T accuracy delta (pp)")
    ax.scatter(m, positions+0.13, marker="s", label="T2I mAP delta (pp)")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Difference from same-seed fixed-alpha=0.5 baseline (percentage points)")
    ax.set_title("Paired validation deltas (positive = better)")
    ax.grid(axis="x", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    save_chart(fig, target)


def plot_tradeoff(summaries, target):
    rows = [s for s in summaries
            if math.isfinite(s["best_error_count"]) and
               math.isfinite(s["best_map_gap_pp"])]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, row in enumerate(rows):
        ax.scatter(row["best_error_count"], row["best_map_gap_pp"], s=55)
        ax.annotate(row["run"].split("/")[-1],
                    (row["best_error_count"], row["best_map_gap_pp"]),
                    xytext=(5, 5+(i%3)*7), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Best validation I2T error count (lower = better)")
    ax.set_ylabel("Best validation T2I (1 - mAP) in pp (lower = better)")
    ax.set_title("Ceiling-aware two-direction trade-off")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_chart(fig, target)


def build_markdown_summary(summaries, paired, warnings, val_size):
    lines = [
        "# CLIP Experiment Comparison",
        "",
        f"Found **{len(summaries)}** runs. Validation sample count assumed: **{val_size}**.",
        "",
        "The input already selects each run's BEST validation epoch. For claims of",
        "method superiority, examine fixed-epoch curves and repeat multiple seeds;",
        "select hyperparameters on validation before using the held-out test set.",
        "",
        "## Runs (best validation)",
        "",
        "| Run | Epoch | I2T acc | I2T errors | T2I mAP | 1-mAP (pp) | Alpha mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        def ff(k, fmt=".5f"):
            v = s.get(k, math.nan)
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
                return "n/a"
            return format(v, fmt)
        lines.append(
            f"| {s['run']} | {s['best_epoch']} | {ff('best_acc_i2t')} | "
            f"{ff('best_error_count', '.0f')} | {ff('best_map_t2i')} | "
            f"{ff('best_map_gap_pp', '.3f')} | {ff('alpha_mean', '.3f')} |"
        )
    lines += [
        "",
        "## What can and cannot be concluded",
        "",
        "- Best validation statistics may be optimistic because epoch selection used validation.",
        "- A single validation mistake changes I2T accuracy by "
          f"{100/val_size:.4f} percentage points.",
        "- Only aggregate scores are in these logs. Per-image paired significance tests",
        "  (McNemar/paired bootstrap) cannot be computed without saved predictions.",
        "- T2I mAP here uses ten class prompts, not real image-caption retrieval.",
        "- Closely saturated results suggest trying CIFAR-100 or true caption datasets",
        "  rather than optimizing tiny differences on CIFAR-10 alone.",
        "",
        "## CSVs",
        "",
        "- `summary_all.csv`: one row per training run.",
        "- `groups_mean_std.csv`: grouped mean/std (requires >=2 runs for std).",
        "- `epochs_all.csv`: all epoch histories.",
        "- `steps_all.csv`: all step histories.",
        "- `paired_vs_fixed_05.csv`: same-seed comparisons to alpha=0.5.",
        "",
        "## Warnings / checks",
        "",
    ]
    lines += [f"- {w}" for w in warnings] if warnings else ["- None."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs", type=Path,
                        help="Directory with subfolders containing summary.json.")
    parser.add_argument("--out", default="reports/all_runs", type=Path,
                        help="Destination for all CSV reports and PNG charts.")
    parser.add_argument("--val-size", type=int, default=5000,
                        help="Number of validation images (default=5000).")
    args = parser.parse_args()
    if args.val_size <= 0:
        parser.error("--val-size must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    try:
        summaries, epochs, steps, runs, warnings = collect(args.root, args.val_size)
    except FileNotFoundError as error:
        parser.error(str(error))
    if not summaries:
        parser.error("No valid experiments found.")

    fixed_baselines = [s for s in summaries if s["method"] == "fixed" and
                       math.isfinite(s["fixed_alpha"]) and abs(s["fixed_alpha"]-0.5)<1e-7]
    if not fixed_baselines:
        warnings.append("No alpha=0.5 run found: paired baseline comparison unavailable")
    if len({s["seed"] for s in summaries}) == 1:
        warnings.append("Only one seed found; results are not evidence of statistical significance")
    if any(math.isfinite(s["best_acc_i2t"]) and s["best_acc_i2t"] > 0.98 for s in summaries):
        warnings.append("I2T accuracy near ceiling: focus on remaining errors and hard datasets")

    grouped = group_table(summaries)
    paired = paired_baseline_table(runs, warnings)

    write_csv(args.out / "summary_all.csv", summaries, SUMMARY_COLS)
    write_csv(args.out / "groups_mean_std.csv", grouped, GROUP_COLS)
    write_csv(args.out / "epochs_all.csv", epochs, EPOCH_COLS)
    write_csv(args.out / "steps_all.csv", steps, STEP_COLS)
    write_csv(args.out / "paired_vs_fixed_05.csv", paired, PAIRED_COLS)

    plot_best_bar(summaries, "best_error_count",
                  "Image-to-text errors (best validation epoch)",
                  f"Incorrect images out of {args.val_size}",
                  args.out / "01_best_i2t_errors.png")
    plot_best_bar(summaries, "best_map_gap_pp",
                  "Text-to-image gap from perfect mAP (best validation epoch)",
                  "100 × (1 - mAP), percentage points",
                  args.out / "02_best_t2i_map_gap.png")
    plot_epoch_lines(runs, "val_error_count",
                     "Validation image-to-text errors by epoch",
                     f"Incorrect images out of {args.val_size}",
                     args.out / "03_validation_i2t_errors.png")
    plot_epoch_lines(runs, "val_map_gap_pp",
                     "Validation text-to-image mAP gap by epoch",
                     "100 × (1 - mAP), percentage points",
                     args.out / "04_validation_t2i_gap.png")
    plot_epoch_lines(runs, "alpha",
                     "Loss-weight alpha over training",
                     "Alpha (weight on image-to-text loss)",
                     args.out / "05_alpha_curves.png", steps=True)
    plot_epoch_lines(runs, "loss_i2t",
                     "Training image-to-text loss over steps",
                     "Image-to-text training loss",
                     args.out / "06_i2t_loss_curves.png", steps=True)
    plot_epoch_lines(runs, "loss_t2i",
                     "Training text-to-image loss over steps",
                     "Text-to-image training loss",
                     args.out / "07_t2i_loss_curves.png", steps=True)
    plot_paired(paired, args.out / "08_paired_deltas_vs_baseline.png")
    plot_tradeoff(summaries, args.out / "09_tradeoff.png")

    (args.out / "REPORT.md").write_text(
        build_markdown_summary(summaries, paired, warnings, args.val_size),
        encoding="utf-8",
    )
    print(f"\nProcessed {len(summaries)} runs across {len(grouped)} groups")
    print(f"Epoch rows={len(epochs)} | Step rows={len(steps)}")
    print(f"Report directory: {args.out.resolve()}")
    print("\nBest validation results (different run definitions; not a hypothesis test):")
    for s in summaries:
        print(f"  {s['run']:<28} acc={s['best_acc_i2t']:.5f} "
              f"errors={s['best_error_count']:>5} mAP={s['best_map_t2i']:.5f} "
              f"alpha={s['alpha_mean']:.3f}")
    if warnings:
        print("\nNotes:")
        for w in warnings:
            print(" -", w)


if __name__ == "__main__":
    main()
