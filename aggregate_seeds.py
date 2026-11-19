#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-seed replication analysis for the adaptive-alpha experiment.

Reads robustness_results{,_s43,_s44}/ produced by evaluate_clip_robustness.py
and checks whether the seed-42 findings replicate:
  primary: fixed alpha=0.3 > fixed alpha=0.5 under noise2 (paired McNemar);
  secondary: alpha=0.7 worse than 0.5; adaptive variants slightly worse;
  also aggregates clean validation accuracy across seeds (mean +/- sd).

Outputs reports/replication/{cross_seed_summary.csv, noise2_replication.csv,
clean_acc_by_seed.csv, REPORT_replication.md}.
"""

import csv
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED_DIRS = {42: ROOT / "robustness_results",
             43: ROOT / "robustness_results_s43",
             44: ROOT / "robustness_results_s44"}
OUT = ROOT / "reports" / "replication"
CONFIGS = ["fixed_a0.30", "fixed_a0.50", "fixed_a0.70",
           "adaptive_tau0.50", "adaptive_tau1.00", "adaptive_tau2.00"]


def short(run_name):
    return re.sub(r"_s\d+$", "", run_name)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [s for s, d in SEED_DIRS.items()
             if (d / "robustness_summary.csv").is_file()]
    if not seeds:
        raise SystemExit("No robustness results found yet.")

    clean_acc = {}      # config -> {seed: acc}
    noise2 = {}         # seed -> {candidate: paired row}
    all_cond_delta = {}  # (config, condition) -> {seed: delta}
    val_indices = {}    # seed -> set of sample indices (clean, one run)

    for seed in seeds:
        d = SEED_DIRS[seed]
        for row in read_csv(d / "robustness_summary.csv"):
            cfg = short(row["run"])
            if row["condition"] == "clean":
                clean_acc.setdefault(cfg, {})[seed] = float(row["acc_i2t"])
        for row in read_csv(d / "paired_vs_fixed_05.csv"):
            cfg = short(row["candidate"])
            rec = {
                "seed": seed,
                "candidate": cfg,
                "condition": row["condition"],
                "delta_acc_pp": float(row["delta_acc_pp"]),
                "wins": int(row["baseline_wrong_candidate_right"]),
                "losses": int(row["baseline_right_candidate_wrong"]),
                "p": float(row["mcnemar_exact_p_two_sided"]),
            }
            all_cond_delta.setdefault((cfg, row["condition"]), {})[seed] = \
                rec["delta_acc_pp"]
            if row["condition"] == "noise2":
                noise2.setdefault(seed, {})[cfg] = rec
        per_image = d / "per_image_predictions.csv"
        if per_image.is_file():
            ids = set()
            for row in read_csv(per_image):
                if row["condition"] == "clean" and \
                        short(row["run"]) == "fixed_a0.50":
                    ids.add(int(row["sample_index"]))
            val_indices[seed] = ids

    # ---- clean accuracy table -------------------------------------------
    rows = []
    for cfg in CONFIGS:
        vals = clean_acc.get(cfg, {})
        accs = [vals[s] for s in seeds if s in vals]
        rows.append({
            "config": cfg,
            **{f"acc_s{s}": vals.get(s, "") for s in seeds},
            "mean_acc": statistics.mean(accs) if accs else "",
            "sd_acc": statistics.stdev(accs) if len(accs) > 1 else "",
            "n_seeds": len(accs),
        })
    with (OUT / "clean_acc_by_seed.csv").open("w", newline="",
                                              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- noise2 replication table ----------------------------------------
    n2_rows = []
    for cfg in CONFIGS:
        if cfg == "fixed_a0.50":
            continue
        per_seed = [noise2.get(s, {}).get(cfg) for s in seeds]
        per_seed = [r for r in per_seed if r]
        deltas = [r["delta_acc_pp"] for r in per_seed]
        n2_rows.append({
            "candidate": cfg,
            **{f"delta_s{s}_pp": (noise2.get(s, {}).get(cfg) or {}).get(
                "delta_acc_pp", "") for s in seeds},
            **{f"p_s{s}": (noise2.get(s, {}).get(cfg) or {}).get("p", "")
               for s in seeds},
            "mean_delta_pp": statistics.mean(deltas) if deltas else "",
            "n_positive": sum(1 if x > 0 else 0 for x in deltas),
            "n_significant_p05": sum(1 if r["p"] < 0.05 else 0
                                     for r in per_seed),
        })
    with (OUT / "noise2_replication.csv").open("w", newline="",
                                               encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(n2_rows[0].keys()))
        w.writeheader()
        w.writerows(n2_rows)

    # ---- full condition x config delta table ------------------------------
    full_rows = []
    for (cfg, cond), vals in sorted(all_cond_delta.items()):
        ds = [vals[s] for s in seeds if s in vals]
        full_rows.append({
            "candidate": cfg, "condition": cond,
            **{f"delta_s{s}_pp": vals.get(s, "") for s in seeds},
            "mean_delta_pp": statistics.mean(ds) if ds else "",
            "n_positive": sum(1 if x > 0 else 0 for x in ds),
        })
    with (OUT / "cross_seed_summary.csv").open("w", newline="",
                                               encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(full_rows[0].keys()))
        w.writeheader()
        w.writerows(full_rows)

    # ---- validation split overlap -----------------------------------------
    overlap_lines = []
    ss = sorted(val_indices)
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            a, b = val_indices[ss[i]], val_indices[ss[j]]
            if a and b:
                inter = len(a & b)
                overlap_lines.append(
                    f"- val(s{ss[i]}) ∩ val(s{ss[j]}): {inter} images "
                    f"({100*inter/min(len(a),len(b)):.1f}% of the smaller split)")

    # ---- markdown report ---------------------------------------------------
    def fmt(x, nd=3):
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)

    lines = ["# Cross-seed replication report", "",
             f"Seeds analyzed: {', '.join(str(s) for s in seeds)}.",
             "Baseline for paired tests: fixed alpha=0.5 within each seed "
             "(same validation images).", ""]
    lines += ["## Clean validation accuracy (mean over seeds)", "",
              "| config | " + " | ".join(f"s{s}" for s in seeds) +
              " | mean | sd |",
              "|---|" + "---:|" * (len(seeds) + 2)]
    for r in rows:
        lines.append("| " + r["config"] + " | " +
                     " | ".join(fmt(r.get(f"acc_s{s}", "")) for s in seeds) +
                     f" | {fmt(r['mean_acc'], 5)} | {fmt(r['sd_acc'], 5)} |")
    lines += ["", "## noise2: delta vs fixed alpha=0.5 (pp) and McNemar p", "",
              "| candidate | " +
              " | ".join(f"s{s} delta / p" for s in seeds) +
              " | mean delta | #positive | #p<0.05 |",
              "|---|" + "---:|" * (len(seeds) + 3)]
    for r in n2_rows:
        cells = []
        for s in seeds:
            rec = noise2.get(s, {}).get(r["candidate"])
            cells.append(f"{rec['delta_acc_pp']:+.2f} / {rec['p']:.2g}"
                         if rec else "n/a")
        lines.append("| " + r["candidate"] + " | " + " | ".join(cells) +
                     f" | {fmt(r['mean_delta_pp'])} | {r['n_positive']}"
                     f" | {r['n_significant_p05']} |")
    primary = [noise2.get(s, {}).get("fixed_a0.30") for s in seeds]
    primary = [r for r in primary if r]
    n_pos = sum(1 if r["delta_acc_pp"] > 0 else 0 for r in primary)
    n_sig = sum(1 if r["p"] < 0.05 else 0 for r in primary)
    lines += ["", "## Primary hypothesis (pre-registered from s42)", "",
              "fixed alpha=0.3 beats fixed alpha=0.5 under noise2.",
              f"Replicated in {n_pos}/{len(primary)} seeds by direction; "
              f"{n_sig}/{len(primary)} seeds with p<0.05.", ""]
    if overlap_lines:
        lines += ["## Validation split overlap across seeds", "",
                  "Splits are drawn independently per seed, so a small "
                  "overlap is expected (~10%); cross-seed McNemar pooling "
                  "is therefore only approximate.", ""] + overlap_lines
    lines += ["", "## Caveats", "",
              "- McNemar tests are per-seed and unadjusted for multiple "
              "comparisons; the primary hypothesis was fixed before looking "
              "at seeds 43/44.",
              "- Different seeds use different validation images; compare "
              "directions and effect sizes, not raw accuracies, across seeds.",
              "- Perturbations are deterministic given (index, condition, "
              "seed); noise realizations differ across seeds."]
    (OUT / "REPORT_replication.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved tables and report to {OUT}")


if __name__ == "__main__":
    main()