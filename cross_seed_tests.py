#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-seed significance testing for the adaptive-alpha robustness results.

Companion to aggregate_seeds.py.  aggregate_seeds.py lists the per-seed
McNemar p-values side by side; with only three seeds that leaves the reader to
eyeball whether the seed-42 finding is real.  This script pools the three
independent seeds properly and adds effect sizes with uncertainty.

Two baselines are used, because they answer two different questions:

  fixed_a0.50  the pre-registered replication baseline.  The seed-42 analysis
               and evaluate_clip_robustness.py both compare against it, and the
               hypothesis fixed before looking at seeds 43/44 was
               "fixed alpha=0.3 beats fixed alpha=0.5 under noise2".
  fixed_a0.30  the empirical champion of the seed-42 robustness sweep (best in
               all five conditions).  Comparing against it asks the sharper
               question: does ANY adaptive setting match the best fixed one?

Statistics
----------
Seeds 42/43/44 use different stratified validation splits and different noise
realisations, so they are independent strata.

  * pooled exact McNemar (stratified sign test).  Under H0 the discordant count
    of stratum i is Binomial(m_i, 1/2) with m_i = wins_i + losses_i, and the
    stratum contributes 2*X_i - m_i to sum(wins - losses).  The exact
    distribution of the total is the convolution of the three stratum
    distributions, giving an exact pooled two-sided p-value.  A normal
    approximation Z = sum(wins-losses) / sqrt(sum(wins+losses)) is reported too.
  * stratified paired bootstrap 95% CI for the accuracy difference in pp.
    Resampling images within a stratum only redistributes counts over the four
    (baseline, candidate) outcome cells, so each stratum is drawn as
    Multinomial(n_i, cell frequencies) - same distribution, much cheaper.  The
    pooled statistic is the MEAN of the stratum deltas, so adding a seed does
    not change the units.
  * mean +/- sd across seeds of accuracy and mAP per config x condition.

mAP cannot be paired per image (it is a ranking metric over the whole subset),
so for mAP only per-seed deltas, mean +/- sd and a direction count are given.

Outputs into reports/replication/:
  cross_seed_condition_summary.csv   config x condition, per seed + mean/sd
  cross_seed_paired_tests.csv        baseline x condition x candidate
  cross_seed_map_deltas.csv          mAP deltas for both baselines
  cross_seed_rank_by_condition.csv   per-seed ranking of configs per condition
  07_forest_delta_acc_vs_a050.png    per-seed deltas + pooled bootstrap CI
  07_forest_delta_acc_vs_a030.png    same, against the champion
  08_condition_acc_cross_seed.png    accuracy per condition, sd across seeds
  REPORT_cross_seed_tests.md         readable summary of all of the above
"""

import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SEED_DIRS = {42: ROOT / "robustness_results",
             43: ROOT / "robustness_results_s43",
             44: ROOT / "robustness_results_s44"}
OUT = ROOT / "reports" / "replication"
CONDITIONS = ["clean", "noise1", "noise2", "blur1", "blur2"]
CONFIGS = ["fixed_a0.30", "fixed_a0.50", "fixed_a0.70",
           "adaptive_tau0.50", "adaptive_tau1.00", "adaptive_tau2.00"]
BASELINES = [("fixed_a0.50", "a050"), ("fixed_a0.30", "a030")]
PRIMARY = ("noise2", "fixed_a0.30", "fixed_a0.50")   # condition, cand, baseline
BOOT_REPS = 20000
BONFERRONI = 0.05 / (len(CONDITIONS) * (len(CONFIGS) - 1) * len(BASELINES))
RNG = np.random.default_rng(20261119)


def short(run_name):
    return re.sub(r"_s\d+$", "", run_name)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- exact stats
def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2.0 ** m
    return min(1.0, 2.0 * tail)


def pooled_exact_mcnemar(pairs):
    """Exact two-sided p-value for sum(wins_i - losses_i) over strata.

    pairs: list of (wins_i, losses_i).  Also returns the normal-approximation
    Z and the total number of discordant images.
    """
    pairs = [(b, c) for b, c in pairs if b + c > 0]
    if not pairs:
        return 1.0, 0.0, 0
    total = {0: 1.0}
    for b, c in pairs:
        m = b + c
        binom = np.array([math.comb(m, k) for k in range(m + 1)]) / 2.0 ** m
        vals = [2 * k - m for k in range(m + 1)]
        new = defaultdict(float)
        for s0, p0 in total.items():
            for v, p1 in zip(vals, binom):
                q = p0 * float(p1)
                if q > 0:
                    new[s0 + v] += q
        total = new
    observed = sum(b - c for b, c in pairs)
    p_exact = sum(p for s, p in total.items() if abs(s) >= abs(observed))
    m_tot = sum(b + c for b, c in pairs)
    z = observed / math.sqrt(m_tot) if m_tot else 0.0
    return min(1.0, p_exact), z, m_tot


def stratified_bootstrap(cells, reps=BOOT_REPS):
    """Bootstrap CI (pp) for the mean paired accuracy difference over strata.

    cells: list of (both_right, wins, losses, both_wrong) per stratum.
    """
    draws = []
    for a, b, c, d in cells:
        n = a + b + c + d
        if n == 0:
            continue
        p = np.array([a, b, c, d], dtype=float) / n
        counts = RNG.multinomial(n, p, size=reps)
        draws.append((counts[:, 1] - counts[:, 2]) / n * 100.0)
    if not draws:
        return float("nan"), float("nan"), float("nan")
    pooled = np.sum(draws, axis=0) / len(draws)
    return (float(pooled.mean()),
            float(np.percentile(pooled, 2.5)),
            float(np.percentile(pooled, 97.5)))


# ------------------------------------------------------------------ load data
def load():
    summary = defaultdict(dict)      # (config, condition) -> {seed: row}
    paired_file = defaultdict(dict)  # (candidate, condition) -> {seed: row} (vs a0.50)
    correct = {}                     # (seed, condition, config) -> {sample_index: 0/1}
    for seed, d in SEED_DIRS.items():
        if not (d / "robustness_summary.csv").is_file():
            print(f"[skip] seed {seed}: {d / 'robustness_summary.csv'} missing")
            continue
        for row in read_csv(d / "robustness_summary.csv"):
            summary[(short(row["run"]), row["condition"])][seed] = row
        pf = d / "paired_vs_fixed_05.csv"
        if pf.is_file():
            for row in read_csv(pf):
                paired_file[(short(row["candidate"]), row["condition"])][seed] = row
        pi = d / "per_image_predictions.csv"
        if pi.is_file():
            table = defaultdict(dict)
            for row in read_csv(pi):
                table[(row["condition"], short(row["run"]))][
                    int(row["sample_index"])] = int(row["correct"])
            for key, m in table.items():
                correct[(seed,) + key] = m
        else:
            print(f"[warn] seed {seed}: per_image_predictions.csv missing; "
                  f"pooled tests unavailable for this seed")
    seeds = sorted({k[0] for k in correct}) if correct else \
        sorted({s for v in summary.values() for s in v})
    return summary, paired_file, correct, seeds


def paired_stats(correct, seeds, cond, cand, base_cfg):
    """Per-seed and pooled paired comparison of cand vs base_cfg under cond."""
    per = {}
    pairs, cells, deltas = [], [], []
    for s in seeds:
        bm = correct.get((s, cond, base_cfg))
        cm = correct.get((s, cond, cand))
        if not bm or not cm:
            continue
        idx = sorted(set(bm) & set(cm))
        bb = np.fromiter((bm[i] for i in idx), dtype=np.int64, count=len(idx))
        cc = np.fromiter((cm[i] for i in idx), dtype=np.int64, count=len(idx))
        wins = int(np.sum((bb == 0) & (cc == 1)))
        losses = int(np.sum((bb == 1) & (cc == 0)))
        both_right = int(np.sum((bb == 1) & (cc == 1)))
        both_wrong = int(np.sum((bb == 0) & (cc == 0)))
        delta = float((cc.mean() - bb.mean()) * 100.0)
        p = mcnemar_exact(wins, losses)
        per[s] = {"n": len(idx), "delta": delta, "wins": wins, "losses": losses,
                  "p": p, "acc_base": float(bb.mean()), "acc_cand": float(cc.mean())}
        pairs.append((wins, losses))
        cells.append((both_right, wins, losses, both_wrong))
        deltas.append(delta)
    p_pool, z, m_tot = pooled_exact_mcnemar(pairs)
    boot = stratified_bootstrap(cells)
    return {"per": per, "pairs": pairs, "deltas": deltas,
            "pooled_p": p_pool, "z": z, "discordant": m_tot, "boot": boot}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, paired_file, correct, seeds = load()
    if not seeds:
        raise SystemExit("No robustness results found; run "
                         "evaluate_clip_robustness.py for each seed first.")
    print(f"Seeds available: {seeds}")

    # ---- 1. condition x config aggregate table ------------------------------
    cond_rows = []
    for cfg in CONFIGS:
        for cond in CONDITIONS:
            rec = {"config": cfg, "condition": cond}
            accs, maps, drops = [], [], []
            for s in seeds:
                r = summary.get((cfg, cond), {}).get(s)
                if not r:
                    rec[f"acc_s{s}"] = rec[f"map_s{s}"] = ""
                    continue
                a, m = float(r["acc_i2t"]), float(r["map_t2i"])
                rec[f"acc_s{s}"] = round(a, 6)
                rec[f"map_s{s}"] = round(m, 6)
                accs.append(a)
                maps.append(m)
                drops.append(float(r["acc_drop_vs_clean_pp"]))
            rec["acc_mean"] = round(statistics.mean(accs), 6) if accs else ""
            rec["acc_sd"] = round(statistics.stdev(accs), 6) if len(accs) > 1 else ""
            rec["map_mean"] = round(statistics.mean(maps), 6) if maps else ""
            rec["map_sd"] = round(statistics.stdev(maps), 6) if len(maps) > 1 else ""
            rec["drop_pp_mean"] = round(statistics.mean(drops), 4) if drops else ""
            rec["drop_pp_sd"] = round(statistics.stdev(drops), 4) if len(drops) > 1 else ""
            rec["n_seeds"] = len(accs)
            cond_rows.append(rec)
    write_csv(OUT / "cross_seed_condition_summary.csv", cond_rows)

    # ---- 2. paired tests vs both baselines ---------------------------------
    test_rows, mismatch, cache = [], [], {}
    for base_cfg, base_tag in BASELINES:
        for cond in CONDITIONS:
            for cand in CONFIGS:
                if cand == base_cfg:
                    continue
                st = paired_stats(correct, seeds, cond, cand, base_cfg)
                cache[(base_cfg, cond, cand)] = st
                rec = {"baseline": base_cfg, "condition": cond, "candidate": cand}
                for s in seeds:
                    d = st["per"].get(s)
                    rec[f"delta_s{s}_pp"] = round(d["delta"], 4) if d else ""
                    rec[f"wins_s{s}"] = d["wins"] if d else ""
                    rec[f"losses_s{s}"] = d["losses"] if d else ""
                    rec[f"p_s{s}"] = f"{d['p']:.4g}" if d else ""
                    rec[f"n_s{s}"] = d["n"] if d else ""
                rec["mean_delta_pp"] = rnd(statistics.mean(st["deltas"]),
                                           bool(st["deltas"]))
                rec["sd_delta_pp"] = rnd(sd(st["deltas"]))
                rec["n_seeds_positive"] = sum(1 for x in st["deltas"] if x > 0)
                rec["n_seeds_negative"] = sum(1 for x in st["deltas"] if x < 0)
                rec["n_seeds_p05"] = sum(1 for d in st["per"].values()
                                         if d["p"] < 0.05)
                rec["n_seeds_p_bonf"] = sum(1 for d in st["per"].values()
                                            if d["p"] < BONFERRONI)
                rec["discordant_total"] = st["discordant"]
                rec["sum_wins_minus_losses"] = sum(b - c for b, c in st["pairs"])
                rec["pooled_exact_p"] = f"{st['pooled_p']:.4g}"
                rec["pooled_z"] = round(st["z"], 4)
                rec["boot_mean_pp"] = rnd(st["boot"][0])
                rec["boot_lo95_pp"] = rnd(st["boot"][1])
                rec["boot_hi95_pp"] = rnd(st["boot"][2])
                rec["boot_excludes_zero"] = bool(
                    st["boot"][1] > 0 or st["boot"][2] < 0)
                rec["sig_after_bonferroni"] = bool(st["pooled_p"] < BONFERRONI)
                test_rows.append(rec)
                # cross-check the vs-0.5 numbers against the eval script's CSV
                if base_cfg == "fixed_a0.50":
                    for s, d in st["per"].items():
                        frow = paired_file.get((cand, cond), {}).get(s)
                        if not frow:
                            continue
                        if int(frow["baseline_wrong_candidate_right"]) != d["wins"] or \
                                int(frow["baseline_right_candidate_wrong"]) != d["losses"] or \
                                abs(float(frow["mcnemar_exact_p_two_sided"]) - d["p"]) > 1e-9 or \
                                abs(float(frow["delta_acc_pp"]) - d["delta"]) > 5e-3:
                            mismatch.append((s, cond, cand))
    write_csv(OUT / "cross_seed_paired_tests.csv", test_rows)
    if mismatch:
        print(f"[warn] {len(mismatch)} disagreements with the per-seed "
              f"paired_vs_fixed_05.csv files: {mismatch[:6]}")
    else:
        print("[ok] recomputed wins/losses, deltas and exact McNemar p-values "
              "reproduce every row of the per-seed paired_vs_fixed_05.csv files")

    # ---- 3. mAP deltas (not pairable per image) -----------------------------
    map_rows = []
    for base_cfg, _ in BASELINES:
        for cond in CONDITIONS:
            for cand in CONFIGS:
                if cand == base_cfg:
                    continue
                row = {"baseline": base_cfg, "condition": cond, "candidate": cand}
                vals = []
                for s in seeds:
                    b = summary.get((base_cfg, cond), {}).get(s)
                    c = summary.get((cand, cond), {}).get(s)
                    v = (float(c["map_t2i"]) - float(b["map_t2i"])) * 100.0 \
                        if b and c else None
                    row[f"delta_map_s{s}_pp"] = round(v, 4) if v is not None else ""
                    if v is not None:
                        vals.append(v)
                row["mean_delta_map_pp"] = rnd(statistics.mean(vals), bool(vals))
                row["sd_delta_map_pp"] = rnd(sd(vals))
                row["n_seeds_positive"] = sum(1 for v in vals if v > 0)
                map_rows.append(row)
    write_csv(OUT / "cross_seed_map_deltas.csv", map_rows)

    # ---- 4. per-seed ranking inside each condition --------------------------
    rank_rows = []
    for cond in CONDITIONS:
        for s in seeds:
            avail = [(cfg, summary[(cfg, cond)][s]) for cfg in CONFIGS
                     if s in summary.get((cfg, cond), {})]
            avail.sort(key=lambda kv: float(kv[1]["acc_i2t"]), reverse=True)
            rank_rows.append({
                "condition": cond, "seed": s,
                "rank": ";".join(cfg for cfg, _ in avail),
                "acc": ";".join(f"{float(r['acc_i2t']):.4f}" for _, r in avail),
                "best": avail[0][0] if avail else "",
                "worst": avail[-1][0] if avail else "",
                "spread_pp": round((float(avail[0][1]["acc_i2t"]) -
                                    float(avail[-1][1]["acc_i2t"])) * 100.0, 3)
                if len(avail) > 1 else "",
            })
    write_csv(OUT / "cross_seed_rank_by_condition.csv", rank_rows)

    # ---- 5. figures ---------------------------------------------------------
    for base_cfg, base_tag in BASELINES:
        plot_forest(test_rows, seeds, base_cfg,
                    OUT / f"07_forest_delta_acc_vs_{base_tag}.png")
    plot_conditions(cond_rows, OUT / "08_condition_acc_cross_seed.png")

    # ---- 6. markdown --------------------------------------------------------
    (OUT / "REPORT_cross_seed_tests.md").write_text(
        build_md(cond_rows, test_rows, map_rows, rank_rows, seeds, cache,
                 mismatch), encoding="utf-8")
    print(f"Saved cross-seed test tables, figures and report to {OUT}")


def rnd(v, ok=True):
    """Round for CSV output; blank when unavailable or not finite."""
    if not ok or not isinstance(v, float) or not math.isfinite(v):
        return ""
    return round(v, 4)


def sd(vals):
    """Sample sd, or NaN when fewer than two values (never raises)."""
    return statistics.stdev(vals) if len(vals) > 1 else float("nan")


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_forest(rows, seeds, base_cfg, path):
    cands = [c for c in CONFIGS if c != base_cfg]
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4.1 * len(CONDITIONS), 4.4),
                             sharey=True)
    ypos = np.arange(len(cands))[::-1]
    for ax, cond in zip(axes, CONDITIONS):
        sub = {r["candidate"]: r for r in rows
               if r["condition"] == cond and r["baseline"] == base_cfg}
        for y, cand in zip(ypos, cands):
            r = sub.get(cand)
            if not r:
                continue
            per = [r.get(f"delta_s{s}_pp") for s in seeds]
            per = [p for p in per if isinstance(p, (int, float))]
            first = (cond == CONDITIONS[0] and y == ypos[0])
            ax.scatter(per, [y] * len(per), s=28, zorder=3, color="#4C72B0",
                       label="per seed" if first else None)
            lo, hi, mu = r.get("boot_lo95_pp"), r.get("boot_hi95_pp"), r.get("boot_mean_pp")
            if all(isinstance(v, (int, float)) for v in (lo, hi, mu)):
                color = "#C44E52" if lo > 0 else ("#55A868" if hi < 0 else "#8C8C8C")
                ax.plot([lo, hi], [y, y], color=color, lw=2.6, zorder=2,
                        label="pooled 95% CI" if first else None)
                ax.scatter([mu], [y], marker="D", s=36, color=color, zorder=4)
        ax.axvline(0, color="black", lw=0.9, ls="--")
        ax.set_yticks(ypos)
        ax.set_title(cond, fontsize=11)
        ax.set_xlabel(f"accuracy - {base_cfg} (pp)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticklabels(cands, fontsize=9)
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Cross-seed paired accuracy difference vs same-seed {base_cfg}\n"
                 f"seeds {', '.join(str(s) for s in seeds)}; diamond = stratified "
                 "bootstrap mean, bar = 95% CI (red = candidate better, "
                 "green = baseline better)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_conditions(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    x = np.arange(len(CONDITIONS))
    width = 0.13
    for ax, metric, label in zip(axes, ["acc", "map"],
                                 ["I2T top-1 accuracy", "T2I mAP (10 class prompts)"]):
        means, sds = [], []
        for k, cfg in enumerate(CONFIGS):
            m_list, s_list = [], []
            for cond in CONDITIONS:
                r = next((r for r in rows if r["config"] == cfg
                          and r["condition"] == cond), None)
                mv = r.get(f"{metric}_mean") if r else None
                sv = r.get(f"{metric}_sd") if r else None
                m_list.append(mv if isinstance(mv, float) else np.nan)
                s_list.append(sv if isinstance(sv, float) else 0.0)
            ax.bar(x + (k - 2.5) * width, m_list, width, yerr=s_list, capsize=2.5,
                   label=cfg, error_kw=dict(lw=0.8))
            means += m_list
            sds += s_list
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        finite = [(m, s) for m, s in zip(means, sds) if isinstance(m, float)
                  and np.isfinite(m)]
        if finite:
            lo = min(m - (s or 0.0) for m, s in finite)
            ax.set_ylim(max(0.0, lo - 0.05), 1.005)
    fig.suptitle("Robustness across conditions, mean over all available seeds "
                 "(error bars = sd across seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_md(cond_rows, test_rows, map_rows, rank_rows, seeds, cache, mismatch):
    seed_head = " | ".join(f"s{s}" for s in seeds)
    ns = len(seeds)
    lines = ["# Cross-seed paired tests (robustness evaluation)", "",
             f"Seeds: {', '.join(str(s) for s in seeds)}. Every comparison is "
             f"paired within a seed: the same perturbed validation images are "
             f"shown to the candidate and to that seed's baseline checkpoint.",
             "", "## Plain summary", ""]

    # headline: the pre-registered primary hypothesis
    cond_p, cand_p, base_p = PRIMARY
    st = cache.get((base_p, cond_p, cand_p))
    if st and st["per"]:
        per = ", ".join(f"s{s}: {d['delta']:+.2f} pp (p={d['p']:.2g})"
                        for s, d in sorted(st["per"].items()))
        lines += [
            f"Pre-registered hypothesis from seed 42: **fixed alpha=0.3 beats "
            f"fixed alpha=0.5 under {cond_p}**.", "",
            f"- Per seed -> {per}.",
            f"- Direction replicated in {sum(1 for x in st['deltas'] if x > 0)}"
            f"/{len(st['deltas'])} seeds.",
            f"- Pooled over seeds: mean {statistics.mean(st['deltas']):+.2f} pp, "
            f"{sum(b - c for b, c in st['pairs']):+d} net images out of "
            f"{st['discordant']} discordant, exact pooled p="
            f"{st['pooled_p']:.3g}, bootstrap 95% CI "
            f"[{st['boot'][1]:+.2f}, {st['boot'][2]:+.2f}] pp.",
            f"- Verdict: {'REPLICATED' if st['boot'][1] > 0 and st['pooled_p'] < BONFERRONI else 'not replicated'}.",
            ""]

    # headline: does any adaptive setting match the champion?
    ad = [(c, cache[("fixed_a0.30", cond_p, c)])
          for c in CONFIGS if c.startswith("adaptive")
          and ("fixed_a0.30", cond_p, c) in cache]
    if ad:
        lines += [f"Sharper question: under {cond_p}, does any adaptive setting "
                  f"match the best fixed one (alpha=0.3)?", ""]
        for c, s in ad:
            lines.append(
                f"- `{c}`: mean {statistics.mean(s['deltas']):+.2f} pp vs "
                f"alpha=0.3, pooled exact p={s['pooled_p']:.3g}, CI "
                f"[{s['boot'][1]:+.2f}, {s['boot'][2]:+.2f}] -> "
                f"{'significantly worse' if s['boot'][2] < 0 else ('significantly better' if s['boot'][1] > 0 else 'indistinguishable')}.")
        lines.append("")

    for base_cfg, base_tag in BASELINES:
        rows = [r for r in test_rows if r["baseline"] == base_cfg]
        lines += [f"## Accuracy vs `{base_cfg}`, pooled over seeds", "",
                  f"| condition | candidate | {seed_head} | mean delta (pp) | "
                  "net images | discordant | pooled exact p | bootstrap 95% CI "
                  "| CI excludes 0 | p < Bonferroni |",
                  "|---|---|" + "---:|" * (ns + 7)]
        for r in rows:
            per = " | ".join(
                f"{r[f'delta_s{s}_pp']:+.2f}"
                if isinstance(r.get(f"delta_s{s}_pp"), float) else "n/a"
                for s in seeds)
            lines.append(
                f"| {r['condition']} | {r['candidate']} | {per} "
                f"| {r['mean_delta_pp']:+.3f} | {r['sum_wins_minus_losses']:+d} "
                f"| {r['discordant_total']} | {r['pooled_exact_p']} "
                f"| [{r['boot_lo95_pp']:+.2f}, {r['boot_hi95_pp']:+.2f}] "
                f"| {'yes' if r['boot_excludes_zero'] else 'no'} "
                f"| {'yes' if r['sig_after_bonferroni'] else 'no'} |")
        lines.append("")

    lines += ["## Absolute performance per condition (mean +/- sd over seeds)", "",
              f"| config | condition | {seed_head} | acc mean | acc sd | "
              "mAP mean | mAP sd | acc drop vs own clean (pp) |",
              "|---|---|" + "---:|" * (ns + 6)]
    for r in cond_rows:
        per = " | ".join(f"{r[f'acc_s{s}']:.4f}" if r.get(f"acc_s{s}") != "" else "n/a"
                         for s in seeds)
        lines.append(
            f"| {r['config']} | {r['condition']} | {per} | {num(r['acc_mean'], 4)} "
            f"| {num(r['acc_sd'], 5)} | {num(r['map_mean'], 4)} | {num(r['map_sd'], 5)} "
            f"| {num(r['drop_pp_mean'], 2)} +/- {num(r['drop_pp_sd'], 2)} |")

    lines += ["", "## T2I mAP delta (pp) - not pairable per image", "",
              f"| baseline | condition | candidate | {seed_head} | mean | sd "
              "| #seeds positive |",
              "|---|---|---|" + "---:|" * (ns + 3)]
    for r in map_rows:
        per = " | ".join(
            f"{r[f'delta_map_s{s}_pp']:+.3f}"
            if isinstance(r.get(f"delta_map_s{s}_pp"), float) else "n/a"
            for s in seeds)
        lines.append(f"| {r['baseline']} | {r['condition']} | {r['candidate']} "
                     f"| {per} | {num(r['mean_delta_map_pp'], 3, signed=True)} "
                     f"| {num(r['sd_delta_map_pp'], 3)} | {r['n_seeds_positive']} |")

    lines += ["", "## Which config ranks first in each seed", "",
              f"| condition | seed | best | worst | spread (pp) | full ranking |",
              "|---|---:|---|---|---:|---|"]
    for r in rank_rows:
        lines.append(f"| {r['condition']} | {r['seed']} | {r['best']} "
                     f"| {r['worst']} | {num(r['spread_pp'], 2)} | {r['rank']} |")

    lines += ["", "## Caveats", "",
              f"- {len(CONDITIONS)} conditions x {len(CONFIGS) - 1} candidates x "
              f"{len(BASELINES)} baselines = "
              f"{len(CONDITIONS) * (len(CONFIGS) - 1) * len(BASELINES)} pooled "
              f"tests; Bonferroni threshold {BONFERRONI:.5f}. The two baselines "
              f"are not independent of each other (they share the same runs), "
              f"so this correction is conservative.",
              "- Accuracy on clean CIFAR-10 is saturated (96.0-96.5%); one image "
              "out of 5000 is 0.02 pp, so clean-set differences carry almost no "
              "information.",
              "- The pooled exact test treats the seeds as independent strata. "
              "Their validation splits overlap by roughly the chance-level 10% "
              "(see REPORT_replication.md), which makes the pooled p-value "
              "mildly anti-conservative.",
              "- mAP is a subset-level ranking metric: no per-image paired test "
              "exists, so only direction and spread across seeds are reported.",
              "- Perturbations are deterministic synthetic Gaussian noise / blur "
              "applied to the 32px images before CLIP's 224px preprocessing. "
              "They are not CIFAR-10-C corruptions.",
              "- Each run's checkpoint is its own best-validation epoch, and the "
              "validation split is also the evaluation split here, so these "
              "numbers are method-selection data rather than a held-out test "
              "estimate.",
              "- Three seeds is the minimum for a mean +/- sd; the sd itself has "
              "very wide uncertainty (n-1 = 2 degrees of freedom)."]
    if mismatch:
        lines += ["", "## Consistency check", "",
                  f"{len(mismatch)} cells disagreed with the per-seed "
                  f"`paired_vs_fixed_05.csv` files: {mismatch[:10]}"]
    else:
        lines += ["", "## Consistency check", "",
                  "Wins/losses, accuracy deltas and exact McNemar p-values "
                  "recomputed from `per_image_predictions.csv` reproduce every "
                  "row of the per-seed `paired_vs_fixed_05.csv` files."]
    return "\n".join(lines) + "\n"


def num(v, nd, signed=False):
    if not isinstance(v, float) or not math.isfinite(v):
        return "n/a"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


if __name__ == "__main__":
    sys.exit(main())
