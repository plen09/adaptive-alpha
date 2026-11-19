#!/usr/bin/env python3
"""Compare validation results across experimental runs, grouped by method."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs")
    args = parser.parse_args()
    files = sorted(Path(args.root).glob("*/summary.json"))
    if not files:
        print("No summaries found. Run adaptive_clip_cifar10.py first.")
        return
    groups = defaultdict(list)
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        if data["method"] == "fixed":
            name = f"fixed alpha={data['alpha_fixed']:.2f}"
        else:
            name = f"adaptive tau={data['tau']:.2f}"
        groups[name].append(data)
    print(f"{'Experiment':<24} {'N':>3} {'Acc mean±sd':>18} {'mAP mean±sd':>18}")
    print("-" * 68)
    for name, runs in sorted(groups.items()):
        accs = [r["best_validation"]["acc_i2t"] for r in runs]
        maps = [r["best_validation"]["map_t2i"] for r in runs]
        def fmt(numbers):
            return f"{statistics.mean(numbers):.4f}±{statistics.stdev(numbers):.4f}" if len(numbers)>1 else f"{numbers[0]:.4f} (n=1)"
        print(f"{name:<24} {len(runs):>3} {fmt(accs):>18} {fmt(maps):>18}")
    print("\nMetrics are on VALIDATION only. Choose hyperparameters before evaluating test.")


if __name__ == "__main__":
    main()
