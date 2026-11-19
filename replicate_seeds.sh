#!/usr/bin/env bash
# Replicate all six configurations with seeds 43/44, then run the full
# 5000-image x 5-condition robustness evaluation per seed.
# Hyperparameters are identical to the original s42 runs (script defaults:
# batch 64, 5 epochs, lr 1e-5, wd 0.05, train_blocks 2, warmup 100).
set -euo pipefail
cd "$(dirname "$0")"

for seed in 43 44; do
  out="runs_s${seed}"
  for a in 0.3 0.5 0.7; do
    echo "=== TRAIN fixed alpha=${a} seed=${seed} ==="
    python adaptive_clip_cifar10.py --mode fixed --alpha "${a}" \
      --seed "${seed}" --epochs 5 --out-dir "${out}"
  done
  for tau in 0.5 1.0 2.0; do
    echo "=== TRAIN adaptive tau=${tau} seed=${seed} ==="
    python adaptive_clip_cifar10.py --mode adaptive --tau "${tau}" \
      --seed "${seed}" --epochs 5 --out-dir "${out}"
  done
done

for seed in 43 44; do
  echo "=== EVAL robustness seed=${seed} ==="
  python evaluate_clip_robustness.py --runs-dir "runs_s${seed}" \
    --seed "${seed}" --per-class 500 --batch-size 64 \
    --out-dir "robustness_results_s${seed}"
done

echo "ALL DONE"