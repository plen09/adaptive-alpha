#!/usr/bin/env bash
# Build runs_all/ : a lightweight view of every training run across seeds
# (42 -> runs/, 43 -> runs_s43/, 44 -> runs_s44/) containing only the
# metadata needed by clip_results_report.py (summary.json, epochs.csv,
# steps.csv). Checkpoints (best.pt, ~58 MB each) and PNGs are NOT copied,
# so the combined tree stays small and read-only w.r.t. the real runs.
#
# clip_results_report.py groups runs by method+hyperparameter and ignores the
# seed, so pointing it at runs_all/ yields cross-seed mean +/- sd of the
# validation metrics (s42 alone could only warn "Only one seed found").
#
# Usage:  bash combine_runs.sh [out_dir]     (default out_dir = runs_all)
set -euo pipefail
cd "$(dirname "$0")"

out="${1:-runs_all}"
rm -rf "${out}"
mkdir -p "${out}"

n=0
for src in runs runs_s43 runs_s44; do
  [ -d "${src}" ] || continue
  for d in "${src}"/*/; do
    name="$(basename "${d}")"
    [ -f "${d}summary.json" ] || continue
    mkdir -p "${out}/${name}"
    cp "${d}summary.json" "${out}/${name}/"
    for f in epochs.csv steps.csv; do
      [ -f "${d}${f}" ] && cp "${d}${f}" "${out}/${name}/"
    done
    n=$((n + 1))
  done
done

echo "Combined ${n} runs into ${out}/"
