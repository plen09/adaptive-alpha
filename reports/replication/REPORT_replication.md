# Cross-seed replication report

Seeds analyzed: 42, 43, 44.
Baseline for paired tests: fixed alpha=0.5 within each seed (same validation images).

## Clean validation accuracy (mean over seeds)

| config | s42 | s43 | s44 | mean | sd |
|---|---:|---:|---:|---:|---:|
| fixed_a0.30 | 0.965 | 0.963 | 0.962 | 0.96307 | 0.00136 |
| fixed_a0.50 | 0.965 | 0.961 | 0.961 | 0.96233 | 0.00197 |
| fixed_a0.70 | 0.964 | 0.961 | 0.962 | 0.96240 | 0.00151 |
| adaptive_tau0.50 | 0.964 | 0.961 | 0.963 | 0.96233 | 0.00155 |
| adaptive_tau1.00 | 0.964 | 0.961 | 0.963 | 0.96247 | 0.00153 |
| adaptive_tau2.00 | 0.964 | 0.961 | 0.961 | 0.96207 | 0.00185 |

## noise2: delta vs fixed alpha=0.5 (pp) and McNemar p

| candidate | s42 delta / p | s43 delta / p | s44 delta / p | mean delta | #positive | #p<0.05 |
|---|---:|---:|---:|---:|---:|---:|
| fixed_a0.30 | +1.00 / 2.5e-06 | +0.50 / 0.011 | +0.06 / 0.82 | 0.520 | 3 | 2 |
| fixed_a0.70 | -0.98 / 6.8e-06 | -0.60 / 0.0032 | +0.12 / 0.82 | -0.487 | 1 | 2 |
| adaptive_tau0.50 | -0.50 / 0.0031 | -0.32 / 0.064 | +0.02 / 1 | -0.267 | 1 | 1 |
| adaptive_tau1.00 | -0.34 / 0.012 | -0.20 / 0.16 | -0.12 / 0.81 | -0.220 | 0 | 1 |
| adaptive_tau2.00 | -0.18 / 0.12 | -0.20 / 0.13 | +0.08 / 0.52 | -0.100 | 1 | 0 |

## Primary hypothesis (pre-registered from s42)

fixed alpha=0.3 beats fixed alpha=0.5 under noise2.
Replicated in 3/3 seeds by direction; 2/3 seeds with p<0.05.

## Validation split overlap across seeds

Splits are drawn independently per seed, so a small overlap is expected (~10%); cross-seed McNemar pooling is therefore only approximate.

- val(s42) ∩ val(s43): 507 images (10.1% of the smaller split)
- val(s42) ∩ val(s44): 515 images (10.3% of the smaller split)
- val(s43) ∩ val(s44): 485 images (9.7% of the smaller split)

## Caveats

- McNemar tests are per-seed and unadjusted for multiple comparisons; the primary hypothesis was fixed before looking at seeds 43/44.
- Different seeds use different validation images; compare directions and effect sizes, not raw accuracies, across seeds.
- Perturbations are deterministic given (index, condition, seed); noise realizations differ across seeds.
