# CLIP Experiment Comparison

Found **18** runs. Validation sample count assumed: **5000**.

The input already selects each run's BEST validation epoch. For claims of
method superiority, examine fixed-epoch curves and repeat multiple seeds;
select hyperparameters on validation before using the held-out test set.

## Runs (best validation)

| Run | Epoch | I2T acc | I2T errors | T2I mAP | 1-mAP (pp) | Alpha mean |
|---|---:|---:|---:|---:|---:|---:|
| fixed_a0.30_s42 | 5 | 0.96460 | 177 | 0.98939 | 1.061 | 0.300 |
| fixed_a0.50_s42 | 5 | 0.96460 | 177 | 0.98909 | 1.091 | 0.500 |
| adaptive_tau2.00_s42 | 5 | 0.96420 | 179 | 0.98885 | 1.115 | 0.531 |
| adaptive_tau1.00_s42 | 5 | 0.96380 | 181 | 0.98873 | 1.127 | 0.561 |
| adaptive_tau0.50_s42 | 5 | 0.96360 | 182 | 0.98855 | 1.145 | 0.615 |
| fixed_a0.70_s42 | 5 | 0.96400 | 180 | 0.98803 | 1.197 | 0.700 |
| fixed_a0.30_s44 | 5 | 0.96200 | 190 | 0.98745 | 1.255 | 0.300 |
| adaptive_tau1.00_s44 | 3 | 0.96280 | 186 | 0.98574 | 1.426 | 0.566 |
| adaptive_tau0.50_s44 | 3 | 0.96280 | 186 | 0.98561 | 1.439 | 0.625 |
| fixed_a0.50_s44 | 5 | 0.96100 | 195 | 0.98672 | 1.328 | 0.500 |
| fixed_a0.70_s44 | 3 | 0.96220 | 189 | 0.98524 | 1.476 | 0.700 |
| fixed_a0.30_s43 | 3 | 0.96260 | 187 | 0.98481 | 1.519 | 0.300 |
| adaptive_tau2.00_s44 | 5 | 0.96100 | 195 | 0.98634 | 1.366 | 0.534 |
| fixed_a0.50_s43 | 3 | 0.96140 | 193 | 0.98513 | 1.487 | 0.500 |
| fixed_a0.70_s43 | 3 | 0.96100 | 195 | 0.98541 | 1.459 | 0.700 |
| adaptive_tau2.00_s43 | 3 | 0.96100 | 195 | 0.98531 | 1.469 | 0.530 |
| adaptive_tau1.00_s43 | 3 | 0.96080 | 196 | 0.98540 | 1.460 | 0.558 |
| adaptive_tau0.50_s43 | 3 | 0.96060 | 197 | 0.98554 | 1.446 | 0.611 |

## What can and cannot be concluded

- Best validation statistics may be optimistic because epoch selection used validation.
- A single validation mistake changes I2T accuracy by 0.0200 percentage points.
- Only aggregate scores are in these logs. Per-image paired significance tests
  (McNemar/paired bootstrap) cannot be computed without saved predictions.
- T2I mAP here uses ten class prompts, not real image-caption retrieval.
- Closely saturated results suggest trying CIFAR-100 or true caption datasets
  rather than optimizing tiny differences on CIFAR-10 alone.

## CSVs

- `summary_all.csv`: one row per training run.
- `groups_mean_std.csv`: grouped mean/std (requires >=2 runs for std).
- `epochs_all.csv`: all epoch histories.
- `steps_all.csv`: all step histories.
- `paired_vs_fixed_05.csv`: same-seed comparisons to alpha=0.5.

## Warnings / checks

- None.
