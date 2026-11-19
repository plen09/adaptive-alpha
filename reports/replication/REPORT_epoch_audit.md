# Best-epoch selection audit

Each `best.pt` is the epoch with the highest validation score (I2T accuracy + T2I mAP) / 2. Accuracy moves in steps of 1/5000 = 0.0002, so a score margin under 0.0005 is decided by one or two images - a coin flip that silently changes which checkpoint enters the robustness evaluation.

## Margin between the best and runner-up epoch

| seed | config | best epoch | best score | runner-up epoch | margin | margin in images (acc) | coin flip |
|---:|---|---:|---:|---:|---:|---:|:--:|
| 42 | fixed_a0.30 | 5 | 0.97699 | 4 | +0.00133 | +1.0 | no |
| 42 | fixed_a0.50 | 5 | 0.97684 | 3 | +0.00101 | -1.0 | no |
| 42 | fixed_a0.70 | 5 | 0.97602 | 3 | +0.00021 | -4.0 | YES |
| 42 | adaptive_tau0.50 | 5 | 0.97607 | 3 | +0.00023 | -6.0 | YES |
| 42 | adaptive_tau1.00 | 5 | 0.97627 | 4 | +0.00043 | -7.0 | YES |
| 42 | adaptive_tau2.00 | 5 | 0.97652 | 4 | +0.00069 | -5.0 | no |
| 43 | fixed_a0.30 | 3 | 0.97371 | 4 | +0.00105 | +17.0 | no |
| 43 | fixed_a0.50 | 3 | 0.97327 | 2 | +0.00044 | +6.0 | YES |
| 43 | fixed_a0.70 | 3 | 0.97321 | 2 | +0.00057 | +6.0 | no |
| 43 | adaptive_tau0.50 | 3 | 0.97307 | 2 | +0.00027 | +2.0 | YES |
| 43 | adaptive_tau1.00 | 3 | 0.97310 | 2 | +0.00017 | +2.0 | YES |
| 43 | adaptive_tau2.00 | 3 | 0.97316 | 2 | +0.00023 | +3.0 | YES |
| 44 | fixed_a0.30 | 5 | 0.97472 | 3 | +0.00110 | +2.0 | no |
| 44 | fixed_a0.50 | 5 | 0.97386 | 3 | +0.00025 | -4.0 | YES |
| 44 | fixed_a0.70 | 3 | 0.97372 | 5 | +0.00057 | +5.0 | no |
| 44 | adaptive_tau0.50 | 3 | 0.97420 | 5 | +0.00078 | +8.0 | no |
| 44 | adaptive_tau1.00 | 3 | 0.97427 | 5 | +0.00079 | +10.0 | no |
| 44 | adaptive_tau2.00 | 5 | 0.97367 | 3 | +0.00007 | -4.0 | YES |

9 of 18 runs (50%) were decided by a margin below 0.0005.

## Epoch selected per seed

- seed 42 -> epoch 5: 6 config(s)
- seed 43 -> epoch 3: 6 config(s)
- seed 44 -> epoch 3: 3 config(s), epoch 5: 3 config(s)  (e3=adaptive_tau0.50+adaptive_tau1.00+fixed_a0.70; e5=adaptive_tau2.00+fixed_a0.30+fixed_a0.50)

Seeds 42 and 43 are homogeneous, so their between-config differences are attributable to alpha. Seed 44 splits, which creates a within-seed natural experiment.

## Seed 44 natural experiment: epoch group vs robustness

Same validation split, same perturbation realisations, so the only systematic difference between the two groups is which epoch was shipped. Gap is later epoch group minus earlier epoch group, so a positive number means the run that trained longer and was selected at that later epoch did better.

| condition | acc e3 | acc e5 | acc gap (e5-e3, pp) | mAP e3 | mAP e5 | mAP gap (e5-e3, pp) |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.9626 | 0.9613 | -0.13 | 0.9855 | 0.9868 | +0.13 |
| noise1 | 0.7995 | 0.7871 | -1.24 | 0.8851 | 0.8716 | -1.35 |
| noise2 | 0.4209 | 0.4213 | +0.04 | 0.5447 | 0.5270 | -1.77 |
| blur1 | 0.9377 | 0.9364 | -0.13 | 0.9705 | 0.9680 | -0.25 |
| blur2 | 0.7080 | 0.7080 | +0.00 | 0.8126 | 0.8091 | -0.35 |

Group membership:

- seed 44 epoch 3: adaptive_tau0.50;adaptive_tau1.00;fixed_a0.70
- seed 44 epoch 5: adaptive_tau2.00;fixed_a0.30;fixed_a0.50

Note the group split is NOT monotone in alpha (alpha=0.70 lands in the same group as the two adaptive runs, while alpha=0.30 and alpha=0.50 land in the other), so the seed-44 corruption ranking cannot be read as an alpha effect.

## What this means for the cross-seed comparison

- Between-config differences measured on `best.pt` mix the alpha effect with whichever epoch won a near-tied validation score. In seed 44 that second term is worth up to ~1.5 pp of accuracy and ~1.8 pp of mAP under noise, i.e. larger than the alpha effect being measured.
- A fixed-epoch protocol (for example always epoch 5, or the epoch chosen by a criterion that is not the clean validation score) would remove this term. It cannot be recovered retrospectively, because only the winning epoch's weights are saved in `best.pt`.
- Seeds 42 and 43 are unaffected by this confound and remain the cleaner evidence for the alpha ordering.
