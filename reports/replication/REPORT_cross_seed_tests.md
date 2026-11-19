# Cross-seed paired tests (robustness evaluation)

Seeds: 42, 43, 44. Every comparison is paired within a seed: the same perturbed validation images are shown to the candidate and to that seed's baseline checkpoint.

## Plain summary

Pre-registered hypothesis from seed 42: **fixed alpha=0.3 beats fixed alpha=0.5 under noise2**.

- Per seed -> s42: +1.00 pp (p=2.5e-06), s43: +0.50 pp (p=0.011), s44: +0.06 pp (p=0.82).
- Direction replicated in 3/3 seeds.
- Pooled over seeds: mean +0.52 pp, +78 net images out of 280 discordant, exact pooled p=3.66e-06, bootstrap 95% CI [+0.30, +0.74] pp.
- Verdict: REPLICATED.

Sharper question: under noise2, does any adaptive setting match the best fixed one (alpha=0.3)?

- `adaptive_tau0.50`: mean -0.79 pp vs alpha=0.3, pooled exact p=1.77e-05, CI [-1.15, -0.43] -> significantly worse.
- `adaptive_tau1.00`: mean -0.74 pp vs alpha=0.3, pooled exact p=3.06e-05, CI [-1.08, -0.39] -> significantly worse.
- `adaptive_tau2.00`: mean -0.62 pp vs alpha=0.3, pooled exact p=6.29e-07, CI [-0.87, -0.38] -> significantly worse.

## Accuracy vs `fixed_a0.50`, pooled over seeds

| condition | candidate | s42 | s43 | s44 | mean delta (pp) | net images | discordant | pooled exact p | bootstrap 95% CI | CI excludes 0 | p < Bonferroni |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | fixed_a0.30 | +0.00 | +0.12 | +0.10 | +0.073 | +11 | 37 | 0.09887 | [-0.01, +0.15] | no | no |
| clean | fixed_a0.70 | -0.06 | -0.04 | +0.12 | +0.007 | +1 | 101 | 1 | [-0.13, +0.14] | no | no |
| clean | adaptive_tau0.50 | -0.10 | -0.08 | +0.18 | +0.000 | +0 | 92 | 1 | [-0.13, +0.13] | no | no |
| clean | adaptive_tau1.00 | -0.08 | -0.06 | +0.18 | +0.013 | +2 | 86 | 0.9142 | [-0.11, +0.13] | no | no |
| clean | adaptive_tau2.00 | -0.04 | -0.04 | +0.00 | -0.027 | -4 | 12 | 0.3877 | [-0.07, +0.02] | no | no |
| noise1 | fixed_a0.30 | +0.30 | +0.08 | +0.06 | +0.147 | +22 | 136 | 0.07135 | [-0.01, +0.30] | no | no |
| noise1 | fixed_a0.70 | -0.48 | -0.44 | +1.54 | +0.207 | +31 | 403 | 0.135 | [-0.05, +0.47] | no | no |
| noise1 | adaptive_tau0.50 | -0.18 | -0.28 | +1.24 | +0.260 | +39 | 331 | 0.03658 | [+0.03, +0.50] | yes | no |
| noise1 | adaptive_tau1.00 | -0.02 | -0.08 | +1.10 | +0.333 | +50 | 298 | 0.004453 | [+0.11, +0.55] | yes | no |
| noise1 | adaptive_tau2.00 | +0.04 | +0.02 | +0.10 | +0.053 | +8 | 36 | 0.243 | [-0.03, +0.13] | no | no |
| noise2 | fixed_a0.30 | +1.00 | +0.50 | +0.06 | +0.520 | +78 | 280 | 3.656e-06 | [+0.30, +0.74] | yes | yes |
| noise2 | fixed_a0.70 | -0.98 | -0.60 | +0.12 | -0.487 | -73 | 679 | 0.005686 | [-0.83, -0.15] | yes | no |
| noise2 | adaptive_tau0.50 | -0.50 | -0.32 | +0.02 | -0.267 | -40 | 578 | 0.1047 | [-0.58, +0.04] | no | no |
| noise2 | adaptive_tau1.00 | -0.34 | -0.20 | -0.12 | -0.220 | -33 | 537 | 0.1672 | [-0.52, +0.09] | no | no |
| noise2 | adaptive_tau2.00 | -0.18 | -0.20 | +0.08 | -0.100 | -15 | 85 | 0.1284 | [-0.22, +0.02] | no | no |
| blur1 | fixed_a0.30 | +0.16 | +0.04 | +0.00 | +0.067 | +10 | 52 | 0.2116 | [-0.03, +0.16] | no | no |
| blur1 | fixed_a0.70 | -0.30 | -0.22 | +0.14 | -0.127 | -19 | 143 | 0.132 | [-0.29, +0.03] | no | no |
| blur1 | adaptive_tau0.50 | -0.18 | -0.14 | +0.18 | -0.047 | -7 | 135 | 0.6057 | [-0.20, +0.11] | no | no |
| blur1 | adaptive_tau1.00 | -0.12 | -0.04 | +0.12 | -0.013 | -2 | 124 | 0.9285 | [-0.16, +0.13] | no | no |
| blur1 | adaptive_tau2.00 | -0.06 | -0.02 | +0.06 | -0.007 | -1 | 15 | 1 | [-0.06, +0.04] | no | no |
| blur2 | fixed_a0.30 | +0.10 | +0.26 | +0.18 | +0.180 | +27 | 217 | 0.07732 | [-0.01, +0.37] | no | no |
| blur2 | fixed_a0.70 | -0.32 | -0.34 | -0.08 | -0.247 | -37 | 583 | 0.1359 | [-0.56, +0.07] | no | no |
| blur2 | adaptive_tau0.50 | -0.22 | -0.14 | +0.06 | -0.100 | -15 | 515 | 0.5373 | [-0.40, +0.20] | no | no |
| blur2 | adaptive_tau1.00 | -0.06 | +0.04 | +0.02 | +0.000 | +0 | 474 | 1 | [-0.29, +0.28] | no | no |
| blur2 | adaptive_tau2.00 | -0.06 | +0.00 | -0.18 | -0.080 | -12 | 52 | 0.1263 | [-0.17, +0.01] | no | no |

## Accuracy vs `fixed_a0.30`, pooled over seeds

| condition | candidate | s42 | s43 | s44 | mean delta (pp) | net images | discordant | pooled exact p | bootstrap 95% CI | CI excludes 0 | p < Bonferroni |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | fixed_a0.50 | +0.00 | -0.12 | -0.10 | -0.073 | -11 | 37 | 0.09887 | [-0.15, +0.01] | no | no |
| clean | fixed_a0.70 | -0.06 | -0.16 | +0.02 | -0.067 | -10 | 116 | 0.4035 | [-0.21, +0.07] | no | no |
| clean | adaptive_tau0.50 | -0.10 | -0.20 | +0.08 | -0.073 | -11 | 109 | 0.3382 | [-0.21, +0.07] | no | no |
| clean | adaptive_tau1.00 | -0.08 | -0.18 | +0.08 | -0.060 | -9 | 101 | 0.4262 | [-0.19, +0.07] | no | no |
| clean | adaptive_tau2.00 | -0.04 | -0.16 | -0.10 | -0.100 | -15 | 45 | 0.0357 | [-0.19, -0.01] | yes | no |
| noise1 | fixed_a0.50 | -0.30 | -0.08 | -0.06 | -0.147 | -22 | 136 | 0.07135 | [-0.30, +0.01] | no | no |
| noise1 | fixed_a0.70 | -0.78 | -0.52 | +1.48 | +0.060 | +9 | 477 | 0.7142 | [-0.21, +0.34] | no | no |
| noise1 | adaptive_tau0.50 | -0.48 | -0.36 | +1.18 | +0.113 | +17 | 405 | 0.4266 | [-0.15, +0.37] | no | no |
| noise1 | adaptive_tau1.00 | -0.32 | -0.16 | +1.04 | +0.187 | +28 | 364 | 0.1569 | [-0.06, +0.43] | no | no |
| noise1 | adaptive_tau2.00 | -0.26 | -0.06 | +0.04 | -0.093 | -14 | 148 | 0.2852 | [-0.25, +0.07] | no | no |
| noise2 | fixed_a0.50 | -1.00 | -0.50 | -0.06 | -0.520 | -78 | 280 | 3.656e-06 | [-0.74, -0.31] | yes | yes |
| noise2 | fixed_a0.70 | -1.98 | -1.10 | +0.06 | -1.007 | -151 | 841 | 2.151e-07 | [-1.39, -0.63] | yes | yes |
| noise2 | adaptive_tau0.50 | -1.50 | -0.82 | -0.04 | -0.787 | -118 | 746 | 1.77e-05 | [-1.15, -0.43] | yes | yes |
| noise2 | adaptive_tau1.00 | -1.34 | -0.70 | -0.18 | -0.740 | -111 | 699 | 3.063e-05 | [-1.08, -0.39] | yes | yes |
| noise2 | adaptive_tau2.00 | -1.18 | -0.70 | +0.02 | -0.620 | -93 | 345 | 6.292e-07 | [-0.87, -0.38] | yes | yes |
| blur1 | fixed_a0.50 | -0.16 | -0.04 | +0.00 | -0.067 | -10 | 52 | 0.2116 | [-0.16, +0.03] | no | no |
| blur1 | fixed_a0.70 | -0.46 | -0.26 | +0.14 | -0.193 | -29 | 173 | 0.03297 | [-0.37, -0.03] | yes | no |
| blur1 | adaptive_tau0.50 | -0.34 | -0.18 | +0.18 | -0.113 | -17 | 167 | 0.2156 | [-0.28, +0.05] | no | no |
| blur1 | adaptive_tau1.00 | -0.28 | -0.08 | +0.12 | -0.080 | -12 | 156 | 0.3785 | [-0.25, +0.08] | no | no |
| blur1 | adaptive_tau2.00 | -0.22 | -0.06 | +0.06 | -0.073 | -11 | 63 | 0.2074 | [-0.18, +0.03] | no | no |
| blur2 | fixed_a0.50 | -0.10 | -0.26 | -0.18 | -0.180 | -27 | 217 | 0.07732 | [-0.38, +0.01] | no | no |
| blur2 | fixed_a0.70 | -0.42 | -0.60 | -0.26 | -0.427 | -64 | 688 | 0.01625 | [-0.77, -0.09] | yes | no |
| blur2 | adaptive_tau0.50 | -0.32 | -0.40 | -0.12 | -0.280 | -42 | 622 | 0.1001 | [-0.61, +0.05] | no | no |
| blur2 | adaptive_tau1.00 | -0.16 | -0.22 | -0.16 | -0.180 | -27 | 579 | 0.2799 | [-0.49, +0.13] | no | no |
| blur2 | adaptive_tau2.00 | -0.16 | -0.26 | -0.36 | -0.260 | -39 | 237 | 0.01341 | [-0.46, -0.06] | yes | no |

## Absolute performance per condition (mean +/- sd over seeds)

| config | condition | s42 | s43 | s44 | acc mean | acc sd | mAP mean | mAP sd | acc drop vs own clean (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_a0.30 | clean | 0.9646 | 0.9626 | 0.9620 | 0.9631 | 0.00136 | 0.9872 | 0.00230 | 0.00 +/- 0.00 |
| fixed_a0.30 | noise1 | 0.8046 | 0.7926 | 0.7872 | 0.7948 | 0.00891 | 0.8764 | 0.01103 | 16.83 +/- 0.76 |
| fixed_a0.30 | noise2 | 0.4426 | 0.4190 | 0.4214 | 0.4277 | 0.01299 | 0.5240 | 0.01418 | 53.54 +/- 1.17 |
| fixed_a0.30 | blur1 | 0.9380 | 0.9354 | 0.9362 | 0.9365 | 0.00133 | 0.9713 | 0.00449 | 2.65 +/- 0.07 |
| fixed_a0.30 | blur2 | 0.7088 | 0.7096 | 0.7098 | 0.7094 | 0.00053 | 0.8057 | 0.00914 | 25.37 +/- 0.19 |
| fixed_a0.50 | clean | 0.9646 | 0.9614 | 0.9610 | 0.9623 | 0.00197 | 0.9870 | 0.00199 | 0.00 +/- 0.00 |
| fixed_a0.50 | noise1 | 0.8016 | 0.7918 | 0.7866 | 0.7933 | 0.00762 | 0.8756 | 0.01099 | 16.90 +/- 0.57 |
| fixed_a0.50 | noise2 | 0.4326 | 0.4140 | 0.4208 | 0.4225 | 0.00941 | 0.5256 | 0.01435 | 53.99 +/- 0.77 |
| fixed_a0.50 | blur1 | 0.9364 | 0.9350 | 0.9362 | 0.9359 | 0.00076 | 0.9713 | 0.00437 | 2.65 +/- 0.17 |
| fixed_a0.50 | blur2 | 0.7078 | 0.7070 | 0.7080 | 0.7076 | 0.00053 | 0.8072 | 0.00770 | 25.47 +/- 0.19 |
| fixed_a0.70 | clean | 0.9640 | 0.9610 | 0.9622 | 0.9624 | 0.00151 | 0.9862 | 0.00156 | 0.00 +/- 0.00 |
| fixed_a0.70 | noise1 | 0.7968 | 0.7874 | 0.8020 | 0.7954 | 0.00740 | 0.8795 | 0.00951 | 16.70 +/- 0.67 |
| fixed_a0.70 | noise2 | 0.4228 | 0.4080 | 0.4220 | 0.4176 | 0.00832 | 0.5307 | 0.01923 | 54.48 +/- 0.71 |
| fixed_a0.70 | blur1 | 0.9334 | 0.9328 | 0.9376 | 0.9346 | 0.00262 | 0.9713 | 0.00299 | 2.78 +/- 0.30 |
| fixed_a0.70 | blur2 | 0.7046 | 0.7036 | 0.7072 | 0.7051 | 0.00186 | 0.8076 | 0.00835 | 25.73 +/- 0.22 |
| adaptive_tau0.50 | clean | 0.9636 | 0.9606 | 0.9628 | 0.9623 | 0.00155 | 0.9866 | 0.00172 | 0.00 +/- 0.00 |
| adaptive_tau0.50 | noise1 | 0.7998 | 0.7890 | 0.7990 | 0.7959 | 0.00602 | 0.8799 | 0.00961 | 16.64 +/- 0.45 |
| adaptive_tau0.50 | noise2 | 0.4276 | 0.4108 | 0.4210 | 0.4198 | 0.00846 | 0.5306 | 0.01824 | 54.25 +/- 0.69 |
| adaptive_tau0.50 | blur1 | 0.9346 | 0.9336 | 0.9380 | 0.9354 | 0.00231 | 0.9719 | 0.00308 | 2.69 +/- 0.21 |
| adaptive_tau0.50 | blur2 | 0.7056 | 0.7056 | 0.7086 | 0.7066 | 0.00173 | 0.8080 | 0.00899 | 25.57 +/- 0.20 |
| adaptive_tau1.00 | clean | 0.9638 | 0.9608 | 0.9628 | 0.9625 | 0.00153 | 0.9866 | 0.00183 | 0.00 +/- 0.00 |
| adaptive_tau1.00 | noise1 | 0.8014 | 0.7910 | 0.7976 | 0.7967 | 0.00526 | 0.8800 | 0.01000 | 16.58 +/- 0.37 |
| adaptive_tau1.00 | noise2 | 0.4292 | 0.4120 | 0.4196 | 0.4203 | 0.00862 | 0.5300 | 0.01788 | 54.22 +/- 0.72 |
| adaptive_tau1.00 | blur1 | 0.9352 | 0.9346 | 0.9374 | 0.9357 | 0.00147 | 0.9721 | 0.00311 | 2.67 +/- 0.17 |
| adaptive_tau1.00 | blur2 | 0.7072 | 0.7074 | 0.7082 | 0.7076 | 0.00053 | 0.8081 | 0.00915 | 25.49 +/- 0.16 |
| adaptive_tau2.00 | clean | 0.9642 | 0.9610 | 0.9610 | 0.9621 | 0.00185 | 0.9868 | 0.00182 | 0.00 +/- 0.00 |
| adaptive_tau2.00 | noise1 | 0.8020 | 0.7920 | 0.7876 | 0.7939 | 0.00738 | 0.8756 | 0.01028 | 16.82 +/- 0.56 |
| adaptive_tau2.00 | noise2 | 0.4308 | 0.4120 | 0.4216 | 0.4215 | 0.00940 | 0.5245 | 0.01423 | 54.06 +/- 0.79 |
| adaptive_tau2.00 | blur1 | 0.9358 | 0.9348 | 0.9368 | 0.9358 | 0.00100 | 0.9708 | 0.00450 | 2.63 +/- 0.21 |
| adaptive_tau2.00 | blur2 | 0.7072 | 0.7070 | 0.7062 | 0.7068 | 0.00053 | 0.8064 | 0.00826 | 25.53 +/- 0.16 |

## T2I mAP delta (pp) - not pairable per image

| baseline | condition | candidate | s42 | s43 | s44 | mean | sd | #seeds positive |
|---|---|---|---:|---:|---:|---:|---:|---:|
| fixed_a0.50 | clean | fixed_a0.30 | +0.030 | -0.032 | +0.072 | +0.024 | 0.053 | 2 |
| fixed_a0.50 | clean | fixed_a0.70 | -0.105 | +0.028 | -0.148 | -0.075 | 0.092 | 1 |
| fixed_a0.50 | clean | adaptive_tau0.50 | -0.054 | +0.040 | -0.111 | -0.042 | 0.077 | 1 |
| fixed_a0.50 | clean | adaptive_tau1.00 | -0.036 | +0.027 | -0.098 | -0.035 | 0.062 | 1 |
| fixed_a0.50 | clean | adaptive_tau2.00 | -0.024 | +0.018 | -0.038 | -0.015 | 0.029 | 1 |
| fixed_a0.50 | noise1 | fixed_a0.30 | +0.059 | -0.009 | +0.173 | +0.074 | 0.092 | 2 |
| fixed_a0.50 | noise1 | fixed_a0.70 | -0.409 | +0.077 | +1.490 | +0.386 | 0.987 | 2 |
| fixed_a0.50 | noise1 | adaptive_tau0.50 | -0.212 | +0.107 | +1.391 | +0.429 | 0.849 | 2 |
| fixed_a0.50 | noise1 | adaptive_tau1.00 | -0.140 | +0.076 | +1.381 | +0.439 | 0.823 | 2 |
| fixed_a0.50 | noise1 | adaptive_tau2.00 | -0.084 | +0.047 | +0.033 | -0.001 | 0.072 | 2 |
| fixed_a0.50 | noise2 | fixed_a0.30 | -0.123 | -0.108 | -0.242 | -0.158 | 0.074 | 0 |
| fixed_a0.50 | noise2 | fixed_a0.70 | -0.241 | -0.079 | +1.849 | +0.509 | 1.163 | 1 |
| fixed_a0.50 | noise2 | adaptive_tau0.50 | -0.113 | -0.023 | +1.627 | +0.497 | 0.980 | 1 |
| fixed_a0.50 | noise2 | adaptive_tau1.00 | -0.101 | -0.051 | +1.481 | +0.443 | 0.899 | 1 |
| fixed_a0.50 | noise2 | adaptive_tau2.00 | -0.126 | -0.100 | -0.101 | -0.109 | 0.015 | 0 |
| fixed_a0.50 | blur1 | fixed_a0.30 | +0.018 | -0.035 | +0.026 | +0.003 | 0.033 | 2 |
| fixed_a0.50 | blur1 | fixed_a0.70 | -0.154 | +0.016 | +0.151 | +0.004 | 0.153 | 2 |
| fixed_a0.50 | blur1 | adaptive_tau0.50 | -0.093 | +0.031 | +0.245 | +0.061 | 0.171 | 2 |
| fixed_a0.50 | blur1 | adaptive_tau1.00 | -0.071 | +0.052 | +0.252 | +0.078 | 0.163 | 2 |
| fixed_a0.50 | blur1 | adaptive_tau2.00 | -0.056 | +0.023 | -0.129 | -0.054 | 0.076 | 1 |
| fixed_a0.50 | blur2 | fixed_a0.30 | -0.070 | -0.324 | -0.069 | -0.154 | 0.147 | 0 |
| fixed_a0.50 | blur2 | fixed_a0.70 | -0.009 | -0.059 | +0.172 | +0.035 | 0.122 | 1 |
| fixed_a0.50 | blur2 | adaptive_tau0.50 | +0.014 | -0.093 | +0.315 | +0.079 | 0.212 | 2 |
| fixed_a0.50 | blur2 | adaptive_tau1.00 | +0.015 | -0.098 | +0.362 | +0.093 | 0.239 | 2 |
| fixed_a0.50 | blur2 | adaptive_tau2.00 | +0.018 | -0.123 | -0.135 | -0.080 | 0.085 | 1 |
| fixed_a0.30 | clean | fixed_a0.50 | -0.030 | +0.032 | -0.072 | -0.024 | 0.053 | 1 |
| fixed_a0.30 | clean | fixed_a0.70 | -0.136 | +0.060 | -0.220 | -0.099 | 0.144 | 1 |
| fixed_a0.30 | clean | adaptive_tau0.50 | -0.084 | +0.072 | -0.184 | -0.065 | 0.129 | 1 |
| fixed_a0.30 | clean | adaptive_tau1.00 | -0.066 | +0.059 | -0.170 | -0.059 | 0.115 | 1 |
| fixed_a0.30 | clean | adaptive_tau2.00 | -0.054 | +0.050 | -0.111 | -0.038 | 0.082 | 1 |
| fixed_a0.30 | noise1 | fixed_a0.50 | -0.059 | +0.009 | -0.173 | -0.074 | 0.092 | 1 |
| fixed_a0.30 | noise1 | fixed_a0.70 | -0.469 | +0.086 | +1.318 | +0.312 | 0.914 | 2 |
| fixed_a0.30 | noise1 | adaptive_tau0.50 | -0.271 | +0.116 | +1.218 | +0.354 | 0.773 | 2 |
| fixed_a0.30 | noise1 | adaptive_tau1.00 | -0.200 | +0.085 | +1.208 | +0.364 | 0.744 | 2 |
| fixed_a0.30 | noise1 | adaptive_tau2.00 | -0.143 | +0.056 | -0.140 | -0.076 | 0.114 | 1 |
| fixed_a0.30 | noise2 | fixed_a0.50 | +0.123 | +0.108 | +0.242 | +0.158 | 0.074 | 3 |
| fixed_a0.30 | noise2 | fixed_a0.70 | -0.119 | +0.029 | +2.091 | +0.667 | 1.235 | 2 |
| fixed_a0.30 | noise2 | adaptive_tau0.50 | +0.010 | +0.084 | +1.869 | +0.654 | 1.052 | 3 |
| fixed_a0.30 | noise2 | adaptive_tau1.00 | +0.022 | +0.057 | +1.723 | +0.601 | 0.972 | 3 |
| fixed_a0.30 | noise2 | adaptive_tau2.00 | -0.004 | +0.008 | +0.141 | +0.048 | 0.080 | 2 |
| fixed_a0.30 | blur1 | fixed_a0.50 | -0.018 | +0.035 | -0.026 | -0.003 | 0.033 | 1 |
| fixed_a0.30 | blur1 | fixed_a0.70 | -0.172 | +0.051 | +0.125 | +0.001 | 0.155 | 2 |
| fixed_a0.30 | blur1 | adaptive_tau0.50 | -0.112 | +0.066 | +0.219 | +0.058 | 0.166 | 2 |
| fixed_a0.30 | blur1 | adaptive_tau1.00 | -0.090 | +0.087 | +0.226 | +0.075 | 0.158 | 2 |
| fixed_a0.30 | blur1 | adaptive_tau2.00 | -0.074 | +0.058 | -0.155 | -0.057 | 0.107 | 1 |
| fixed_a0.30 | blur2 | fixed_a0.50 | +0.070 | +0.324 | +0.069 | +0.154 | 0.147 | 3 |
| fixed_a0.30 | blur2 | fixed_a0.70 | +0.061 | +0.265 | +0.241 | +0.189 | 0.112 | 3 |
| fixed_a0.30 | blur2 | adaptive_tau0.50 | +0.084 | +0.231 | +0.384 | +0.233 | 0.150 | 3 |
| fixed_a0.30 | blur2 | adaptive_tau1.00 | +0.084 | +0.226 | +0.430 | +0.247 | 0.174 | 3 |
| fixed_a0.30 | blur2 | adaptive_tau2.00 | +0.087 | +0.202 | -0.066 | +0.074 | 0.135 | 2 |

## Which config ranks first in each seed

| condition | seed | best | worst | spread (pp) | full ranking |
|---|---:|---|---|---:|---|
| clean | 42 | fixed_a0.30 | adaptive_tau0.50 | 0.10 | fixed_a0.30;fixed_a0.50;adaptive_tau2.00;fixed_a0.70;adaptive_tau1.00;adaptive_tau0.50 |
| clean | 43 | fixed_a0.30 | adaptive_tau0.50 | 0.20 | fixed_a0.30;fixed_a0.50;fixed_a0.70;adaptive_tau2.00;adaptive_tau1.00;adaptive_tau0.50 |
| clean | 44 | adaptive_tau0.50 | adaptive_tau2.00 | 0.18 | adaptive_tau0.50;adaptive_tau1.00;fixed_a0.70;fixed_a0.30;fixed_a0.50;adaptive_tau2.00 |
| noise1 | 42 | fixed_a0.30 | fixed_a0.70 | 0.78 | fixed_a0.30;adaptive_tau2.00;fixed_a0.50;adaptive_tau1.00;adaptive_tau0.50;fixed_a0.70 |
| noise1 | 43 | fixed_a0.30 | fixed_a0.70 | 0.52 | fixed_a0.30;adaptive_tau2.00;fixed_a0.50;adaptive_tau1.00;adaptive_tau0.50;fixed_a0.70 |
| noise1 | 44 | fixed_a0.70 | fixed_a0.50 | 1.54 | fixed_a0.70;adaptive_tau0.50;adaptive_tau1.00;adaptive_tau2.00;fixed_a0.30;fixed_a0.50 |
| noise2 | 42 | fixed_a0.30 | fixed_a0.70 | 1.98 | fixed_a0.30;fixed_a0.50;adaptive_tau2.00;adaptive_tau1.00;adaptive_tau0.50;fixed_a0.70 |
| noise2 | 43 | fixed_a0.30 | fixed_a0.70 | 1.10 | fixed_a0.30;fixed_a0.50;adaptive_tau1.00;adaptive_tau2.00;adaptive_tau0.50;fixed_a0.70 |
| noise2 | 44 | fixed_a0.70 | adaptive_tau1.00 | 0.24 | fixed_a0.70;adaptive_tau2.00;fixed_a0.30;adaptive_tau0.50;fixed_a0.50;adaptive_tau1.00 |
| blur1 | 42 | fixed_a0.30 | fixed_a0.70 | 0.46 | fixed_a0.30;fixed_a0.50;adaptive_tau2.00;adaptive_tau1.00;adaptive_tau0.50;fixed_a0.70 |
| blur1 | 43 | fixed_a0.30 | fixed_a0.70 | 0.26 | fixed_a0.30;fixed_a0.50;adaptive_tau2.00;adaptive_tau1.00;adaptive_tau0.50;fixed_a0.70 |
| blur1 | 44 | adaptive_tau0.50 | fixed_a0.50 | 0.18 | adaptive_tau0.50;fixed_a0.70;adaptive_tau1.00;adaptive_tau2.00;fixed_a0.30;fixed_a0.50 |
| blur2 | 42 | fixed_a0.30 | fixed_a0.70 | 0.42 | fixed_a0.30;fixed_a0.50;adaptive_tau1.00;adaptive_tau2.00;adaptive_tau0.50;fixed_a0.70 |
| blur2 | 43 | fixed_a0.30 | fixed_a0.70 | 0.60 | fixed_a0.30;adaptive_tau1.00;fixed_a0.50;adaptive_tau2.00;adaptive_tau0.50;fixed_a0.70 |
| blur2 | 44 | fixed_a0.30 | adaptive_tau2.00 | 0.36 | fixed_a0.30;adaptive_tau0.50;adaptive_tau1.00;fixed_a0.50;fixed_a0.70;adaptive_tau2.00 |

## Caveats

- 5 conditions x 5 candidates x 2 baselines = 50 pooled tests; Bonferroni threshold 0.00100. The two baselines are not independent of each other (they share the same runs), so this correction is conservative.
- Accuracy on clean CIFAR-10 is saturated (96.0-96.5%); one image out of 5000 is 0.02 pp, so clean-set differences carry almost no information.
- The pooled exact test treats the seeds as independent strata. Their validation splits overlap by roughly the chance-level 10% (see REPORT_replication.md), which makes the pooled p-value mildly anti-conservative.
- mAP is a subset-level ranking metric: no per-image paired test exists, so only direction and spread across seeds are reported.
- Perturbations are deterministic synthetic Gaussian noise / blur applied to the 32px images before CLIP's 224px preprocessing. They are not CIFAR-10-C corruptions.
- Each run's checkpoint is its own best-validation epoch, and the validation split is also the evaluation split here, so these numbers are method-selection data rather than a held-out test estimate.
- Three seeds is the minimum for a mean +/- sd; the sd itself has very wide uncertainty (n-1 = 2 degrees of freedom).

## Consistency check

Wins/losses, accuracy deltas and exact McNemar p-values recomputed from `per_image_predictions.csv` reproduce every row of the per-seed `paired_vs_fixed_05.csv` files.
