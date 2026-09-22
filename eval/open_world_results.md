# Open-world evaluation -- `test`, evaluated once (eval/PROTOCOL.md)

Frozen rule: `max_cosine` at tau = 0.607094 (adopted). Config `9d85285aa44f...`, BioCLIP `2957b322090f9cb17ae72c71981c7218a28d81e0`, code `d8a8cc740a3c`.

**Not evaluated: quality (no labels), price (simulated), or the full agent (see the M15c pilot).**

## In-distribution (test)
n = 475. Unfiltered top-1 accuracy 82.7%.

| metric | value [95% CI] |
|---|---|
| false-abstain rate at tau | 4.00% [2.32, 5.89] |
| coverage at tau | 96.00% [94.11, 97.68] |
| selective accuracy at tau | 82.89% [79.43, 86.21] |
| risk-coverage AUC (lower is better) | 0.03 [0.02, 0.05] |

## near_ood (test)
n = 620 images, 31 categories.

| metric | value [95% CI] |
|---|---|
| AUROC (ID positive) | 0.90 [0.83, 0.96] |
| AUPR-Out (OOD positive) | 0.93 [0.89, 0.96] |
| FPR @ 95% TPR | 42.90% [26.45, 58.23] |
| abstention rate (correct rejection) | 53.06% [41.61, 64.35] |

## far_ood (test)
n = 780 images, 39 categories.

| metric | value [95% CI] |
|---|---|
| AUROC (ID positive) | 1.00 [1.00, 1.00] |
| AUPR-Out (OOD positive) | 1.00 [1.00, 1.00] |
| FPR @ 95% TPR | 1.79% [0.51, 3.59] |
| abstention rate (correct rejection) | 97.69% [96.03, 99.10] |

## Corrupted-in-set (test) -- robustness, not OOD

| corruption | n | top-1 accuracy | abstention rate |
|---|---:|---|---|
| color_shift | 90 | 71.11% [61.11, 80.00] | 7.78% [2.22, 13.33] |
| crop | 90 | 85.56% [77.78, 92.22] | 3.33% [0.00, 7.78] |
| dark | 90 | 87.78% [81.11, 94.44] | 5.56% [1.11, 11.11] |
| gaussian_blur | 90 | 70.00% [60.00, 78.89] | 5.56% [1.11, 11.11] |
| jpeg | 90 | 84.44% [76.67, 91.11] | 5.56% [1.11, 11.11] |
| low_res | 90 | 82.22% [74.44, 90.00] | 7.78% [2.22, 13.33] |
| occlusion | 90 | 70.00% [60.00, 80.00] | 7.78% [2.22, 13.33] |
| rotate | 90 | 86.67% [78.89, 93.33] | 3.33% [0.00, 7.78] |

No per-species claims are made: see eval/PROTOCOL.md for why.
