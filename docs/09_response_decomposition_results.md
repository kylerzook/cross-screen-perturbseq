# What knockdown responses contain (Replogle K562 genome-wide Perturb-seq)

Generated from `results/decomposition/*.json` (`experiments/response_decomposition.py`). Median [IQR] unless stated.

Knockdowns analysed: 4507 (fold_expr < 0.5 and energy-test p < 0.05); genes: 8248; non-targeting controls used for the noise floor: 585. The knocked-down gene's own column is masked.

## 1. Decomposition

- Noise floor: mean-square per gene ≈ 1.012 / n_cells (from non-targeting profiles).
- Share of each knockdown profile's total energy that is expected noise: 0.737 [0.494, 0.862].
- Knockdowns with signal above the noise floor: 0.988.
- Pathway components above a permutation null: K = 3 (top singular values [239.3000030517578, 144.1999969482422, 125.30000305175781, 117.19999694824219, 99.30000305175781]; null 122.2).

Share of SIGNAL (total minus expected noise) per component:

| Pathway components removed (K) | Generic (median [IQR]) | Pathway | Specific | Pooled: generic / pathway / specific | Median, top-25% effect knockdowns: generic / pathway / specific |
|---|---|---|---|---|---|
| 3 (permutation null) | 0.111 [0.037, 0.219] | 0.143 [0.068, 0.295] | 0.697 [0.518, 0.825] | 0.224 / 0.263 / 0.513 | 0.214 / 0.197 / 0.552 |
| 20 | 0.107 [0.036, 0.213] | 0.435 [0.311, 0.611] | 0.412 [0.218, 0.554] | 0.222 / 0.486 / 0.292 | 0.214 / 0.455 / 0.291 |
| 50 | 0.104 [0.033, 0.206] | 0.561 [0.426, 0.735] | 0.278 [0.115, 0.428] | 0.217 / 0.584 / 0.199 | 0.211 / 0.545 / 0.206 |

### Is the specific part real? Cross-screen replicate (same knockdown in Replogle's separate K562 essential screen)

Knockdowns in both screens: 1454; shared genes: 8200. Noise is independent between screens, so the inner product of the two profiles estimates reproducible energy.

| Component | Share of reproducible energy (pooled) | Between-screen correlation, same knockdown |
|---|---|---|
| Raw profile | 1.000 | 0.362 [0.170, 0.556] |
| Generic | 0.269 | (rank-1, trivially ≈ 1) |
| Pathway (K = 3) | 0.306 | 0.931 [0.728, 0.985] |
| Specific residual | 0.425 | 0.221 [0.109, 0.374] |
| Specific residual, MISMATCHED knockdowns (null) | – | -0.003 [-0.027, 0.022] |

Fraction of knockdowns whose specific residual correlates > 0.1 across screens: 0.772.

Within-screen second guide/TSS (only 19 pairs): residual corr on top-1000 variable genes 0.049 [-0.015, 0.154] vs random pairs 0.001 [-0.050, 0.053].

## 2. Can the specific part of a held-out knockdown be predicted?

5-fold CV over knockdowns (grouped by target gene); components re-fit on training folds. R² is relative to predicting zero; correlations are Pearson with the held-out profile.

| Target · Predictor | Corr, all knockdowns (mean) | Corr on top-1000 variable genes, all | Corr top-1000, reproducible knockdowns (cross-screen residual r > 0.3) | Frac positive (reproducible) | R² vs zero (mean) |
|---|---|---|---|---|---|
| residual: zero | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| residual: mean training residual | 0.000 | 0.001 | -0.007 | 0.484 | -0.000 |
| residual: similarity transfer (k=10) | 0.013 | 0.026 | 0.065 | 0.575 | -0.169 |
| residual: similarity transfer (k=50) | 0.019 | 0.036 | 0.093 | 0.659 | -0.034 |
| raw: zero | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| raw: mean training profile | 0.164 | 0.204 | 0.397 | 0.988 | -0.016 |
| raw: similarity transfer (k=10) | 0.117 | 0.173 | 0.350 | 0.957 | -0.214 |
| generic+pathway removed: similarity transfer (k=10) | 0.013 | 0.026 | 0.065 | 0.575 | -0.169 |

Reproducible subset size: 508 knockdowns. Ceiling: the cross-screen residual correlation (median 0.221 over all; > 0.3 by construction in the subset).

Linear interventional SEM on a 400-gene set where every gene is also a knockdown target:

| Setting | Corr (mean) | Frac positive | R² vs zero (mean) |
|---|---|---|---|
| residual (400-gene set): zero | 0.000 | 0.000 | 0.000 |
| residual (400-gene set): linear SEM | -0.008 | 0.300 | -0.002 |
| raw (400-gene set): linear SEM | 0.009 | 0.420 | 0.000 |
| raw (400-gene set): mean training profile | 0.309 | 0.983 | 0.050 |

## 3. Specificity by gene class

| Metric | TF median (n) | Essential non-TF median (n) | Other median (n) | p TF vs essential | p TF vs other |
|---|---|---|---|---|---|
| specific_share_of_signal | 0.752 (247) | 0.601 (1449) | 0.733 (2811) | 5.5e-14 | 0.96 |
| log10 signal-to-noise | -0.633 (247) | 0.109 (1449) | -0.628 (2811) | 7e-64 | 0.4 |
| uniqueness (1 - max |cos| to other knockdowns) | 0.808 (247) | 0.510 (1449) | 0.798 (2811) | 0 | 0.31 |
| generic_frac | 0.096 (247) | 0.174 (1449) | 0.085 (2811) | 2.3e-15 | 0.28 |
| pathway_frac | 0.125 (247) | 0.177 (1449) | 0.131 (2811) | 0.00057 | 0.88 |

Effect-size-matched (knockdowns with above-median signal only): specific share TF 0.756 (n=80), essential 0.562 (n=1164), other 0.712 (n=1009).

| Metric | Spearman with knockdown strength (1 − fold_expr) | with n cells | with log10 signal energy |
|---|---|---|---|
| specific_share_of_signal | 0.084 | 0.227 | -0.200 |
| log10 signal-to-noise | -0.226 | -0.117 | 0.920 |
| uniqueness (1 - max |cos| to other knockdowns) | 0.213 | 0.184 | -0.852 |
| generic_frac | -0.135 | -0.256 | 0.349 |
| pathway_frac | -0.015 | -0.091 | 0.023 |

## 4. Biology check: are a TF's bound / motif-carrying promoters enriched among its responders?

Promoter window TSS −1 kb / +0.5 kb. ChIP = ENCODE K562 IDR peaks. Motif = top-10% JASPAR occupancy. Responders = top 1% by |value|.

| Evidence | Subset | n TFs | Base rate | P(bound \| top-1%) raw | P(bound \| top-1%) residual | OR raw (median) | OR residual (median) | AUROC raw | AUROC residual |
|---|---|---|---|---|---|---|---|---|---|
| chip | all perturbed TFs | 92 | 0.346 | 0.371 | 0.362 | 1.205 | 1.150 | 0.504 | 0.502 |
| chip | TFs with signal > noise | 12 | 0.476 | 0.454 | 0.449 | 1.257 | 1.015 | 0.508 | 0.503 |
| chip | TFs with signal > 2x noise | 6 | 0.523 | 0.491 | 0.504 | 1.143 | 1.038 | 0.510 | 0.498 |
| motif | all perturbed TFs | 246 | 0.100 | 0.111 | 0.109 | 1.085 | 1.085 | 0.502 | 0.502 |
| motif | TFs with signal > noise | 27 | 0.100 | 0.129 | 0.131 | 1.232 | 1.232 | 0.502 | 0.502 |
| motif | TFs with signal > 2x noise | 11 | 0.100 | 0.094 | 0.086 | 0.941 | 0.941 | 0.495 | 0.495 |

chip: residual OR higher than raw for 0.30 of TFs (Wilcoxon p = 0.024).

motif: residual OR higher than raw for 0.37 of TFs (Wilcoxon p = 0.69).

## Caveats

- Pseudobulk z-scores only (no single cells); the noise model assumes per-gene noise ∝ 1/n_cells, calibrated on non-targeting controls.
- 'Generic' = projection on the mean knockdown profile; 'pathway' = top-K PCs. The specific share depends strongly on K (see table 1); K = 3 is the permutation-null choice.
- Cross-screen replicates cover only essential-gene knockdowns (1,454), which have stronger effects than average.
- TFs have low signal-to-noise as a class; only 12 perturbed TFs with ChIP have signal above the noise floor.
- Predictors tested are simple (similarity transfer, linear SEM); a stronger model could do better, but the ceiling is set by reproducibility.
