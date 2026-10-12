# Context-dependence of perturbation effects — Norman et al. 2019 (numeric results)

Generated from `results/context/part1.json` and `part2.json` (`experiments/context_dependence.py`). K562, CRISPRa. Median [IQR] unless stated.

Cells 111445 (controls 11855); single perturbations 105; pairs with both singles 131; genes scored 2000 highly variable genes (perturbed genes masked).
Noise calibration: control half-vs-half z-scores have SD 0.987 (ideal 1.0); fraction |z| > 3 = 0.0010 (ideal 0.0027).

## Part 1 — does activating gene C change what activating gene A does?

### 1a. Size of non-additivity

| Metric | Value |
|---|---|
| Interaction energy / expected noise energy | 1.941 [1.555, 3.703] |
| Fraction of pairs with interaction energy > 2× noise | 0.496 |
| Interaction share of pair effect (noise-corrected) | 0.140 [0.080, 0.298] |
| Significant interaction genes per pair (FDR < 0.05) | 14.000 [4.500, 40.500] |
| Fraction of pairs with ≥ 1 significant interaction gene | 0.931 |
| Spearman(interaction share, pair signal-to-noise) | -0.325 |

### 1b. Which simple model explains the pair? (built from half A of cells, scored on half B)

| Model | Median correlation with pair profile | Median R² |
|---|---|---|
| Replicate (half A pair vs half B pair) — ceiling | 0.869 | – |
| additive | 0.819 | 0.576 |
| scaled (a*A + c*C) | 0.839 | 0.688 |
| dominant single | 0.793 | 0.519 |

Fitted scaling a + c: median 1.462; fraction of pairs with a + c > 1.2 (synergy/amplification): 0.679; < 0.8 (buffering): 0.076.

### 1c. Held-out pairs (5-fold over pairs; predictions from half A, scored on half B)

| Predictor | Mean corr | Median corr | n pairs | Mean gain vs additive | Fraction of pairs beating additive |
|---|---|---|---|---|---|
| additive | 0.785 | 0.819 | 131 | – | – |
| global saturation (isotonic on A+C) | 0.799 | 0.839 | 131 | 0.015 | 0.855 |
| modulator low-rank (K=1) | 0.794 | 0.813 | 106 | -0.007 | 0.336 |
| modulator low-rank (K=3) | 0.792 | 0.831 | 106 | -0.009 | 0.359 |
| modulator low-rank (K=5) | 0.791 | 0.831 | 106 | -0.010 | 0.366 |
| replicate ceiling | 0.820 | 0.869 | 131 | – | – |

### 1d. Structure

| Metric | Observed | Null | Note |
|---|---|---|---|
| Spread (SD) of per-gene modulator scores | 0.272 | 0.380 | permutation p = 0.892 |
| Interaction-matrix components above null | 2 | – | singular values [289.20001220703125, 153.1999969482422, 140.0] vs null top 146.60000610351562 |
| Variance share of top interaction component | 0.187 | 0.048 | |

Most frequently interacting target genes (count of pairs with |z| > 4): HBZ (46), HBG2 (44), AIF1 (43), ALAS2 (37), HBA1 (33), LST1 (31), TYROBP (29), HBG1 (27), LGALS1 (27), CORO1A (26), CD52 (25), FAM178B (25).

Top modulator scores (gene, score, n pairs): ZC3HAV1 1.188 (n=3), CEBPE 1.172 (n=11), BCL2L11 1.122 (n=2), OSR2 0.941 (n=4), MAPK1 0.85 (n=8), DUSP9 0.727 (n=6), BAK1 0.719 (n=3), C3orf72 0.682 (n=1).

Most interacting pairs (pair, interaction share, n sig genes): ZC3HAV1_CEBPE 3.224 (191), DUSP9_MAPK1 3.06 (134), CEBPE_PTPN12 2.399 (171), PTPN12_OSR2 2.339 (59), BCL2L11_BAK1 2.132 (8), CEBPE_KLF1 2.07 (312), SET_CEBPE 1.889 (378), CEBPE_CNN1 1.52 (112).

### 1e. Rewiring of target sets

| Comparison | Jaccard of significant targets (mean) | Fraction of shared targets that flip sign (mean) | n |
|---|---|---|---|
| A alone vs A in C's context | 0.185 | 0.015 | 260 |
| A alone, half A vs half B (reproducibility) | 0.430 | 0.000 | 251 |

## Part 2 — does cell state change a perturbation's effect?

Perturbations with ≥ 300 cells; genes defining the states excluded from scoring (1978 genes). Heterogeneity = Cochran-Q-like statistic across state bins, compared with a label-permutation null.

| State | n perturbations | Fraction state-dependent (q < 0.05) | Heterogeneity / null (median) | Between-state corr of top-50 targets (median) | Sign flips / strong targets | Effect size max/min state (median) |
|---|---|---|---|---|---|---|
| cell-cycle phase | 152 | 0.724 | 1.110 | 0.959 | 5 / 10223 | 1.258 |
| ISR level | 152 | 0.000 | 1.025 | 0.985 | 1 / 5651 | 1.116 |

Most state-dependent (cell-cycle phase): CEBPE 3.17, CEBPA 2.21, KLF1_CEBPA 1.83, SPI1 1.81, CEBPB 1.7, FOXA3_HOXB9 1.66, IKZF3 1.64, CEBPE_RUNX1T1 1.62. Permutations: 50 (minimum attainable p = 0.020).

Most state-dependent (ISR level): CEBPE 2.42, CEBPA 1.47, KLF1_CEBPA 1.43, COL1A1 1.34, CEBPE_RUNX1T1 1.33, IKZF3 1.31, OSR2 1.31, SPI1 1.28. Permutations: 50 (minimum attainable p = 0.020).
