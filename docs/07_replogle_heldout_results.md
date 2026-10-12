# Real-data test: Replogle K562 Perturb-seq, held-out knockdowns (numeric results)

Generated from `results/replogle/replogle_heldout.json` (`experiments/replogle_heldout.py`). Steady-state linear SEM fit by KD-aware nodewise Lasso on pseudobulk z-score profiles; 5-fold CV over knockdowns (each knockdown held out once); mean ± sd over held-out knockdowns (edge metrics: over the 5 folds).

Genes (strong knockdown, fold_expr < 0.5): 203; TFs with K562 ChIP-seq among them: 11; ChIP-bound base rate for TF→gene pairs: 0.391; mean |z| over all held-out (regulator, gene) pairs: 0.256.

| Method | Held-out profile corr | Sign acc. (abs z ≥ 1) | Top-20 responder overlap | Mean abs z of top-200 edges from held-out regulators | ChIP-bound frac of top-200 TF edges |
|---|---|---|---|---|---|
| KD-aware, no prior | 0.012 ± 0.298 | 0.437 ± 0.388 | 0.256 ± 0.163 | 0.384 ± 0.055 | 0.409 ± 0.029 |
| KD-aware + ChIP prior (a=0.8) | 0.011 ± 0.296 | 0.444 ± 0.384 | 0.257 ± 0.161 | 0.356 ± 0.076 | 0.891 ± 0.022 |
| KD-aware + JASPAR occupancy prior (a=0.8) | 0.011 ± 0.295 | 0.456 ± 0.386 | 0.255 ± 0.162 | 0.362 ± 0.066 | 0.426 ± 0.009 |
| KD-aware + shuffled ChIP prior (a=0.8) | 0.013 ± 0.296 | 0.442 ± 0.387 | 0.257 ± 0.159 | 0.358 ± 0.075 | 0.630 ± 0.014 |
| KD-unaware, no prior | 0.003 ± 0.309 | 0.429 ± 0.394 | 0.269 ± 0.163 | 0.403 ± 0.061 | 0.408 ± 0.028 |
| baseline: mean training profile | 0.422 ± 0.402 | 0.732 ± 0.405 | 0.520 ± 0.223 | – | – |
| baseline: target only (no propagation) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.047 ± 0.054 | – | – |

Note: ChIP-bound fraction for prior-using models is partly circular (the prior pushes ChIP-bound edges up); the informative value is the no-prior model vs the base rate.
