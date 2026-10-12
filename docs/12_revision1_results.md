# Revision 1 — new analyses (numeric results)

Script: `experiments/revision.py` (+ `final_synthetic.py f5masks|depthctrl`). JSON in `results/revision/` and `results/final/`. Manuscript: `paper/main.pdf`; reply: `paper/response_to_reviewers.pdf`.

| Analysis | Metric | Result | Meaning |
|---|---|---|---|
| Decomposition, pooled vs median | expected noise share of energy | pooled 0.47; median knockdown 0.74 | first version mixed the two |
| Decomposition vs K | specific share of signal, K = 0 / 3 / 50 | 0.78 / 0.51 / 0.18 | "specific" depends strongly on K |
| Noise model across screens | observed / predicted noise energy of gw−es difference | 1.30 (controls 1.18) | noise model roughly right, likely underestimates perturbed noise |
| Axis stability | cosines generic, axis 1–3 (separate fits per screen) | 0.91; 0.97, 0.94, 0.01 | 2 of 3 axes reproduce (ribosomal, mito-encoded); axis 3 (erythroid) doesn't |
| Benchmark (leak fixed) | best corr on independent-screen specific target | kNN-100 0.071 [0.062, 0.080] | ≈11% of equal-reliability ceiling √r = 0.66 |
| Benchmark | R² vs zero, independent-screen targets | ≤ 0 for all predictors (mean profile on raw: 0.007) | no predictor beats zero on magnitude |
| Cascade test, real | cascade − random-mediator corr: residual / generic-removed / raw | −0.004 [−0.012, 0.004] / +0.133 / +0.137 | cascade-like structure lives on shared axes |
| Cascade positive control | detection at α = 0.05, coverage 1 / 0.5 / 0.25 | 0.044 / 0.021 / 0.011 (all CIs > 0) | residual cascade share ≥5% excluded if ≥25% of mediators perturbed |
| Binding (promoters) | AUROC occ×ATAC vs PWM max (binding) | 0.610 vs 0.566 (paired +0.045, 97% of TFs) | occupancy helps binding a little |
| Binding | hub AUROC overall / within ATAC / within expression quintiles | 0.761 / 0.753 / 0.758 | hub not explained by accessibility or activity |
| Function | AUROC for responders: measured ChIP / occupancy | 0.496 / 0.518 (CIs include 0.5) | measured binding fails too → not the occupancy approximation |
| Context, noise-matched | significant changes obs vs null: Norman / Wessels | 12,084 vs 1,421 / 1,692 vs 565 | context dependence common |
| Context | reversal share of significant changes | 1.8% / 1.1% | signs of strong shared responses preserved |
| Imperfect masks (AUPRC×) | exact 22.2; −25% regulators 15.9; +200% false 15.9; −25% & +100% false 10.7 (no prior 15.5) | each missing regulator costs far more than a false one | hard masks need high-recall annotation |
| Propagation, depth ≥3 regulators | corr at D=1 → D≥3: true operator 0.61 → 0.78; estimated (stabilised) 0.70 → 0.66–0.71 | depth helps only with an accurate operator | limitation is operator accuracy |
