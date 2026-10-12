# Review-driven analyses (numeric results)

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

## Round 2

| Analysis | Metric | Result | Meaning |
|---|---|---|---|
| Noise-constant estimators | pooled / median noise share | 0.47–0.50 / 0.74–0.78 (bootstrap over controls 0.467–0.473 / 0.733–0.742) | headline noise fractions robust to the estimator |
| Matched axes | \|cos\| GW axis k vs essential axis k | 0.95, 0.91, ≤0.12 | axes 1–2 individually reproduce |
| Benchmark, calibrated | R²₀ after training-only scalar calibration (raw / centred / specific) | 0.033–0.047 / 0.005–0.013 / 0.002–0.004 | negative scores were mostly over-scaling |
| Benchmark, paired | k=100 − k=10 on specific target | Δcorr +0.023 [0.016, 0.030]; calibrated ΔR²₀ +0.0024 [0.0016, 0.0031] | more neighbours helps beyond shrinkage |
| Positive control, exact α | detection at α=0.05 (base generator) | coverage 1 / 0.5 / 0.25: 0.047 / 0.026 / 0.008 | sensitive under favourable generator |
| Positive control, robustness | random mediator ranking; invisible mediators; all adverse | 0.006 at α=0.05; none even at α=0.5; none | no calibrated exclusion possible |
| Context, all categories | other (intermediate) share of significant changes | Norman 23.6%, Wessels 36.3% | categories now sum to total |
| Propagation, same targets | depth ≥3: mean(D=2–4) − D=1 | true +0.19 (noise-free) / +0.07 (measured); estimated stabilised +0.04 / −0.02 | depth helps only with the true operator |
