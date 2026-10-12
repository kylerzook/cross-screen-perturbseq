# BOND: honest re-evaluation, ablations, novelty and plan (after the medium-scale and real-data runs)

Numbers come from `docs/06_medium_synthetic_results.md` (10 seeds) and `docs/07_replogle_heldout_results.md`.

## What held up and what didn't

| Component | Medium synthetic (snapshot, noisy, 70 measured genes) | Real data (Replogle K562) | Verdict |
|---|---|---|---|
| Binding prior | AUPRC 15.5× → 18.6×, precision 0.38 → 0.47, held-out effect r 0.33 → 0.42. But a shuffled prior already gives 17.7× (the gain mostly comes from knowing that only TFs regulate); a clean prior gives 20.3× | No change in held-out prediction (r 0.011 vs 0.012). Without the prior, top TF edges are no more often ChIP-bound than chance (0.41 vs 0.39) | Small and real in simulation, but mostly the "only TFs regulate" effect. Not useful on real data in this form |
| Knockdown-aware fitting | AUPRC 17.6× → 18.6×, precision 0.45 → 0.47 | r 0.003 → 0.012 (both ≈ 0) | Marginal |
| Velocity / dynamics (vs a static model) | Static wins on AUROC (0.94 vs 0.87), indirect FP (0.9% vs 9.3%) and sign accuracy (0.89 vs 0.71); velocity wins only on AUPRC (18.6× vs 16.4×). Degrades as velocity noise grows | Not testable (pseudobulk only) | Doesn't help under snapshot data with realistic velocity noise |
| Active knockdown selection | Area under the coverage curve 17.0 vs 16.1 (random) and 15.6 (degree); better in 10/10 seeds against each. Degree-based is worse than random | Not tested | Consistent but modest. The most positive new result |
| Overall model vs simple baselines | KD mean difference (total effects) has the best ranking (AUPRC 26.4×, precision 0.64) but 36% of its top edges are indirect. BOND-V: 18.6×, 9% indirect | Mean training profile (r = 0.42) beats every model (r ≈ 0.01) | BOND doesn't beat the simplest baselines on their own metric |

## Novelty after the literature check

Already published (verify the latest versions):
- PHOENIX (Genome Biology 2024): a sparse neural ODE GRN with a motif-derived prior.
- PerturbODE (arXiv 2501.02409): interpretable neural ODEs trained on perturbations.
- scKINETICS: velocity combined with ATAC/motif priors.
- Dictys: multiome footprints combined with dynamic GRNs.
- DCDI/DCDFG: interventional structure learning with known targets.
- inspre (Nature Communications 2025): large-scale interventional structure learning on Replogle K562.
- Active intervention design: Zhang et al., Nature Machine Intelligence 2023; Bayesian active GRN learning, BMC Bioinformatics 2025.

What's left as possibly new is narrow:
1. Thermodynamic occupancy priors instead of motif hits. Small gains in simulation, none detected on real data.
2. The empirical finding that interventions pull indirect edges onto the perturbed node, with structure resolved only where perturbations land (toy result).
3. A disambiguation-driven selection rule that beats random and degree-based choices on structure recovery. Existing active-design work mainly targets reaching a desired cell state or general posterior uncertainty, so this framing is new-ish, not clearly new.

None of these is method-paper novelty on its own.

## Plan

- **Not worth pursuing as a method paper in its current form.** The combined model is incremental over PHOENIX + PerturbODE + DCDI. It fails the real-data test against a trivial baseline, and the prior's benefit is mostly the "only TFs regulate" effect.
- **Worth pursuing: "design principles for interventional GRN inference" as a benchmark and analysis paper.** This fits your BEELINE-successor work:
  - perturbation coverage curves;
  - a measure of indirect-edge attraction toward perturbed nodes;
  - active vs random vs degree selection;
  - total-effect baselines that win on ranking yet are 36% indirect;
  - the mean-profile baseline that beats models on held-out prediction.
- **Investigate further, in order:**
  1. Fix the real-data predictor. Fit nonlinear models and models that include the generic response, and test whether anything beats the mean training profile on held-out knockdowns. If nothing does, that is itself the headline.
  2. Evaluate on edges with direct-binding-plus-response evidence: your benchmark's ChIP + RNA + ATAC-matched edges, with the 22% bound-responder overlap built in.
  3. Push active selection harder: larger networks, batch selection, and comparison against published acquisition functions.
  4. Use genuine dynamics data (metabolic labelling) only if a dataset with perturbations exists. Otherwise drop velocity.
- **Leave behind:**
  - per-cell-line activity (μ) estimates from two-state bulk data;
  - velocity-based fitting on snapshot data;
  - tuning the sequence prior further;
  - hidden-unit "mediator recovery" outside toy settings.
