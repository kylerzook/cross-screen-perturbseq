# Interpretation — are regulatory edges fixed or context-dependent?

Companion to `10_context_dependence_results.md`.

## Short answer
Edges in this data behave like **state-dependent weights, not rewired circuits** (hypothesis H1, not H2).
Context changes *how strongly* a perturbation acts, and *which genes cross a detection threshold*, but almost
never *which direction* it acts in. The non-additivity that exists is mostly amplification/saturation along
one shared differentiation axis, not gene-specific rewiring by "modulator" genes.

## Evidence, one line each
- **Non-additivity is real.** Noise is well calibrated (control z SD 0.99), and interaction energy is ~1.9× noise
  in the median pair; 93% of pairs have at least one significant interaction gene.
- **But it is mostly scaling.** Letting each pair rescale A and C (a·A + c·C) lifts R² from 0.58 to 0.69, and the
  fitted a + c is > 1.2 in 68% of pairs — pairs act like *amplified* sums (synergy), not new programs.
- **A shared saturation curve beats additive on held-out pairs** (85% of pairs, +0.015 corr), and closes part of
  the gap to the replicate ceiling (additive 0.78 → saturation 0.80; ceiling 0.82).
- **Modulator-specific interaction models do not help** on held-out pairs (−0.007 vs additive, win 34%), and
  per-gene modulator scores are *less* spread than a permutation null (p = 0.89): no special "rewiring" genes.
- **The interactions that exist share one main pattern** (top component 19% of variance vs 5% null), dominated by
  erythroid (HBZ, HBG2, ALAS2, HBA1) and myeloid (AIF1, LST1, TYROBP) genes — i.e. pairs push cells further or
  less far along the same differentiation axis, as Norman et al. described.
- **Target sets change, signs don't.** A's significant targets overlap only 0.19 (Jaccard) between alone and in C's
  context vs 0.43 for split-half reproducibility — but only 1.5% of shared targets flip sign. Membership changes
  because effect *sizes* shift around detection thresholds.
- **Cell state modulates gain slightly.** Across cell-cycle phases, 72% of perturbations show detectable
  heterogeneity, but it is small (1.11× null), top-target effects correlate 0.96 between phases, effect size
  varies ~1.26× between phases, and only 5 of 10,223 strong targets flip sign. ISR level: no detectable
  dependence (1.03× null, corr 0.985, 1 flip).

## What this means for the "dynamic edge" idea (Gemini discussion)
- **H1 (edges depend on state) is supported, but in its mildest form**: a smooth gain/saturation that a
  nonlinear dynamical model with one weight per edge already captures. No evidence here for edges carrying their
  own state or memory (H2).
- **Decision rule outcome:** Part 1 = outcome B (scaling/saturation), with a shared low-rank axis but no
  predictive modulator structure; Part 2 = strength changes, essentially no direction flips. → Do **not** launch a
  wet-lab edge-memory experiment on this basis. Fold "edges are state-dependent gains, ~1.1–1.5× modulation,
  almost never sign-flipping" into the design-principles project as a measured fact.
- **For GRN modeling:** one signed weight per edge plus a saturating response (and a cell-state/differentiation
  axis) is a defensible simplification for this system; sign-flipping, context-rewired edges are rare.

## What this does NOT rule out
- **CRISPRa overexpression, one late timepoint, RNA only, K562.** Fast, transient, or protein-level (PTM,
  complex-formation) context effects — the core of the Gemini idea — are invisible here.
- **Only 131 pairs**, biased toward hematopoietic TFs and signaling genes chosen by Norman et al.
- **Part 2 permutation resolution** (50 permutations) limits p-values; magnitudes, not p-values, carry the result.
- **Thresholded target-set overlaps** are sensitive to effect size; the sign-flip rate is the robust readout.

## Suggested next step
If you still want to test H2 (edge memory), the cheapest real-data version is a **same-state, different-history**
comparison: cells at the same expression state reached via different trajectories (e.g. time-course perturbation
data or lineage-traced data) — does the same perturbation act differently? Otherwise, move to the domino
(indirect-effect) analysis, which the last two experiments point to as the larger source of error.
