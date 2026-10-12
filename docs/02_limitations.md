# Key limitations and restrictions

Each item is tagged with what it affects: **[All]**, **[T1]**, **[T2a–d]**, **[T3]**.

## Model and physics
- **[All] Approximate energies.** E comes from JASPAR log-odds under the Berg–von Hippel approximation, assuming 1 log-odds unit = 1 kT. These are not ProBound or SELEX-fitted free-energy models, so the absolute energy scale is unknown.
- **[All] Independent sites.** Both strands and overlapping positions are summed as if independent. The model has no steric exclusion, cooperativity, competition between TFs, or DNA methylation.
- **[All] Accessibility form.** Accessibility enters as log(a). Peak signal is scaled by its 99th percentile, and positions outside peaks get an arbitrary floor of a = 0.02.
- **[T2] mu is not calibrated.** Fixed mu values are set relative to each motif's E_thr (−4, −2 or 0), not to real TF concentrations. mu = E_thr − 2 was chosen as the default in advance.

## ProBound add-on
- **[ProBound] Model choice.** One model per gene symbol: single-factor fits only, human first (otherwise mouse), then the lowest fit_id. This rule was fixed before looking at results. Every model is a single mononucleotide binding mode, with no dinucleotide terms, position bias or methylation effects. 120 of the 122 common TFs use human models.
- **[ProBound] Coverage.** ProBound covers 128 of the 526 K562 ChIP targets, against 197 for JASPAR. The comparison is only valid on the 122-TF overlap, and only 13 of those TFs pass the Perturb-seq function filters.
- **[ProBound] mu grid not equivalent.** mu is set relative to each model's own E_thr (median 2.3 kT for ProBound vs 4.7 for JASPAR), so "mu = E_thr − 2" means a different occupancy regime in each model. ProBound's calibrated absolute energy scale is therefore not fully exploited.
- **[T3 ProBound] Site pruning.** To fit the compute budget, sites whose accessibility-adjusted energy exceeds E_thr + 4 in both cell lines were dropped (16.8M → 7.0M sites). The JASPAR run did not drop these. The TF sets also differ (90 vs 131), so the two Test 3 runs aren't paired.

## Latent-dynamics prototype
- **[Latent] Toy setting.** There are 3 genes, one cascade, known Hill kinetics and smooth pulse dynamics. Cells are full time-series trajectories, whereas real scRNA-seq gives destructive snapshots with no per-cell trajectories.
- **[Latent] Model matches the simulator.** The model's sigmoid activations are close to the simulator's Hill functions. Edge calls depend on the 10% threshold and on λ, so results are reported over the whole λ grid rather than for one tuned value.
- **[Latent] Initialisation leaks information.** When a free latent unit is initialised from (x_t, x_t − x_{t−1}), it can read the hidden B directly from C's velocity, so it never has to be produced by A. The `fromstart` condition removes that shortcut.
- **[Latent] Weak Markov test.** In these toys, A is a deterministic decay, so most of the system is predictable from the current observed state. The model-free ΔR² is therefore insensitive, even in the randB0 positive control. The model-based residual test is the more sensitive readout.
- **[Latent Q3] The target is assumed known.** The `modeled` runs are told which latent unit the knockdown hits. Real CRISPRi screens do know the target gene, but mapping a target onto an unobserved latent unit is an assumption. The knockdown is also clean, uniform and constant over time. The learned efficiency was 0.11–0.13 with 50% knockdown cells and about 0.07 with 25%, against a true 0.10.
- **[Latent] Bug fixed during Q3.** An edge-strength bug zeroed every edge leaving a constant-initialised unit. All `fromstart` and knockdown results were rerun with the fix.
- **[Latent] Small numbers.** 3 seeds, 1,200–2,000 Adam iterations and no hyperparameter search beyond λ.

## BOND-D prototype
- **[BOND-D] The prior is partly circular.** The planted sites are derived from the true edges. Decoy and weak sites make it imperfect, but it is still far cleaner than real binding data, where binding rarely implies function.
- **[BOND-D] Every true regulator is a TF.** The prior only rewards TF columns, so even a shuffled prior carries "only TFs regulate" information. This explains why the shuffled prior still helps (AUROC 0.92 vs 0.86).
- **[BOND-D] Unidentifiable efficiencies.** T0 and T3 have no regulated production, so their knockdown efficiency can't be identified (it is learned as about 0.9). A knockdown of them acts through the observed initial level only.
- **[BOND-D] Fixed settings.** λ is fixed at 1e-3 and prior strength at 0.9, both untuned. There are 3 seeds and a single held-out knockdown (T2). The "+T2KD" variants see T2 knockdowns in training, so their T2 metrics are not held-out.

## Medium synthetic + real-data runs
- **[Medium] In-house simulator.** It is SERGIO-style, not SERGIO itself, and its network is acyclic (no feedback loops). The baselines are re-implementations of the core idea of PHOENIX, DCDI, CellOracle and GENIE3, not their original code.
- **[Medium] The prior is easier than real data.** In the "realistic" prior, 36% of a TF's responders are bound, against a target of about 22%. Every true regulator column is still a TF, or a non-TF regulator with no motif.
- **[Medium] Two bugs found and fixed before the final runs:**
  - the first simulator version was trivially easy (AUROC ≈ 1.0), so realism was added;
  - an adaptive-lasso bug cancelled the prior's effect.
- **[Medium] Coverage runs.** Curves run to 12 knockdowns, with a 3-model bootstrap ensemble for active selection. Seeds 0–1 (active) and all random/degree runs were done on your computer; seeds 2–9 (active) in the cloud, on identical data (seed 0 was verified to reproduce exactly).
- **[Real] Model limits.** The real-data model is a linear steady-state model on pseudobulk z-scores, using 203 genes, of which only 11 are TFs with ChIP. The prior can therefore touch few edges. There are no single cells and no dynamics.

## Regions and labels
- **[T2, T3] Promoters only.** Each gene gets one RefSeq 1-kb upstream window. Enhancers, the TSS core and downstream sequence, and alternative TSSs are all ignored.
- **[T2, T3] Binary ChIP labels.** A promoter counts as bound if any IDR peak overlaps the window. Each TF uses one ENCODE file (conservative IDR preferred, largest file kept), and antibody and experiment quality vary between TFs.
- **[T2, T3] Name matching.** ChIP targets are matched to JASPAR by uppercase gene symbol. JASPAR dimer motifs are excluded.

## Test-specific
- **[T1] Optimistic by design.** The simulator uses the same functional form as the model. Background sequence is i.i.d. random. The fit is initialised and regularised around the true mu centre (E_thr − 2). w and mu are identifiable only up to a joint sign flip, which was resolved using the noisy TF proxy.
- **[T2c] Hub prior advantage.** The hub prior uses ChIP-seq for other TFs in the same cell line, which captures highly-occupied (HOT) promoters. Most real held-out settings won't have this information.
- **[T2a, T2b] Accessibility barely filters.** 96% of these promoters (genes expressed in K562) overlap an ATAC peak. So the "ATAC-open" subset in 2b is nearly the full set and doesn't control for accessibility; the informative comparison is OCC_SUB_ACC vs ACC_only. For the same reason, motif-in-peak filtering removes very few promoter hits.
- **[T2d] Small function set.** Only 21 TFs passed every filter (motif, ChIP, fold_expr < 0.85, ≥ 20 responders). The |z| ≥ 0.5 threshold is arbitrary. Responses mix direct and indirect effects, and CRISPRi knockdown strength differs across TFs.
- **[T2d] Strong co-response baseline.** Coexpr_LOO uses about 11k other perturbations from the same screen. That much data isn't available in an ordinary scRNA-seq setting.
- **[T3] Two cell lines only.** Each TF's Δmu rests on a single contrast. The bulk RNA-seq for the two lines comes from different experiments, so batch effects are confounded with cell identity.
- **[T3] Regularisation dominates.** With L2 = 1e-3, Δmu barely moves (sd ≈ 0.08 kT). With L2 = 1e-6, the fit did not converge within the 3,000-iteration cap (12 warm-started chunks), 34% of mu values hit the ±2 bound, and train R² rises while test R² falls.
- **[T3] Truncation and limited validation.** Sites with E > E_thr + 4 are dropped. The proxies agree only weakly with each other (Spearman 0.27), which caps how well any Δmu can validate. Key marker TFs (HNF4A, HNF1A, GATA1, TAL1) are missing from the TFs that have ChIP in both lines plus a JASPAR motif.
- **[T3] One split.** Genes were split 80/20 once at random, with no repeated cross-validation.

## Restrictions (environment and data)
- **Compute.** Everything ran on a 4-core, 3 GB RAM shell with a 180-second limit per command. That is why per-TF features are cached, Test 3 fits are chunked, and the held-out logistic regression subsamples 3,000 genes per training TF.
- **Data.** All inputs are public:
  - JASPAR 2024 CORE vertebrates (non-redundant)
  - UCSC hg38 upstream1000 and refGene
  - GENCODE v44 attributes (gene ID to symbol mapping)
  - ENCODE GRCh38 TF ChIP-seq IDR peaks, ATAC pseudoreplicated peaks and total RNA-seq gene quantifications for K562 and HepG2 (accessions in `data/raw/encode_manifest.tsv`)
  - Replogle et al. 2022 K562 genome-wide normalized pseudobulk (figshare file 35773217)
- **Statistics.** There is no multiple-testing correction. Per-TF metrics are averaged with equal weight per TF.
