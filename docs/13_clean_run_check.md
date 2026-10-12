# Clean-environment re-run of the main-text pipeline

Procedure: fresh `python -m venv` from `requirements.txt` (pip install succeeded), fresh `git clone` of the repository (commit 0d720d8), raw inputs linked under `data/raw/`, cached per-TF promoter features copied to `data/processed/` (their generation, Supplement S1, was not re-run). Commands, in order: `response_decomposition.py decompose`, `revision.py decomp`, `revision.py noise_pcs`, `revision.py binding`, `context_dependence.py prep`, `revision.py context`, `revision.py f2`, `revision.py f1` (all exit 0; 760 s on 2 cores).

Comparison with committed results (numeric leaves, excluding run times):

```
results/decomposition/decomp_summary.json: 115 numeric values compared, max |diff| = 0, non-numeric mismatches = 0, keys only committed = 0, only rerun = 0
results/revision/decomp_sensitivity.json: 160 numeric values compared, max |diff| = 0, non-numeric mismatches = 0, keys only committed = 0, only rerun = 0
results/revision/noise_pcs.json: 22 numeric values compared, max |diff| = 0, non-numeric mismatches = 0, keys only committed = 5, only rerun = 2
results/revision/binding.json: 87 numeric values compared, max |diff| = 0, non-numeric mismatches = 0, keys only committed = 0, only rerun = 0
results/revision/context.json: 70 numeric values compared, max |diff| = 19.4, non-numeric mismatches = 0, keys only committed = 0, only rerun = 0
   worst: ("/Wessels 2023 (A alone vs A in C's context)/counts/lost (strong in ctx1, ns in ctx2)/null_sd", 70.51765736324485, 51.12677869766489)
results/revision/f2_revised.json: 251 numeric values compared, max |diff| = 0, non-numeric mismatches = 0, keys only committed = 0, only rerun = 0
results/revision/f1_revised.json: 434 numeric values compared, max |diff| = 0, non-numeric mismatches = 0, keys only committed = 0, only rerun = 0
```

Notes:
- context.json: observed counts identical; null replicates differed because comparison order came from Python set iteration (hash-seed dependent). Fixed (sorted order), re-run, and verified identical output under different PYTHONHASHSEED values. Wessels null changed from 564.7 to 562.4 significant changes (ratio 3.0 unchanged).
- noise_pcs.json: the committed file had extra gene-specific estimators added after the code run; these are now computed by `revision.py noise_pcs` itself and the file was regenerated (profile-level values unchanged).
- Not re-verified: Supplement S1–S9 simulations and the per-TF promoter-feature generation (`test2_k562.py features`).
