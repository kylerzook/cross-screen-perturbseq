#!/bin/bash
# Commands used to produce every result, table and figure, in order (see README for which steps were re-verified
# from a clean environment). Supplement S1-S4 steps use requirements-s1-s4.txt; everything else requirements.txt.
# Inputs: data/raw/ as listed in data/README.md.  Outputs: results/, paper/tables, paper/figures, paper/main.pdf
set -euo pipefail
cd "$(dirname "$0")"

## Supplement S1: occupancy model (T1-T3, ProBound)
for s in 0 1 2; do python experiments/test1_synthetic.py --seed $s; done
python experiments/test1_synthetic.py report
for s in 0 40 80 120 160 200; do python experiments/test2_k562.py features $s:$((s+40)); done   # per-TF promoter features
python experiments/test2_k562.py analyze
python experiments/test2_k562.py --model probound features 0:250 && python experiments/test2_k562.py --model probound analyze
python experiments/test2_k562.py analyze --common && python experiments/test2_k562.py --model probound analyze --common
python experiments/test2b_probound_vs_jaspar.py
python experiments/test3_cell_state_mu.py && python experiments/test3_cell_state_mu.py   # first call caches the setup, second fits
python experiments/test3_cell_state_mu.py --model probound && python experiments/test3_cell_state_mu.py --model probound

## Supplement S2-S4: latent ODE, toy, 80-gene simulator, linear SEM on Replogle
for c in full_lownoise full_highnoise hiddenB_noslot hiddenB_slot hiddenB_slot_fromstart hiddenB_randB0 hiddenB_KD50_unmodeled hiddenB_KD50_modeled hiddenB_KD25_modeled; do
  for s in 0 1 2; do python experiments/latent_cascade.py run $c $s; done; done
python experiments/latent_cascade.py report
for c in observed hiddenT1; do for s in 0 1 2; do
  python experiments/bond_dynamics.py run $c $s base,prior,perturb,prior+perturb,shufprior+perturb,prior+perturb+T2KD,perturb+T2KD
  python experiments/bond_dynamics.py baselines $c $s; done; done
python experiments/bond_dynamics.py report
for s in 0 1 2 3 4 5 6 7 8 9; do python experiments/medium_synthetic.py ablate $s; TIME_LIMIT=100000 python experiments/medium_synthetic.py coverage $s; done
python experiments/medium_synthetic.py report
python experiments/replogle_heldout.py

## Main text: Replogle decomposition, benchmark, cascade test, binding, context
python experiments/response_decomposition.py decompose
python experiments/response_decomposition.py heldout
python experiments/response_decomposition.py specificity
python experiments/response_decomposition.py biology
python experiments/revision.py decomp
python experiments/revision.py noise_pcs     # noise-constant sensitivity, matched axes
python experiments/revision.py f2
python experiments/revision.py f2cal         # training-only calibration, paired comparisons
python experiments/revision.py f1
python experiments/revision.py f1pc          # positive-control generator robustness
python experiments/revision.py binding        # needs data/processed/test2 and test2_acc_mean.npy (see experiments/test2_k562.py)
python experiments/context_dependence.py prep && python experiments/context_dependence.py part1 && python experiments/context_dependence.py part2
python experiments/revision.py context
python experiments/final_context.py shifrut && python experiments/final_context.py wessels
python experiments/final_replogle.py f1 && python experiments/final_replogle.py f2      # first-version F1/F2 (kept for comparison)

## Supplement S5-S8: memory detector, errors-in-variables, prior forms, masks, propagation depth
for p in main noise dt rank; do python experiments/final_synthetic.py f3 $p; done
python experiments/final_synthetic.py f4
for s in 0 1 2 3 4 5 6 7 8 9; do
  python experiments/final_synthetic.py f5 $s
  python experiments/final_synthetic.py f5masks $s
  python experiments/final_synthetic.py depthctrl $s
  python experiments/final_synthetic.py depthctrl2 $s     # both operators on the same targets
done

## Tables, figures, manuscript
python scripts/make_tables.py && python scripts/make_tables_rev.py
python scripts/make_figures.py && python scripts/make_figures_rev.py
cd paper && latexmk -pdf main.tex supplement.tex && latexmk -pdf main.tex   # second pass resolves cross-document references
