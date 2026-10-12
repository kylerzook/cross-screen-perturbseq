# cross-screen-perturbseq

Code, results and manuscript for **"What genome-scale Perturb-seq responses contain: an independent-screen analysis and
the limits of reading perturbation responses as regulatory networks"** (Kyler Zook, University of Missouri).

The project began as **BOND** (Boltzmann Occupancy Network Discovery), a thermodynamic-occupancy prior for gene
regulatory network inference; those experiments are reported in the manuscript's Supplement (S1–S9).

**📄 Read the paper: [main manuscript](paper/main.pdf) · [supplementary material](paper/supplement.pdf)** (LaTeX source in [`paper/`](paper/))

## What is here
| Path | Contents |
|---|---|
| `bond/` | small library: sequence/PWM scanning, occupancy, intervals, metrics, data loaders |
| `experiments/` | one script per experiment; `revision.py` holds the main-text Replogle/Norman/Wessels analyses |
| `results/` | JSON outputs of every experiment (`results/revision/` = main-text analyses, `results/final/` = simulations and context replication) |
| `paper/` | `main.tex` and `supplement.tex` (compiled: [`main.pdf`](paper/main.pdf), [`supplement.pdf`](paper/supplement.pdf)), generated `tables/` and `figures/` |
| `scripts/` | table and figure generators |
| `docs/` | short numeric notes per experiment |
| `data/README.md` | identifiers for every public input; `data/encode_manifest.tsv`, `data/download_list.txt` list ENCODE accessions |

## Environments
* `requirements.txt` — main environment (Python 3.13, CPU): main text and Supplement S5–S10.
* `requirements-s1-s4.txt` — environment used for most Supplement S1–S4 runs (Python 3.10.12, PyTorch 2.14.1+cpu).

## Reproduce
```bash
python -m venv env && env/bin/pip install -r requirements.txt
# place the public inputs listed in data/README.md under data/raw/
bash run_all.sh            # commands used, in order
```
**Verification status.** The main-text pipeline (decomposition, noise checks, binding from cached promoter features,
Norman preprocessing and context analysis, independent-screen benchmark, cascade test) was re-run from a fresh clone
in a clean environment built from `requirements.txt`, and its outputs were compared with the committed results. The remaining commands in `run_all.sh` (Supplement S1–S9 simulations, per-TF
promoter-feature generation) are supplied as used but were not re-verified end to end.

## Main-text results → code
| Section | Command | Output |
|---|---|---|
| 2.1 decomposition, sensitivity, noise and axis checks | `response_decomposition.py decompose`; `revision.py decomp`; `revision.py noise_pcs` | `results/decomposition/`, `results/revision/decomp_sensitivity.json`, `noise_pcs.json` |
| 2.2 independent-screen benchmark, calibration | `revision.py f2`; `revision.py f2cal` | `results/revision/f2_revised.json`, `f2_calibrated.json` |
| 2.3 cascade test, positive controls | `revision.py f1`; `revision.py f1pc` | `results/revision/f1_revised.json`, `f1_pc_round2.json` |
| 2.4 promoter binding | `test2_k562.py`; `revision.py binding` | `results/test2_k562.json`, `results/revision/binding.json` |
| 2.5 context | `context_dependence.py`; `revision.py context`; `final_context.py` | `results/context/`, `results/revision/context.json`, `results/final/f6_*.json` |

An archived release with a DOI will be added on acceptance.
