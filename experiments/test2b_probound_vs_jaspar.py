"""Test 2 add-on: paired comparison of ProBound free-energy models vs JASPAR PWM log-odds.

Same K562 promoters, same ChIP labels, same ATAC, same occupancy formula; only the energy
model changes. Restricted to TFs that have both models. Paired per-TF statistics.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import test2_k562 as T2
from bond.metrics import auroc, auprc_ratio
from bond.intervals import to_log_acc

ROOT = T2.ROOT
SCORES = ["PWM_max", "PWM_hits", "PWM_hits_in_ATAC", "OCC_mu-4", "OCC_mu-2", "OCC_mu+0",
          "OCC_ACC_mu-4", "OCC_ACC_mu-2", "OCC_ACC_mu+0"]


def per_tf(model, tfs):
    out = {s: [] for s in SCORES + ["OCC_SUB_mu-2|zero-hit", "OCC_SUB_ACC_mu-2|zero-hit"]}
    for tf in tfs:
        f = dict(np.load(T2.cache_dir(model) / f"{tf}.npz"))
        y = f["chip_bound"]
        for s in SCORES:
            out[s].append((auroc(y, f[s]), auprc_ratio(y, f[s])))
        z = f["PWM_hits"] == 0
        for s in ["OCC_SUB_mu-2", "OCC_SUB_ACC_mu-2"]:
            out[f"{s}|zero-hit"].append((auroc(y[z], f[s][z]), np.nan))
    return {k: np.array(v, float) for k, v in out.items()}


def main():
    j = {p.stem for p in T2.cache_dir("jaspar").glob("*.npz")}
    p = {p.stem for p in T2.cache_dir("probound").glob("*.npz")}
    tfs = sorted(j & p)
    J, P = per_tf("jaspar", tfs), per_tf("probound", tfs)
    R = {"n_tfs_common": len(tfs), "scores": {}}
    for s in J:
        for mi, mname in [(0, "AUROC"), (1, "AUPRC_ratio")]:
            a, b = J[s][:, mi], P[s][:, mi]
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() < 5:
                continue
            a, b = a[ok], b[ok]
            R["scores"][f"{s} [{mname}]"] = {
                "JASPAR_mean": float(a.mean()), "ProBound_mean": float(b.mean()),
                "JASPAR_median": float(np.median(a)), "ProBound_median": float(np.median(b)),
                "ProBound_wins_frac": float((b > a).mean()),
                "wilcoxon_p": float(wilcoxon(b, a).pvalue), "n": int(ok.sum())}
    # E_thr / energy-scale diagnostics
    R["Ethr_median_kT"] = {m: float(np.median([float(np.load(T2.cache_dir(m) / f"{t}.npz")["Ethr"]) for t in tfs]))
                           for m in ["jaspar", "probound"]}
    R["motif_width_median"] = {m: float(np.median([int(np.load(T2.cache_dir(m) / f"{t}.npz")["width"]) for t in tfs]))
                               for m in ["jaspar", "probound"]}
    from bond.probound import load_models
    PB = load_models()
    R["probound_model_source_counts"] = {
        "human": sum(PB[t]["tax_id"] == 9606 for t in tfs), "mouse": sum(PB[t]["tax_id"] == 10090 for t in tfs),
        "other": sum(PB[t]["tax_id"] not in (9606, 10090) for t in tfs)}
    json.dump(R, open(ROOT / "results/test2b_probound_vs_jaspar.json", "w"), indent=2)
    for k, v in R["scores"].items():
        print(f"{k:42s} J={v['JASPAR_mean']:.3f} PB={v['ProBound_mean']:.3f} win={v['ProBound_wins_frac']:.2f} p={v['wilcoxon_p']:.2g}")
    print({k: v for k, v in R.items() if k != "scores"})


if __name__ == "__main__":
    main()
