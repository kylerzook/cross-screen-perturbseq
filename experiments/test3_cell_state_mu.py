"""Test 3 - cell-state TF activity (mu) as a chemical potential: K562 vs HepG2.

Model (fit on 80% of genes, evaluated on held-out 20%):
    y[g,c] = b_c + sum_t w_t * occ[g,t,c],   occ = sum_sites sigmoid(mu[t,c] - E + log a_c)
    y = log1p(TPM) from ENCODE total RNA-seq; a_c = ATAC accessibility in state c.
Variants: mu 'state' (one mu per TF per cell line), 'shared' (one mu per TF), 'fixed' (mu = E_thr-2).
Baseline: linear regression on promoter accessibility only.
Validation of learned delta_mu = mu[HepG2] - mu[K562] against independent activity proxies:
    ChIP-seq peak-count log2 ratio, promoter-binding-fraction log2 ratio, TF mRNA log2 ratio.
"""
import gzip, json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bond import data as D
from bond.seq import background
from bond.pwm import logodds, scan_energies
from bond.intervals import read_bed, overlap_any, position_signal, to_log_acc, count_peaks
from bond.fit import OccupancyModel, sparse_sites

ROOT = D.ROOT
CELLS = ["K562", "HepG2"]
FLOOR = 0.02
CUT = 4.0       # keep sites with E < E_thr + CUT
MU_BOUND = 2.0  # mu in [E_thr-2-2, E_thr-2+2]
MODEL = "jaspar"
MARKERS = ["HNF4A", "HNF1A", "FOXA1", "FOXA2", "CEBPB", "NR2F2", "GATA1", "TAL1", "GATA2", "NFE2"]


def ensg2sym():
    m = {}
    with gzip.open(D.RAW / "gencodeAttrsV44.txt.gz", "rt") as fh:
        for line in fh:
            f = line.split("\t")
            m.setdefault(f[0].split(".")[0], f[1])
    return m


def rna(cell, e2s):
    tpm = {}
    p = D.other_file(cell, "rna")
    with open(p) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ti = hdr.index("TPM")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            s = e2s.get(f[0].split(".")[0])
            if s:
                tpm[s] = tpm.get(s, 0.0) + float(f[ti])
    return tpm


def setup():
    t0 = time.time()
    e2s = ensg2sym()
    tpm = {c: rna(c, e2s) for c in CELLS}
    var = set(D.perturb_var_genes())
    keep = var & set(tpm["K562"]) & set(tpm["HepG2"])
    P = D.promoters(keep_genes=keep)
    genes = P["symbols"]; G = len(genes)
    bg = background(P["seqs"])
    y = np.stack([np.log1p([tpm[c][g] for g in genes]) for c in CELLS], 1)
    possig = {c: position_signal(read_bed(D.other_file(c, "atac"), signal_col=6),
                                 P["chrom"], P["start"], P["end"], P["strand"], 1000) for c in CELLS}
    src = D.motif_source(MODEL)
    chipK, chipH = D.chip_files_all("K562"), D.chip_files_all("HepG2")
    tfs = sorted(t for t in set(chipK) & set(chipH) if t in src)
    T = len(tfs)
    eg, et, Ls, Ethr = [], [], [], np.zeros(T)
    for t, tf in enumerate(tfs):
        W = src[tf](bg); w = W.shape[0]
        Ethr[t] = D.energy_threshold(W, bg)
        E, _, _ = scan_energies(P["seqs"], W)
        g, flat, e = sparse_sites(E, Ethr[t] + CUT)
        pos = flat % E.shape[2]
        la = np.stack([to_log_acc(possig[c], w, floor=FLOOR)[g, pos] for c in CELLS], 1)
        eg.append(g); et.append(np.full(g.size, t)); Ls.append(-e[:, None] + la)
    eg, et, L = np.concatenate(eg), np.concatenate(et), np.vstack(Ls).astype(np.float32)
    return dict(genes=genes, y=y, eg=eg, et=et, L=L, Ethr=Ethr, tfs=np.array(tfs),
                acc=np.stack([np.exp(to_log_acc(possig[c], 1, floor=FLOOR)).mean(1) for c in CELLS], 1),
                chrom=P["chrom"], start=P["start"], end=P["end"]), tpm


def main():
    t0 = time.time()
    sfx = "" if MODEL == "jaspar" else f"_{MODEL}"
    cache = ROOT / f"data/processed/test3_setup{sfx}.npz"
    e2s = ensg2sym()
    tpm = {c: rna(c, e2s) for c in CELLS}
    if not cache.exists():
        S, _ = setup()
        np.savez(cache, **S)
        print(f"setup cached {time.time() - t0:.0f}s"); return
    S = dict(np.load(cache, allow_pickle=True))
    genes, y, eg, et, L, Ethr = S["genes"], S["y"], S["eg"], S["et"], S["L"], S["Ethr"]
    tfs = list(S["tfs"]); G, T = len(genes), len(tfs)
    if MODEL != "jaspar":
        # compute budget: drop sites whose accessibility-adjusted energy exceeds E_thr + CUT in
        # both cell lines (each contributes < sigmoid(-2) even at the mu upper bound)
        keep = (-L.max(1)) < (Ethr[et] + CUT)
        eg, et, L = eg[keep], et[keep], L[keep]
    P = dict(chrom=S["chrom"], start=S["start"], end=S["end"])
    chipK, chipH = D.chip_files_all("K562"), D.chip_files_all("HepG2")
    print(f"setup {time.time() - t0:.0f}s genes={G} tfs={T} sites={eg.size}", flush=True)

    rng = np.random.default_rng(0)
    test = np.zeros(G, bool); test[rng.choice(G, G // 5, replace=False)] = True
    mask = np.repeat((~test)[:, None], 2, axis=1).astype(float)

    def r2(pred, obs):
        return float(1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())

    R = {"n_genes": int(G), "n_test_genes": int(test.sum()), "n_tfs": T, "n_sites": int(eg.size)}
    # accessibility-only baseline
    acc = S["acc"]
    Xb = np.c_[np.ones(G * 2), acc.ravel(), np.repeat([[0], [1]], G, axis=1).T.ravel()]
    tr = np.repeat(~test, 2)
    beta = np.linalg.lstsq(Xb[tr], y.ravel()[tr], rcond=None)[0]
    pb = (Xb @ beta).reshape(G, 2)
    R["ACC_only"] = {"R2_test_y": r2(pb[test], y[test]),
                     "R2_test_delta_y": r2((pb[:, 1] - pb[:, 0])[test], (y[:, 1] - y[:, 0])[test])}
    fits = {}
    variants = [("fixed", "fixed", 1e-3), ("shared", "shared", 1e-3), ("state", "state", 1e-3)]
    if MODEL == "jaspar":
        variants.append(("state_lam1e-6", "state", 1e-6))
    for name, mode, lam_mu in variants:
        t1 = time.time()
        m = OccupancyModel(eg, et, L, G, T, mu_mode=mode, intercept="state", lam_w=1e-4, lam_mu=lam_mu,
                           mu_bound=MU_BOUND, mu0=Ethr - 2, mu_center=Ethr - 2)
        pf = ROOT / f"data/processed/test3_params{sfx}_{name}.npy"
        meta_f = pf.with_suffix(".json")
        meta = json.load(open(meta_f)) if meta_f.exists() else None
        if meta is not None and meta.get("done", True):
            m.p = np.load(pf)
            m.res = type("r", (), {"success": meta["converged"]})(); secs = meta["seconds"]
        else:
            # chunked L-BFGS (warm-started) so each call stays inside the shell time limit
            init = None
            if meta is not None:
                init = np.load(pf)
            elif name == "state_lam1e-6" and (ROOT / "data/processed/test3_params_state.npy").exists():
                init = np.load(ROOT / "data/processed/test3_params_state.npy")
            m.fit(y, mask=mask, seed=0, maxiter=250, p_init=init)
            np.save(pf, m.p)
            prev = meta["seconds"] if meta else 0.0
            secs = prev + time.time() - t1
            n_chunks = (meta["chunks"] + 1) if meta else 1
            done = bool(m.res.success) or m.res.nit < 250 or n_chunks >= 12  # cap: 3000 L-BFGS iterations
            json.dump({"converged": bool(m.res.success), "seconds": secs, "done": done,
                       "chunks": (meta["chunks"] + 1) if meta else 1}, open(meta_f, "w"))
            if not done:
                print(f"{name}: chunk saved, rerun to continue"); return
        pred = m.predict(m.p)
        fits[name] = m
        R[f"OCC_mu_{name}"] = {"R2_test_y": r2(pred[test], y[test]),
                               "R2_train_y": r2(pred[~test], y[~test]),
                               "R2_test_delta_y": r2((pred[:, 1] - pred[:, 0])[test], (y[:, 1] - y[:, 0])[test]),
                               "converged": bool(m.res.success), "fit_seconds": secs}
        print(name, R[f"OCC_mu_{name}"], flush=True)
    nK = np.array([count_peaks(read_bed(chipK[t])) for t in tfs], float)
    nH = np.array([count_peaks(read_bed(chipH[t])) for t in tfs], float)
    fK = np.array([overlap_any(read_bed(chipK[t]), P["chrom"], P["start"], P["end"]).mean() for t in tfs])
    fH = np.array([overlap_any(read_bed(chipH[t]), P["chrom"], P["start"], P["end"]).mean() for t in tfs])
    mrna = np.array([np.log2((tpm["HepG2"].get(t, 0) + 1) / (tpm["K562"].get(t, 0) + 1)) for t in tfs])
    prox = {"ChIP_peak_count_log2ratio": np.log2((nH + 1) / (nK + 1)),
            "ChIP_promoter_bound_frac_log2ratio": np.log2((fH + 1e-3) / (fK + 1e-3)),
            "TF_mRNA_log2ratio": mrna}
    R["reference: ChIP_peak_ratio vs TF_mRNA_ratio (spearman)"] = float(spearmanr(prox["ChIP_peak_count_log2ratio"], mrna)[0])
    for fname in [v[0] for v in variants if v[1] == "state"]:
        _, w, mu = fits[fname].unpack(fits[fname].p)
        dmu = mu[:, 1] - mu[:, 0]
        strong = np.abs(w) >= np.median(np.abs(w))
        V = {}
        for k, v in prox.items():
            rho = spearmanr(dmu, v)[0]
            perm = np.array([spearmanr(rng.permutation(dmu), v)[0] for _ in range(1000)])
            V[k] = {"spearman_all_TFs": float(rho), "perm_p_two_sided": float((np.abs(perm) >= abs(rho)).mean()),
                    "spearman_top_half_|w|": float(spearmanr(dmu[strong], v[strong])[0])}
        R[f"delta_mu_validation[{fname}]"] = V
        R[f"delta_mu_summary[{fname}]"] = {"mean": float(dmu.mean()), "sd": float(dmu.std()),
                                 "frac_mu_at_bound": float(np.mean(np.abs(mu - (Ethr - 2)[:, None]) > MU_BOUND - 1e-3)),
                                 "frac_|w|<1e-3": float(np.mean(np.abs(w) < 1e-3))}
        R[f"markers_delta_mu(HepG2-K562)[{fname}]"] = {t: {"delta_mu": float(dmu[tfs.index(t)]), "w": float(w[tfs.index(t)]),
                                                 "mRNA_log2ratio": float(mrna[tfs.index(t)])}
                                             for t in MARKERS if t in tfs}
        R[f"per_tf[{fname}]"] = {t: {"delta_mu": float(dmu[i]), "w": float(w[i]),
                                     "chip_log2ratio": float(prox["ChIP_peak_count_log2ratio"][i]),
                                     "mRNA_log2ratio": float(mrna[i])} for i, t in enumerate(tfs)}
    R["motif_model"] = MODEL
    json.dump(R, open(ROOT / f"results/test3_cell_state_mu{sfx}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in R.items() if not k.startswith("per_tf")}, indent=1))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if "--model" in sys.argv:
        MODEL = sys.argv[sys.argv.index("--model") + 1]
    main()
