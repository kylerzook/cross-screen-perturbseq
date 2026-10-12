"""Final real-data tests on Replogle 2022 K562 (genome-wide screen = 'gw', essential screen = 'es').

F1  Domino share: how much of a knockdown's reproducible specific signal is explained by it acting
    THROUGH its own measured responders?  For knockdown j with responders k (top-K by |specific
    residual|, each itself knocked down in the gw screen):
        r_hat_j = sum_k (a_jk / d_k) * r_k
    a_jk = j's raw effect on k (z), d_k = k's own self-knockdown (z), r_k = k's specific-residual
    knockdown profile (which already contains everything downstream of k). Scored against the
    INDEPENDENT es-screen residual of j (noise independent of everything used to build r_hat), on
    genes other than j and its responders, normalised by the cross-screen reliability.
    Null: same weights, responders replaced by random knocked-down genes.
F2  Reproducible benchmark target: predictors built from gw training knockdowns only, scored against
    the es-screen profile of held-out knockdowns (raw, and specific residual). Shows how the choice of
    target changes which predictor 'wins'.

Usage: python experiments/final_replogle.py f1 | f2
"""
import json, sys, time
from pathlib import Path
import numpy as np, h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.response_decomposition import load, split, fit_components, RAW
OUT = ROOT / "results/final"; OUT.mkdir(parents=True, exist_ok=True)


def load_es(col, G):
    f = h5py.File(RAW / "K562_essential_normalized_bulk_01.h5ad", "r")
    cats = [x.decode() if isinstance(x, bytes) else x for x in f["var/__categories/gene_name"][:]]
    var_e = np.array([cats[c] for c in f["var/gene_name"][:]])
    tg = np.array([x.decode().split("_")[1] for x in f["obs/gene_transcript"][:]])
    fe = f["obs/fold_expr"][:]
    Xe = np.nan_to_num(f["X"][:].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    shared = np.array([col[g] for g in var_e if g in col]); src = np.array([k for k, g in enumerate(var_e) if g in col])
    best = {}
    for k, t in enumerate(tg):
        if fe[k] == fe[k] and fe[k] < 0.5 and (t not in best or fe[k] < fe[best[t]]):
            best[t] = k
    prof = {}
    for t, k in best.items():
        v = np.zeros(G, np.float32); v[shared] = Xe[k, src]
        if t in col:
            v[col[t]] = 0.0
        prof[t] = v
    M = np.zeros(G, bool); M[shared] = True
    return prof, M


def raw_self_z(rows, tg, col):
    f = h5py.File(RAW / "K562_gwps_normalized_bulk_01.h5ad", "r")
    X = f["X"]
    out = {}
    order = np.argsort(rows)
    for i in order:
        r, t = rows[i], tg[i]
        if t in col and t not in out:
            out[t] = float(X[r, col[t]])
    return out


def f1():
    t0 = time.time()
    d = load(); X, P, col, var = d["X"], d["P"], d["col"], d["var"]; G = X.shape[1]
    dz = np.load(ROOT / "results/decomposition/decomp.npz"); u, V = dz["u"], dz["V"]
    rows = np.nonzero(P)[0]; tg = d["tgt"][rows]
    first = {}
    for k, t in enumerate(tg):
        first.setdefault(t, k)
    E = X[rows]; g_, p_, R = split(E, u, V)
    es, M = load_es(col, G)
    selfz = raw_self_z(rows, tg, col)
    kd_genes = [t for t in first if t in col and t in selfz and selfz[t] < -1.0]  # usable mediators
    kd_set = set(kd_genes)
    kd_cols = np.array([col[t] for t in kd_genes]); kd_rowidx = np.array([first[t] for t in kd_genes])
    cs = np.load(ROOT / "results/decomposition/cross_screen.npz", allow_pickle=True)
    rel = dict(zip(cs["targets"], cs["res_corr"]))
    targets = [t for t in first if t in es and t in col and t in rel]
    rng = np.random.default_rng(0)
    res = {K: {"norm_expl": [], "corr": [], "null_corr": [], "reliab": [], "reproducible": []} for K in (5, 10, 20, 50)}
    for t in targets:
        j = first[t]
        rj = R[j]
        ej = es[t]; ej_r = ej - (ej @ u) * u; ej_r = ej_r - (ej_r @ V.T) @ V
        cand = np.array([c for c in kd_cols if c != col[t]])
        cand_t = [var[c] for c in cand]
        order = np.argsort(-np.abs(rj[cand]))
        for K in res:
            sel = order[:K]; ks = [cand_t[s] for s in sel]
            w = np.array([E[j, col[k]] / selfz[k] for k in ks])
            pred = (w[:, None] * R[[first[k] for k in ks]]).sum(0)
            score_mask = M.copy(); score_mask[col[t]] = False
            for k in ks:
                score_mask[col[k]] = False
            rel_j = np.corrcoef(rj[score_mask], ej_r[score_mask])[0, 1]
            c = np.corrcoef(pred[score_mask], ej_r[score_mask])[0, 1] if pred[score_mask].std() > 0 else 0.0
            # null: same weights, random mediators (not among responders)
            pool = [k for k in kd_genes if k != t and k not in ks]
            nk = list(rng.choice(pool, K, replace=False))
            predn = (w[:, None] * R[[first[k] for k in nk]]).sum(0)
            cn = np.corrcoef(predn[score_mask], ej_r[score_mask])[0, 1] if predn[score_mask].std() > 0 else 0.0
            res[K]["corr"].append(float(c)); res[K]["null_corr"].append(float(cn)); res[K]["reliab"].append(float(rel_j))
            res[K]["norm_expl"].append(float(max(c, 0) ** 2 / max(rel_j, 1e-3)) if rel_j > 0 else np.nan)
            res[K]["reproducible"].append(bool(rel.get(t, 0) > 0.3))
    out = {"n_targets": len(targets), "n_mediators_available": len(kd_genes), "by_K": {}}
    for K, r in res.items():
        rep = np.array(r["reproducible"])
        c, cn, ne, rl = map(np.array, (r["corr"], r["null_corr"], r["norm_expl"], r["reliab"]))
        out["by_K"][str(K)] = {
            "corr_mean_all": float(np.mean(c)), "null_corr_mean_all": float(np.mean(cn)),
            "corr_mean_reproducible": float(np.mean(c[rep])), "null_corr_mean_reproducible": float(np.mean(cn[rep])),
            "frac_beats_null_reproducible": float(np.mean(c[rep] > cn[rep])),
            "reliability_mean_reproducible": float(np.mean(rl[rep])),
            "domino_share_median_reproducible": float(np.nanmedian(ne[rep])),
            "domino_share_mean_reproducible": float(np.nanmean(np.minimum(ne[rep], 1))),
            "domino_share_q25_q75_reproducible": [float(np.nanpercentile(ne[rep], 25)), float(np.nanpercentile(ne[rep], 75))],
            "n_reproducible": int(rep.sum())}
    out["seconds"] = time.time() - t0
    np.savez(OUT / "f1_vectors.npz", **{f"K{K}_{k}": np.array(v) for K, r in res.items() for k, v in r.items()})
    json.dump(out, open(OUT / "f1_domino.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def f2():
    t0 = time.time()
    d = load(); X, P, col, var = d["X"], d["P"], d["col"], d["var"]; G = X.shape[1]
    rows = np.nonzero(P)[0]; tg = d["tgt"][rows]
    keep = np.array([t in col for t in tg]); rows, tg = rows[keep], tg[keep]
    es, M = load_es(col, G)
    genes_u = np.unique(tg); r = np.random.default_rng(1); r.shuffle(genes_u)
    fold_of = {g: k % 5 for k, g in enumerate(genes_u)}
    folds = np.array([fold_of[t] for t in tg])
    hv = np.argsort(-X[rows].var(0))[:1000]
    preds = ["zero", "mean training profile", "generic only (mean amplitude x generic direction)",
             "kNN raw (k=50)", "kNN raw (k=10)", "factorised (generic mean + pathway kNN50 + specific kNN10)"]
    targets = ["gw raw (same screen)", "es raw (independent screen)", "es specific residual (independent screen)"]
    S = {p: {tt: [] for tt in targets} for p in preds}
    for fo in range(5):
        tr = folds != fo; te = folds == fo
        Etr = X[rows[tr]]
        u, V, _, _, K = fit_components(Etr, seed=fo)
        gtr, ptr, rtr = split(Etr, u, V)
        Z = Etr - Etr.mean(0); Z /= (Z.std(0) + 1e-6)
        tcol_tr = np.array([col[t] for t in tg[tr]])
        mean_raw = Etr.mean(0); amp_mean = (Etr @ u).mean()
        te_idx = np.nonzero(te)[0]
        for a in te_idx:
            t = tg[a]
            if t not in es:
                continue
            sim = Z[:, col[t]] @ Z[:, tcol_tr] / Z.shape[0]
            o = np.argsort(-sim)

            def knn(Mat, k):
                nb = o[:k]; w = np.maximum(sim[nb], 0)
                return (w[:, None] * Mat[nb]).sum(0) / (w.sum() + 1e-9)
            pr = {"zero": np.zeros(G), "mean training profile": mean_raw,
                  "generic only (mean amplitude x generic direction)": amp_mean * u,
                  "kNN raw (k=50)": knn(Etr, 50), "kNN raw (k=10)": knn(Etr, 10),
                  "factorised (generic mean + pathway kNN50 + specific kNN10)": amp_mean * u + knn(ptr, 50) + knn(rtr, 10)}
            gw = X[rows[a]]; ej = es[t]
            ej_r = ej - (ej @ u) * u; ej_r = ej_r - (ej_r @ V.T) @ V
            msk = M.copy(); msk[col[t]] = False
            sel = np.intersect1d(hv, np.nonzero(msk)[0])
            for pname, p in pr.items():
                for tname, y in [("gw raw (same screen)", gw), ("es raw (independent screen)", ej), ("es specific residual (independent screen)", ej_r)]:
                    yy = y[sel]; pp = p[sel]
                    if tname.startswith("es specific"):
                        pp = pp - (p @ u) * u[sel]; pp = pp - ((p - (p @ u) * u) @ V.T) @ V[:, sel]
                    S[pname][tname].append(float(np.corrcoef(pp, yy)[0, 1]) if pp.std() > 1e-12 else 0.0)
        print(f"fold {fo} {time.time() - t0:.0f}s", flush=True)
    out = {"n_heldout_knockdowns_scored": len(S["zero"]["es raw (independent screen)"]),
           "scores_mean_corr": {p: {tt: float(np.mean(v)) for tt, v in d_.items()} for p, d_ in S.items()},
           "scores_sem": {p: {tt: float(np.std(v) / np.sqrt(len(v))) for tt, v in d_.items()} for p, d_ in S.items()},
           "rank_by_target": {tt: sorted(preds, key=lambda p: -np.mean(S[p][tt])) for tt in targets},
           "seconds": time.time() - t0}
    json.dump(out, open(OUT / "f2_benchmark.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"f1": f1, "f2": f2}[sys.argv[1]]()
