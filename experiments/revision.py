"""Analyses added in response to review (revision 1).

  binding   per-TF uncertainty, hub-prior characterisation, binding vs occupancy failure (promoter scope)
  context   gained / lost / shared / reversed responses with a null matched to each comparison's noise
            (Norman 2019, Wessels 2023)
  f2        independent-screen benchmark, revised: per-fold gene selection, magnitude metric, mean-centring
            comparison, k sensitivity, reliability ceiling sqrt(r12) on the same panel
  f1        domino test: raw vs residual targets, bootstrap CIs, and a positive control (semi-synthetic data
            matched to the real mediator coverage, knockdown strengths and noise, with known cascade share)
  decomp    decomposition sensitivity (K, gene panel, cell count, knockdown strength), consistently pooled
            shares, noise-model check on perturbed profiles across screens, cross-screen stability of PCs

Usage: python experiments/revision.py <part>
"""
import json, sys, time
from pathlib import Path
import numpy as np, h5py
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/revision"; OUT.mkdir(parents=True, exist_ok=True)
RAW = ROOT / "data/raw"
rng0 = np.random.default_rng(0)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    return float(roc_auc_score(y, s))


def boot_ci(v, n=2000, seed=0):
    v = np.asarray([x for x in v if x == x]); r = np.random.default_rng(seed)
    m = [r.choice(v, len(v)).mean() for _ in range(n)]
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(len(v))]


# =============================================================================== binding
def binding():
    t0 = time.time()
    C = ROOT / "data/processed/test2"
    genes = np.load(C / "_genes.npy", allow_pickle=True)
    acc = np.load(ROOT / "data/processed/test2_acc_mean.npy")
    F = {p.stem: dict(np.load(p)) for p in sorted(C.glob("*.npz"))}
    G = len(genes)
    tfs = [t for t in sorted(F) if 20 <= F[t]["chip_bound"].sum() <= G - 20]
    f = h5py.File(RAW / "K562_gwps_normalized_bulk_01.h5ad", "r")
    cats = [x.decode() if isinstance(x, bytes) else x for x in f["var/__categories/gene_name"][:]]
    var = np.array([cats[c] for c in f["var/gene_name"][:]]); col = {g: i for i, g in enumerate(var)}
    vmean = f["var/mean"][:]
    expr = np.array([vmean[col[g]] if g in col else np.nan for g in genes]); lexpr = np.log10(expr + 1e-3)
    Y = np.stack([F[t]["chip_bound"] for t in tfs], 1).astype(bool)
    nb = Y.sum(1)
    res = {"scope": "1-kb upstream promoters of Perturb-seq-measured genes, K562", "n_genes": int(G), "n_tfs": len(tfs)}
    S = {"PWM_max": lambda t: F[t]["PWM_max"], "occupancy (mu=Ethr-2)": lambda t: F[t]["OCC_mu-2"],
         "occupancy x ATAC": lambda t: F[t]["OCC_ACC_mu-2"], "ATAC only": lambda t: acc}
    A = {k: [auroc(F[t]["chip_bound"], s(t)) for t in tfs] for k, s in S.items()}
    hub = {t: (nb - Y[:, j]) / (len(tfs) - 1) for j, t in enumerate(tfs)}  # leave-own-TF-out hub
    A["hub (leave-TF-out binding frequency)"] = [auroc(F[t]["chip_bound"], hub[t]) for t in tfs]
    res["binding_auroc_mean_ci"] = {k: boot_ci(v) for k, v in A.items()}
    d = np.array(A["occupancy x ATAC"]) - np.array(A["PWM_max"])
    res["paired_occATAC_minus_PWMmax"] = {"mean_ci": boot_ci(d), "wilcoxon_p": float(wilcoxon(d).pvalue), "frac_TFs_positive": float((d > 0).mean())}
    d2 = np.array(A["occupancy x ATAC"]) - np.array(A["ATAC only"])
    res["paired_occATAC_minus_ATAC"] = {"mean_ci": boot_ci(d2), "wilcoxon_p": float(wilcoxon(d2).pvalue), "frac_TFs_positive": float((d2 > 0).mean())}
    # hub characterisation
    hall = nb / len(tfs); ok = np.isfinite(lexpr)
    res["hub_characterisation"] = {
        "spearman(hub, ATAC)": float(spearmanr(hall, acc).statistic),
        "spearman(hub, log expression)": float(spearmanr(hall[ok], lexpr[ok]).statistic),
        "spearman(ATAC, log expression)": float(spearmanr(acc[ok], lexpr[ok]).statistic),
        "frac_promoters_bound_by_>=50%_of_TFs": float((hall >= 0.5).mean()),
        "frac_promoters_bound_by_no_TF": float((hall == 0).mean())}
    strat = {}
    for nm, v in [("ATAC quintile", acc), ("expression quintile", np.where(ok, lexpr, np.nanmedian(lexpr)))]:
        q = np.digitize(v, np.quantile(v, [0.2, 0.4, 0.6, 0.8]))
        for sc in ["hub (leave-TF-out binding frequency)", "occupancy x ATAC", "PWM_max"]:
            per = []
            for t in tfs:
                y = F[t]["chip_bound"]
                s = hub[t] if sc.startswith("hub") else (F[t]["OCC_ACC_mu-2"] if sc.startswith("occ") else F[t]["PWM_max"])
                per.append(np.nanmean([auroc(y[q == k], s[q == k]) for k in range(5) if 5 <= y[q == k].sum() < (q == k).sum() - 5]))
            strat[f"{sc} | within {nm}"] = boot_ci(per)
    res["binding_auroc_within_strata"] = strat
    # function (same labels as Test 2: |z| >= 0.5 in the TF's strongest knockdown, fold_expr < 0.85)
    gt = [x.decode() for x in f["obs/gene_transcript"][:]]; fe = f["obs/fold_expr"][:]
    gcols = np.array([col[g] for g in genes])
    rows = {}
    for i, s in enumerate(gt):
        g = s.split("_")[1]
        if g in F and fe[i] == fe[i] and fe[i] < 0.85 and (g not in rows or fe[i] < fe[rows[g]]):
            rows[g] = i
    X = f["X"]
    FN = {k: [] for k in ["ChIP bound (measured binding)", "occupancy (mu=Ethr-2)", "occupancy x ATAC", "PWM_max",
                          "ChIP bound | within expression quintiles", "occupancy x ATAC | within expression quintiles"]}
    qe = np.digitize(np.where(ok, lexpr, np.nanmedian(lexpr)), np.quantile(np.where(ok, lexpr, np.nanmedian(lexpr)), [0.2, 0.4, 0.6, 0.8]))
    ftfs = []
    for tf, i in sorted(rows.items()):
        z = np.nan_to_num(X[i][gcols]); y = np.abs(z) >= 0.5; y[genes == tf] = False
        if y.sum() < 20:
            continue
        ftfs.append(tf); b = F[tf]["chip_bound"]
        FN["ChIP bound (measured binding)"].append(auroc(y, b.astype(float)))
        FN["occupancy (mu=Ethr-2)"].append(auroc(y, F[tf]["OCC_mu-2"]))
        FN["occupancy x ATAC"].append(auroc(y, F[tf]["OCC_ACC_mu-2"]))
        FN["PWM_max"].append(auroc(y, F[tf]["PWM_max"]))
        FN["ChIP bound | within expression quintiles"].append(np.nanmean([auroc(y[qe == k], b[qe == k].astype(float)) for k in range(5) if 3 <= y[qe == k].sum()]))
        FN["occupancy x ATAC | within expression quintiles"].append(np.nanmean([auroc(y[qe == k], F[tf]["OCC_ACC_mu-2"][qe == k]) for k in range(5) if 3 <= y[qe == k].sum()]))
    res["function_tfs"] = ftfs
    res["function_auroc_mean_ci"] = {k: boot_ci(v) for k, v in FN.items()}
    res["responders_vs_expression_spearman_median"] = None
    res["seconds"] = time.time() - t0
    json.dump(res, open(OUT / "binding.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


# =============================================================================== context
def categorize(d1, s1, d2, s2, zt=3.0, zn=1.96):
    z1, z2 = d1 / s1, d2 / s2
    zd = (d2 - d1) / np.sqrt(s1 ** 2 + s2 ** 2)
    a1, a2 = np.abs(z1) > zt, np.abs(z2) > zt
    same = np.sign(d1) == np.sign(d2)
    c = {"strong_in_ctx1": a1, "strong_in_ctx2": a2,
         "shared_same_sign": a1 & a2 & same, "reversed": a1 & a2 & ~same,
         "gained (ns in ctx1, strong in ctx2)": a2 & (np.abs(z1) < zn), "lost (strong in ctx1, ns in ctx2)": a1 & (np.abs(z2) < zn),
         "significant context difference |z_diff|>3": np.abs(zd) > zt}
    D = c["significant context difference |z_diff|>3"]
    c["  of which: reversal"] = D & c["reversed"]
    c["  of which: same sign, different magnitude"] = D & c["shared_same_sign"]
    c["  of which: gained"] = D & c["gained (ns in ctx1, strong in ctx2)"]
    c["  of which: lost"] = D & c["lost (strong in ctx1, ns in ctx2)"]
    return {k: int(v.sum()) for k, v in c.items()}


def null_counts(d1, s1, d2, s2, reps=20, seed=0):
    r = np.random.default_rng(seed)
    w1, w2 = 1 / s1 ** 2, 1 / s2 ** 2
    t = (d1 * w1 + d2 * w2) / (w1 + w2)  # common truth under 'no context effect' (precision-weighted)
    out = []
    for _ in range(reps):
        out.append(categorize(t + s1 * r.standard_normal(len(t)), s1, t + s2 * r.standard_normal(len(t)), s2))
    return out


def context_dataset(comps):
    """comps: list of (d1, s1, d2, s2) arrays over genes."""
    obs = {}; nul = []
    for k, (d1, s1, d2, s2) in enumerate(comps):
        for kk, v in categorize(d1, s1, d2, s2).items():
            obs[kk] = obs.get(kk, 0) + v
        nul.append(null_counts(d1, s1, d2, s2, seed=k))
    reps = len(nul[0])
    nsum = [{kk: sum(nul[c][r][kk] for c in range(len(comps))) for kk in obs} for r in range(reps)]
    return {kk: {"observed": obs[kk], "null_mean": float(np.mean([n[kk] for n in nsum])), "null_sd": float(np.std([n[kk] for n in nsum]))} for kk in obs}


def context():
    t0 = time.time()
    res = {}
    # ---- Norman 2019
    d = dict(np.load(ROOT / "results/context/prep.npz", allow_pickle=True))
    M, genes, pert, hv = d["M"], d["genes"], d["pert"], d["hvg_mask"]
    perts = [p for p in np.unique(pert) if p != "control"]
    singles = [p for p in perts if "_" not in p]
    pairs = [p for p in perts if "_" in p and all(g in singles for g in p.split("_"))]
    idx = {p: np.nonzero(pert == p)[0] for p in perts + ["control"]}
    gidx = {g: i for i, g in enumerate(genes)}
    st = {p: (M[idx[p]].mean(0), M[idx[p]].var(0) / len(idx[p])) for p in singles + pairs + ["control"]}
    comps = []
    for p in pairs:
        a, c = p.split("_")
        mask = hv.copy()
        for g in (a, c):
            if g in gidx:
                mask[gidx[g]] = False
        for A_, C_ in ((a, c), (c, a)):
            d1 = (st[A_][0] - st["control"][0])[mask]; s1 = np.sqrt(st[A_][1] + st["control"][1])[mask]
            d2 = (st[p][0] - st[C_][0])[mask]; s2 = np.sqrt(st[p][1] + st[C_][1])[mask]
            comps.append((d1, s1 + 1e-9, d2, s2 + 1e-9))
    res["Norman 2019 (A alone vs A in C's context)"] = {"n_comparisons": len(comps), "n_genes": int(hv.sum()), "counts": context_dataset(comps)}
    print("norman", time.time() - t0, flush=True)
    # ---- Wessels 2023
    from experiments.final_context import cat, load as sload
    f0 = h5py.File(RAW / "scperturb/WesselsSatija2023.h5ad", "r")
    pw = cat(f0, "perturbation")
    sing = sorted({p for p in pw if p and p != "control" and "_" not in p})
    _, X, _ = sload("WesselsSatija2023", exclude=sing)
    ctl = np.nonzero(pw == "control")[0]
    cnt = {p: int((pw == p).sum()) for p in set(pw) if p}
    ms = lambda ii: (X[ii].mean(0), X[ii].var(0, ddof=1) / len(ii))
    comps = []
    mc = ms(ctl)
    for p, n in cnt.items():
        if "_" not in p or n < 100:
            continue
        a, c = p.split("_")
        for A_, C_ in ((a, c), (c, a)):
            if cnt.get(A_, 0) < 60 or cnt.get(C_, 0) < 60:
                continue
            mA, mC, mAC = ms(np.nonzero(pw == A_)[0]), ms(np.nonzero(pw == C_)[0]), ms(np.nonzero(pw == p)[0])
            comps.append((mA[0] - mc[0], np.sqrt(mA[1] + mc[1]) + 1e-9, mAC[0] - mC[0], np.sqrt(mAC[1] + mC[1]) + 1e-9))
    res["Wessels 2023 (A alone vs A in C's context)"] = {"n_comparisons": len(comps), "n_genes": int(X.shape[1]), "counts": context_dataset(comps)}
    res["note"] = ("Null: for every gene and comparison both contexts share the precision-weighted mean effect; each context's "
                   "estimate is redrawn with its own standard error (20 replicates). Using the noisy mean as truth makes the null "
                   "slightly generous to 'gained'/'lost' counts, i.e. conservative for claiming excess context dependence.")
    res["seconds"] = time.time() - t0
    json.dump(res, open(OUT / "context.json", "w"), indent=1)
    for k, v in res.items():
        if isinstance(v, dict) and "counts" in v:
            print(k, v["n_comparisons"])
            for kk, c in v["counts"].items():
                print(f"   {kk:48s} obs {c['observed']:7d}   null {c['null_mean']:9.1f} ± {c['null_sd']:.1f}")



# =============================================================================== shared Replogle helpers
def rep_data():
    from experiments.response_decomposition import load
    from experiments.final_replogle import load_es
    d = load(); X, P, col, var = d["X"], d["P"], d["col"], d["var"]; G = X.shape[1]
    es, M = load_es(col, G)
    return d, X, P, col, var, G, es, M


def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan  # undefined (e.g. the zero predictor); reported as n/a
    return float(np.corrcoef(a, b)[0, 1])


def r2z(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y ** 2).sum() + 1e-12))  # R^2 relative to predicting zero


# =============================================================================== F2 revised
def f2():
    from experiments.response_decomposition import fit_components, split
    t0 = time.time()
    d, X, P, col, var, G, es, M = rep_data()
    rows = np.nonzero(P)[0]; tg = d["tgt"][rows]
    keep = np.array([t in col for t in tg]); rows, tg = rows[keep], tg[keep]
    genes_u = np.unique(tg); r = np.random.default_rng(1); r.shuffle(genes_u)
    fold_of = {g: k % 5 for k, g in enumerate(genes_u)}
    folds = np.array([fold_of[t] for t in tg])
    KS = (10, 20, 50, 100)
    preds = ["zero", "mean training profile"] + [f"kNN k={k}" for k in KS] + ["factorised (generic mean + pathway kNN50 + specific kNN50)"]
    targets = ["same screen, raw", "independent screen, raw", "independent screen, mean-centred", "independent screen, specific residual"]
    S = {p: {t: {"corr": [], "R2_vs_zero": []} for t in targets} for p in preds}
    ceil = {t: [] for t in targets}
    proto = []
    for fo in range(5):
        tr = folds != fo; te = folds == fo
        Etr = X[rows[tr]]
        hv = np.argsort(-Etr.var(0))[:1000]  # gene panel chosen on TRAINING knockdowns only
        u, V, _, _, K = fit_components(Etr, seed=fo)
        gtr, ptr, rtr = split(Etr, u, V)
        Z = Etr - Etr.mean(0); Z /= (Z.std(0) + 1e-6)
        tcol_tr = np.array([col[t] for t in tg[tr]])
        mean_raw = Etr.mean(0); amp_mean = (Etr @ u).mean()
        proto.append({"fold": fo, "n_train": int(tr.sum()), "n_test": int(te.sum()), "K_pathway": int(K)})
        spec = lambda v: (lambda w: w - (w @ V.T) @ V)(v - (v @ u) * u)
        for a in np.nonzero(te)[0]:
            t = tg[a]
            if t not in es:
                continue
            sim = Z[:, col[t]] @ Z[:, tcol_tr] / Z.shape[0]  # co-response of the held-out TARGET GENE in training KDs
            o = np.argsort(-sim)

            def knn(Mat, k):
                nb = o[:k]; w = np.maximum(sim[nb], 0)
                return (w[:, None] * Mat[nb]).sum(0) / (w.sum() + 1e-9)
            pr = {"zero": np.zeros(G), "mean training profile": mean_raw}
            for k in KS:
                pr[f"kNN k={k}"] = knn(Etr, k)
            pr["factorised (generic mean + pathway kNN50 + specific kNN50)"] = amp_mean * u + knn(ptr, 50) + knn(rtr, 50)
            gw = X[rows[a]]; ej = es[t]
            msk = M.copy(); msk[col[t]] = False
            sel = np.intersect1d(hv, np.nonzero(msk)[0])
            Y = {"same screen, raw": gw, "independent screen, raw": ej,
                 "independent screen, mean-centred": ej - mean_raw, "independent screen, specific residual": spec(ej)}
            T = {"same screen, raw": lambda p: p, "independent screen, raw": lambda p: p,
                 "independent screen, mean-centred": lambda p: p - mean_raw, "independent screen, specific residual": spec}
            # reliability reference on the same panel: corr between the two screens' versions of this target
            ref = {"same screen, raw": None, "independent screen, raw": corr(gw[sel], ej[sel]),
                   "independent screen, mean-centred": corr((gw - mean_raw)[sel], (ej - mean_raw)[sel]),
                   "independent screen, specific residual": corr(spec(gw)[sel], spec(ej)[sel])}
            for tn in targets:
                if ref[tn] is not None:
                    ceil[tn].append(ref[tn])
                yy = Y[tn][sel]
                for pn, p in pr.items():
                    pp = T[tn](p)[sel] if pn != "zero" else np.zeros(len(sel))
                    S[pn][tn]["corr"].append(corr(pp, yy)); S[pn][tn]["R2_vs_zero"].append(r2z(yy, pp))
        print(f"fold {fo} {time.time() - t0:.0f}s", flush=True)
    out = {"protocol": proto, "n_scored": len(S["zero"][targets[0]]["corr"]), "panel": "1000 most variable genes among TRAINING knockdowns, per fold; target gene masked",
           "scores": {p: {t: {m: boot_ci(v) if not all(x != x for x in v) else None for m, v in d_.items()} for t, d_ in dd.items()} for p, dd in S.items()}}
    rel = {}
    for tn, v in ceil.items():
        if not v:
            continue
        v = np.array(v); vp = np.sqrt(np.clip(v, 0, None))
        rel[tn] = {"cross_screen_r_mean": float(np.nanmean(v)), "cross_screen_r_median": float(np.nanmedian(v)),
                   "ceiling_sqrt_r_mean (equal-reliability assumption)": float(np.nanmean(vp))}
        for pn in preds[1:]:
            c = np.array(S[pn][tn]["corr"], float)
            rel[tn][f"{pn}: mean corr / mean ceiling"] = float(np.nanmean(c) / np.nanmean(vp))
    out["reliability"] = rel
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT / "f2_revised.json", "w"), indent=1)
    for pn in preds:
        print(pn[:40].ljust(42), " | ".join(f"{(out['scores'][pn][t]['corr'] or [np.nan])[0]:.3f}/{out['scores'][pn][t]['R2_vs_zero'][0]:.3f}" for t in targets))
    print(json.dumps(rel, indent=1))



# =============================================================================== F1 revised + positive control
def f1_setup():
    from experiments.response_decomposition import split
    from experiments.final_replogle import raw_self_z
    d, X, P, col, var, G, es, M = rep_data()
    dz = np.load(ROOT / "results/decomposition/decomp.npz"); u, V = dz["u"], dz["V"]
    rows = np.nonzero(P)[0]; tg = d["tgt"][rows]
    first = {}
    for k, t in enumerate(tg):
        first.setdefault(t, k)
    E = X[rows]; g_, p_, R = split(E, u, V)
    selfz = raw_self_z(rows, tg, col)
    med = [t for t in first if t in col and t in selfz and selfz[t] < -1.0]
    cs = np.load(ROOT / "results/decomposition/cross_screen.npz", allow_pickle=True)
    rel = dict(zip(cs["targets"], cs["res_corr"]))
    targets = [t for t in first if t in es and t in col and t in rel]
    return dict(E=E, R=R, Gr=E - g_, u=u, V=V, first=first, selfz=selfz, med=med, rel=rel, targets=targets, es=es, M=M, col=col, var=var, G=G)


def domino_eval(S, prof, tgt_vec, sel_vec, raw_for_w, targets, K, seed=0, cand_pool=None, rel_obs=None):
    """Generic F1 estimator. prof[t] mediator profile; tgt_vec[t] independent-screen target; sel_vec[t] screen-1
    profile used to select responders; raw_for_w[t] screen-1 raw profile for weights."""
    col, selfz, M = S["col"], S["selfz"], S["M"]
    r = np.random.default_rng(seed)
    out = {"corr": [], "null": [], "share": [], "rel": []}
    for t in targets:
        pool = cand_pool[t] if cand_pool is not None else [m for m in S["med"] if m != t]
        cc = np.array([col[m] for m in pool])
        order = np.argsort(-np.abs(sel_vec(t)[cc]))[:K]
        ks = [pool[i] for i in order]
        w = np.array([raw_for_w(t)[col[k]] / selfz[k] for k in ks])
        pred = (w[:, None] * np.stack([prof(k) for k in ks])).sum(0)
        msk = M.copy(); msk[col[t]] = False
        for k in ks:
            msk[col[k]] = False
        y = tgt_vec(t)[msk]
        c = corr(pred[msk], y)
        nk = list(r.choice([m for m in pool if m not in ks], K, replace=False))
        cn = corr((w[:, None] * np.stack([prof(k) for k in nk])).sum(0)[msk], y)
        rl = rel_obs(t, msk) if rel_obs else S["rel"][t]
        out["corr"].append(c); out["null"].append(cn); out["rel"].append(rl)
        out["share"].append(max(c, 0) ** 2 / rl if rl > 0 else np.nan)
    c, n = np.array(out["corr"]), np.array(out["null"])
    return {"corr_mean_ci": boot_ci(c), "null_mean_ci": boot_ci(n), "diff_mean_ci": boot_ci(c - n),
            "frac_beats_null": float(np.mean(c > n)), "share_mean": float(np.nanmean(np.minimum(out["share"], 1))),
            "share_median": float(np.nanmedian(out["share"])), "n": len(c)}


def f1():
    t0 = time.time()
    S = f1_setup(); E, R, Gr, u, V, first, es = S["E"], S["R"], S["Gr"], S["u"], S["V"], S["first"], S["es"]
    rep = [t for t in S["targets"] if S["rel"][t] > 0.3]
    res = {"n_targets_reproducible": len(rep), "n_mediators": len(S["med"]), "real": {}, "positive_control": {}}
    gen_rm = lambda v: v - (v @ u) * u
    spec = lambda v: (lambda w: w - (w @ V.T) @ V)(gen_rm(v))
    variants = {
        "specific residual (original)": (lambda k: R[first[k]], lambda t: spec(es[t]), lambda t: R[first[t]]),
        "generic removed only": (lambda k: Gr[first[k]], lambda t: gen_rm(es[t]), lambda t: Gr[first[t]]),
        "raw profiles": (lambda k: E[first[k]], lambda t: es[t], lambda t: E[first[t]])}
    for vn, (prof, tv, sv) in variants.items():
        res["real"][vn] = {str(K): domino_eval(S, prof, tv, sv, lambda t: E[first[t]], rep, K) for K in (5, 10, 20)}
        print(vn, json.dumps(res["real"][vn]["10"]), flush=True)
    # ---------------- positive control (semi-synthetic, residual pipeline)
    col, G, M = S["col"], S["G"], S["M"]
    med = S["med"]; medset = set(med)
    r12_med = np.median([S["rel"][t] for t in S["targets"]])
    rng = np.random.default_rng(7)
    sig = lambda v: np.sqrt((v[M] ** 2).mean())
    def noisy(v, rel_):
        R_ = max(rel_, 0.05); s_ = sig(v) * np.sqrt((1 - R_) / R_)
        n = rng.standard_normal(G) * s_; n[~M] = 0
        return v + n
    # mediator observed profiles (screen 1) = 'truth' (real residual shape) + noise at matched reliability
    truth_m = {k: R[first[k]].copy() for k in med}
    obs_m = {k: noisy(truth_m[k], S["rel"].get(k, r12_med)) for k in med}
    for cov in (1.0, 0.5, 0.25):
        for alpha in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
            T1, T2, pools = {}, {}, {}
            for t in rep:
                cand = [m for m in med if m != t]
                aa = np.array([E[first[t], col[m]] for m in cand])
                true_k = [cand[i] for i in np.argsort(-np.abs(aa))[:10]]  # true mediators: strongest real effects
                hidden = set(rng.choice(true_k, int(round((1 - cov) * len(true_k))), replace=False)) if cov < 1 else set()
                pools[t] = [m for m in cand if m not in hidden]
                c = sum((E[first[t], col[k]] / S["selfz"][k]) * truth_m[k] for k in true_k)
                o = R[first[t]].copy()
                for k in true_k:
                    c[col[k]] = 0; o[col[k]] = 0
                c[col[t]] = 0
                nc_, no_ = np.linalg.norm(c[M]) + 1e-12, np.linalg.norm(o[M]) + 1e-12
                tt = np.linalg.norm(o[M]) * (np.sqrt(alpha) * c / nc_ + np.sqrt(1 - alpha) * o / no_)
                for k in true_k:
                    tt[col[k]] = E[first[t], col[k]]  # the target's effect on its mediators
                T1[t] = noisy(tt, S["rel"][t]); T2[t] = noisy(tt, S["rel"][t])
            def rel_obs(t, msk):
                return corr(T1[t][msk], T2[t][msk])
            key = f"coverage={cov} alpha={alpha}"
            res["positive_control"][key] = domino_eval(S, lambda k: obs_m[k], lambda t: T2[t], lambda t: T1[t], lambda t: T1[t],
                                                       rep, 10, seed=1, cand_pool=pools, rel_obs=rel_obs)
            pc = res["positive_control"][key]
            print(key, "corr", round(pc["corr_mean_ci"][0], 3), "null", round(pc["null_mean_ci"][0], 3), "diff CI", [round(x, 3) for x in pc["diff_mean_ci"][1:3]],
                  "share", round(pc["share_mean"], 3), flush=True)
    res["note"] = ("Positive control: target truth = sqrt(alpha) x cascade through its 10 strongest measured mediators (weights a_jk/d_k "
                   "from the real data, mediator profiles = real residual shapes) + sqrt(1-alpha) x its own real residual shape; "
                   "Gaussian noise added independently for the two screens and for mediator profiles at each profile's real "
                   "cross-screen reliability; coverage < 1 hides a fraction of true mediators from the candidate pool. The generator is "
                   "additive like the estimator (favourable case: no path double counting or nonlinearity).")
    res["seconds"] = time.time() - t0
    json.dump(res, open(OUT / "f1_revised.json", "w"), indent=1)



# =============================================================================== decomposition sensitivity
def decomp():
    from sklearn.utils.extmath import randomized_svd
    t0 = time.time()
    d, X, P, col, var, G, es, M = rep_data()
    nc, nt, fe = d["nc"], d["nt"], d["fe"]
    E = X[P]; ncP = nc[P]; feP = fe[P]
    c = float(np.median((X[nt] ** 2).mean(1) * nc[nt]))
    m = E.mean(0); u = m / np.linalg.norm(m); Rg = E - np.outer(E @ u, u)
    _, S50, Vt = randomized_svd(Rg, n_components=50, random_state=0)
    def shares(rows, K, genes=None):
        gsel = np.arange(G) if genes is None else genes
        Gs = len(gsel)
        Er = E[rows][:, gsel]; uu = u[gsel] / (np.linalg.norm(u[gsel]) + 1e-12)
        g = np.outer(Er @ uu, uu); Rr = Er - g
        if K:
            Vk = Vt[:K][:, gsel]; Vk, _ = np.linalg.qr(Vk.T); Vk = Vk.T
            p = (Rr @ Vk.T) @ Vk
        else:
            p = np.zeros_like(Rr)
        r = Rr - p
        noise = Gs * c / ncP[rows]; noise_r = noise * (Gs - 1 - K) / Gs
        tot = (Er ** 2).sum(1)
        pooled = {"noise": noise.sum() / tot.sum(), "generic": (g ** 2).sum() / tot.sum(), "pathway": (p ** 2).sum() / tot.sum(),
                  "specific (residual minus expected noise)": ((r ** 2).sum() - noise_r.sum()) / tot.sum()}
        sig = tot.sum() - noise.sum()
        of_signal = {"generic": (g ** 2).sum() / sig, "pathway": (p ** 2).sum() / sig, "specific": ((r ** 2).sum() - noise_r.sum()) / sig}
        return {"pooled_share_of_total_energy": {k: float(v) for k, v in pooled.items()},
                "pooled_share_of_signal": {k: float(v) for k, v in of_signal.items()},
                "median_noise_share_per_knockdown": float(np.median(noise / tot)), "n": int(len(rows))}
    allr = np.arange(len(E))
    res = {"noise_c": c, "K_sensitivity": {str(K): shares(allr, K) for K in (0, 1, 3, 5, 10, 20, 50)}}
    hv2000 = np.argsort(-E.var(0))[:2000]
    expr_hi = np.nonzero(h5py.File(RAW / "K562_gwps_normalized_bulk_01.h5ad", "r")["var/mean"][:] > 1.0)[0]
    res["gene_panel_sensitivity (K=3)"] = {"all genes": shares(allr, 3), "2000 most variable": shares(allr, 3, hv2000),
                                           f"mean UMI > 1 ({len(expr_hi)} genes)": shares(allr, 3, expr_hi)}
    qn = np.quantile(ncP, [1 / 3, 2 / 3]); qf = np.quantile(feP, [1 / 3, 2 / 3])
    res["cell_count_tertiles (K=3)"] = {f"tertile {i + 1}": shares(np.nonzero(np.digitize(ncP, qn) == i)[0], 3) for i in range(3)}
    res["knockdown_strength_tertiles (K=3; tertile 1 = strongest)"] = {f"tertile {i + 1}": shares(np.nonzero(np.digitize(feP, qf) == i)[0], 3) for i in range(3)}
    # noise model on PERTURBED profiles: knockdowns in both screens, E||x_gw - x_es||^2 >= noise_gw + noise_es
    fe_ = h5py.File(RAW / "K562_essential_normalized_bulk_01.h5ad", "r")
    tg_e = np.array([x.decode().split("_")[1] for x in fe_["obs/gene_transcript"][:]])
    gt_e = [x.decode() for x in fe_["obs/gene_transcript"][:]]
    nc_e = fe_["obs/num_cells_filtered"][:]; nc_e = np.where(np.isfinite(nc_e) & (nc_e > 0), nc_e, np.nanmedian(nc_e))
    cats = [x.decode() if isinstance(x, bytes) else x for x in fe_["var/__categories/gene_name"][:]]
    var_e = np.array([cats[k] for k in fe_["var/gene_name"][:]])
    Xe = np.nan_to_num(fe_["X"][:].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    nt_e = np.array(["non-targeting" in s for s in gt_e])
    shared_e = np.array([k for k, g in enumerate(var_e) if g in col]); shared_g = np.array([col[var_e[k]] for k in shared_e])
    Gs = len(shared_g)
    c_e = float(np.median((Xe[nt_e][:, shared_e] ** 2).mean(1) * nc_e[nt_e]))
    c_g = float(np.median((X[nt][:, shared_g] ** 2).mean(1) * nc[nt]))
    rows = np.nonzero(P)[0]; tg = d["tgt"][rows]; fe_e = fe_["obs/fold_expr"][:]
    erow = {}
    for k, t in enumerate(tg_e):
        if fe_e[k] == fe_e[k] and fe_e[k] < 0.5 and (t not in erow or fe_e[k] < fe_e[erow[t]]):
            erow[t] = k
    ratios, ratio_ctrl = [], []
    seen = set()
    for k, t in enumerate(tg):
        if t in erow and t not in seen and t in col:
            seen.add(t)
            a = X[rows[k]][shared_g].copy(); b = Xe[erow[t]][shared_e].copy()
            j = np.nonzero(shared_g == col[t])[0]; a[j] = 0; b[j] = 0
            pred = Gs * c_g / nc[rows[k]] + Gs * c_e / nc_e[erow[t]]
            ratios.append(float(((a - b) ** 2).sum() / pred))
    # controls: non-targeting profiles paired at random across screens (no true difference expected beyond batch)
    ig = np.nonzero(nt)[0]; ie = np.nonzero(nt_e)[0]; r = np.random.default_rng(0)
    for _ in range(300):
        a = X[r.choice(ig)][shared_g]; b = Xe[r.choice(ie)][shared_e]
        ratio_ctrl.append(float(((a - b) ** 2).sum() / (Gs * c_g / np.median(nc[ig]) + Gs * c_e / np.median(nc_e[ie]))))
    res["noise_model_check_across_screens"] = {
        "n_knockdowns": len(ratios), "observed/predicted noise energy of (gw - es) difference, median [IQR]":
            [float(np.median(ratios)), float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))],
        "non-targeting controls, random cross-screen pairs, median [IQR]":
            [float(np.median(ratio_ctrl)), float(np.percentile(ratio_ctrl, 25)), float(np.percentile(ratio_ctrl, 75))],
        "interpretation": "ratio 1 = difference fully explained by the control-calibrated noise model; >1 = extra noise in perturbed profiles and/or real between-screen differences"}
    # cross-screen stability of the shared low-rank axes: PCs fitted separately in each screen on the knockdowns in both
    both = [(rows[k], erow[t]) for k, t in enumerate(tg) if t in erow]
    A_ = X[[a for a, _ in both]][:, shared_g]; B_ = Xe[[b for _, b in both]][:, shared_e]
    def axes(Z, K=3):
        mm = Z.mean(0); uu = mm / np.linalg.norm(mm); Rz = Z - np.outer(Z @ uu, uu)
        return uu, randomized_svd(Rz, n_components=K, random_state=0)[2]
    ua, Va = axes(A_); ub, Vb = axes(B_)
    cosg = float(abs(ua @ ub))
    sv = np.linalg.svd(Va @ Vb.T, compute_uv=False)
    Qr = np.linalg.qr(r.standard_normal((Gs, 3)))[0].T
    sv_null = np.linalg.svd(Va @ Qr.T, compute_uv=False)
    names = var[shared_g]
    res["axis_stability"] = {"n_knockdowns_in_both": len(both), "cos(generic_gw, generic_es)": cosg,
                             "principal_angle_cosines_pathway3": sv.round(3).tolist(), "random_subspace_cosines": sv_null.round(3).tolist(),
                             "top_loading_genes_gw": {f"PC{i + 1}": names[np.argsort(-np.abs(Va[i]))[:12]].tolist() for i in range(3)},
                             "top_loading_genes_generic": names[np.argsort(-np.abs(ua))[:12]].tolist()}
    res["seconds"] = time.time() - t0
    json.dump(res, open(OUT / "decomp_sensitivity.json", "w"), indent=1)
    print(json.dumps(res, indent=1))



# =============================================================================== round 2: positive-control robustness
def f1pc():
    """Positive control, round 2. alpha is defined exactly: the residual component is orthogonalised against the
    cascade direction, so alpha = fraction of the target's (noise-free) signal energy along the cascade direction.
    Generator variants: mediator ranking, coordinate restoration, heterogeneous strengths, unequal screen reliability,
    and the original non-orthogonal construction (realized alpha reported)."""
    t0 = time.time()
    S = f1_setup(); E, R, first = S["E"], S["R"], S["first"]
    col, G, M, med = S["col"], S["G"], S["M"], S["med"]
    rep = [t for t in S["targets"] if S["rel"][t] > 0.3]
    r12_med = float(np.median([S["rel"][t] for t in S["targets"]]))
    real = json.load(open(OUT / "f1_revised.json"))["real"]["specific residual (original)"]["10"]["diff_mean_ci"]
    sig = lambda v: np.sqrt((v[M] ** 2).mean())

    def run(alpha, cov, rank="strongest", restore=True, hetero=False, rel1=1.0, rel2=1.0, orth=True, seed=0):
        rng = np.random.default_rng(seed)
        def noisy(v, rel_):
            R_ = float(np.clip(rel_, 0.05, 0.95)); s_ = sig(v) * np.sqrt((1 - R_) / R_)
            n = rng.standard_normal(G) * s_; n[~M] = 0
            return v + n
        truth_m = {k: R[first[k]] for k in med}
        obs_m = {k: noisy(truth_m[k], S["rel"].get(k, r12_med)) for k in med}
        T1, T2, pools, realized = {}, {}, {}, []
        for t in rep:
            cand = [m for m in med if m != t]
            aa = np.array([E[first[t], col[m]] for m in cand])
            if rank == "strongest":
                true_k = [cand[i] for i in np.argsort(-np.abs(aa))[:10]]
            else:  # true mediators drawn at random among measured genes (mediator ranking errors)
                true_k = list(rng.choice(cand, 10, replace=False))
            w = np.array([E[first[t], col[k]] / S["selfz"][k] for k in true_k])
            if hetero:
                w = w * rng.lognormal(0, 1.0, len(w))
            hidden = set(rng.choice(true_k, int(round((1 - cov) * len(true_k))), replace=False)) if cov < 1 else set()
            pools[t] = [m for m in cand if m not in hidden]
            c = (w[:, None] * np.stack([truth_m[k] for k in true_k])).sum(0)
            o = R[first[t]].copy()
            zero = [col[k] for k in true_k] + [col[t]]
            c[zero] = 0; o[zero] = 0; c[~M] = 0; o[~M] = 0
            ch = c / (np.linalg.norm(c) + 1e-12)
            if orth:
                o = o - (o @ ch) * ch
            oh = o / (np.linalg.norm(o) + 1e-12)
            tt = np.linalg.norm(R[first[t]][M]) * (np.sqrt(alpha) * ch + np.sqrt(1 - alpha) * oh)
            realized.append(float((tt @ ch) ** 2 / (tt @ tt + 1e-12)))
            if restore:
                for k, wk in zip(true_k, w):
                    tt[col[k]] = E[first[t], col[k]]
            T1[t] = noisy(tt, S["rel"][t] * rel1); T2[t] = noisy(tt, S["rel"][t] * rel2)
        out = domino_eval(S, lambda k: obs_m[k], lambda t: T2[t], lambda t: T1[t], lambda t: T1[t], rep, 10, seed=seed + 1,
                          cand_pool=pools, rel_obs=lambda t, msk: corr(T1[t][msk], T2[t][msk]))
        out["realized_alpha_mean_sd"] = [float(np.mean(realized)), float(np.std(realized))]
        return out
    variants = {
        "base (orthogonal, strongest mediators, coordinates restored, equal reliability)": {},
        "mediators ranked at random (ranking errors)": {"rank": "random"},
        "mediator coordinates not restored": {"restore": False},
        "heterogeneous mediator strengths (lognormal, sd 1)": {"hetero": True},
        "screen-2 reliability halved": {"rel2": 0.5},
        "screen-1 reliability halved": {"rel1": 0.5},
        "non-orthogonal construction (original)": {"orth": False},
        "all adverse: random ranking, no restore, heterogeneous, screen-2 halved": {"rank": "random", "restore": False, "hetero": True, "rel2": 0.5},
    }
    res = {"real_data_diff_ci": real, "definition": "alpha = fraction of noise-free target signal energy along the cascade direction (exact when orthogonalised)", "variants": {}}
    for vn, kw in variants.items():
        res["variants"][vn] = {}
        covs = (1.0, 0.5, 0.25) if vn.startswith("base") else (1.0, 0.25)
        for cov in covs:
            for a in (0.0, 0.05, 0.1, 0.25, 0.5):
                o = run(a, cov, **kw)
                res["variants"][vn][f"coverage={cov} alpha={a}"] = o
                print(vn[:40], cov, a, "diff", [round(x, 3) for x in o["diff_mean_ci"][:3]], "realized", round(o["realized_alpha_mean_sd"][0], 3), flush=True)
        json.dump(res, open(OUT / "f1_pc_round2.json", "w"), indent=1)
    res["seconds"] = time.time() - t0
    json.dump(res, open(OUT / "f1_pc_round2.json", "w"), indent=1)


# =============================================================================== round 2: calibrated benchmark
def f2cal():
    from experiments.response_decomposition import fit_components, split
    t0 = time.time()
    d, X, P, col, var, G, es, M = rep_data()
    rows = np.nonzero(P)[0]; tg = d["tgt"][rows]
    keep = np.array([t in col for t in tg]); rows, tg = rows[keep], tg[keep]
    genes_u = np.unique(tg); r = np.random.default_rng(1); r.shuffle(genes_u)
    fold_of = {g: k % 5 for k, g in enumerate(genes_u)}
    folds = np.array([fold_of[t] for t in tg])
    KS = (10, 20, 50, 100)
    preds = ["mean training profile"] + [f"kNN k={k}" for k in KS] + ["factorised (generic mean + pathway kNN50 + specific kNN50)"]
    targets = ["independent screen, raw", "independent screen, mean-centred", "independent screen, specific residual"]
    S = {p: {t: {"corr": [], "R2_raw": [], "R2_cal": []} for t in targets} for p in preds}
    lam_log = {p: {t: [] for t in targets} for p in preds}
    for fo in range(5):
        tr = folds != fo; te = folds == fo
        Etr = X[rows[tr]]; tg_tr = tg[tr]
        hv = np.argsort(-Etr.var(0))[:1000]
        u, V, _, _, K = fit_components(Etr, seed=fo)
        gtr, ptr, rtr = split(Etr, u, V)
        Z = Etr - Etr.mean(0); Z /= (Z.std(0) + 1e-6)
        tcol_tr = np.array([col[t] for t in tg_tr])
        mean_raw = Etr.mean(0); amp_mean = (Etr @ u).mean()
        spec = lambda v: (lambda w: w - (w @ V.T) @ V)(v - (v @ u) * u)
        T = {"independent screen, raw": lambda p: p, "independent screen, mean-centred": lambda p: p - mean_raw,
             "independent screen, specific residual": spec}
        def predict(t, exclude=None):
            sim = Z[:, col[t]] @ Z[:, tcol_tr] / Z.shape[0]
            if exclude is not None:
                sim = sim.copy(); sim[exclude] = -np.inf
            o = np.argsort(-sim)
            def knn(Mat, k):
                nb = o[:k]; w = np.maximum(sim[nb], 0)
                return (w[:, None] * Mat[nb]).sum(0) / (w.sum() + 1e-9)
            pr = {"mean training profile": mean_raw}
            for k in KS:
                pr[f"kNN k={k}"] = knn(Etr, k)
            pr["factorised (generic mean + pathway kNN50 + specific kNN50)"] = amp_mean * u + knn(ptr, 50) + knn(rtr, 50)
            return pr
        def panel(t):
            msk = M.copy(); msk[col[t]] = False
            return np.intersect1d(hv, np.nonzero(msk)[0])
        # training-only calibration: leave-one-out predictions for up to 300 training knockdowns with an es profile
        cal_idx = [i for i in np.random.default_rng(fo).permutation(len(tg_tr)) if tg_tr[i] in es][:300]
        num = {p: {t: 0.0 for t in targets} for p in preds}; den = {p: {t: 0.0 for t in targets} for p in preds}
        for i in cal_idx:
            t = tg_tr[i]; sel = panel(t)
            same = np.nonzero(tg_tr == t)[0]
            pr = predict(t, exclude=same)
            ej = es[t]
            for tn in targets:
                y = T[tn](ej)[sel] if tn != "independent screen, raw" else ej[sel]
                yy = float(y @ y) + 1e-12
                for pn, p in pr.items():
                    pp = T[tn](p)[sel]
                    # weights 1/||y||^2 make lambda maximise the MEAN per-knockdown R^2_0 (the reported metric)
                    num[pn][tn] += float(pp @ y) / yy; den[pn][tn] += float(pp @ pp) / yy
        lam = {pn: {tn: float(np.clip(num[pn][tn] / (den[pn][tn] + 1e-12), 0.0, 10.0)) for tn in targets} for pn in preds}
        for pn in preds:
            for tn in targets:
                lam_log[pn][tn].append(lam[pn][tn])
        for a in np.nonzero(te)[0]:
            t = tg[a]
            if t not in es:
                continue
            sel = panel(t); pr = predict(t); ej = es[t]
            for tn in targets:
                y = T[tn](ej)[sel]
                for pn, p in pr.items():
                    pp = T[tn](p)[sel]
                    S[pn][tn]["corr"].append(corr(pp, y)); S[pn][tn]["R2_raw"].append(r2z(y, pp))
                    S[pn][tn]["R2_cal"].append(r2z(y, lam[pn][tn] * pp))
        print(f"fold {fo} {time.time() - t0:.0f}s", flush=True)
    out = {"n_scored": len(S[preds[0]][targets[0]]["corr"]), "calibration": "per fold, per predictor and target: lambda in [0, 10] maximising the mean per-knockdown R^2_0 over leave-one-out predictions for 300 training knockdowns (training data only)",
           "lambda_per_fold": lam_log, "scores": {}, "paired_differences": {}}
    for pn in preds:
        out["scores"][pn] = {}
        for tn in targets:
            v = S[pn][tn]
            out["scores"][pn][tn] = {"corr": boot_ci(v["corr"]) if not all(x != x for x in v["corr"]) else None,
                                     "R2_vs_zero_uncalibrated": boot_ci(v["R2_raw"]), "R2_vs_zero_calibrated": boot_ci(v["R2_cal"])}
    for a_, b_ in [("kNN k=100", "kNN k=10"), ("kNN k=50", "kNN k=10"), ("kNN k=100", "kNN k=50"), ("factorised (generic mean + pathway kNN50 + specific kNN50)", "kNN k=50")]:
        for tn in targets:
            dc = np.array(S[a_][tn]["corr"]) - np.array(S[b_][tn]["corr"])
            dr = np.array(S[a_][tn]["R2_cal"]) - np.array(S[b_][tn]["R2_cal"])
            out["paired_differences"][f"{a_} minus {b_} | {tn}"] = {"corr": boot_ci(dc), "R2_calibrated": boot_ci(dr)}
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT / "f2_calibrated.json", "w"), indent=1)
    for pn in preds:
        print(pn[:30].ljust(32), " | ".join(f"r={out['scores'][pn][t]['corr'][0] if out['scores'][pn][t]['corr'] else float('nan'):.3f} R2c={out['scores'][pn][t]['R2_vs_zero_calibrated'][0]:.4f}" for t in targets))
    print(json.dumps(out["paired_differences"], indent=0)[:3000])


# =============================================================================== round 2: noise uncertainty + matched PCs
def noise_pcs():
    from sklearn.utils.extmath import randomized_svd
    t0 = time.time()
    d, X, P, col, var, G, es, M = rep_data()
    nc, nt = d["nc"], d["nt"]
    E = X[P]; ncP = nc[P]; tot = (E ** 2).sum(1)
    Xn = X[nt]; nn = nc[nt]
    def shares(cval=None, cg=None):
        noise = (G * cval / ncP) if cg is None else (cg.sum() / ncP)
        return float(noise.sum() / tot.sum()), float(np.median(noise / tot))
    per = (Xn ** 2).mean(1) * nn
    res = {"estimators": {}}
    res["estimators"]["median over controls (used)"] = shares(float(np.median(per)))
    res["estimators"]["mean over controls"] = shares(float(np.mean(per)))
    from scipy.stats import trim_mean
    res["estimators"]["20% trimmed mean"] = shares(float(trim_mean(per, 0.2)))
    cg = np.median(Xn ** 2 * nn[:, None], 0)
    res["estimators"]["gene-specific (median per gene)"] = shares(cg=cg)
    rng = np.random.default_rng(0); B = []
    for _ in range(500):
        b = rng.choice(len(per), len(per)); B.append(shares(float(np.median(per[b]))))
    B = np.array(B)
    res["bootstrap_over_controls"] = {"pooled": [float(np.percentile(B[:, 0], 2.5)), float(np.percentile(B[:, 0], 97.5))],
                                      "median": [float(np.percentile(B[:, 1], 2.5)), float(np.percentile(B[:, 1], 97.5))]}
    # matched PCs across screens
    fe_ = h5py.File(RAW / "K562_essential_normalized_bulk_01.h5ad", "r")
    tg_e = np.array([x.decode().split("_")[1] for x in fe_["obs/gene_transcript"][:]]); fe_e = fe_["obs/fold_expr"][:]
    cats = [x.decode() if isinstance(x, bytes) else x for x in fe_["var/__categories/gene_name"][:]]
    var_e = np.array([cats[k] for k in fe_["var/gene_name"][:]])
    Xe = np.nan_to_num(fe_["X"][:].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    shared_e = np.array([k for k, g in enumerate(var_e) if g in col]); shared_g = np.array([col[var_e[k]] for k in shared_e])
    rows = np.nonzero(P)[0]; tg = d["tgt"][rows]
    erow = {}
    for k, t in enumerate(tg_e):
        if fe_e[k] == fe_e[k] and fe_e[k] < 0.5 and (t not in erow or fe_e[k] < fe_e[erow[t]]):
            erow[t] = k
    both = [(rows[k], erow[t]) for k, t in enumerate(tg) if t in erow]
    A_ = X[[a for a, _ in both]][:, shared_g]; B_ = Xe[[b for _, b in both]][:, shared_e]
    def axes(Zm, K=3):
        mm = Zm.mean(0); uu = mm / np.linalg.norm(mm); Rz = Zm - np.outer(Zm @ uu, uu)
        return randomized_svd(Rz, n_components=K, random_state=0)[2]
    Va, Vb = axes(A_), axes(B_)
    Cm = np.abs(Va @ Vb.T)
    U_, s_, Wt = np.linalg.svd(Va @ Vb.T)
    Ca = U_.T @ Va  # aligned (canonical) directions in the GW subspace
    names = var[shared_g]
    res["matched_pc_abs_cosine_gw_rows_es_cols"] = Cm.round(3).tolist()
    res["canonical_cosines"] = s_.round(3).tolist()
    res["canonical_direction_top_genes"] = {f"canonical {i + 1}": names[np.argsort(-np.abs(Ca[i]))[:10]].tolist() for i in range(3)}
    res["seconds"] = time.time() - t0
    json.dump(res, open(OUT / "noise_pcs.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    globals()[sys.argv[1]]()
