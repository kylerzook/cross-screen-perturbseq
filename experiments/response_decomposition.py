"""What do knockdown experiments actually contain?  Replogle 2022 K562 genome-wide Perturb-seq (pseudobulk).

Stages (cached; run in order):
  decompose  - split every knockdown profile into
                 generic  : projection on the mean knockdown profile (rank 1, per-knockdown amplitude)
                 pathway  : top-K principal components of what is left (K by a permutation null, cap 100)
                 residual : the rest  (= specific part + noise)
               noise floor from the 585 NON-TARGETING control profiles (E[x^2] ~ c / n_cells), so
               'specific above noise' = residual fraction - expected noise fraction.
               Replicate check: knockdowns of the same gene with two different guides/TSSs.
  heldout    - 5-fold CV over knockdowns (grouped by target gene). Components are re-fit on the
               training folds only; the held-out knockdown's specific residual is the prediction target.
               Predictors: zero, similarity transfer (target genes with similar co-response profiles),
               linear interventional SEM on a 400-gene perturbed set. Also the raw-profile task.
  specificity- per-knockdown specificity; TF vs essential vs other; vs knockdown strength / cells / effect size.
  biology    - K562 ChIP-seq (ENCODE) and JASPAR motif occupancy at promoters: are a TF's bound /
               motif-carrying genes enriched among its responders in the RAW vs the RESIDUAL profile?
  report     - write docs/09_response_decomposition_results.md

Knockdowns used: fold_expr < 0.5 (strong knockdown) and energy-test p < 0.05 (detectable response),
non-targeting excluded. The target gene's own column is masked (set to 0) everywhere.
"""
import gzip, json, sys, time, urllib.parse, urllib.request
from pathlib import Path
import numpy as np, h5py
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"; OUT = ROOT / "results/decomposition"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
KMAX = 100
rng0 = np.random.default_rng(0)


# ----------------------------------------------------------------------------- loading
def load():
    f = h5py.File(RAW / "K562_gwps_normalized_bulk_01.h5ad", "r")
    cats = [x.decode() if isinstance(x, bytes) else x for x in f["var/__categories/gene_name"][:]]
    var = np.array([cats[c] for c in f["var/gene_name"][:]])
    gt = [x.decode() for x in f["obs/gene_transcript"][:]]
    tgt = np.array([s.split("_")[1] for s in gt])
    fe = f["obs/fold_expr"][:]; nc = f["obs/num_cells_filtered"][:]; pe = f["obs/energy_test_p_value"][:]
    nc = np.where(np.isfinite(nc) & (nc > 0), nc, np.nanmedian(nc))
    X = np.nan_to_num(f["X"][:].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    col = {g: i for i, g in enumerate(var)}
    for r, g in enumerate(tgt):  # mask the target's own column (self-knockdown dominates otherwise)
        if g in col:
            X[r, col[g]] = 0.0
    nt = np.array(["non-targeting" in s for s in gt])
    P = (~nt) & (fe < 0.5) & (pe < 0.05)
    return dict(X=X, var=var, tgt=tgt, fe=fe, nc=nc, pe=pe, nt=nt, P=P, col=col)


def gene_sets(var):
    from bond.pwm import parse_jaspar
    tfs = {m["name"].upper() for m in parse_jaspar(RAW / "jaspar2024_vert_nr.txt") if "::" not in m["name"]}
    f = h5py.File(RAW / "K562_essential_normalized_bulk_01.h5ad", "r")
    ess = {x.decode().split("_")[1] for x in f["obs/gene_transcript"][:]}
    return tfs, ess


# ----------------------------------------------------------------------------- decomposition
def fit_components(E, kmax=KMAX, seed=0):
    """Generic direction u (mean profile) + pathway PCs V (K by permutation null)."""
    m = E.mean(0); u = m / (np.linalg.norm(m) + 1e-12)
    R = E - np.outer(E @ u, u)
    from sklearn.utils.extmath import randomized_svd
    U, S, Vt = randomized_svd(R, n_components=kmax, random_state=seed)
    # permutation null: shuffle each column independently, top singular value
    r = np.random.default_rng(seed)
    Rp = np.stack([r.permutation(R[:, j]) for j in range(R.shape[1])], 1)
    s_null = randomized_svd(Rp, n_components=1, random_state=seed)[1][0]
    K = int((S > s_null).sum())
    return u, Vt[:K], S, s_null, K


def split(X, u, V):
    g = np.outer(X @ u, u)
    Rg = X - g
    p = (Rg @ V.T) @ V
    return g, p, Rg - p


def decompose():
    d = load(); X, P, nc, nt = d["X"], d["P"], d["nc"], d["nt"]
    t0 = time.time()
    E = X[P]; G = X.shape[1]
    # noise floor from non-targeting profiles: mean square per gene ~ c / n_cells
    c = float(np.median((X[nt] ** 2).mean(1) * nc[nt]))
    noise = G * c / nc[P]
    tot = (E ** 2).sum(1)
    signal = np.maximum(tot - noise, 1e-9)
    u, V_full, S, s_null, K_null = fit_components(E)
    q = lambda v: [float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75))]
    summary = {"n_knockdowns": int(P.sum()), "n_genes": G, "n_nontargeting": int(nt.sum()), "K_null": K_null,
               "top_singular_values": S[:10].round(1).tolist(), "null_top_singular_value": float(s_null),
               "noise_c": c, "noise_frac_of_total_energy": q(noise / tot),
               "frac_knockdowns_signal_above_noise": float(np.mean(tot > noise)), "by_K": {}}
    keepers = {}
    # fit up to KMAX components once, then evaluate K = null-based, 20, 50
    from sklearn.utils.extmath import randomized_svd
    m = E.mean(0); uu = m / np.linalg.norm(m); Rg = E - np.outer(E @ uu, uu)
    _, S100, Vt100 = randomized_svd(Rg, n_components=KMAX, random_state=0)
    for K in sorted({K_null, 20, 50}):
        V = Vt100[:K]
        g, p, r = split(E, uu, V)
        noise_r = noise * (G - 1 - K) / G
        gen, pat = (g ** 2).sum(1), (p ** 2).sum(1)
        spec = np.maximum((r ** 2).sum(1) - noise_r, 0)
        sig = gen + pat + spec
        fr = dict(generic=gen / sig, pathway=pat / sig, specific=spec / sig)
        big = signal > np.percentile(signal, 75)
        summary["by_K"][str(K)] = {
            "share_of_signal_q25_median_q75": {k: q(v) for k, v in fr.items()},
            "share_of_signal_pooled": {k: float(v.sum() / sig.sum()) for k, v in dict(generic=gen, pathway=pat, specific=spec).items()},
            "share_of_signal_median_top25pct_effect": {k: float(np.median(v[big])) for k, v in fr.items()}}
        keepers[K] = (V, fr, r)
    K = K_null if K_null >= 3 else 20
    V, fr, r = keepers[K]
    summary["K_used_downstream"] = K
    # replicate check: same target, two different guides / TSS (whole transcriptome and top-1000 variable genes)
    tg = d["tgt"][P]; idx = {}
    for k, t in enumerate(tg):
        idx.setdefault(t, []).append(k)
    pairs = [(v[0], v[1]) for v in idx.values() if len(v) >= 2]
    hv = np.argsort(-E.var(0))[:1000]
    rr = lambda A, i, j, cols=slice(None): float(np.corrcoef(A[i, cols], A[j, cols])[0, 1])
    g_, p_, _ = split(E, uu, V)
    rnd = [(i, j) for i, j in rng0.choice(len(tg), (500, 2)) if tg[i] != tg[j]]
    rep = {"n_pairs": len(pairs)}
    for nm, A in [("raw", E), ("generic_removed", E - g_), ("residual", r)]:
        rep[f"{nm}_corr_all_genes"] = q([rr(A, i, j) for i, j in pairs])
        rep[f"{nm}_corr_top1000var"] = q([rr(A, i, j, hv) for i, j in pairs])
        rep[f"random_{nm}_corr_top1000var"] = q([rr(A, i, j, hv) for i, j in rnd])
    summary["replicates_q25_median_q75"] = rep
    # cross-screen replicate: the same knockdown in Replogle's separate K562 ESSENTIAL screen.
    # Noise in the two screens is independent, so inner products between the two profiles estimate the
    # REPRODUCIBLE (noise-free) energy of each component.
    fe_ = h5py.File(RAW / "K562_essential_normalized_bulk_01.h5ad", "r")
    cats = [x.decode() if isinstance(x, bytes) else x for x in fe_["var/__categories/gene_name"][:]]
    var_e = np.array([cats[c] for c in fe_["var/gene_name"][:]])
    tg_e = np.array([x.decode().split("_")[1] for x in fe_["obs/gene_transcript"][:]])
    fold_e = fe_["obs/fold_expr"][:]
    Xe = np.nan_to_num(fe_["X"][:].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    colg = d["col"]; shared = np.array([colg[g] for g in var_e if g in colg]); src = np.array([k for k, g in enumerate(var_e) if g in colg])
    erow = {}
    for k, t in enumerate(tg_e):
        if fold_e[k] == fold_e[k] and fold_e[k] < 0.5 and (t not in erow or fold_e[k] < fold_e[erow[t]]):
            erow[t] = k
    both = [(k, erow[t]) for k, t in enumerate(tg) if t in erow and k == idx[t][0]]
    M = np.zeros(G, bool); M[shared] = True
    xg_rows = np.stack([E[k] for k, _ in both])
    xe_rows = np.zeros((len(both), G), np.float32); xe_rows[:, shared] = Xe[[e for _, e in both]][:, src]
    for n_, (k, _) in enumerate(both):  # mask the target column in both
        tcol = colg.get(tg[k]);
        if tcol is not None:
            xg_rows[n_, tcol] = 0; xe_rows[n_, tcol] = 0
    xg_rows[:, ~M] = 0
    ga, pa, ra = split(xg_rows, uu, V); gb, pb, rb = split(xe_rows, uu, V)
    dot = lambda A, B: (A * B).sum(1)
    cross = {"n_knockdowns_in_both_screens": len(both), "n_shared_genes": int(M.sum())}
    tot_rep = dot(xg_rows, xe_rows)
    for nm, a_, b_ in [("generic", ga, gb), ("pathway", pa, pb), ("specific residual", ra, rb)]:
        cross[f"reproducible_energy_share_pooled_{nm}"] = float(dot(a_, b_).sum() / tot_rep.sum())
        cross[f"corr_between_screens_{nm}_q25_median_q75"] = q([float(np.corrcoef(a_[i], b_[i])[0, 1]) for i in range(len(both))])
    cross["corr_between_screens_raw_q25_median_q75"] = q([float(np.corrcoef(xg_rows[i], xe_rows[i])[0, 1]) for i in range(len(both))])
    perm = np.random.default_rng(3).permutation(len(both))
    cross["corr_mismatched_pairs_residual_q25_median_q75"] = q([float(np.corrcoef(ra[i], rb[perm[i]])[0, 1]) for i in range(len(both)) if perm[i] != i])
    cross["frac_knockdowns_residual_corr_gt_0.1"] = float(np.mean([np.corrcoef(ra[i], rb[i])[0, 1] > 0.1 for i in range(len(both))]))
    summary["cross_screen_replicate"] = cross
    np.savez(OUT / "cross_screen.npz", targets=np.array([tg[k] for k, _ in both]),
             res_corr=np.array([np.corrcoef(ra[i], rb[i])[0, 1] for i in range(len(both))]),
             spec_rep_energy=dot(ra, rb), tot_rep_energy=tot_rep)
    summary["seconds"] = time.time() - t0
    np.savez(OUT / "decomp.npz", u=uu, V=V, K=K, noise_c=c, P=P, frac_generic=fr["generic"], frac_pathway=fr["pathway"],
             frac_specific=fr["specific"], signal=signal, noise=noise, total=tot)
    json.dump(summary, open(OUT / "decomp_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


# ----------------------------------------------------------------------------- held-out test
def heldout():
    d = load(); X, P, col, var = d["X"], d["P"], d["col"], d["var"]
    rows = np.nonzero(P)[0]; tg = d["tgt"][rows]
    keep = np.array([t in col for t in tg]); rows, tg = rows[keep], tg[keep]
    genes_u = np.unique(tg); r = np.random.default_rng(1); r.shuffle(genes_u)
    fold_of = {g: k % 5 for k, g in enumerate(genes_u)}
    folds = np.array([fold_of[t] for t in tg])
    G = X.shape[1]
    res = {k: {"R2_vs_zero": [], "corr": []} for k in
           ["residual: zero", "residual: mean training residual", "residual: similarity transfer (k=10)",
            "residual: similarity transfer (k=50)",
            "raw: zero", "raw: mean training profile", "raw: similarity transfer (k=10)",
            "generic+pathway removed: similarity transfer (k=10)"]}
    sem = {"residual (400-gene set): zero": [], "residual (400-gene set): linear SEM": [],
           "raw (400-gene set): linear SEM": [], "raw (400-gene set): mean training profile": []}
    sem_c = {k: [] for k in sem}

    hv = np.argsort(-X[rows].var(0))[:1000]
    per = []
    cur = {"t": None}

    def score(name, y, yh, store=res):
        ss = (y ** 2).sum()
        store[name]["R2_vs_zero"].append(1 - ((y - yh) ** 2).sum() / ss)
        c_all = float(np.corrcoef(y, yh)[0, 1]) if yh.std() > 1e-12 else 0.0
        c_hv = float(np.corrcoef(y[hv], yh[hv])[0, 1]) if yh[hv].std() > 1e-12 else 0.0
        store[name]["corr"].append(c_all)
        per.append((cur["t"], name, c_all, c_hv))

    t0 = time.time()
    for fo in range(5):
        tr, te = rows[folds != fo], rows[folds == fo]
        Etr = X[tr]
        u, V, S, s_null, K = fit_components(Etr, seed=fo)
        gtr, ptr, rtr = split(Etr, u, V)
        Ete = X[te]; gte, pte, rte = split(Ete, u, V)
        # similarity of target genes: co-response profiles across TRAINING knockdowns
        Z = Etr - Etr.mean(0); Z /= (Z.std(0) + 1e-6)
        tcols_tr = np.array([col[t] for t in tg[folds != fo]]); tcols_te = np.array([col[t] for t in tg[folds == fo]])
        A = Z[:, tcols_te].T; B = Z[:, tcols_tr].T
        sim = (A @ B.T) / Z.shape[0]  # (n_te, n_tr) correlation of target genes' co-response
        mean_r = rtr.mean(0); mean_raw = Etr.mean(0); gp_removed_tr = Etr - gtr - ptr
        for a in range(len(te)):
            cur["t"] = tg[folds == fo][a]
            for kk, name in [(10, "residual: similarity transfer (k=10)"), (50, "residual: similarity transfer (k=50)")]:
                nb = np.argsort(-sim[a])[:kk]; w = np.maximum(sim[a, nb], 0)
                pred = (w[:, None] * rtr[nb]).sum(0) / (w.sum() + 1e-9)
                score(name, rte[a], pred)
            nb = np.argsort(-sim[a])[:10]; w = np.maximum(sim[a, nb], 0)
            score("raw: similarity transfer (k=10)", Ete[a], (w[:, None] * Etr[nb]).sum(0) / (w.sum() + 1e-9))
            score("generic+pathway removed: similarity transfer (k=10)", Ete[a] - gte[a] - pte[a],
                  (w[:, None] * gp_removed_tr[nb]).sum(0) / (w.sum() + 1e-9))
            score("residual: zero", rte[a], np.zeros(G)); score("residual: mean training residual", rte[a], mean_r)
            score("raw: zero", Ete[a], np.zeros(G)); score("raw: mean training profile", Ete[a], mean_raw)
        # linear interventional SEM on a gene set where every gene is also a knockdown target
        tr_genes = list(dict.fromkeys(tg[folds != fo])); te_genes = list(dict.fromkeys(tg[folds == fo]))
        en_r = {t: (rtr[i] ** 2).sum() for i, t in enumerate(tg[folds != fo])}
        S_tr = sorted(tr_genes, key=lambda t: -en_r[t])[:320]
        en_te = {t: (rte[i] ** 2).sum() for i, t in enumerate(tg[folds == fo])}
        S_te = sorted(te_genes, key=lambda t: -en_te[t])[:80]
        Sg = S_tr + S_te; cols = np.array([col[t] for t in Sg]); n = len(Sg)
        tr_row = {t: i for i, t in enumerate(tg[folds != fo])}; te_row = {t: i for i, t in enumerate(tg[folds == fo])}
        for space, Mtr, Mte in [("residual", rtr, rte), ("raw", Etr, Ete)]:
            Rtr = np.stack([Mtr[tr_row[t], cols] for t in S_tr])  # training knockdowns x gene set
            Bm = np.zeros((n, n))
            from sklearn.linear_model import Lasso
            for i in range(n):
                rws = [k for k, t in enumerate(S_tr) if t != Sg[i]]
                F = np.delete(Rtr[rws], i, axis=1); y = Rtr[rws, i]
                sd = F.std(0) + 1e-9
                m_ = Lasso(alpha=0.05, max_iter=2000).fit(F / sd, y)
                Bm[i, np.arange(n) != i] = m_.coef_ / sd
            for t in S_te:
                j = Sg.index(t); obs = Mte[te_row[t], cols]; keepg = np.arange(n) != j
                e = np.zeros(n); e[j] = 1.0; term = e.copy(); pr = e.copy()
                for _ in range(4):
                    term = Bm @ term; pr = pr + term
                zs = X[te[te_row[t]], col[t]] if False else -1.0  # self column is masked; use unit knockdown
                pr = pr * zs
                y = obs[keepg]; yh = pr[keepg]
                # scale-free comparison: best non-negative scaling of the propagated pattern (fit on the
                # held-out profile itself would leak, so report correlation and R2 with scale fixed to -1)
                key = f"{space} (400-gene set): linear SEM"
                sem[key].append(1 - ((y - yh) ** 2).sum() / (y ** 2).sum()); sem_c[key].append(float(np.corrcoef(y, yh)[0, 1]) if yh.std() > 1e-12 else 0.0)
                if space == "residual":
                    sem["residual (400-gene set): zero"].append(0.0); sem_c["residual (400-gene set): zero"].append(0.0)
                else:
                    mp = Etr[:, cols].mean(0)[keepg]
                    sem["raw (400-gene set): mean training profile"].append(1 - ((y - mp) ** 2).sum() / (y ** 2).sum())
                    sem_c["raw (400-gene set): mean training profile"].append(float(np.corrcoef(y, mp)[0, 1]))
        print(f"fold {fo} done {time.time() - t0:.0f}s K={K}", flush=True)
    summ = lambda v: [float(np.nanmean(v)), float(np.nanmedian(v)), float(np.nanstd(v)), int(len(v)),
                      float(np.mean(np.array(v) > 0))]
    out = {"note": "values: [mean, median, sd, n, fraction > 0]",
           "predictors": {k: {m: summ(v) for m, v in r_.items()} for k, r_ in res.items()},
           "sem": {k: {"R2_vs_zero": summ(sem[k]), "corr": summ(sem_c[k])} for k in sem}}
    # re-score on knockdowns whose specific residual REPRODUCES across screens (cross-screen corr > 0.3)
    cs = np.load(OUT / "cross_screen.npz", allow_pickle=True)
    rel = dict(zip(cs["targets"], cs["res_corr"]))
    sub = {}
    for name in res:
        rows_ = [(c1, c2, rel[t]) for t, n_, c1, c2 in per if n_ == name and t in rel]
        hi = [r for r in rows_ if r[2] > 0.3]
        sub[name] = {"n_reproducible": len(hi),
                     "corr_all_genes_mean_reproducible": float(np.mean([r[0] for r in hi])) if hi else None,
                     "corr_top1000var_mean_reproducible": float(np.mean([r[1] for r in hi])) if hi else None,
                     "corr_top1000var_mean_all": float(np.mean([c2 for t, n_, c1, c2 in per if n_ == name])),
                     "frac_positive_top1000var_reproducible": float(np.mean([r[1] > 0 for r in hi])) if hi else None}
    out["reproducible_subset"] = sub
    out["ceiling_note"] = "cross-screen residual corr median (see decomp_summary) bounds what any predictor can reach"
    json.dump(out, open(OUT / "heldout.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- specificity
def specificity():
    d = load(); X, P = d["X"], d["P"]
    dz = np.load(OUT / "decomp.npz")
    tfs, ess = gene_sets(d["var"])
    rows = np.nonzero(P)[0]; tg = d["tgt"][rows]
    E = X[rows]
    Zn = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    C = Zn @ Zn.T
    same = tg[:, None] == tg[None, :]
    C[same] = -np.inf
    uniq = 1 - np.max(np.abs(np.where(np.isinf(C), 0, C)), 1)
    spec = dz["frac_specific"]
    grp = np.array(["TF" if t in tfs else ("essential (non-TF)" if t in ess else "other") for t in tg])
    eff = np.log10(dz["signal"])
    out = {"groups": {}, "tests": {}, "correlations": {}}
    snr = np.log10(dz["signal"] / dz["noise"])
    for name, v in [("specific_share_of_signal", spec), ("log10 signal-to-noise", snr), ("uniqueness (1 - max |cos| to other knockdowns)", uniq),
                    ("generic_frac", dz["frac_generic"]), ("pathway_frac", dz["frac_pathway"])]:
        out["groups"][name] = {gname: [float(np.median(v[grp == gname])), float(np.mean(v[grp == gname])), int((grp == gname).sum())]
                               for gname in ["TF", "essential (non-TF)", "other"]}
        out["tests"][name] = {f"TF vs essential": float(mannwhitneyu(v[grp == "TF"], v[grp == "essential (non-TF)"]).pvalue),
                              f"TF vs other": float(mannwhitneyu(v[grp == "TF"], v[grp == "other"]).pvalue)}
        out["correlations"][name] = {
            "knockdown strength (1 - fold_expr)": float(spearmanr(v, 1 - d["fe"][rows])[0]),
            "n cells": float(spearmanr(v, d["nc"][rows])[0]),
            "log10 signal energy (above noise)": float(spearmanr(v, eff)[0])}
    order = np.argsort(-spec)
    out["top20_most_specific"] = [[tg[k], grp[k], round(float(spec[k]), 3)] for k in order[:20]]
    out["top20_least_specific"] = [[tg[k], grp[k], round(float(spec[k]), 3)] for k in order[::-1][:20]]
    # strong responders only (effect above median), to separate specificity from effect size
    big = eff > np.median(eff)
    out["groups_large_effect_only"] = {gname: [float(np.median(spec[big & (grp == gname)])), int((big & (grp == gname)).sum())]
                                       for gname in ["TF", "essential (non-TF)", "other"]}
    json.dump(out, open(OUT / "specificity.json", "w"), indent=1)
    np.savez(OUT / "specificity_vectors.npz", spec=spec, uniq=uniq, grp=grp, tg=tg)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("top")}, indent=1))


# ----------------------------------------------------------------------------- biology check
def promoters(keep):
    nm2sym = {}
    with gzip.open(RAW / "refGene.txt.gz", "rt") as fh:
        for line in fh:
            f = line.split("\t"); nm2sym.setdefault(f[1], f[12])
    lut = np.full(256, 4, np.int8)
    for i, c in enumerate("ACGT"):
        lut[ord(c)] = i; lut[ord(c.lower())] = i
    out, name, buf = {}, None, []

    def flush():
        if name is None:
            return
        acc, loc = name.split()[0], name.split()[1]
        nm = "_".join(acc.split("_")[:2]); sym = nm2sym.get(nm); chrom = loc.split(":")[0]; seq = "".join(buf)
        if nm.startswith("NM_") and sym in keep and sym not in out and "_" not in chrom and len(seq) == 1000:
            a, b = loc.split(":")[1].split("-"); a, b = int(a) - 1, int(b)
            strand = "+" if acc.endswith("_f") else "-"
            win = (b - 1000, b + 500) if strand == "+" else (a - 500, a + 1000)  # TSS -1 kb / +0.5 kb
            out[sym] = (lut[np.frombuffer(seq.encode(), np.uint8)], chrom, win)
    with gzip.open(RAW / "upstream1000.fa.gz", "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                flush(); name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    flush()
    return out


def chip_k562(tfs):
    q = [("type", "File"), ("status", "released"), ("assembly", "GRCh38"), ("format", "json"), ("limit", "all"),
         ("assay_title", "TF ChIP-seq"), ("biosample_ontology.term_name", "K562"), ("file_format", "bed"),
         ("field", "href"), ("field", "file_size"), ("field", "target.label")]
    best = {}
    for ot, rank in [("conservative IDR thresholded peaks", 0), ("IDR thresholded peaks", 1)]:
        url = "https://www.encodeproject.org/search/?" + urllib.parse.urlencode(q + [("output_type", ot)])
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=120) as r:
            for f in json.load(r)["@graph"]:
                t = (f.get("target") or {}).get("label", "").upper()
                if t in tfs and (t not in best or (rank, -f.get("file_size", 0)) < best[t][:2]):
                    best[t] = (rank, -f.get("file_size", 0), f["href"])
    peaks = {}
    (RAW / "encode").mkdir(exist_ok=True)
    for t, (_, _, href) in best.items():
        p = RAW / "encode" / f"K562_{t}.bed.gz"
        if not p.exists():
            urllib.request.urlretrieve("https://www.encodeproject.org" + href, p)
        dd = {}
        with gzip.open(p, "rt") as fh:
            for line in fh:
                c, s, e = line.split("\t")[:3]; dd.setdefault(c, []).append((int(s), int(e)))
        peaks[t] = {c: np.array(sorted(v)) for c, v in dd.items()}
    return peaks


def biology():
    from bond.pwm import parse_jaspar, logodds, scan_energies
    from bond.occupancy import sigmoid
    d = load(); X, P, var, col = d["X"], d["P"], d["var"], d["col"]
    dz = np.load(OUT / "decomp.npz"); u, V = dz["u"], dz["V"]
    rows = np.nonzero(P)[0]; tg = d["tgt"][rows]
    J = {m["name"].upper(): m for m in parse_jaspar(RAW / "jaspar2024_vert_nr.txt") if "::" not in m["name"]}
    tf_rows = {}
    for k, t in enumerate(tg):
        if t in J and t not in tf_rows:
            tf_rows[t] = rows[k]
    t0 = time.time()
    peaks = chip_k562(set(tf_rows))
    prom = promoters(set(var))
    gidx = np.array([i for i, g in enumerate(var) if g in prom]); gnames = var[gidx]
    seqs = np.stack([prom[g][0] for g in gnames])
    bg = np.bincount(seqs.ravel(), minlength=5)[:4].astype(float); bg = (bg + bg[::-1]) / 2; bg /= bg.sum()
    print(f"ChIP files {len(peaks)}  TFs perturbed {len(tf_rows)}  promoters {len(gidx)}  {time.time() - t0:.0f}s", flush=True)

    def bound_vec(pk):
        out = np.zeros(len(gnames), bool)
        cache = {c: (a[:, 0], np.maximum.accumulate(a[:, 1])) for c, a in pk.items()}
        for k, g in enumerate(gnames):
            _, c, (s, e) = prom[g]
            if c in cache:
                st, pm = cache[c]; j = np.searchsorted(st, e) - 1
                out[k] = j >= 0 and pm[j] > s
        return out
    res = {"chip": [], "motif": []}
    for t, r_ in tf_rows.items():
        x = X[r_][gidx]
        xg = X[r_] - (X[r_] @ u) * u
        rr = (xg - (xg @ V.T) @ V)[gidx]
        selfmask = gnames != t
        top_n = int(0.01 * selfmask.sum())

        def stats(lab):
            out = {}
            for nm, v in [("raw", np.abs(x)), ("residual", np.abs(rr))]:
                vv, ll = v[selfmask], lab[selfmask]
                out[f"AUROC_{nm}"] = float(roc_auc_score(ll, vv))
                top = np.zeros(len(vv), bool); top[np.argsort(-vv)[:top_n]] = True
                a, b_, c_, d_ = (top & ll).sum() + .5, (~top & ll).sum() + .5, (top & ~ll).sum() + .5, (~top & ~ll).sum() + .5
                out[f"OR_top1pct_{nm}"] = float(a * d_ / (b_ * c_))
                out[f"P(bound|top1pct)_{nm}"] = float(ll[top].mean())
            out["base_rate"] = float(lab[selfmask].mean())
            return out
        if t in peaks:
            lab = bound_vec(peaks[t])
            if 20 <= lab[selfmask].sum() <= selfmask.sum() - 20:
                res["chip"].append({"tf": t, **stats(lab)})
        W = logodds(J[t]["pfm"], bg)
        s = np.random.default_rng(0).choice(4, size=(3000, 100 + W.shape[0] - 1), p=bg).astype(np.int8)
        et = float(np.quantile(scan_energies(s, W)[0][:, 0, :].ravel(), 1e-4))
        occ = sigmoid(et - 2 - scan_energies(seqs, W)[0]).sum(axis=(1, 2))
        lab = occ >= np.quantile(occ[selfmask], 0.9)  # top-10% motif occupancy = 'motif-carrying'
        res["motif"].append({"tf": t, **stats(lab)})
    summ = {}
    for kind, L in res.items():
        if not L:
            continue
        s_ = {}
        for m in ["AUROC", "OR_top1pct", "P(bound|top1pct)"]:
            a = np.array([z[f"{m}_raw"] for z in L]); b = np.array([z[f"{m}_residual"] for z in L])
            s_[m] = {"raw_mean": float(a.mean()), "raw_median": float(np.median(a)), "residual_mean": float(b.mean()),
                     "residual_median": float(np.median(b)), "frac_TFs_residual_higher": float((b > a).mean()),
                     "wilcoxon_p": float(wilcoxon(b, a).pvalue)}
        s_["base_rate_mean"] = float(np.mean([z["base_rate"] for z in L])); s_["n_TFs"] = len(L)
        summ[kind] = s_
    json.dump({"summary": summ, "per_tf": res}, open(OUT / "biology.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    {"decompose": decompose, "heldout": heldout, "specificity": specificity, "biology": biology}[sys.argv[1]]()
