"""Test 2 - one real cell type (K562): binding (ENCODE ChIP-seq) and function (Perturb-seq).

Stage 'features' (chunked, cached per TF in data/processed/test2/<TF>.npz):
  scan the 1-kb upstream promoter of every Perturb-seq-measured gene with the TF's JASPAR PWM
  (log-odds -> approximate energy E, kT) and compute per-gene scores:
    PWM_max, PWM_hits (FIMO-like p<1e-4), PWM_hits_in_ATAC (standard motif-in-peak filter),
    OCC (Boltzmann occupancy, no accessibility), OCC_ACC (occupancy with soft ATAC modulation),
    OCC_SUB_ACC (occupancy from sub-threshold sites only), at mu = E_thr + {-4,-2,0}.
  Default (pre-registered) mu = E_thr - 2.
Stage 'analyze': binding metrics, weak-site test, held-out-TF test, function metrics.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import h5py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bond import data as D
from bond.seq import background
from bond.pwm import logodds, scan_energies
from bond.occupancy import sigmoid
from bond.intervals import read_bed, overlap_any, position_signal, to_log_acc
from bond.metrics import auroc, auprc_ratio, summarize

ROOT = D.ROOT
MODEL = "jaspar"
COMMON = False


def cache_dir(model=None):
    m = model or MODEL
    return ROOT / ("data/processed/test2" if m == "jaspar" else f"data/processed/test2_{m}")
MU_OFFS = (-4, -2, 0)
DEFAULT = -2
FLOOR = 0.02


def setup():
    var = D.perturb_var_genes()
    P = D.promoters(keep_genes=set(var))
    bg = background(P["seqs"])
    atac = read_bed(D.other_file("K562", "atac"), signal_col=6)
    possig = position_signal(atac, P["chrom"], P["start"], P["end"], P["strand"], 1000)
    return P, bg, possig


def features(lo, hi):
    CACHE = cache_dir(); CACHE.mkdir(parents=True, exist_ok=True)
    P, bg, possig = setup()
    np.save(CACHE / "_genes.npy", P["symbols"])
    src = D.motif_source(MODEL)
    chips = D.chip_files_all("K562")
    tfs = sorted(t for t in chips if t in src)
    for tf in tfs[lo:hi]:
        out = CACHE / f"{tf}.npz"
        if out.exists():
            continue
        t0 = time.time()
        W = src[tf](bg)
        w = W.shape[0]
        Ethr = D.energy_threshold(W, bg)
        E, _, _ = scan_energies(P["seqs"], W)
        la = to_log_acc(possig, w, floor=FLOOR)
        inpeak = la > np.log(FLOOR) + 1e-6
        hit = E < Ethr
        f = dict(Ethr=Ethr, width=w,
                 PWM_max=-E.min(axis=(1, 2)),
                 PWM_hits=hit.sum(axis=(1, 2)).astype(np.float32),
                 PWM_hits_in_ATAC=(hit & inpeak[:, None, :]).sum(axis=(1, 2)).astype(np.float32))
        for off in MU_OFFS:
            mu = Ethr + off
            f[f"OCC_mu{off:+d}"] = sigmoid(mu - E).sum(axis=(1, 2))
            x = sigmoid(mu - E + la[:, None, :])
            f[f"OCC_ACC_mu{off:+d}"] = x.sum(axis=(1, 2))
            f[f"OCC_SUB_ACC_mu{off:+d}"] = np.where(hit, 0, x).sum(axis=(1, 2))
            f[f"OCC_SUB_mu{off:+d}"] = np.where(hit, 0, sigmoid(mu - E)).sum(axis=(1, 2))
        peaks = read_bed(chips[tf], signal_col=6)
        f["chip_bound"] = overlap_any(peaks, P["chrom"], P["start"], P["end"])
        np.savez_compressed(out, **f)
        print(f"{tf} {time.time() - t0:.1f}s bound={f['chip_bound'].sum()}", flush=True)


def load_cache():
    CACHE = cache_dir()
    genes = np.load(CACHE / "_genes.npy")
    names = sorted(p.stem for p in CACHE.glob("*.npz"))
    if COMMON:  # restrict to TFs that have both a JASPAR and a ProBound model
        other = {p.stem for p in cache_dir("probound" if MODEL == "jaspar" else "jaspar").glob("*.npz")}
        names = [n for n in names if n in other]
    F = {n: dict(np.load(CACHE / f"{n}.npz")) for n in names}
    return genes, F


def lr_fit_predict(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def analyze():
    genes, F = load_cache()
    P, bg, possig = setup()
    assert (P["symbols"] == genes).all()
    acc_mean = np.exp(to_log_acc(possig, 1, floor=FLOOR)).mean(1)
    G = len(genes)
    tfs = [t for t in sorted(F) if 20 <= F[t]["chip_bound"].sum() <= G - 20]
    d = f"{DEFAULT:+d}"
    score_names = ["PWM_max", "PWM_hits", "PWM_hits_in_ATAC", "ACC_only"] + \
        [f"OCC_mu{o:+d}" for o in MU_OFFS] + [f"OCC_ACC_mu{o:+d}" for o in MU_OFFS]

    def get(tf, s):
        return acc_mean if s == "ACC_only" else F[tf][s]

    R = {"n_genes": int(G), "n_tfs_total": len(F), "n_tfs_binding_eval": len(tfs),
         "chip_bound_frac_median": float(np.median([F[t]["chip_bound"].mean() for t in tfs])),
         "promoters_overlapping_ATAC_peak_frac": float((possig.max(1) > 0).mean())}
    # ---- (a) binding ----
    B = {s: {"AUROC": [], "AUPRC_ratio": []} for s in score_names}
    for tf in tfs:
        y = F[tf]["chip_bound"]
        for s in score_names:
            B[s]["AUROC"].append(auroc(y, get(tf, s)))
            B[s]["AUPRC_ratio"].append(auprc_ratio(y, get(tf, s)))
    R["binding"] = {s: {k: summarize(v) for k, v in B[s].items()} for s in score_names}
    R["binding_win_frac"] = {
        f"OCC_ACC_mu{d} > PWM_hits_in_ATAC (AUPRC)": float(np.mean(np.array(B[f"OCC_ACC_mu{d}"]["AUPRC_ratio"]) > np.array(B["PWM_hits_in_ATAC"]["AUPRC_ratio"]))),
        f"OCC_mu{d} > PWM_max (AUPRC)": float(np.mean(np.array(B[f"OCC_mu{d}"]["AUPRC_ratio"]) > np.array(B["PWM_max"]["AUPRC_ratio"]))),
        f"OCC_ACC_mu{d} > ACC_only (AUPRC)": float(np.mean(np.array(B[f"OCC_ACC_mu{d}"]["AUPRC_ratio"]) > np.array(B["ACC_only"]["AUPRC_ratio"]))),
    }
    # ---- (b) weak-site test: genes with zero thresholded hits ----
    W = {"OCC_SUB (all zero-hit genes)": [], "OCC_SUB (zero-hit, ATAC-open genes)": [],
         "ACC_only (zero-hit genes)": [], "OCC_SUB_ACC (zero-hit genes)": []}
    for tf in tfs:
        z = F[tf]["PWM_hits"] == 0
        y = F[tf]["chip_bound"]
        W["OCC_SUB (all zero-hit genes)"].append(auroc(y[z], F[tf][f"OCC_SUB_mu{d}"][z]))
        W["ACC_only (zero-hit genes)"].append(auroc(y[z], acc_mean[z]))
        W["OCC_SUB_ACC (zero-hit genes)"].append(auroc(y[z], F[tf][f"OCC_SUB_ACC_mu{d}"][z]))
        zo = z & (acc_mean > FLOOR * 1.5)
        if y[zo].sum() >= 10 and (~y[zo]).sum() >= 10:
            W["OCC_SUB (zero-hit, ATAC-open genes)"].append(auroc(y[zo], F[tf][f"OCC_SUB_mu{d}"][zo]))
    R["weak_sites_AUROC"] = {k: summarize(v) for k, v in W.items()}
    R["zero_hit_gene_frac_median"] = float(np.median([(F[t]["PWM_hits"] == 0).mean() for t in tfs]))
    R["bound_genes_with_zero_hits_frac_median"] = float(np.median(
        [((F[t]["PWM_hits"] == 0) & F[t]["chip_bound"]).sum() / F[t]["chip_bound"].sum() for t in tfs]))
    # ---- (c) held-out TFs: hub prior vs occupancy (5 TF folds) ----
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(tfs)), 5)
    Y = np.stack([F[t]["chip_bound"] for t in tfs], 1).astype(float)
    H = {"HUB_prior(train-TF binding freq)": [], f"OCC_ACC_mu{d}": [],
         "LR[hub+acc]": [], f"LR[hub+acc+logOCC_ACC]": []}
    Hp = {k: [] for k in H}
    for fo in folds:
        tr = np.setdiff1d(np.arange(len(tfs)), fo)
        hub = Y[:, tr].mean(1)
        sub = rng.choice(G, min(G, 3000), replace=False)
        Xa_tr, Xb_tr, ytr = [], [], []
        for j in tr:
            occ = np.log1p(F[tfs[j]][f"OCC_ACC_mu{d}"])
            # leave this TF out of the hub feature used for its own training rows
            hub_j = (Y[:, tr].sum(1) - Y[:, j]) / (len(tr) - 1)
            Xa_tr.append(np.c_[hub_j, acc_mean][sub]); Xb_tr.append(np.c_[hub_j, acc_mean, occ][sub])
            ytr.append(Y[sub, j])
        Xa_tr, Xb_tr, ytr = np.vstack(Xa_tr), np.vstack(Xb_tr), np.concatenate(ytr)
        for j in fo:
            y = Y[:, j].astype(bool); occ = np.log1p(F[tfs[j]][f"OCC_ACC_mu{d}"])
            pa = lr_fit_predict(Xa_tr, ytr, np.c_[hub, acc_mean])
            pb = lr_fit_predict(Xb_tr, ytr, np.c_[hub, acc_mean, occ])
            for k, s in [("HUB_prior(train-TF binding freq)", hub), (f"OCC_ACC_mu{d}", occ),
                         ("LR[hub+acc]", pa), ("LR[hub+acc+logOCC_ACC]", pb)]:
                H[k].append(auroc(y, s)); Hp[k].append(auprc_ratio(y, s))
    R["heldout_TF"] = {k: {"AUROC": summarize(H[k]), "AUPRC_ratio": summarize(Hp[k])} for k in H}
    R["heldout_TF"]["occupancy_adds_over_hub_frac_TFs(AUPRC)"] = float(
        np.mean(np.array(Hp["LR[hub+acc+logOCC_ACC]"]) > np.array(Hp["LR[hub+acc]"])))
    # ---- (d) function: Perturb-seq knockdown responses ----
    with h5py.File(D.PERTURB, "r") as f:
        gt = [x.decode() for x in f["obs/gene_transcript"][:]]
        fe = f["obs/fold_expr"][:]
        var = D.perturb_var_genes()
        col = {g: i for i, g in enumerate(var)}
        gcols = np.array([col[g] for g in genes])
        rows = {}
        for i, s in enumerate(gt):
            g = s.split("_")[1]
            if g in F and fe[i] == fe[i] and fe[i] < 0.85:
                if g not in rows or fe[i] < fe[rows[g]]:
                    rows[g] = i
        X = np.nan_to_num(f["X"][:], nan=0.0, posinf=0.0, neginf=0.0)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Xg = Xz[:, gcols]
    pert_gene = np.array([s.split("_")[1] for s in gt])
    ZTHR = 0.5
    fn_names = score_names + ["ChIP_bound", "Coexpr_LOO(|r| across other perturbations)"]
    Fm = {s: {"AUROC": [], "AUPRC_ratio": []} for s in fn_names}
    Dm = {s: {"AUROC": [], "AUPRC_ratio": []} for s in fn_names if s != "ChIP_bound"}
    ov = {"P(resp|bound)": [], "P(resp|unbound)": [], "P(bound|resp)": [], "P(bound)": [], "odds_ratio": []}
    fn_tfs = []
    for tf, i in sorted(rows.items()):
        z = X[i, gcols]
        y = np.abs(z) >= ZTHR
        y[genes == tf] = False
        if y.sum() < 20:
            continue
        fn_tfs.append(tf)
        b = F[tf]["chip_bound"]
        coex = np.full(len(genes), 0.0)
        if tf in col:
            ex = pert_gene == tf  # drop the TF's own knockdown rows (leave-TF-out)
            v = Xz[:, col[tf]]
            dot = v @ Xg - v[ex] @ Xg[ex]
            coex = np.abs(dot) / (len(gt) - ex.sum())
        for s in fn_names:
            sc = b.astype(float) if s == "ChIP_bound" else (coex if s.startswith("Coexpr") else get(tf, s))
            Fm[s]["AUROC"].append(auroc(y, sc)); Fm[s]["AUPRC_ratio"].append(auprc_ratio(y, sc))
            if s != "ChIP_bound":
                yd = y & b
                if yd.sum() >= 5:
                    Dm[s]["AUROC"].append(auroc(yd, sc)); Dm[s]["AUPRC_ratio"].append(auprc_ratio(yd, sc))
        ov["P(resp|bound)"].append(y[b].mean()); ov["P(resp|unbound)"].append(y[~b].mean())
        ov["P(bound|resp)"].append(b[y].mean()); ov["P(bound)"].append(b.mean())
        a, bb, c, dd = (y & b).sum() + .5, (~y & b).sum() + .5, (y & ~b).sum() + .5, (~y & ~b).sum() + .5
        ov["odds_ratio"].append(a * dd / (bb * c))
    R["function"] = {"n_tfs": len(fn_tfs), "tfs": fn_tfs, "z_threshold": ZTHR,
                     "scores": {s: {k: summarize(v) for k, v in Fm[s].items()} for s in fn_names},
                     "binding_function_overlap_median": {k: float(np.median(v)) for k, v in ov.items()},
                     "direct_proxy(bound&responsive)": {s: {k: summarize(v) for k, v in Dm[s].items()} for s in Dm}}
    R["motif_model"] = MODEL; R["common_tf_set"] = COMMON
    tag = "" if (MODEL == "jaspar" and not COMMON) else f"_{MODEL}" + ("_common" if COMMON else "")
    json.dump(R, open(ROOT / f"results/test2_k562{tag}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in R.items() if k != "function"}, indent=1)[:4000])
    print(json.dumps(R["function"]["binding_function_overlap_median"]))
    for s in fn_names:
        print("FN", s, R["function"]["scores"][s]["AUROC"]["mean"], R["function"]["scores"][s]["AUPRC_ratio"]["mean"])


if __name__ == "__main__":
    if "--model" in sys.argv:
        MODEL = sys.argv[sys.argv.index("--model") + 1]
    COMMON = "--common" in sys.argv
    if sys.argv[1] == "features":
        lo, hi = map(int, sys.argv[2].split(":"))
        features(lo, hi)
    else:
        analyze()
