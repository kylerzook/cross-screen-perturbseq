"""Are regulatory effects fixed or context-dependent?  Norman et al. 2019 (K562, CRISPRa single + double).

Part 1 (other genes as context): does activating C change what activating A does?
  * effects = perturbation mean - control mean (log-normalised expression, HVGs; perturbed genes masked)
  * every perturbation's cells (and the controls) are split into two random halves A/B, so that
    predictions built from half A are scored against half B -> noise in the target is independent,
    and 'replicate' = half-A pair profile vs half-B pair profile gives the ceiling.
  * interaction I_AC = D_AC - (D_A + D_C); noise-corrected interaction strength; gene-level z (analytic SE)
  * models for the pair: additive, fitted scaling (a*D_A + c*D_C, Norman-style), global saturation
    (isotonic, fit on training pairs), dominant single; low-rank 'modulator' model for HELD-OUT pairs.
  * structure: modulator scores vs permutation, low-rank structure vs null, target rewiring vs split-half.
Part 2 (cell state as context): does a perturbation's effect differ across cell-cycle phase or
  integrated-stress-response (ISR) level, beyond a bin-label permutation null?

Usage: python experiments/context_dependence.py prep | part1 | part2
"""
import json, sys, time
from pathlib import Path
import numpy as np
import anndata as ad
from scipy import sparse
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/norman/norman2019.h5ad"
OUT = ROOT / "results/context"; OUT.mkdir(parents=True, exist_ok=True)
N_HVG = 2000
rng = np.random.default_rng(0)

S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1",
           "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3",
           "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1",
           "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M_GENES = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67",
             "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E",
             "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2",
             "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE",
             "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]
ISR_GENES = ["DDIT3", "ATF3", "ATF4", "ASNS", "TRIB3", "CHAC1", "SESN2", "VEGFA", "PSAT1", "PHGDH", "SLC7A11",
             "CEBPB", "PPP1R15A", "XBP1", "HSPA5", "HERPUD1", "DDIT4", "STC2", "CARS1", "WARS1", "SARS1", "GPT2"]


# ------------------------------------------------------------------------------------------- prep
def prep():
    t0 = time.time()
    a = ad.read_h5ad(DATA, backed="r")
    obs = a.obs.copy()
    var = np.array(a.var_names)
    pert = obs["perturbation"].astype(str).values
    n = a.n_obs
    # pass 1: library size + per-gene mean / mean-square of log-normalised expression (chunked)
    sums = np.zeros(len(var)); sq = np.zeros(len(var)); lib = np.zeros(n)
    for s in range(0, n, 10000):
        X = a.X[s:s + 10000]; X = sparse.csr_matrix(X)
        lib[s:s + 10000] = np.asarray(X.sum(1)).ravel()
        Xn = X.multiply(1e4 / lib[s:s + 10000][:, None]).tocsr(); Xn.data = np.log1p(Xn.data)
        sums += np.asarray(Xn.sum(0)).ravel(); sq += np.asarray(Xn.multiply(Xn).sum(0)).ravel()
    mu = sums / n; vv = sq / n - mu ** 2
    expressed = mu > 0.05
    disp = np.where(expressed, vv / (mu + 1e-9), 0)
    hvg = set(np.argsort(-disp)[:N_HVG])
    pert_genes = set(g for p in np.unique(pert) if p != "control" for g in p.split("_"))
    extra = set(np.nonzero(np.isin(var, list(pert_genes | set(S_GENES) | set(G2M_GENES) | set(ISR_GENES))))[0])
    keep = np.array(sorted(hvg | extra))
    # pass 2: dense log-normalised matrix on kept genes
    M = np.zeros((n, len(keep)), np.float32)
    for s in range(0, n, 10000):
        X = sparse.csr_matrix(a.X[s:s + 10000])
        Xn = X.multiply(1e4 / lib[s:s + 10000][:, None]).tocsr(); Xn.data = np.log1p(Xn.data)
        M[s:s + 10000] = Xn[:, keep].toarray()
    half = rng.integers(0, 2, n)  # random half assignment for split-half analyses
    np.savez(OUT / "prep.npz", M=M, genes=var[keep], pert=pert, half=half, umi=lib,
             hvg_mask=np.isin(keep, list(hvg)))
    print(f"prep done {time.time() - t0:.0f}s cells={n} genes kept={len(keep)} perts={len(np.unique(pert))}", flush=True)


def load():
    d = dict(np.load(OUT / "prep.npz", allow_pickle=True))
    return d


def means(M, idx):
    x = M[idx]
    return x.mean(0), x.var(0) / max(len(idx), 1), len(idx)


# ------------------------------------------------------------------------------------------- part 1
def part1():
    t0 = time.time()
    d = load(); M, genes, pert, half = d["M"], d["genes"], d["pert"], d["half"]
    hv = d["hvg_mask"]
    perts = [p for p in np.unique(pert) if p != "control"]
    singles = [p for p in perts if "_" not in p]
    pairs = [p for p in perts if "_" in p and all(g in singles for g in p.split("_"))]
    ctrl = np.nonzero(pert == "control")[0]
    idx = {p: np.nonzero(pert == p)[0] for p in perts}
    gidx = {g: i for i, g in enumerate(genes)}
    # summary stats: full, half 0, half 1
    def stats(cells):
        return {h: means(M, cells if h == "all" else cells[half[cells] == h]) for h in ["all", 0, 1]}
    C = stats(ctrl)
    S = {p: stats(idx[p]) for p in singles + pairs}
    eff = lambda p, h: S[p][h][0] - C[h][0]
    effvar = lambda p, h: S[p][h][1] + C[h][1]

    res = {"n_cells": int(len(pert)), "n_control": int(len(ctrl)), "n_singles": len(singles), "n_pairs": len(pairs),
           "n_genes_scored": int(hv.sum()), "pairs": {}}
    iso_x, iso_y = [], []
    # ---- per pair: interaction, noise, model comparison (half 0 builds the prediction, half 1 is the target)
    for p in pairs:
        A, Cc = p.split("_")
        mask = hv.copy()
        for g in (A, Cc):
            if g in gidx:
                mask[gidx[g]] = False
        DA, DC, DAC = eff(A, "all")[mask], eff(Cc, "all")[mask], eff(p, "all")[mask]
        I = DAC - DA - DC
        varI = (S[p]["all"][1] + S[A]["all"][1] + S[Cc]["all"][1] + C["all"][1])[mask]  # ctrl enters with coef +1
        noiseI = varI.sum()
        sigAC = max((DAC ** 2).sum() - effvar(p, "all")[mask].sum(), 1e-9)
        z = I / np.sqrt(varI + 1e-12)
        # BH FDR on two-sided normal p-values
        from scipy.stats import norm
        pv = 2 * norm.sf(np.abs(z)); o = np.argsort(pv); m_ = len(pv)
        bh = np.minimum.accumulate((pv[o] * m_ / (np.arange(m_) + 1))[::-1])[::-1]
        q = np.empty_like(bh); q[o] = bh
        # split-half evaluation
        a0, c0, ac0 = eff(A, 0)[mask], eff(Cc, 0)[mask], eff(p, 0)[mask]
        a1, c1, ac1 = eff(A, 1)[mask], eff(Cc, 1)[mask], eff(p, 1)[mask]
        X2 = np.c_[a0, c0]; coef = np.linalg.lstsq(X2, ac0, rcond=None)[0]
        dom = a0 if (a0 ** 2).sum() >= (c0 ** 2).sum() else c0
        preds = {"additive": a0 + c0, "scaled (a*A + c*C)": X2 @ coef, "dominant single": dom}
        corr = lambda u, v: float(np.corrcoef(u, v)[0, 1])
        r2 = lambda y, yh: float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        rec = {"interaction_strength_noise_corrected": float(max((I ** 2).sum() - noiseI, 0) / sigAC),
               "interaction_energy_over_noise": float((I ** 2).sum() / noiseI),
               "n_sig_genes_q05": int((q < 0.05).sum()), "pair_effect_snr": float((DAC ** 2).sum() / effvar(p, "all")[mask].sum()),
               "scaling_coefs": [float(coef[0]), float(coef[1])], "replicate_corr": corr(ac0, ac1), "replicate_R2": r2(ac1, ac0),
               "models_corr": {k: corr(v, ac1) for k, v in preds.items()},
               "models_R2": {k: r2(ac1, v) for k, v in preds.items()},
               "n_cells": [int(len(idx[A])), int(len(idx[Cc])), int(len(idx[p]))]}
        res["pairs"][p] = rec
        iso_x.append((a0 + c0)); iso_y.append(ac0)
    # ---- calibration: control split (half0 - half1) z-scores should be N(0,1)
    zc = (C[0][0] - C[1][0])[hv] / np.sqrt(C[0][1][hv] + C[1][1][hv])
    res["calibration_control_split_z_sd"] = float(zc.std())
    res["calibration_control_split_frac_absz_gt3"] = float((np.abs(zc) > 3).mean())

    # ---- held-out pairs: saturation (global isotonic) and low-rank modulator model, 5-fold over pairs
    from sklearn.isotonic import IsotonicRegression
    plist = list(res["pairs"]); folds = np.array_split(rng.permutation(len(plist)), 5)
    ho = {k: [] for k in ["additive", "global saturation (isotonic on A+C)", "modulator low-rank (K=1)",
                          "modulator low-rank (K=3)", "modulator low-rank (K=5)", "replicate ceiling"]}
    Imat = {}
    for p in plist:
        A, Cc = p.split("_"); mask = hv.copy()
        for g in (A, Cc):
            if g in gidx:
                mask[gidx[g]] = False
        Imat[p] = (eff(p, 0) - eff(A, 0) - eff(Cc, 0)) * mask, mask
    for fo in folds:
        test = [plist[k] for k in fo]; train = [p for p in plist if p not in test]
        xs = np.concatenate([(eff(p.split("_")[0], 0) + eff(p.split("_")[1], 0))[Imat[p][1]] for p in train])
        ys = np.concatenate([eff(p, 0)[Imat[p][1]] for p in train])
        sub = rng.choice(len(xs), min(len(xs), 200000), replace=False)
        iso = IsotonicRegression(out_of_bounds="clip").fit(xs[sub], ys[sub])
        # modulator model: I_AC ~ (w_A + w_C) V, alternating least squares on training pairs
        gl = sorted({g for p in train for g in p.split("_")})
        gi = {g: k for k, g in enumerate(gl)}
        Itr = np.stack([Imat[p][0] for p in train])
        Z = np.zeros((len(train), len(gl)))
        for r_, p in enumerate(train):
            for g in p.split("_"):
                Z[r_, gi[g]] = 1
        fits = {}
        for K in (1, 3, 5):
            U, s_, Vt = np.linalg.svd(Itr, full_matrices=False)
            V = Vt[:K]
            for _ in range(30):
                Wg = np.linalg.lstsq(Z, Itr @ V.T, rcond=None)[0]          # per-gene modulator loadings
                Lw = Z @ Wg
                V = np.linalg.lstsq(Lw, Itr, rcond=None)[0]
            fits[K] = (Wg, V)
        for p in test:
            A, Cc = p.split("_"); m_ = Imat[p][1]
            y = eff(p, 1)[m_]; base = (eff(A, 0) + eff(Cc, 0))[m_]
            ho["additive"].append(float(np.corrcoef(base, y)[0, 1]))
            ho["global saturation (isotonic on A+C)"].append(float(np.corrcoef(iso.predict(base), y)[0, 1]))
            for K, (Wg, V) in fits.items():
                if A in gi and Cc in gi:
                    inter = ((Wg[gi[A]] + Wg[gi[Cc]]) @ V)[m_]
                    ho[f"modulator low-rank (K={K})"].append(float(np.corrcoef(base + inter, y)[0, 1]))
                else:
                    ho[f"modulator low-rank (K={K})"].append(np.nan)
            ho["replicate ceiling"].append(float(np.corrcoef(eff(p, 0)[m_], y)[0, 1]))
    res["heldout_pairs_corr"] = {k: [float(np.nanmean(v)), float(np.nanmedian(v)), int(np.sum(~np.isnan(v)))] for k, v in ho.items()}
    res["heldout_pairs_paired_gain_vs_additive"] = {
        k: [float(np.nanmean(np.array(v) - np.array(ho["additive"]))), float(np.nanmean((np.array(v) - np.array(ho["additive"])) > 0))]
        for k, v in ho.items() if k not in ("additive", "replicate ceiling")}

    # ---- structure: modulator scores vs permutation
    strength = {p: r["interaction_strength_noise_corrected"] for p, r in res["pairs"].items()}
    gl = sorted({g for p in strength for g in p.split("_")})
    def mod_scores(st):
        return {g: np.mean([v for p, v in st.items() if g in p.split("_")]) for g in gl}
    ms = mod_scores(strength)
    obs_sd = np.std(list(ms.values()))
    null = []
    keys = list(strength); vals = np.array([strength[k] for k in keys])
    for _ in range(500):
        perm = dict(zip(keys, rng.permutation(vals)))
        null.append(np.std(list(mod_scores(perm).values())))
    res["modulator_score_sd_observed"] = float(obs_sd)
    res["modulator_score_sd_null_mean"] = float(np.mean(null))
    res["modulator_score_perm_p"] = float((np.array(null) >= obs_sd).mean())
    res["top_modulators"] = sorted([[g, round(float(v), 3), sum(g in p.split("_") for p in strength)] for g, v in ms.items()],
                                   key=lambda x: -x[1])[:10]
    # ---- low-rank structure of the interaction matrix vs per-gene permutation null
    Iz = []
    for p in plist:
        A, Cc = p.split("_"); m_ = Imat[p][1]
        varI = (S[p]["all"][1] + S[A]["all"][1] + S[Cc]["all"][1] + C["all"][1])
        Iz.append(np.where(m_, (eff(p, "all") - eff(A, "all") - eff(Cc, "all")) / np.sqrt(varI + 1e-12), 0)[hv])
    Iz = np.stack(Iz)
    sv = np.linalg.svd(Iz, compute_uv=False)
    Ip = np.stack([rng.permutation(Iz[:, j]) for j in range(Iz.shape[1])], 1)
    svn = np.linalg.svd(Ip, compute_uv=False)
    res["interaction_matrix_top5_singular_values"] = sv[:5].round(1).tolist()
    res["interaction_matrix_null_top5_singular_values"] = svn[:5].round(1).tolist()
    res["interaction_n_components_above_null_top"] = int((sv > svn[0]).sum())
    res["interaction_variance_top1_share_obs_vs_null"] = [float(sv[0] ** 2 / (sv ** 2).sum()), float(svn[0] ** 2 / (svn ** 2).sum())]
    # target sensitivity: genes most often significantly interacting
    sig_counts = (np.abs(Iz) > 4).sum(0)
    hv_genes = genes[hv]
    res["top_interacting_target_genes"] = [[str(hv_genes[k]), int(sig_counts[k])] for k in np.argsort(-sig_counts)[:15]]
    # ---- rewiring: A's targets alone vs A's effect in C's context, compared with split-half reproducibility
    rw = {"jaccard_alone_vs_inC": [], "jaccard_splithalf_alone": [], "frac_sign_flips_inC": [], "frac_sign_flips_splithalf": []}
    for p in plist:
        A, Cc = p.split("_"); m_ = hv.copy()
        for g in (A, Cc):
            if g in gidx:
                m_[gidx[g]] = False
        for X_, Y_ in [(A, Cc), (Cc, A)]:
            zA = eff(X_, "all") / np.sqrt(effvar(X_, "all")); tA = m_ & (np.abs(zA) > 4)
            inC = eff(p, "all") - eff(Y_, "all"); zin = inC / np.sqrt(S[p]["all"][1] + S[Y_]["all"][1] + 1e-12); tin = m_ & (np.abs(zin) > 4)
            z0 = eff(X_, 0) / np.sqrt(effvar(X_, 0)); z1 = eff(X_, 1) / np.sqrt(effvar(X_, 1))
            t0_, t1_ = m_ & (np.abs(z0) > 4), m_ & (np.abs(z1) > 4)
            if tA.sum() >= 5:
                rw["jaccard_alone_vs_inC"].append((tA & tin).sum() / max((tA | tin).sum(), 1))
                both = tA & tin; rw["frac_sign_flips_inC"].append(float((np.sign(zA[both]) != np.sign(zin[both])).mean()) if both.any() else np.nan)
            if t0_.sum() >= 5:
                rw["jaccard_splithalf_alone"].append((t0_ & t1_).sum() / max((t0_ | t1_).sum(), 1))
                both = t0_ & t1_; rw["frac_sign_flips_splithalf"].append(float((np.sign(z0[both]) != np.sign(z1[both])).mean()) if both.any() else np.nan)
    res["rewiring"] = {k: [float(np.nanmean(v)), float(np.nanmedian(v)), int(np.sum(~np.isnan(v)))] for k, v in rw.items()}
    # ---- summaries over pairs
    P = res["pairs"]
    q = lambda v: [float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75))]
    res["summary"] = {
        "interaction_strength_noise_corrected_q25_med_q75": q([r["interaction_strength_noise_corrected"] for r in P.values()]),
        "interaction_energy_over_noise_q25_med_q75": q([r["interaction_energy_over_noise"] for r in P.values()]),
        "frac_pairs_interaction_energy_gt_2x_noise": float(np.mean([r["interaction_energy_over_noise"] > 2 for r in P.values()])),
        "n_sig_interaction_genes_q05_q25_med_q75": q([r["n_sig_genes_q05"] for r in P.values()]),
        "frac_pairs_with_any_sig_gene": float(np.mean([r["n_sig_genes_q05"] > 0 for r in P.values()])),
        "replicate_corr_q25_med_q75": q([r["replicate_corr"] for r in P.values()]),
        "models_corr_median": {k: float(np.median([r["models_corr"][k] for r in P.values()])) for k in P[plist[0]]["models_corr"]},
        "models_R2_median": {k: float(np.median([r["models_R2"][k] for r in P.values()])) for k in P[plist[0]]["models_R2"]},
        "scaling_coef_sum_median": float(np.median([sum(r["scaling_coefs"]) for r in P.values()])),
        "frac_pairs_scaling_sum_lt_0.8": float(np.mean([sum(r["scaling_coefs"]) < 0.8 for r in P.values()])),
        "frac_pairs_scaling_sum_gt_1.2": float(np.mean([sum(r["scaling_coefs"]) > 1.2 for r in P.values()])),
        "spearman_interaction_strength_vs_pair_snr": float(spearmanr([r["interaction_strength_noise_corrected"] for r in P.values()],
                                                                     [r["pair_effect_snr"] for r in P.values()])[0])}
    res["most_interacting_pairs"] = sorted([[p, round(r["interaction_strength_noise_corrected"], 3), r["n_sig_genes_q05"]] for p, r in P.items()],
                                           key=lambda x: -x[1])[:10]
    res["seconds"] = time.time() - t0
    json.dump(res, open(OUT / "part1.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=1))


# ------------------------------------------------------------------------------------------- part 2
def part2():
    t0 = time.time()
    d = load(); M, genes, pert, half, hv = d["M"], d["genes"], d["pert"], d["half"], d["hvg_mask"]
    gidx = {g: i for i, g in enumerate(genes)}
    def score(gs):
        ii = [gidx[g] for g in gs if g in gidx]
        Z = M[:, ii]; Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
        return Z.mean(1), len(ii)
    s_sc, ns = score(S_GENES); g_sc, ng = score(G2M_GENES); isr, ni = score(ISR_GENES)
    phase = np.where((s_sc < 0) & (g_sc < 0), "G1", np.where(s_sc > g_sc, "S", "G2M"))
    isr_bin = np.where(isr > np.median(isr), "ISR-high", "ISR-low")
    ctrl = pert == "control"
    perts = [p for p in np.unique(pert) if p != "control" and (pert == p).sum() >= 300]
    state_genes = set(S_GENES) | set(G2M_GENES) | set(ISR_GENES)
    mask_base = hv & ~np.isin(genes, list(state_genes))  # exclude the genes that DEFINE the states
    out = {"n_genes_scored": int(mask_base.sum()), "phase_counts_control": {k: int(((phase == k) & ctrl).sum()) for k in ["G1", "S", "G2M"]},
           "genes_used": {"S": ns, "G2M": ng, "ISR": ni}, "by_state": {}}
    Msub = M[:, mask_base]; sub_genes = genes[mask_base]; sgi = {g: i for i, g in enumerate(sub_genes)}
    cc = np.nonzero(ctrl)[0]; Xc = Msub[cc]; Xc2 = Xc ** 2
    NPERM = 50

    def bin_stats(X, X2, lab, levels):
        H = np.stack([(lab == lv) for lv in levels], 1).astype(np.float32)  # n x L
        n_ = H.sum(0)
        mean = (H.T @ X) / n_[:, None]; var = (H.T @ X2) / n_[:, None] - mean ** 2
        return mean, var / n_[:, None], n_
    for state_name, lab in [("cell-cycle phase", phase), ("ISR level", isr_bin)]:
        levels = sorted(set(lab))
        recs = []
        for p in perts:
            keep_g = np.ones(len(sub_genes), bool)
            for g in p.split("_"):
                if g in sgi:
                    keep_g[sgi[g]] = False
            pc = np.nonzero(pert == p)[0]; Xp = Msub[pc]; Xp2 = Xp ** 2
            lp, lc = lab[pc], lab[cc]
            if min(np.sum(lp == lv) for lv in levels) < 30 or min(np.sum(lc == lv) for lv in levels) < 30:
                continue

            def het_of(lp_, lc_):
                mp, vp, _ = bin_stats(Xp, Xp2, lp_, levels); mc, vc, _ = bin_stats(Xc, Xc2, lc_, levels)
                E = (mp - mc)[:, keep_g]; Vv = (vp + vc)[:, keep_g] + 1e-9
                pooled = (E / Vv).sum(0) / (1 / Vv).sum(0)
                return E, Vv, float((((E - pooled) ** 2) / Vv).sum())
            E, Vv, het = het_of(lp, lc)
            nullh = [het_of(rng.permutation(lp), rng.permutation(lc))[2] for _ in range(NPERM)]
            pval = (np.sum(np.array(nullh) >= het) + 1) / (NPERM + 1)
            pooled_z = E.mean(0) / np.sqrt(Vv.mean(0) / len(levels))
            top = np.argsort(-np.abs(pooled_z))[:50]
            cors = [float(np.corrcoef(E[i][top], E[j][top])[0, 1]) for i in range(len(levels)) for j in range(i + 1, len(levels))]
            strong = np.abs(E / np.sqrt(Vv)) > 4
            flips = denom = 0
            for i in range(len(levels)):
                for j in range(i + 1, len(levels)):
                    both = strong[i] & strong[j]; denom += int(both.sum()); flips += int((np.sign(E[i][both]) != np.sign(E[j][both])).sum())
            gain = [float(np.linalg.norm(E[i][top])) for i in range(len(levels))]
            recs.append({"pert": p, "het_over_null_mean": float(het / np.mean(nullh)), "perm_p": float(pval),
                         "between_state_corr_top50": float(np.mean(cors)), "sign_flips": flips, "both_strong": denom,
                         "effect_size_by_state_top50": dict(zip(levels, gain))})
        ps = np.array([r["perm_p"] for r in recs])
        o = np.argsort(ps); bh = np.minimum.accumulate((ps[o] * len(ps) / (np.arange(len(ps)) + 1))[::-1])[::-1]
        qv = np.empty_like(bh); qv[o] = np.minimum(bh, 1)
        for r, qq in zip(recs, qv):
            r["q"] = float(qq)
        ratio = [max(r["effect_size_by_state_top50"].values()) / max(min(r["effect_size_by_state_top50"].values()), 1e-9) for r in recs]
        out["by_state"][state_name] = {
            "n_perturbations_tested": len(recs), "n_permutations": NPERM,
            "frac_state_dependent_q<0.05": float(np.mean(qv < 0.05)),
            "frac_perm_p_at_minimum": float(np.mean(ps <= 1 / (NPERM + 1) + 1e-12)),
            "heterogeneity_over_null_median": float(np.median([r["het_over_null_mean"] for r in recs])),
            "between_state_corr_top50_median": float(np.median([r["between_state_corr_top50"] for r in recs])),
            "total_sign_flips_among_strong_targets": int(sum(r["sign_flips"] for r in recs)),
            "total_strong_in_both_states": int(sum(r["both_strong"] for r in recs)),
            "effect_size_max_over_min_state_median": float(np.median(ratio)),
            "most_state_dependent": sorted([[r["pert"], round(r["het_over_null_mean"], 2), round(r["q"], 3)] for r in recs], key=lambda x: -x[1])[:10],
            "per_perturbation": recs}
        print(state_name, json.dumps({k: v for k, v in out["by_state"][state_name].items() if k != "per_perturbation"}), flush=True)
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT / "part2.json", "w"), indent=1)


if __name__ == "__main__":
    {"prep": prep, "part1": part1, "part2": part2}[sys.argv[1]]()
