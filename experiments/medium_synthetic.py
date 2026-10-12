"""Medium-scale, more honest synthetic test of BOND (perturbation-aware vector field + binding prior).

What changed vs the toy (bond_dynamics.py), following the agreed next steps:
  * 80 genes: 16 TFs (have motifs), 8 non-TF regulators (no motif), 56 targets; ~140 edges;
    TF cascades of depth >= 3, repressors, multi-input targets.  In-house SERGIO-style Hill simulator
    (SERGIO itself was not used).
  * SNAPSHOT sampling: each cell is measured once, at an unknown time, with a noisy
    metabolic-labelling-style velocity estimate  v = (x(t) - x(t - 0.3)) / 0.3 + noise.  No trajectories.
  * Binding prior from NOISY sequences: a site is planted for a true TF->gene edge with prob p1 and
    for a non-edge with prob p0 (decoys); strong or weak sites; occupancy scanned with real JASPAR
    motifs.  'realistic' (p1=0.5, p0=0.15) is calibrated so ~25% of a TF's knockdown responders are
    bound (real K562: ~22%).
  * Knockdowns of regulators with heterogeneous, unknown efficiency U(0.05, 0.3).
  * Realism added after a first pass came out trivially easy (AUROC ~1.0): per-edge Hill coefficient
    n in {1..4} and threshold K ~ U(0.5, 2) (model uses one fixed Hill-2 basis -> misspecified),
    a hidden per-cell 'cell size' factor scaling all production (confounder), Poisson counts with
    variable library size + log-normalisation, velocity noise 1.0 x sd (swept 0.3-2.0), and 10
    UNMEASURED genes (1 TF, 1 non-TF regulator, 8 targets). Indirect pairs = reachable by a path of
    length >= 2 without a direct edge (paths through unmeasured genes count as indirect).
  * 10 seeds; lambda chosen by BIC for every Lasso-type method; prior strength and quality swept.

Model (BOND-V): for each gene i, regress velocity on regulator features
    v_i = b_i + sum_j W_ij phi(x_j) - g_i x_i,     phi(x) = x^2/(K^2+x^2), K = median(x_j)
  perturbation-aware: cells in which gene i itself was knocked down are dropped from gene i's
    equation (the interventional-likelihood trick of DCDI/GIES);
  prior: adaptive-lasso penalty factor 1 - alpha * p_ij for TF regulators (p = occupancy rank).
Baselines (re-implementations of the core idea, NOT the original code bases):
  PHOENIX-style (velocity + prior, interventions ignored), DCDI-style (static nodewise, known
  targets, no prior, no DAG constraint), CellOracle-style (motif-restricted base GRN + Bayesian ridge),
  GENIE3-style (extra-trees importance), |corr|, KD mean-difference (total effects), prior only.
Coverage / experimental design: start from wild type only and add knockdowns one at a time
  (pool of 20 regulators; 4 always held out) chosen at random, by predicted out-degree, or actively
  (max disagreement of predicted knockdown effects across a 3-model bootstrap ensemble).

Usage: python experiments/medium_synthetic.py ablate <seed> | coverage <seed> [strategies] | report
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LassoLarsIC, BayesianRidge
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bond.pwm import parse_jaspar, logodds, scan_energies
from bond.occupancy import sigmoid as np_sigmoid

OUT = ROOT / "results/medium_synthetic"
G, NTF, NREG = 80, 16, 8
TFS = list(range(NTF)); NONTF_REGS = list(range(NTF, NTF + NREG)); REGS = TFS + NONTF_REGS
N_WT, N_KD = 1500, 80
TAU, T_MAX, DT = 0.3, 5.0, 0.01
VEL_NOISE, PROC_NOISE, LIB = 1.0, 0.05, 20.0
VEL_NOISES = [0.3, 1.0, 2.0]
NOMINAL_KD = 0.15
PRIORS = {"uninformative": (0.15, 0.15), "realistic": (0.5, 0.15), "clean": (0.8, 0.05)}
ALPHAS = [0.0, 0.5, 0.8, 0.95]
SEEDS = list(range(10))
COV_STEPS = [0, 1, 2, 4, 8, 12, 20]
BG = np.array([0.29, 0.21, 0.21, 0.29])


# ------------------------------------------------------------------ ground truth + simulation
def make_network(rng):
    E = []
    for t in range(4, NTF):  # TF layer (DAG -> cascades)
        for j in rng.choice(t, rng.integers(1, 3), replace=False):
            E.append((t, int(j)))
    for r in NONTF_REGS:
        for j in rng.choice(NTF, rng.integers(1, 3), replace=False):
            E.append((r, int(j)))
    for g in range(NTF + NREG, G):
        k = rng.integers(1, 4); regs = set()
        while len(regs) < k:
            regs.add(int(rng.choice(TFS)) if rng.random() < 0.75 else int(rng.choice(NONTF_REGS)))
        E += [(g, j) for j in regs]
    A = np.zeros((G, G)); NH = np.full((G, G), 2.0); KH = np.ones((G, G))
    for i, j in E:
        A[i, j] = (1 if rng.random() < 0.7 else -1) * rng.uniform(0.8, 2.0)
        NH[i, j] = rng.integers(1, 5); KH[i, j] = rng.uniform(0.5, 2.0)
    return A, NH, KH


def simulate(rng, A, NH, KH, kd_target, eff):
    """Snapshot simulation. kd_target (N,) with -1 = wild type."""
    N = len(kd_target)
    gamma = rng.uniform(0.5, 1.0, G)
    u = rng.uniform(0.1, 3.0, (N, 4))  # cell-state heterogeneity: master-regulator input
    x = rng.uniform(0, 1.5, (N, G))
    s = np.ones((N, G))
    kd = kd_target >= 0
    s[np.nonzero(kd)[0], kd_target[kd]] = eff[kd_target[kd]]
    x = x * np.where(s < 1, s, 1)
    t_obs = rng.uniform(0.5, T_MAX, N)
    ii, jj = np.nonzero(A); a_e, n_e, k_e = A[ii, jj], NH[ii, jj], KH[ii, jj]
    size = rng.lognormal(0, 0.3, (N, 1))  # hidden confounder: scales all production
    M_scatter = np.eye(G)[ii]
    x_prev = np.zeros_like(x); x_now = np.zeros_like(x)
    steps = int(T_MAX / DT)
    for k in range(1, steps + 1):
        xe = x[:, jj]; h = xe ** n_e / (k_e ** n_e + xe ** n_e)
        contrib = np.where(a_e > 0, a_e * h, -a_e * (1 - h))
        prod = np.full((N, G), 0.05)
        prod += contrib @ M_scatter
        prod[:, :4] += u
        prod *= size
        x = np.maximum(x + (s * prod - gamma * x) * DT + PROC_NOISE * np.sqrt(DT) * rng.standard_normal((N, G)), 0)
        t = k * DT
        m1 = np.abs(t - (t_obs - TAU)) < DT / 2; x_prev[m1] = x[m1]
        m2 = np.abs(t - t_obs) < DT / 2; x_now[m2] = x[m2]
    v = (x_now - x_prev) / TAU
    lib = rng.lognormal(np.log(LIB), 0.4, (N, 1))
    counts = rng.poisson(x_now * lib)
    X = np.log1p(counts / lib * LIB) / np.log1p(LIB) * 2  # log-normalised, rescaled to ~[0, 4]
    # labelling velocity: true velocity mapped to the log scale (chain rule) + noise
    dlog = (2 / np.log1p(LIB)) * LIB / (1 + x_now * LIB)
    vt = v * dlog
    return X, vt, rng.standard_normal(vt.shape) * (vt.std(0) + 1e-6)


def make_data(seed, vel_noise=VEL_NOISE):
    f = OUT / f"data_v2_s{seed}.npz"
    if f.exists():
        d = dict(np.load(f))
    else:
        rng = np.random.default_rng(seed)
        A, NH, KH = make_network(rng)
        eff = np.ones(G); eff[REGS] = rng.uniform(0.05, 0.3, len(REGS))
        hidden = np.r_[rng.choice(range(4, NTF), 1), rng.choice(NONTF_REGS, 1), rng.choice(range(NTF + NREG, G), 8, replace=False)]
        pool = [j for j in REGS if j not in hidden]
        heldout = np.r_[rng.choice([j for j in pool if 4 <= j < NTF], 2, replace=False), rng.choice([j for j in pool if j >= NTF], 2, replace=False)]
        kd = np.r_[np.full(N_WT, -1), np.repeat(pool, N_KD)]
        X, Vt, Z = simulate(rng, A, NH, KH, kd, eff)
        OUT.mkdir(parents=True, exist_ok=True)
        d = dict(A=A, X=X, Vt=Vt, Z=Z, kd=kd, eff=eff, heldout=heldout, hidden=hidden)
        np.savez(f, **d)
    obs = np.array([g for g in range(G) if g not in d["hidden"]])
    idx = {g: k for k, g in enumerate(obs)}
    kd_o = np.array([idx.get(int(k), -1) if k >= 0 else -1 for k in d["kd"]])
    Ao = d["A"][np.ix_(obs, obs)]
    reach = (d["A"] != 0).astype(int)  # transitive closure on the FULL graph
    R = reach.copy()
    for _ in range(G):
        R2 = ((R + R @ reach) > 0).astype(int)
        if (R2 == R).all():
            break
        R = R2
    IND = (R > 0) & (d["A"] == 0); np.fill_diagonal(IND, False)
    return dict(A=Ao, X=d["X"][:, obs], V=(d["Vt"] + vel_noise * d["Z"])[:, obs], kd=kd_o,
                heldout=np.array([idx[int(j)] for j in d["heldout"]]), IND=IND[np.ix_(obs, obs)], obs=obs,
                tf_obs=[idx[j] for j in TFS if j in idx], reg_obs=[idx[j] for j in REGS if j in idx], A_full=d["A"])


def binding_prior(seed, p1, p0, A, obs):
    rng = np.random.default_rng(5000 + seed)
    ms = [m for m in parse_jaspar(ROOT / "data/raw/jaspar2024_vert_nr.txt")
          if "::" not in m["name"] and 9 <= m["pfm"].shape[1] <= 14]
    Ws = [logodds(ms[i]["pfm"], BG) for i in rng.choice(len(ms), NTF, replace=False)]
    seqs = rng.choice(4, size=(G, 1000), p=BG).astype(np.int8)
    Ethr = []
    for W in Ws:
        s = rng.choice(4, size=(3000, 100 + W.shape[0] - 1), p=BG).astype(np.int8)
        Ethr.append(float(np.quantile(scan_energies(s, W)[0][:, 0, :].ravel(), 1e-4)))

    def site(W, lo, hi):
        p = np.exp(W) * BG; p /= p.sum(1, keepdims=True); smax = W.max(1).sum(); best, bd = None, 1e9
        for _ in range(40):
            s = np.array([rng.choice(4, 2000, p=p[k]) for k in range(W.shape[0])]).T
            mut = rng.random(s.shape) < rng.uniform(0, 0.6, (2000, 1)); s[mut] = rng.integers(0, 4, mut.sum())
            E = smax - W[np.arange(W.shape[0]), s].sum(1); ok = np.nonzero((E >= lo) & (E <= hi))[0]
            if ok.size:
                return s[ok[0]].astype(np.int8)
            k = np.abs(E - (lo + hi) / 2).argmin()
            if abs(E[k] - (lo + hi) / 2) < bd:
                best, bd = s[k].astype(np.int8), abs(E[k] - (lo + hi) / 2)
        return best
    planted = np.zeros((G, G), bool)
    for j in TFS:
        W = Ws[j]; w = W.shape[0]
        for i in range(G):
            if i == j:
                continue
            if rng.random() < (p1 if A[i, j] != 0 else p0):
                planted[i, j] = True
                strong = rng.random() < 0.5
                for _ in range(1 if strong else 6):
                    pos = rng.integers(0, 1000 - w)
                    st = site(W, 0, 1.5) if strong else site(W, Ethr[j] + 0.5, Ethr[j] + 2.5)
                    seqs[i, pos:pos + w] = st if rng.random() < 0.5 else (3 - st)[::-1]
    O = np.zeros((G, G))
    for j in TFS:
        O[:, j] = np_sigmoid(Ethr[j] - 2 - scan_energies(seqs, Ws[j])[0]).sum(axis=(1, 2))
    np.fill_diagonal(O, 0)
    O = O[np.ix_(obs, obs)]; planted = planted[np.ix_(obs, obs)]; n = len(obs)
    P = np.zeros((n, n))
    for k, g in enumerate(obs):
        if g in TFS:
            col = O[:, k].copy(); col[k] = -np.inf
            P[:, k] = col.argsort().argsort() / (n - 1); P[k, k] = 0
    return O, P, planted


# ------------------------------------------------------------------ helpers
def phi_fn(X, K):
    return X ** 2 / (K ** 2 + X ** 2)


def indirect_pairs(A):
    B = (A != 0).astype(int); two = (B @ B) > 0; np.fill_diagonal(two, False)
    return two & (B == 0)


def edge_metrics(S, A, IND, decoy):
    off = ~np.eye(A.shape[0], dtype=bool)
    y = (A != 0)[off]; s = S[off] + 1e-12 * np.random.default_rng(0).random(off.sum())
    k = int(y.sum()); top = np.zeros(off.sum(), bool); top[np.argsort(-s)[:k]] = True
    return {"AUROC": float(roc_auc_score(y, s)), "AUPRC": float(average_precision_score(y, s)),
            "AUPRC_ratio": float(average_precision_score(y, s) / y.mean()),
            "precision_at_E": float(y[top].mean()),
            "indirect_frac_of_topE": float(IND[off][top].mean()),
            "decoy_frac_of_topE": float(decoy[off][top].mean())}


# ------------------------------------------------------------------ models
def _fit_one(i, Phi, X, V, kd, pf_all, use_kd, use_velocity):
    n = X.shape[1]
    m = (kd != i) if use_kd else np.ones(len(kd), bool)
    cols = [j for j in range(n) if j != i]
    pf = pf_all[i, cols]
    F = Phi[m][:, cols]
    if use_velocity:
        F = np.c_[F, X[m, i]]; pfv = np.r_[pf, 0.01]  # decay term effectively unpenalised
        y = V[m, i]
    else:
        pfv = pf; y = X[m, i]
    mu, sd = F.mean(0), F.std(0) + 1e-9
    # adaptive lasso: standardise FIRST, then divide by the penalty factor
    mdl = LassoLarsIC(criterion="bic").fit((F - mu) / sd / pfv, y)
    coef = mdl.coef_ / pfv / sd
    b = mdl.intercept_ - (coef * mu).sum()
    g = 0.0
    if use_velocity:
        g = -coef[-1]; coef = coef[:-1]
    return i, cols, coef, b, g


def fit_vector_field(X, V, kd, P, alpha, use_kd=True, use_velocity=True, rows=None):
    """Nodewise adaptive-lasso (parallel over genes). Returns dict(W, b, g, K, dynamic)."""
    from joblib import Parallel, delayed
    if rows is not None:
        X, V, kd = X[rows], V[rows], kd[rows]
    n = X.shape[1]
    K = np.maximum(np.median(X, 0), 1e-3)
    Phi = phi_fn(X, K)
    W = np.zeros((n, n)); b = np.zeros(n); g = np.zeros(n)
    pf_all = 1 - alpha * P
    out = Parallel(n_jobs=4, prefer="threads")(delayed(_fit_one)(i, Phi, X, V, kd, pf_all, use_kd, use_velocity) for i in range(n))
    for i, cols, coef, bi, gi in out:
        W[i, cols] = coef; b[i] = bi; g[i] = gi
    return dict(W=W, b=b, g=g, K=K, dynamic=use_velocity)


def strength(model, X):
    return np.abs(model["W"]) * phi_fn(X, model["K"]).std(0)[None, :]


def predict_kd_effect(model, X0, j, n_steps=60, dt=0.05):
    """Mean shift in expression predicted for a knockdown of j (nominal efficiency)."""
    W, b, g, K = model["W"], model["b"], model["g"], model["K"]
    n = W.shape[0]
    s = np.ones(n); s[j] = NOMINAL_KD

    def run(scale):
        x = X0.copy()
        if model["dynamic"]:
            gg = np.maximum(g, 0.05)
            for _ in range(n_steps):
                prod = b + phi_fn(x, K) @ W.T
                x = np.maximum(x + dt * (scale * prod - gg * x), 0)
        else:
            for _ in range(30):
                xn = b + phi_fn(x, K) @ W.T
                xn[:, j] = x[:, j] * scale[j] if scale[j] < 1 else xn[:, j]
                x = np.maximum(0.5 * x + 0.5 * xn, 0)
        return x
    return run(s).mean(0) - run(np.ones(n)).mean(0)


def linear_kd_effect(model, X, j):
    """Fast linearised steady-state effect (used only inside the active-selection criterion)."""
    xb = X.mean(0); K = model["K"]
    dphi = 2 * xb * K ** 2 / (K ** 2 + xb ** 2) ** 2
    J = model["W"] * dphi[None, :] - np.diag(np.maximum(model["g"], 0.05))
    prod_j = model["b"][j] + model["W"][j] @ phi_fn(xb, K)
    n = J.shape[0]
    rhs = np.zeros(n); rhs[j] = -(1 - NOMINAL_KD) * max(prod_j, 0.05)
    try:
        return -np.linalg.solve(J, rhs)
    except np.linalg.LinAlgError:
        return np.zeros(n)


def effect_scores(model, d, train_rows):
    X, kd = d["X"], d["kd"]
    wt = np.nonzero(kd == -1)[0]
    X0 = X[np.random.default_rng(1).choice(wt, 400, replace=False)]
    cors, signs = [], []
    for j in d["heldout"]:
        kj = kd == j
        true = X[kj].mean(0) - X[wt].mean(0)
        se = np.sqrt(X[kj].var(0) / kj.sum() + X[wt].var(0) / len(wt))
        pred = predict_kd_effect(model, X0, j)
        keep = np.arange(X.shape[1]) != j
        cors.append(pearsonr(pred[keep], true[keep])[0] if pred[keep].std() > 1e-9 else 0.0)
        big = keep & (np.abs(true) > 3 * se)
        signs.append(float((np.sign(pred[big]) == np.sign(true[big])).mean()) if big.any() else np.nan)
    return {"heldout_effect_corr": float(np.nanmean(cors)), "heldout_effect_sign_acc": float(np.nanmean(signs))}


# ------------------------------------------------------------------ experiment A: ablations + baselines
def ablate(seed):
    d = make_data(seed); A, X, V, kd, IND = d["A"], d["X"], d["V"], d["kd"], d["IND"]
    n = A.shape[0]; tfo = d["tf_obs"]
    train = ~np.isin(kd, d["heldout"])  # WT + training knockdowns; 4 knockdowns held out
    res = {"seed": seed, "n_genes_measured": n, "n_edges": int((A != 0).sum()), "n_indirect": int(IND.sum()),
           "methods": {}, "sweeps": {}}
    priors = {k: binding_prior(seed, *v, d["A_full"], d["obs"]) for k, v in PRIORS.items()}
    O, Pr, planted = priors["realistic"]
    decoy = planted & (A == 0)
    wt = kd == -1; fr = []
    for j in tfo:
        kj = kd == j
        if kj.sum() == 0:
            continue
        dz = np.abs(X[kj].mean(0) - X[wt].mean(0)) / np.sqrt(X[kj].var(0) / kj.sum() + X[wt].var(0) / wt.sum() + 1e-12)
        resp = (dz > 3) & (np.arange(n) != j)
        if resp.sum() >= 3:
            fr.append(planted[resp, j].mean())
    res["calib_frac_responders_bound_realistic"] = float(np.mean(fr))
    ti, tj = np.nonzero(A[:, tfo] != 0)
    res["calib_frac_true_targets_bound_realistic"] = float(planted[ti, np.array(tfo)[tj]].mean())

    def record(name, model=None, S=None):
        if S is None:
            S = strength(model, X[train])
        r = edge_metrics(S, A, IND, decoy)
        if model is not None:
            r.update(effect_scores(model, d, train))
        res["methods"][name] = r
    record("BOND-V (velocity + KD-aware + prior a=0.8)", fit_vector_field(X, V, kd, Pr, 0.8, rows=train))
    record("  - prior (a=0)", fit_vector_field(X, V, kd, Pr, 0.0, rows=train))
    record("  - KD-aware [= PHOENIX-style: velocity + prior]", fit_vector_field(X, V, kd, Pr, 0.8, use_kd=False, rows=train))
    record("  - velocity [static, KD-aware + prior]", fit_vector_field(X, V, kd, Pr, 0.8, use_velocity=False, rows=train))
    record("  - KD-aware - prior [velocity only]", fit_vector_field(X, V, kd, Pr, 0.0, use_kd=False, rows=train))
    record("DCDI-style (static, KD-aware, no prior)", fit_vector_field(X, V, kd, Pr, 0.0, use_velocity=False, rows=train))
    S = np.zeros((n, n)); Xt = X[train]
    for i in range(n):
        cand = [j for j in tfo if j != i and Pr[i, j] >= 0.7]
        if cand:
            br = BayesianRidge().fit(Xt[:, cand], Xt[:, i])
            S[i, cand] = np.abs(br.coef_) * Xt[:, cand].std(0)
    record("CellOracle-style (motif base GRN + ridge)", S=S)
    from sklearn.ensemble import ExtraTreesRegressor
    S = np.zeros((n, n)); sub = np.random.default_rng(seed).choice(np.nonzero(train)[0], 1500, replace=False)
    for i in range(n):
        cols = [j for j in range(n) if j != i]
        et = ExtraTreesRegressor(n_estimators=40, max_features="sqrt", min_samples_leaf=5, random_state=0, n_jobs=2).fit(X[sub][:, cols], X[sub, i])
        S[i, cols] = et.feature_importances_
    record("GENIE3-style (extra trees)", S=S)
    C = np.abs(np.corrcoef(Xt.T)); np.fill_diagonal(C, 0); record("|corr|", S=np.nan_to_num(C))
    MD = np.zeros((n, n))
    for j in np.unique(kd[train & (kd >= 0)]):
        kj = kd == j
        MD[:, j] = np.abs(X[kj].mean(0) - X[wt].mean(0)) / np.sqrt(X[kj].var(0) / kj.sum() + X[wt].var(0) / wt.sum() + 1e-12)
    np.fill_diagonal(MD, 0); record("KD mean-difference (total effect)", S=MD)
    record("binding prior only (realistic)", S=O)

    def sweep(name, P_, pl_, a, Vv=None):
        m = fit_vector_field(X, V if Vv is None else Vv, kd, P_, a, rows=train)
        res["sweeps"][name] = {**edge_metrics(strength(m, X[train]), A, IND, pl_ & (A == 0)), **effect_scores(m, d, train)}
    for a in ALPHAS:
        sweep(f"prior strength a={a} | realistic prior", Pr, planted, a)
    for q, (O_, P_, pl_) in priors.items():
        sweep(f"prior quality {q} | a=0.8", P_, pl_, 0.8)
    Ps = Pr.copy(); r = np.random.default_rng(77 + seed)
    for j in tfo:
        idx = [i for i in range(n) if i != j]; Ps[idx, j] = Ps[r.permutation(idx), j]
    sweep("prior quality shuffled | a=0.8", Ps, planted, 0.8)
    for vn in VEL_NOISES:
        sweep(f"velocity noise {vn} x sd | a=0.8 realistic", Pr, planted, 0.8, Vv=make_data(seed, vn)["V"])
    json.dump(res, open(OUT / f"ablate_s{seed}.json", "w"), indent=1)
    for k, v in res["methods"].items():
        print(f"{k:48s} AUROC={v['AUROC']:.3f} AUPRCx={v['AUPRC_ratio']:.2f} P@E={v['precision_at_E']:.2f} ind={v['indirect_frac_of_topE']:.2f} "
              f"dec={v['decoy_frac_of_topE']:.2f} eff={v.get('heldout_effect_corr', float('nan')):.2f}")


# ------------------------------------------------------------------ experiment B: coverage + design
def coverage(seed, strategies=("random", "degree", "active")):
    d = make_data(seed); A, X, V, kd, IND = d["A"], d["X"], d["V"], d["kd"], d["IND"]
    _, P, planted = binding_prior(seed, *PRIORS["realistic"], d["A_full"], d["obs"])
    pool = [j for j in d["reg_obs"] if j not in d["heldout"]]
    wt_rows = np.nonzero(kd == -1)[0]
    res = {"seed": seed, "pool_size": len(pool), "strategies": {}}
    steps_max = min(max(COV_STEPS), len(pool))
    t_start = time.time()
    for strat in strategies:
        prog = OUT / f"coverage_s{seed}_{strat}.progress.json"
        if (OUT / f"coverage_s{seed}_{strat}.json").exists():
            continue
        st = json.load(open(prog)) if prog.exists() else {"chosen": [], "curve": {}, "step": 0}
        rng = np.random.default_rng(100 + seed)
        for k in range(len(st["chosen"])):  # replay the random stream exactly when resuming
            if strat == "random":
                rng.choice([j for j in pool if j not in st["chosen"][:k]])
        chosen, curve = st["chosen"], {int(k): v for k, v in st["curve"].items()}
        done = False
        for step in range(st["step"], steps_max + 1):
            if time.time() - t_start > 140:
                json.dump({"chosen": chosen, "curve": curve, "step": step}, open(prog, "w"))
                print(f"{strat}: paused at step {step}; rerun to continue", flush=True)
                return
            rows = np.r_[wt_rows, np.nonzero(np.isin(kd, chosen))[0]]
            m = fit_vector_field(X, V, kd, P, 0.8, rows=rows)
            if step in COV_STEPS or step == steps_max:
                curve[step] = {**edge_metrics(strength(m, X[rows]), A, IND, planted & (A == 0)), **effect_scores(m, d, rows)}
            if step == steps_max:
                done = True
                break
            cand = [j for j in pool if j not in chosen]
            if strat == "random":
                nxt = int(rng.choice(cand))
            elif strat == "degree":
                S = strength(m, X[rows]); nxt = int(cand[np.argmax([S[:, j].sum() for j in cand])])
            else:
                ens = []
                for bsd in range(3):
                    br = np.random.default_rng(1000 * seed + 10 * step + bsd).choice(rows, len(rows), replace=True)
                    ens.append(fit_vector_field(X, V, kd, P, 0.8, rows=br))
                scores = [np.var(np.stack([linear_kd_effect(e, X[rows], j) for e in ens]), 0).sum() for j in cand]
                nxt = int(cand[int(np.argmax(scores))])
            chosen.append(nxt)
        res["strategies"][strat] = {"curve": curve, "order": chosen,
                                    "order_is_TF": [int(j in d["tf_obs"]) for j in chosen]}
        json.dump({"seed": seed, "pool_size": len(pool), "strategy": strat, **res["strategies"][strat]},
                  open(OUT / f"coverage_s{seed}_{strat}.json", "w"), indent=1)
        print(strat, {k: (round(v["AUPRC_ratio"], 2), round(v["indirect_frac_of_topE"], 2), round(v["heldout_effect_corr"], 2)) for k, v in curve.items()}, flush=True)


# ------------------------------------------------------------------ report
def report():
    ms = lambda xs: f"{np.nanmean(xs):.3f} ± {np.nanstd(xs):.3f}"
    AB = [json.load(open(OUT / f"ablate_s{s}.json")) for s in SEEDS if (OUT / f"ablate_s{s}.json").exists()]
    CV = []
    for s_ in SEEDS:
        fs = {st: OUT / f"coverage_s{s_}_{st}.json" for st in ["random", "degree", "active"]}
        if all(f.exists() for f in fs.values()):
            CV.append({"strategies": {st: json.load(open(f)) for st, f in fs.items()}})
    R = ["# Medium synthetic test (numeric results)", "",
         f"Generated by `experiments/medium_synthetic.py report`. {len(AB)} seeds (ablation), {len(CV)} seeds (coverage); mean ± sd. "
         f"80 genes simulated, {AB[0]['n_genes_measured']} measured, ~{np.mean([a['n_edges'] for a in AB]):.0f} true edges among measured genes, ~{np.mean([a['n_indirect'] for a in AB]):.0f} indirect pairs. "
         "Snapshot cells, Poisson counts, noisy labelling velocity (1.0 x sd unless swept), hidden cell-size confounder, misspecified Hill shapes. Training: WT + 18 regulator knockdowns; 4 knockdowns held out for effect prediction. "
         "Top-E = as many top-scored edges as true edges.", "",
         f"Prior calibration (realistic): fraction of a TF's knockdown responders that are bound = {ms([a['calib_frac_responders_bound_realistic'] for a in AB])}; "
         f"fraction of true direct TF targets bound = {ms([a['calib_frac_true_targets_bound_realistic'] for a in AB])}.", "",
         "## A. Ablations and baselines (full training coverage)", "",
         "| Method | AUROC | AUPRC× | Precision@E | Indirect frac of top-E | Decoy frac of top-E | Held-out KD effect corr | Held-out KD sign acc |",
         "|---|---|---|---|---|---|---|---|"]
    for k in AB[0]["methods"]:
        g = lambda kk: ms([a["methods"][k].get(kk, np.nan) for a in AB]) if kk in AB[0]["methods"][k] else "–"
        R.append(f"| {k} | {g('AUROC')} | {g('AUPRC_ratio')} | {g('precision_at_E')} | {g('indirect_frac_of_topE')} | {g('decoy_frac_of_topE')} | {g('heldout_effect_corr')} | {g('heldout_effect_sign_acc')} |")
    R += ["", "## B. Prior strength and prior quality sweeps (BOND-V)", "",
          "| Setting | AUROC | AUPRC× | Precision@E | Indirect frac | Decoy frac | Held-out KD effect corr |", "|---|---|---|---|---|---|---|"]
    for k in AB[0]["sweeps"]:
        g = lambda kk: ms([a["sweeps"][k][kk] for a in AB])
        R.append(f"| {k} | {g('AUROC')} | {g('AUPRC_ratio')} | {g('precision_at_E')} | {g('indirect_frac_of_topE')} | {g('decoy_frac_of_topE')} | {g('heldout_effect_corr')} |")
    if CV:
        R += ["", "## C. Perturbation coverage and experiment selection (BOND-V, realistic prior, a = 0.8)", "",
              "| Strategy | # training KDs | AUPRC× | Precision@E | Indirect frac of top-E | Held-out KD effect corr |", "|---|---|---|---|---|---|"]
        for st in ["random", "degree", "active"]:
            for k in COV_STEPS:
                if str(k) not in CV[0]["strategies"][st]["curve"]:
                    continue
                g = lambda kk: ms([c["strategies"][st]["curve"][str(k)][kk] for c in CV if str(k) in c["strategies"][st]["curve"]])
                R.append(f"| {st} | {k} | {g('AUPRC_ratio')} | {g('precision_at_E')} | {g('indirect_frac_of_topE')} | {g('heldout_effect_corr')} |")
        R += ["", "| Strategy | Fraction of first 4 picks that are TFs | Fraction of first 8 picks that are TFs |", "|---|---|---|"]
        for st in ["random", "degree", "active"]:
            R.append(f"| {st} | {ms([np.mean(c['strategies'][st]['order_is_TF'][:4]) for c in CV])} | {ms([np.mean(c['strategies'][st]['order_is_TF'][:8]) for c in CV])} |")
        steps = [k for k in COV_STEPS if all(str(k) in c["strategies"]["random"]["curve"] for c in CV)]
        aucs = {st: [np.trapz([c["strategies"][st]["curve"][str(k)]["AUPRC_ratio"] for k in steps], steps) / max(steps) for c in CV] for st in ["random", "degree", "active"]}
        R += ["", "Area under the AUPRC× coverage curve (0–" + str(max(steps)) + " KDs, normalised): " + ", ".join(f"{k} = {ms(v)}" for k, v in aucs.items()),
              "Paired differences per seed: active − random = " + ms(np.array(aucs["active"]) - np.array(aucs["random"])) +
              "; active − degree = " + ms(np.array(aucs["active"]) - np.array(aucs["degree"]))]
    (ROOT / "docs/06_medium_synthetic_results.md").write_text("\n".join(R) + "\n")
    print("\n".join(R))


if __name__ == "__main__":
    {"ablate": lambda: ablate(int(sys.argv[2])),
     "coverage": lambda: coverage(int(sys.argv[2]), sys.argv[3].split(",") if len(sys.argv) > 3 else ("random", "degree", "active")),
     "report": report}[sys.argv[1]]()
