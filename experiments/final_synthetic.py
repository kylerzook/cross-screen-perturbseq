"""Final synthetic tests (80-gene Hill simulator from medium_synthetic.py).

F3  Memory functions as a hidden-regulator detector. Dense, noisy time courses; for each observed gene
    compare a Markov (current-state) predictor of its next increment with one that also sees the past
    (lags 1,2,4,8 samples). Memory gain = 1 - MSE_lag / MSE_markov on held-out trajectories.
    (a) detection AUROC for 'gene has a hidden direct regulator', vs an all-observed control and vs a
        snapshot detector (1 - R^2 of the gene from all other observed genes);
    (b) residual-autocorrelation timescale vs the hidden regulator's decay rate;
    (c) sampling interval sweep;  (d) latent size: # hidden regulators vs rank of lagged residual
        cross-covariance above a circular-shift null.
F4  Intervention attraction. (a) 3-gene errors-in-variables formula check (coefficient c*b*lambda,
    lambda = s_e^2 / (s_e^2 + Var(B|A))) and Lasso detection rates; (b) 80-gene simulation: false-
    positive rate of indirect pairs j->i by (source perturbed?) x (2-hop mediator perturbed/unperturbed/
    hidden), random half perturbation subsets; (c) identifiability flag.
F5  Prior forms: none / soft site prior / soft identity prior / hard identity mask / hard TF-only mask /
    hard binding mask / identity mask + soft site prior.  Plus propagation-depth sweep for held-out KDs.

Usage: python experiments/final_synthetic.py f3 [part] | f4 | f5 <seed> | f5report
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV, LassoLarsIC
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import medium_synthetic as ms

OUT = ROOT / "results/final"; OUT.mkdir(parents=True, exist_ok=True)
G, TFS, NONTF, REGS = ms.G, ms.TFS, ms.NONTF_REGS, ms.REGS


# =============================================================== F3 memory
def sim_traj(rng, A, NH, KH, gamma, n_traj=100, n_pts=300, dt_save=0.1, proc=0.25, burn=15.0, DT=0.01, proc_reg=None):
    N = n_traj
    pn = np.full(G, proc)
    if proc_reg is not None:
        pn[REGS] = proc_reg
    u = rng.uniform(0.1, 3.0, (N, 4)); size = rng.lognormal(0, 0.3, (N, 1))
    x = rng.uniform(0, 1.5, (N, G))
    ii, jj = np.nonzero(A); a_e, n_e, k_e = A[ii, jj], NH[ii, jj], KH[ii, jj]
    Msc = np.eye(G)[ii]
    every = max(1, int(round(dt_save / DT)))
    nb = int(burn / DT); total = nb + every * n_pts
    out = np.zeros((N, n_pts, G)); c = 0
    for k in range(1, total + 1):
        xe = x[:, jj]; h = xe ** n_e / (k_e ** n_e + xe ** n_e)
        prod = 0.05 + np.where(a_e > 0, a_e * h, -a_e * (1 - h)) @ Msc
        prod[:, :4] += u; prod *= size
        x = np.maximum(x + (prod - gamma * x) * DT + pn * np.sqrt(DT) * rng.standard_normal((N, G)), 0)
        if k > nb and (k - nb) % every == 0:
            out[:, c] = x; c += 1
    return out


def pick_hidden(rng, A, k):
    """k hidden regulators, each with >= 2 children that are not genes 0-3."""
    cand = [j for j in REGS if j >= 4 and ((A[4:, j] != 0).sum() >= 2)]
    return list(rng.choice(cand, k, replace=False)) if k else []


SMOOTH = 1


def lag_design(Y, lags, phiK=None, H=1):
    if SMOOTH > 1:  # causal moving average (denoises measurement noise; target stays raw)
        cs = np.cumsum(np.concatenate([np.zeros_like(Y[:, :1]), Y], 1), 1)
        Ys = Y.copy(); w = SMOOTH
        Ys[:, w - 1:] = (cs[:, w:] - cs[:, :-w]) / w
    else:
        Ys = Y
    """Y (N, T, n) -> rows for t in [maxlag, T-1): markov feats, lag feats, target increments, traj id."""
    L = max(lags) + SMOOTH; N, T, n = Y.shape
    t = np.arange(L, T - H)
    cur = Ys[:, t]                                     # (N, nt, n)
    K = np.maximum(np.median(Y.reshape(-1, n), 0), 1e-3) if phiK is None else phiK
    mk = np.concatenate([cur, cur ** 2 / (K ** 2 + cur ** 2)], -1)
    lg = np.concatenate([Ys[:, t - l] for l in lags], -1)
    dy = Y[:, t + H] - Y[:, t]
    tid = np.repeat(np.arange(N), len(t))
    f = lambda Z: Z.reshape(-1, Z.shape[-1])
    return f(mk), f(lg), f(dy), tid


HORIZON = 1


LAGS = (1, 2, 4, 8)


def memory_scores(Y, lags=None, seed=0, H=None):
    lags = LAGS if lags is None else lags
    N = Y.shape[0]
    mk, lg, dy, tid = lag_design(Y, lags, H=HORIZON if H is None else H)
    rng = np.random.default_rng(seed); te_traj = rng.permutation(N)[: int(0.3 * N)]
    te = np.isin(tid, te_traj); tr = ~te
    mu, sd = mk[tr].mean(0), mk[tr].std(0) + 1e-9; Mk = (mk - mu) / sd
    mu2, sd2 = lg[tr].mean(0), lg[tr].std(0) + 1e-9; Lg = (lg - mu2) / sd2
    al = np.logspace(-2, 4, 13)
    r1 = RidgeCV(alphas=al, alpha_per_target=True).fit(Mk[tr], dy[tr])
    F2 = np.c_[Mk, Lg]
    r2 = RidgeCV(alphas=al, alpha_per_target=True).fit(F2[tr], dy[tr])
    e1 = dy[te] - r1.predict(Mk[te]); e2 = dy[te] - r2.predict(F2[te])
    gain = 1 - (e2 ** 2).mean(0) / ((e1 ** 2).mean(0) + 1e-12)
    # residuals on training trajectories, reshaped (traj, time, gene) for timescale / rank
    etr = dy[tr] - r1.predict(Mk[tr])
    ntr = N - len(te_traj); E = etr.reshape(ntr, -1, Y.shape[2])
    return gain, E


def acf_rate(E, dt, max_lag=6):
    """Decay rate from the residual autocorrelation (log-linear fit over lags 1..max_lag)."""
    Z = E - E.mean(1, keepdims=True); v = (Z ** 2).mean((0, 1)) + 1e-12
    rates = []
    ac = np.array([(Z[:, l:] * Z[:, :Z.shape[1] - l]).mean((0, 1)) / v for l in range(1, max_lag + 1)])  # (L, n)
    for i in range(E.shape[2]):
        a = ac[:, i]; ok = a > 0.02
        if ok.sum() >= 2:
            ls = np.arange(1, max_lag + 1)[ok] * dt
            rates.append(-np.polyfit(ls, np.log(a[ok]), 1)[0])
        else:
            rates.append(np.nan)
    return np.array(rates), ac[0]


def lagged_rank(E, lag=2, n_null=30, seed=0):
    Z = E - E.mean(1, keepdims=True); Z /= Z.std((0, 1), keepdims=True) + 1e-12
    def sv(Zs):
        C = np.einsum("ntg,nth->gh", Zs[:, lag:], Zs[:, :-lag]) / (Zs.shape[0] * (Zs.shape[1] - lag))
        return np.linalg.svd(C, compute_uv=False)
    obs = sv(Z)
    rng = np.random.default_rng(seed); nulls = []
    T = Z.shape[1]
    for _ in range(n_null):
        Zs = np.empty_like(Z)
        for g in range(Z.shape[2]):
            Zs[:, :, g] = np.roll(Z[:, :, g], rng.integers(lag + 5, T - lag - 5), axis=1)
        nulls.append(sv(Zs))
    thr = np.quantile(np.array(nulls), 0.95, axis=0)
    k = 0
    while k < len(obs) and obs[k] > thr[k]:
        k += 1
    return int(k), obs[:8].tolist(), thr[:8].tolist()


def snapshot_detector(Y, seed=0):
    """1 - R^2 of each gene from all other observed genes, one random time point per trajectory... use
    many independent cells: take every 25th sample (approx independent snapshots)."""
    S = Y[:, ::25].reshape(-1, Y.shape[2])
    K = np.maximum(np.median(S, 0), 1e-3); Phi = np.c_[S, S ** 2 / (K ** 2 + S ** 2)]
    n = S.shape[1]; out = np.zeros(n)
    for i in range(n):
        cols = [c for c in range(2 * n) if c % n != i]
        F = Phi[:, cols]; F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        r = RidgeCV(alphas=np.logspace(-2, 4, 7)).fit(F, S[:, i])
        out[i] = 1 - r.score(F, S[:, i])
    return out


def oracle_gain(Y, obs, hidden, H):
    mk, _, dy, tid = lag_design(Y[:, :, obs], (1,), H=H)
    mkh, _, _, _ = lag_design(Y[:, :, hidden], (1,), H=H)
    tr = tid < int(0.7 * Y.shape[0]); te = ~tr
    def fit(Fm):
        Fm = (Fm - Fm[tr].mean(0)) / (Fm[tr].std(0) + 1e-9)
        r = RidgeCV(alphas=np.logspace(-2, 4, 7), alpha_per_target=True).fit(Fm[tr], dy[tr])
        return ((dy[te] - r.predict(Fm[te])) ** 2).mean(0)
    return 1 - fit(np.c_[mk, mkh]) / fit(mk)


PROC_REG = None


def f3_world(seed, k_hidden=2, dt_save=0.1, meas=0.5, n_traj=100, n_pts=300, control=False, extra=False):
    rng = np.random.default_rng(9000 + seed)
    A, NH, KH = ms.make_network(rng)
    gamma = rng.uniform(0.5, 1.0, G); gamma[REGS] = rng.uniform(0.15, 1.5, len(REGS))
    hidden = pick_hidden(rng, A, k_hidden)
    Y = sim_traj(rng, A, NH, KH, gamma, n_traj=n_traj, n_pts=n_pts, dt_save=dt_save, proc_reg=PROC_REG, DT=0.02 if dt_save >= 0.5 else 0.01)
    Yclean = Y.copy()
    if meas > 0:
        within_sd = Y.std(1).mean(0)
        Y = Y + meas * within_sd[None, None, :] * rng.standard_normal(Y.shape)
    obs = [g for g in range(G) if (control or g not in hidden)]
    Yo = Y[:, :, obs]
    hidden_parent = np.array([any(A[g, h] != 0 for h in hidden) for g in obs])
    evalm = np.array([g >= 4 for g in obs])  # genes 0-3 have an unmeasured external input u
    gain, E = memory_scores(Yo, seed=seed)
    res = {"seed": seed, "k_hidden": k_hidden, "dt": dt_save, "meas": meas, "control": control,
           "n_pos": int((hidden_parent & evalm).sum()), "n_eval": int(evalm.sum())}
    y = hidden_parent[evalm]
    lab_hidden = np.array([any(A[g, h] != 0 for h in hidden) for g in range(G)])
    if control:  # label = would-be hidden-parent genes; the hidden genes themselves are excluded
        keep = np.array([g not in hidden for g in obs]) & evalm
        y = lab_hidden[np.array(obs)][keep]; gsc = gain[keep]
    else:
        gsc = gain[evalm]
    res["memory_gain_pos_median"] = float(np.median(gsc[y])) if y.any() else None
    res["memory_gain_neg_median"] = float(np.median(gsc[~y]))
    res["auroc_memory"] = float(roc_auc_score(y, gsc)) if 0 < y.sum() < len(y) else None
    if extra and not control:
        og = oracle_gain(Yclean, obs, hidden, HORIZON)[evalm]
        res["oracle_gain_pos_median"] = float(np.median(og[y])); res["oracle_gain_neg_median"] = float(np.median(og[~y]))
        snap = snapshot_detector(Yo)[evalm]
        res["auroc_snapshot_unexplained_var"] = float(roc_auc_score(y, snap))
        rates, ac1 = acf_rate(E, dt_save)
        pairs = []
        for h in hidden:
            ch = [k for k, g in enumerate(obs) if A[g, h] != 0 and g >= 4]
            r = np.nanmedian(rates[ch]) if ch else np.nan
            pairs.append((float(gamma[h]), float(r)))
        res["timescale_pairs_true_gamma_vs_est_rate"] = pairs
    return res


def f3(part="all"):
    global PROC_REG, HORIZON, SMOOTH, LAGS
    t0 = time.time()
    fo = OUT / f"f3_memory_{part}.json"
    R = {}
    PROC_REG, HORIZON, SMOOTH, LAGS = 1.0, 5, 1, (1, 2, 4, 8)  # main regime: fluctuating regulators
    R["settings"] = {"regulator_process_noise": PROC_REG, "other_process_noise": 0.25, "horizon_samples": HORIZON,
                     "lags_samples": LAGS, "dt": 0.1, "n_traj": 100, "n_points": 300}
    if part == "main":
        R["main_clean"] = [f3_world(s, meas=0.0, extra=True) for s in range(10)]
        R["control_all_observed_clean"] = [f3_world(s, meas=0.0, control=True) for s in range(10)]
        PROC_REG = None
        R["weak_regulator_noise_clean"] = [f3_world(s, meas=0.0, extra=True) for s in range(10)]
        PROC_REG = 1.0
    if part == "noise":
        R["meas_sweep"] = {str(m): [f3_world(s, meas=m) for s in range(6)] for m in (0.1, 0.25, 0.5, 1.0)}
        R["meas_sweep_control"] = {str(m): [f3_world(s, meas=m, control=True) for s in range(6)] for m in (0.25, 0.5)}
        SMOOTH, LAGS = 4, (4, 8, 16, 32)
        R["meas0.5_with_causal_smoothing"] = [f3_world(s, meas=0.5) for s in range(6)]
        SMOOTH, LAGS = 1, (1, 2, 4, 8)
    if part == "dt":
        R["dt_sweep_clean"] = {str(dt): [f3_world(s, meas=0.0, dt_save=dt, n_traj=50) for s in range(4)] for dt in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)}
        R["settings"]["dt_sweep_note"] = "50 trajectories, 4 seeds, 300 points each; Euler step 0.02 when sampling interval >= 0.5"
    if part == "rank":
        rk = {}
        for k in (0, 1, 2, 3, 4):
            rk[str(k)] = []
            for s in range(5):
                rng = np.random.default_rng(9500 + 10 * k + s)
                A, NH, KH = ms.make_network(rng)
                gamma = rng.uniform(0.5, 1.0, G); gamma[REGS] = rng.uniform(0.15, 1.5, len(REGS))
                hidden = pick_hidden(rng, A, k)
                Y = sim_traj(rng, A, NH, KH, gamma, proc_reg=PROC_REG)
                obs = [g for g in range(G) if g not in hidden and g >= 4]  # drop externally driven genes 0-3
                _, E = memory_scores(Y[:, :, obs], seed=s)
                est, sv, thr = lagged_rank(E, seed=s)
                rk[str(k)].append({"est_rank": est, "sv": sv, "null95": thr})
            print("rank k", k, [r["est_rank"] for r in rk[str(k)]], flush=True)
        R["latent_rank_clean"] = rk
    R["seconds"] = time.time() - t0
    json.dump(R, open(fo, "w"), indent=1)
    print(part, "done", R["seconds"], flush=True)


# =============================================================== F4 intervention attraction
def f4_three_gene(n=2000, reps=300, seed=0):
    rng = np.random.default_rng(seed)
    b, c, sB, sC, se = 1.0, 1.0, 0.5, 0.5, 0.5
    out = {}
    conds = {"observational": (0, 0), "perturb A": (1, 0), "perturb B": (0, 1), "perturb A and B": (1, 1)}
    for name, (pA, pB) in conds.items():
        coefs, sel, tst = [], [], []
        for _ in range(reps):
            A = rng.standard_normal(n)
            if pA:
                A[: n // 3] -= 3.0  # knockdown of A in a third of the cells
            B = b * A + sB * rng.standard_normal(n)
            if pB:
                B[n // 3: 2 * n // 3] -= 3.0  # knockdown of B in a different third
            C = c * B + sC * rng.standard_normal(n)
            Bo = B + se * rng.standard_normal(n)
            F = np.c_[A, Bo]
            beta, *_ = np.linalg.lstsq(np.c_[np.ones(n), F], C, rcond=None)
            res = C - np.c_[np.ones(n), F] @ beta
            cov = np.linalg.inv(np.cov(F.T) * (n - 1)) * res.var()
            coefs.append(beta[1]); tst.append(beta[1] / np.sqrt(cov[0, 0]))
            m = LassoLarsIC(criterion="bic").fit((F - F.mean(0)) / F.std(0), C)
            sel.append(abs(m.coef_[0]) > 1e-8)
        varB_A = sB ** 2 + (pB * (1 / 3) * (2 / 3) * 9.0)
        lam = se ** 2 / (se ** 2 + varB_A)
        out[name] = {"theory_coef_c_b_lambda": c * b * lam, "empirical_coef_mean": float(np.mean(coefs)),
                     "empirical_coef_sd": float(np.std(coefs)), "mean_t_stat": float(np.mean(tst)),
                     "lasso_selects_spurious_A_to_C": float(np.mean(sel)), "sd_A": float(np.sqrt(1 + pA * 2 / 9 * 9))}
    # sweep measurement noise
    sweep = []
    for se_ in (0.0, 0.1, 0.25, 0.5, 1.0):
        A = rng.standard_normal(20000); B = A + sB * rng.standard_normal(20000); C = B + sC * rng.standard_normal(20000)
        Bo = B + se_ * rng.standard_normal(20000)
        beta, *_ = np.linalg.lstsq(np.c_[np.ones(20000), A, Bo], C, rcond=None)
        sweep.append({"sigma_e": se_, "theory": se_ ** 2 / (se_ ** 2 + sB ** 2), "empirical": float(beta[1])})
    out["noise_sweep_observational"] = sweep
    return out


def f4_network(seeds=range(10), n_sub=3):
    rows = []; flag_stats = []
    for seed in seeds:
        d = ms.make_data(seed); A, X, V, kd, IND = d["A"], d["X"], d["V"], d["kd"], d["IND"]
        obs = d["obs"]; Af = d["A_full"]; n = len(obs)
        idx = {g: k for k, g in enumerate(obs)}
        pert_all = [int(j) for j in np.unique(kd[kd >= 0])]
        nE = int((A != 0).sum())
        for r in range(n_sub):
            rng = np.random.default_rng(100 * seed + r)
            S = set(rng.choice(pert_all, len(pert_all) // 2, replace=False).tolist())
            rws = (kd == -1) | np.isin(kd, list(S))
            m = ms.fit_vector_field(X, V, kd, np.zeros((n, n)), 0.0, rows=rws)
            Sc = ms.strength(m, X[rws]); np.fill_diagonal(Sc, -1)
            thr = np.sort(Sc.ravel())[-nE]; top = Sc >= thr
            thr2 = np.sort(Sc.ravel())[-2 * nE]; top2 = Sc >= thr2
            for jo in range(n):
                j = obs[jo]
                if j not in REGS:
                    continue
                for io in range(n):
                    if io == jo or not IND[io, jo]:
                        continue
                    i = obs[io]
                    meds = [mm for mm in range(G) if Af[mm, j] != 0 and Af[i, mm] != 0]
                    if not meds:
                        st = "longer path only"
                    elif any(mm not in idx for mm in meds):
                        st = "mediator hidden"
                    elif all(idx[mm] in S for mm in meds):
                        st = "mediator perturbed"
                    else:
                        st = "mediator unperturbed"
                    rows.append((seed, jo in S, st, bool(top[io, jo])))
            # identifiability flag on predicted edges
            for io in range(n):
                for jo in range(n):
                    if not top[io, jo]:
                        continue
                    fl = any(top2[mo, jo] and top2[io, mo] and mo not in S for mo in range(n) if mo not in (io, jo))
                    flag_stats.append((seed, bool(fl), A[io, jo] != 0, bool(IND[io, jo]), jo in S, obs[jo] in REGS))
        print("seed", seed, flush=True)
    rows = np.array(rows, dtype=object); fs = np.array(flag_stats, dtype=object)
    table = {}
    for sp in (True, False):
        for st in ("mediator unperturbed", "mediator perturbed", "mediator hidden", "longer path only"):
            m = (rows[:, 1] == sp) & (rows[:, 2] == st)
            if m.sum():
                seedrates = [rows[m & (rows[:, 0] == s), 3].astype(float).mean() for s in seeds if (m & (rows[:, 0] == s)).sum()]
                table[f"source {'perturbed' if sp else 'unperturbed'} | {st}"] = {
                    "n_pairs": int(m.sum()), "false_positive_rate": float(rows[m, 3].astype(float).mean()),
                    "seed_sd": float(np.std(seedrates))}
    fl = fs[:, 1].astype(bool); tp = fs[:, 2].astype(bool); ind = fs[:, 3].astype(bool); sp = fs[:, 4].astype(bool)
    flag = {"n_predicted_edges": int(len(fs)), "fraction_flagged": float(fl.mean()),
            "precision_flagged": float(tp[fl].mean()), "precision_unflagged": float(tp[~fl].mean()),
            "indirect_frac_flagged": float(ind[fl].mean()), "indirect_frac_unflagged": float(ind[~fl].mean()),
            "share_of_indirect_FPs_that_are_flagged": float(fl[ind].mean()),
            "share_of_TPs_flagged": float(fl[tp].mean()),
            "precision_source_perturbed": float(tp[sp].mean()), "precision_source_unperturbed": float(tp[~sp].mean())}
    rg = fs[:, 5].astype(bool)
    flag.update({"precision_regulator_source_perturbed": float(tp[sp & rg].mean()),
                 "precision_regulator_source_unperturbed": float(tp[~sp & rg].mean()),
                 "n_edges_from_unperturbed_regulators": int((~sp & rg).sum()), "n_edges_from_perturbed": int(sp.sum()),
                 "fraction_of_edges_from_non_regulators": float((~rg).mean()),
                 "precision_flagged_among_regulator_sources": float(tp[fl & rg].mean()),
                 "precision_unflagged_among_regulator_sources": float(tp[~fl & rg].mean())})
    return table, flag


def f4():
    t0 = time.time()
    R = {"three_gene": f4_three_gene()}
    print(json.dumps(R["three_gene"], indent=1), flush=True)
    R["network_fp_table"], R["identifiability_flag"] = f4_network()
    R["seconds"] = time.time() - t0
    json.dump(R, open(OUT / "f4_attraction.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "three_gene"}, indent=1))


# =============================================================== F5 prior forms + propagation depth
def f5(seed):
    t0 = time.time()
    d = ms.make_data(seed); A, X, V, kd, IND = d["A"], d["X"], d["V"], d["kd"], d["IND"]
    n = A.shape[0]; tfo = d["tf_obs"]; rego = d["reg_obs"]
    train = ~np.isin(kd, d["heldout"])
    O, P, planted = ms.binding_prior(seed, *ms.PRIORS["realistic"], d["A_full"], d["obs"])
    decoy = planted & (A == 0)
    BIG = 1e6
    PF = {}
    PF["no prior"] = np.ones((n, n))
    PF["soft site prior (a=0.8)"] = 1 - 0.8 * P
    s = np.full((n, n), 3.0); s[:, rego] = 1.0; PF["soft identity prior (non-regulators x3 penalty)"] = s
    h = np.full((n, n), BIG); h[:, rego] = 1.0; PF["hard identity mask (TFs + other regulators)"] = h
    h = np.full((n, n), BIG); h[:, tfo] = 1.0; PF["hard TF-only mask"] = h
    h = np.full((n, n), BIG); h[:, tfo] = np.where(P[:, tfo] >= 0.85, 1.0, BIG); PF["hard binding mask (TF x top-15% occupancy)"] = h
    h = np.full((n, n), BIG); h[:, rego] = 1.0; h[:, tfo] = 1 - 0.8 * P[:, tfo]; PF["hard identity mask + soft site prior"] = h
    res = {"seed": seed, "variants": {}, "depth": {}}
    models = {}
    for name, pf in PF.items():
        m = ms.fit_vector_field(X, V, kd, 1 - pf, 1.0, rows=train)
        Sc = ms.strength(m, X[train]) * (pf < BIG / 2)
        r = ms.edge_metrics(Sc, A, IND, decoy)
        r["recall_ceiling"] = float(((pf < BIG / 2) & (A != 0)).sum() / (A != 0).sum())
        r.update(ms.effect_scores(m, d, train))
        res["variants"][name] = r; models[name] = m
    # propagation depth: linearised steady-state response truncated at depth D
    wt = kd == -1
    for name in ("no prior", "hard identity mask + soft site prior", "no prior, stabilised (rho=0.8)",
                 "hard identity mask + soft site prior, stabilised (rho=0.8)"):
        m = models[name.split(", stabilised")[0]]; xb = X[train].mean(0); K = m["K"]
        dphi = 2 * xb * K ** 2 / (K ** 2 + xb ** 2) ** 2
        g = np.maximum(m["g"], 0.05)
        Jn = (m["W"] * dphi[None, :]) / g[:, None]  # dx_i = sum_j Jn_ij dx_j at steady state
        if "stabilised" in name:
            rho = np.abs(np.linalg.eigvals(Jn)).max()
            if rho > 0.8:
                Jn = Jn * (0.8 / rho)
        cors = {D: [] for D in (1, 2, 3, 4, 6, 8, 12, "inf")}
        for j in d["heldout"]:
            kj = kd == j; true = X[kj].mean(0) - X[wt].mean(0)
            keep = np.arange(n) != j
            Jc = Jn.copy(); Jc[j, :] = 0  # j clamped
            e = np.zeros(n); e[j] = true[j]
            acc = e.copy(); term = e.copy()
            for D in range(1, 13):
                term = Jc @ term; acc = acc + term
                if D in cors:
                    cors[D].append(float(pearsonr(acc[keep], true[keep])[0]) if acc[keep].std() > 0 else 0.0)
            try:
                full = np.linalg.solve(np.eye(n) - Jc, e)
                cors["inf"].append(float(pearsonr(full[keep], true[keep])[0]))
            except np.linalg.LinAlgError:
                cors["inf"].append(np.nan)
        res["depth"][name] = {str(k): float(np.nanmean(v)) for k, v in cors.items()}
        # spectral radius (does the series converge?)
        res["depth"][name]["spectral_radius"] = float(np.abs(np.linalg.eigvals(Jn)).max())
    # true graph depth of held-out KDs' downstream (longest shortest path)
    Ab = (A != 0)
    depths = []
    for j in d["heldout"]:
        frontier = {j}; seen = {j}; Dd = 0
        while True:
            nxt = {i for f in frontier for i in np.nonzero(Ab[:, f])[0]} - seen
            if not nxt:
                break
            Dd += 1; seen |= nxt; frontier = nxt
        depths.append(Dd)
    res["true_downstream_depth_heldout"] = depths
    res["seconds"] = time.time() - t0
    json.dump(res, open(OUT / f"f5_s{seed}.json", "w"), indent=1)
    for k, v in res["variants"].items():
        print(f"{k:50s} AUPRCx={v['AUPRC_ratio']:.2f} P@E={v['precision_at_E']:.2f} ceil={v['recall_ceiling']:.2f} eff={v['heldout_effect_corr']:.2f}")
    print(res["depth"], depths, res["seconds"])


# =============================================================== revision: imperfect masks
def f5_masks(seed):
    t0 = time.time()
    d = ms.make_data(seed); A, X, V, kd, IND = d["A"], d["X"], d["V"], d["kd"], d["IND"]
    n = A.shape[0]; tfo = d["tf_obs"]; rego = list(d["reg_obs"])
    train = ~np.isin(kd, d["heldout"])
    O, P, planted = ms.binding_prior(seed, *ms.PRIORS["realistic"], d["A_full"], d["obs"])
    decoy = planted & (A == 0)
    rng = np.random.default_rng(300 + seed)
    nonreg = [g for g in range(n) if g not in rego]
    BIG = 1e6
    res = {"seed": seed, "variants": {}}
    def run(name, allowed, site=False):
        pf = np.full((n, n), BIG); pf[:, allowed] = 1.0
        if site:
            tf_al = [j for j in allowed if j in tfo]
            pf[:, tf_al] = 1 - 0.8 * P[:, tf_al]
        m = ms.fit_vector_field(X, V, kd, 1 - pf, 1.0, rows=train)
        Sc = ms.strength(m, X[train]) * (pf < BIG / 2)
        r = ms.edge_metrics(Sc, A, IND, decoy)
        r["recall_ceiling"] = float(((pf < BIG / 2) & (A != 0)).sum() / (A != 0).sum())
        r["n_allowed_regulators"] = len(allowed)
        r.update(ms.effect_scores(m, d, train))
        res["variants"][name] = r
    run("exact regulator set", rego)
    for miss in (0.1, 0.25, 0.5):
        keep = list(rng.choice(rego, int(round(len(rego) * (1 - miss))), replace=False))
        run(f"missing {int(miss * 100)}% of regulators", keep)
    for add in (0.5, 1.0, 2.0):
        extra = list(rng.choice(nonreg, min(len(nonreg), int(round(len(rego) * add))), replace=False))
        run(f"+{int(add * 100)}% false regulators", rego + extra)
    keep = list(rng.choice(rego, int(round(len(rego) * 0.75)), replace=False))
    extra = list(rng.choice(nonreg, len(rego), replace=False))
    run("missing 25% + 100% false", keep + extra)
    run("missing 25% + 100% false, + soft site prior", keep + extra, site=True)
    res["seconds"] = time.time() - t0
    json.dump(res, open(OUT / f"f5masks_s{seed}.json", "w"), indent=1)
    for k, v in res["variants"].items():
        print(f"{k:45s} AUPRCx={v['AUPRC_ratio']:.2f} ceil={v['recall_ceiling']:.2f} eff={v['heldout_effect_corr']:.2f}", flush=True)


# =============================================================== revision: propagation with true vs estimated operator
def replay_params(seed):
    """Re-draw the exact simulator parameters used by medium_synthetic.make_data(seed)."""
    rng = np.random.default_rng(seed)
    A, NH, KH = ms.make_network(rng)
    eff = np.ones(G); eff[REGS] = rng.uniform(0.05, 0.3, len(REGS))
    hidden = np.r_[rng.choice(range(4, ms.NTF), 1), rng.choice(NONTF, 1), rng.choice(range(ms.NTF + ms.NREG, G), 8, replace=False)]
    pool = [j for j in REGS if j not in hidden]
    heldout = np.r_[rng.choice([j for j in pool if 4 <= j < ms.NTF], 2, replace=False), rng.choice([j for j in pool if j >= ms.NTF], 2, replace=False)]
    gamma = rng.uniform(0.5, 1.0, G)
    return A, NH, KH, eff, hidden, gamma


def prod_fn(x, A, NH, KH, u):
    ii, jj = np.nonzero(A); a, n_, k_ = A[ii, jj], NH[ii, jj], KH[ii, jj]
    h = x[jj] ** n_ / (k_ ** n_ + x[jj] ** n_)
    p = np.full(G, 0.05); np.add.at(p, ii, np.where(a > 0, a * h, -a * (1 - h))); p[:4] += u
    return p


def steady(A, NH, KH, gamma, u, s, T=60.0, dt=0.01):
    x = np.ones(G)
    for _ in range(int(T / dt)):
        x = np.maximum(x + dt * (s * prod_fn(x, A, NH, KH, u) - gamma * x), 0)
    return x


def downstream_depth(Ab, j):
    frontier = {j}; seen = {j}; D = 0
    while True:
        nxt = {i for f in frontier for i in np.nonzero(Ab[:, f])[0]} - seen
        if not nxt:
            return D
        D += 1; seen |= nxt; frontier = nxt


def trunc(Jn, j, dxj, Ds):
    n = Jn.shape[0]; Jc = Jn.copy(); Jc[j, :] = 0
    e = np.zeros(n); e[j] = dxj; acc = e.copy(); term = e.copy(); out = {}
    for D in range(1, max(Ds) + 1):
        term = Jc @ term; acc = acc + term
        if D in Ds:
            out[D] = acc.copy()
    return out


def depth_control(seed):
    t0 = time.time()
    Ds = (1, 2, 3, 4, 6, 8)
    A, NH, KH, eff, hidden, gamma = replay_params(seed)
    obs = np.array([g for g in range(G) if g not in hidden])
    u = np.full(4, 1.55)
    x0 = steady(A, NH, KH, gamma, u, np.ones(G))
    # true Jacobian of production at WT steady state (finite differences), normalised by decay
    J = np.zeros((G, G)); p0 = prod_fn(x0, A, NH, KH, u)
    for j in range(G):
        xe = x0.copy(); h = 1e-4 * max(1.0, x0[j]); xe[j] += h
        J[:, j] = (prod_fn(xe, A, NH, KH, u) - p0) / h
    Jn_true = J / gamma[:, None]
    Ab = A != 0
    res = {"seed": seed, "true_spectral_radius": float(np.abs(np.linalg.eigvals(Jn_true)).max()), "true": [], "estimated": []}
    regs = [j for j in REGS if j not in hidden]
    for j in regs:
        s = np.ones(G); s[j] = eff[j]
        xk = steady(A, NH, KH, gamma, u, s)
        dx = xk - x0
        pr = trunc(Jn_true, j, dx[j], Ds)
        keep = np.array([g for g in obs if g != j])
        rec = {"reg": int(j), "depth": downstream_depth(Ab, j)}
        for D in Ds:
            rec[str(D)] = corr_(pr[D][keep], dx[keep])
        rec["full_inverse"] = corr_(np.linalg.solve(np.eye(G) - np.where(np.arange(G)[:, None] == j, 0, Jn_true), np.eye(G)[j] * dx[j])[keep], dx[keep])
        res["true"].append(rec)
    # estimated operator: 5-fold over knockdowns, BOND-V without prior, effects in the measured (log) space
    d = ms.make_data(seed); X, V, kd = d["X"], d["V"], d["kd"]
    n = X.shape[0]; idx = {g: k for k, g in enumerate(d["obs"])}
    pert = [int(j) for j in np.unique(kd[kd >= 0])]
    r = np.random.default_rng(seed); r.shuffle(pert); folds = [pert[k::5] for k in range(5)]
    wt = kd == -1
    for fo in folds:
        rows = ~np.isin(kd, fo)
        m = ms.fit_vector_field(X, V, kd, np.zeros((X.shape[1], X.shape[1])), 0.0, rows=rows)
        xb = X[rows].mean(0); K = m["K"]; dphi = 2 * xb * K ** 2 / (K ** 2 + xb ** 2) ** 2
        Jn = (m["W"] * dphi[None, :]) / np.maximum(m["g"], 0.05)[:, None]
        rho = np.abs(np.linalg.eigvals(Jn)).max(); Jst = Jn * (0.8 / rho) if rho > 0.8 else Jn
        for jo in fo:
            true = X[kd == jo].mean(0) - X[wt].mean(0)
            keep = np.arange(X.shape[1]) != jo
            pr, ps = trunc(Jn, jo, true[jo], Ds), trunc(Jst, jo, true[jo], Ds)
            rec = {"reg": int(d["obs"][jo]), "depth": downstream_depth(Ab, int(d["obs"][jo])), "rho": float(rho)}
            for D in Ds:
                rec[str(D)] = corr_(pr[D][keep], true[keep]); rec[f"stab_{D}"] = corr_(ps[D][keep], true[keep])
            res["estimated"].append(rec)
    res["seconds"] = time.time() - t0
    json.dump(res, open(OUT / f"depthctrl_s{seed}.json", "w"), indent=1)
    print(seed, "rho_true", round(res["true_spectral_radius"], 2), round(res["seconds"]), flush=True)


def depth_control2(seed):
    """Round 2: both operators scored against the SAME targets on the SAME (log-normalised) scale.
    Targets: (i) noise-free steady-state effect mapped to log scale; (ii) measured mean effect in the simulated data.
    True-operator predictions (raw scale) are mapped to log scale with the same first-order transform."""
    t0 = time.time()
    Ds = (1, 2, 3, 4, 6, 8)
    A, NH, KH, eff, hidden, gamma = replay_params(seed)
    obs = np.array([g for g in range(G) if g not in hidden]); oi = {g: k for k, g in enumerate(obs)}
    u = np.full(4, 1.55)
    x0 = steady(A, NH, KH, gamma, u, np.ones(G))
    J = np.zeros((G, G)); p0 = prod_fn(x0, A, NH, KH, u)
    for j in range(G):
        xe = x0.copy(); h = 1e-4 * max(1.0, x0[j]); xe[j] += h
        J[:, j] = (prod_fn(xe, A, NH, KH, u) - p0) / h
    Jn_true = J / gamma[:, None]
    LIB = ms.LIB
    to_log = (2 / np.log1p(LIB)) * LIB / (1 + x0 * LIB)  # d log-normalised / d raw at the WT state
    d = ms.make_data(seed); X, V, kd = d["X"], d["V"], d["kd"]
    wt = kd == -1
    Ab = A != 0
    pert = [int(j) for j in np.unique(kd[kd >= 0])]
    r = np.random.default_rng(seed); order = list(pert); r.shuffle(order); folds = [order[k::5] for k in range(5)]
    est = {}
    for fo in folds:
        rows = ~np.isin(kd, fo)
        m = ms.fit_vector_field(X, V, kd, np.zeros((X.shape[1], X.shape[1])), 0.0, rows=rows)
        xb = X[rows].mean(0); K = m["K"]; dphi = 2 * xb * K ** 2 / (K ** 2 + xb ** 2) ** 2
        Jn = (m["W"] * dphi[None, :]) / np.maximum(m["g"], 0.05)[:, None]
        rho = np.abs(np.linalg.eigvals(Jn)).max(); Jst = Jn * (0.8 / rho) if rho > 0.8 else Jn
        for jo in fo:
            est[jo] = (Jn, Jst)
    recs = []
    for jo in pert:
        j = int(obs[jo])
        s = np.ones(G); s[j] = eff[j]
        dx = steady(A, NH, KH, gamma, u, s) - x0
        clean = (dx * to_log)[obs]                       # noise-free effect, log scale, observed genes
        meas = X[kd == jo].mean(0) - X[wt].mean(0)       # measured effect, log scale
        keep = np.arange(len(obs)) != jo
        pt = trunc(Jn_true, j, dx[j], Ds)
        Jn, Jst = est[jo]
        pe, ps = trunc(Jn, jo, meas[jo], Ds), trunc(Jst, jo, meas[jo], Ds)
        rec = {"reg": j, "depth": downstream_depth(Ab, j)}
        for D in Ds:
            ptl = (pt[D] * to_log)[obs]
            rec[f"true|clean|{D}"] = corr_(ptl[keep], clean[keep]); rec[f"true|meas|{D}"] = corr_(ptl[keep], meas[keep])
            rec[f"est|clean|{D}"] = corr_(pe[D][keep], clean[keep]); rec[f"est|meas|{D}"] = corr_(pe[D][keep], meas[keep])
            rec[f"stab|clean|{D}"] = corr_(ps[D][keep], clean[keep]); rec[f"stab|meas|{D}"] = corr_(ps[D][keep], meas[keep])
        rec["clean_vs_meas"] = corr_(clean[keep], meas[keep])
        recs.append(rec)
    json.dump({"seed": seed, "records": recs, "seconds": time.time() - t0}, open(OUT / f"depthctrl2_s{seed}.json", "w"), indent=1)
    print(seed, round(time.time() - t0), flush=True)


def corr_(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "f3":
        f3(sys.argv[2] if len(sys.argv) > 2 else "all")
    elif a == "f4":
        f4()
    elif a == "f5":
        f5(int(sys.argv[2]))
    elif a == "f5masks":
        f5_masks(int(sys.argv[2]))
    elif a == "depthctrl":
        depth_control(int(sys.argv[2]))
    elif a == "depthctrl2":
        depth_control2(int(sys.argv[2]))
