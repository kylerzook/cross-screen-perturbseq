"""BOND-D prototype: perturbation-aware latent dynamical GRN with a binding-physics prior (toy data).

Ground truth: 10 genes (TFs T0-T3, targets g4-g9), 9 directed edges incl. a cascade
T0->T1->T2->{g4,g7}, a feed-forward loop T0->g5<-T1, and repressors T3-|g6, T1-|g9.
  dx_i = kd_i * (b0 + sum_act V*hill(x_j) + sum_rep V*(1 - hill(x_j))) - g_i x_i  (+ process noise)
Cells: full trajectories (31 points, dt_obs 0.2). Wild type plus CRISPRi-like knockdowns of
T0, T1, T3 (training) and T2 (held out entirely: used only to test prediction of an unseen perturbation).
Knockdown = production x0.1 and initial level x0.1 of the target.

Binding prior: a 1-kb promoter per gene with planted sites for each TF->gene edge (half strong single
sites, half weak 6-site clusters) plus DECOYS: strong sites on non-target genes (bound, no function).
Occupancy O[i,j] = sum_sites sigmoid(E_thr_j - 2 - E) using real JASPAR motifs.

Model: gene-aligned latent ODE  dz/dt = eff(kd) * (W @ sigmoid(a(z-c)) + b) - softplus(g) z ;  W = GRN.
  observed genes: z0 = per-gene encoder(x0); unobserved genes: z0 = learned constant; rollout from t=0.
  'perturb' variants: known knockdown target, learned efficiency per target (sigmoid).
  'prior' variants: edge-wise L1 weight lam * (1 - 0.9 * p_ij), p_ij = occupancy rank-scaled per TF;
  non-TF regulators get full weight.
Variants: base | prior | perturb | prior+perturb | shuffled-prior+perturb (prior permuted within TF).
Conditions: 'observed' (all 10 genes measured), 'hiddenT1' (T1 never measured).
Metrics: edge AUROC/AUPRC, precision@9 (= recall@9), indirect-pair and decoy false-positive rates in the
top 9, sign accuracy on true edges, WT test MSE, held-out T2-knockdown trajectory MSE and effect
correlation, hidden-unit vs truth correlation.
Baselines: |correlation|, best lagged correlation, Lasso-VAR on velocity, KD mean-difference (total
effects), occupancy prior alone.

Usage: python experiments/bond_dynamics.py run <condition> <seed> <variant[,variant...]>
       python experiments/bond_dynamics.py baselines <condition> <seed>
       python experiments/bond_dynamics.py report
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bond.pwm import parse_jaspar, logodds, scan_energies
from bond.occupancy import sigmoid as np_sigmoid

OUT = ROOT / "results/bond_dynamics"
NAMES = ["T0", "T1", "T2", "T3", "g4", "g5", "g6", "g7", "g8", "g9"]
G, TFS = 10, [0, 1, 2, 3]
EDGES = [(1, 0, +1), (2, 1, +1), (4, 2, +1), (5, 0, +1), (5, 1, +1), (6, 3, -1), (7, 2, +1), (8, 3, +1), (9, 1, -1)]
GAMMA = np.array([0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.7, 0.6, 0.6, 0.7])
V, K_H, B0, SIG_P, NOISE, KD_STR = 2.0, 0.6, 0.05, 0.03, 0.05, 0.1
T_OBS, DT_OBS, DT_SIM = 31, 0.2, 0.005
N_WT, N_KD = 160, 60                       # per knockdown target
TRAIN_KD, HELDOUT_KD = [0, 1, 3], 2
CONDS = {"observed": [g for g in range(G)], "hiddenT1": [g for g in range(G) if g != 1]}
VARIANTS = {"base": (False, False, False), "prior": (True, False, False), "perturb": (False, True, False),
            "prior+perturb": (True, True, False), "shufprior+perturb": (True, True, True),
            # mechanism check: also train on T2 knockdowns (so T2 is no longer a held-out perturbation)
            "prior+perturb+T2KD": (True, True, False, True), "perturb+T2KD": (False, True, False, True)}
LAM, ITERS, SEEDS = 1e-3, 1500, [0, 1, 2]
BG = np.array([0.29, 0.21, 0.21, 0.29])


def adjacency():
    A = np.zeros((G, G), int)
    for i, j, s in EDGES:
        A[i, j] = s
    return A


def indirect_pairs(A):
    B = (A != 0).astype(int)
    two = (B @ B) > 0
    np.fill_diagonal(two, False)
    return two & (B == 0)


# ------------------------------------------------------------------ simulation
def simulate(rng, kd_target):
    """kd_target: (N,) int, -1 = wild type."""
    N = len(kd_target)
    hill = lambda x: x ** 2 / (K_H ** 2 + x ** 2)
    x = rng.uniform(0, 0.3, (N, G))
    x[:, 0] = rng.uniform(0.5, 3.0, N); x[:, 3] = rng.uniform(0.5, 3.0, N)
    kdm = np.ones((N, G))
    for n, t in enumerate(kd_target):
        if t >= 0:
            kdm[n, t] = KD_STR; x[n, t] *= KD_STR
    every = int(round(DT_OBS / DT_SIM))
    traj = np.zeros((N, T_OBS, G)); traj[:, 0] = x
    for step in range(1, (T_OBS - 1) * every + 1):
        prod = np.full((N, G), B0)
        for i, j, s in EDGES:
            prod[:, i] += V * (hill(x[:, j]) if s > 0 else 1 - hill(x[:, j]))
        x = x + (kdm * prod - GAMMA * x) * DT_SIM + SIG_P * np.sqrt(DT_SIM) * rng.standard_normal((N, G))
        x = np.maximum(x, 0)
        if step % every == 0:
            traj[:, step // every] = x
    return traj


def make_data(seed):
    rng = np.random.default_rng(seed)
    kd = np.r_[np.full(N_WT, -1), np.repeat(TRAIN_KD + [HELDOUT_KD], N_KD)]
    truth = simulate(rng, kd)
    X = truth + NOISE * truth.std((0, 1)) * rng.standard_normal(truth.shape)
    split = np.zeros(len(kd), bool)  # True = test
    for t in [-1] + TRAIN_KD:
        idx = np.nonzero(kd == t)[0]; split[rng.choice(idx, len(idx) // 4, replace=False)] = True
    split[kd == HELDOUT_KD] = True
    return truth, X, kd, split


# ------------------------------------------------------------------ binding prior from sequence
def binding_prior(seed, n_decoy=4):
    rng = np.random.default_rng(1000 + seed)
    ms = [m for m in parse_jaspar(ROOT / "data/raw/jaspar2024_vert_nr.txt")
          if "::" not in m["name"] and 9 <= m["pfm"].shape[1] <= 14]
    motifs = [ms[i] for i in rng.choice(len(ms), 4, replace=False)]
    Ws = [logodds(m["pfm"], BG) for m in motifs]

    def ethr(W):
        s = rng.choice(4, size=(4000, 100 + W.shape[0] - 1), p=BG).astype(np.int8)
        return float(np.quantile(scan_energies(s, W)[0][:, 0, :].ravel(), 1e-4))
    Ethr = [ethr(W) for W in Ws]

    def site(W, lo, hi):
        p = np.exp(W) * BG; p /= p.sum(1, keepdims=True); smax = W.max(1).sum()
        best, bestd = None, np.inf
        for _ in range(60):
            s = np.array([rng.choice(4, 3000, p=p[k]) for k in range(W.shape[0])]).T
            mut = rng.random(s.shape) < rng.uniform(0, 0.6, (3000, 1)); s[mut] = rng.integers(0, 4, mut.sum())
            E = smax - W[np.arange(W.shape[0]), s].sum(1); ok = np.nonzero((E >= lo) & (E <= hi))[0]
            if ok.size:
                return s[ok[0]].astype(np.int8)
            d = np.abs(E - (lo + hi) / 2); k = d.argmin()
            if d[k] < bestd:
                best, bestd = s[k].astype(np.int8), d[k]
        return best  # closest available energy (rare)
    seqs = rng.choice(4, size=(G, 1000), p=BG).astype(np.int8)
    used = np.zeros((G, 1000), bool)

    def plant(i, j, strong):
        W = Ws[j]; w = W.shape[0]
        for _ in range(1 if strong else 6):
            for _t in range(500):
                pos = rng.integers(0, 1000 - w)
                if not used[i, pos:pos + w].any():
                    break
            s = site(W, 0.0, 1.5) if strong else site(W, Ethr[j] + 0.5, Ethr[j] + 2.5)
            seqs[i, pos:pos + w] = s if rng.random() < 0.5 else (3 - s)[::-1]; used[i, pos:pos + w] = True
    kinds = {}
    for i, j, s in EDGES:
        if j in TFS:
            strong = bool(rng.random() < 0.5); plant(i, j, strong); kinds[(i, j)] = "strong" if strong else "weak"
    A = adjacency()
    cand = [(i, j) for i in range(G) for j in TFS if i != j and A[i, j] == 0]
    decoys = [cand[k] for k in rng.choice(len(cand), n_decoy, replace=False)]
    for i, j in decoys:
        plant(i, j, True)
    O = np.zeros((G, G))
    for j in TFS:
        E = scan_energies(seqs, Ws[j])[0]
        O[:, j] = np_sigmoid(Ethr[j] - 2 - E).sum(axis=(1, 2))
    np.fill_diagonal(O, 0)
    P = np.zeros((G, G))
    for j in TFS:  # rank-scale per TF column to [0, 1]
        col = O[:, j].copy(); col[j] = -np.inf
        r = col.argsort().argsort().astype(float) / (G - 1)
        P[:, j] = r; P[j, j] = 0
    D = np.zeros((G, G), bool)
    for i, j in decoys:
        D[i, j] = True
    return O, P, D, kinds


# ------------------------------------------------------------------ model
class GeneMLP(nn.Module):
    def __init__(self, n, h=16):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n, h) * 0.5); self.b1 = nn.Parameter(torch.zeros(n, h))
        self.w2 = nn.Parameter(torch.zeros(n, h))

    def forward(self, x):
        return x + (torch.tanh(x[..., None] * self.w1 + self.b1) * self.w2).sum(-1)


class BondDyn(nn.Module):
    def __init__(self, obs, kd_targets):
        super().__init__()
        self.obs = torch.tensor(obs); self.hid = torch.tensor([g for g in range(G) if g not in obs], dtype=torch.long)
        self.enc, self.dec = GeneMLP(len(obs)), GeneMLP(len(obs))
        self.z0 = nn.Parameter(torch.zeros(G))
        self.W = nn.Parameter(0.01 * torch.randn(G, G)); self.register_buffer("mask", 1 - torch.eye(G))
        self.a = nn.Parameter(torch.zeros(G)); self.c = nn.Parameter(torch.zeros(G))
        self.g = nn.Parameter(torch.zeros(G)); self.b = nn.Parameter(torch.zeros(G))
        self.kd_logit = nn.Parameter(torch.zeros(G))  # one learned efficiency per possible target

    def Wm(self):
        return self.W * self.mask

    def act(self, z):
        return torch.sigmoid((F.softplus(self.a) + 0.1) * (z - self.c))

    def kd_scale(self, kd, eff_override=None):
        """kd: (B,) long, -1 = WT. Returns (B, G) production multipliers."""
        s = torch.ones(kd.shape[0], G)
        m = kd >= 0
        if m.any():
            eff = torch.sigmoid(self.kd_logit) if eff_override is None else torch.full((G,), eff_override)
            s = s.clone(); s[m, kd[m]] = eff[kd[m]]
        return s

    def init(self, x0):
        z = self.z0.expand(x0.shape[0], G).clone()
        z[:, self.obs] = self.enc(x0[:, self.obs])
        return z

    def rollout(self, x0, steps, kd=None, eff_override=None, sub=3):
        z = self.init(x0); scale = None if kd is None else self.kd_scale(kd, eff_override)
        out, h = [z], DT_OBS / sub
        for _ in range(steps):
            for _ in range(sub):
                prod = self.act(z) @ self.Wm().T + self.b
                if scale is not None:
                    prod = prod * scale
                z = z + h * (prod - F.softplus(self.g) * z)
            out.append(z)
        return torch.stack(out, 1)

    def decode(self, Z):
        return self.dec(Z[..., self.obs])


def train(X, kd, obs, use_kd, lam_matrix, seed):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32); KD = torch.tensor(kd, dtype=torch.long)
    m = BondDyn(obs, TRAIN_KD)
    L = torch.tensor(lam_matrix, dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    ob = torch.tensor(obs)
    for it in range(ITERS):
        ci = torch.randint(0, X.shape[0], (128,))
        x = Xt[ci]
        Z = m.rollout(x[:, 0], T_OBS - 1, kd=KD[ci] if use_kd else None)
        pred = m.decode(Z)
        loss = F.mse_loss(pred, x[..., ob]) + F.mse_loss(m.dec(m.enc(x[:, 0, ob])), x[:, 0, ob]) + (L * m.Wm().abs()).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m


# ------------------------------------------------------------------ metrics
def edge_metrics(S, A, IND, D, k=9):
    from sklearn.metrics import roc_auc_score, average_precision_score
    off = ~np.eye(G, dtype=bool)
    y = (A != 0)[off]; s = S[off]
    order = np.argsort(-S[off], kind="stable")[:k]
    top = np.zeros(off.sum(), bool); top[order] = True
    return {"AUROC": float(roc_auc_score(y, s)), "AUPRC": float(average_precision_score(y, s)),
            "precision_at_9": float(y[top].mean()),
            "indirect_FP_in_top9": float(IND[off][top].sum()), "indirect_pairs_total": int(IND[off].sum()),
            "decoy_FP_in_top9": float(D[off][top].sum()), "decoys_total": int(D[off].sum()),
            "indirect_mean_rank_pct": float(np.mean([(s > v).mean() for v in s[IND[off]]])) if IND[off].any() else None}


def evaluate(m, X, truth, kd, split, obs, use_kd):
    A = adjacency()
    tr = ~split
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32); KD = torch.tensor(kd, dtype=torch.long)
        Ztr = m.rollout(Xt[tr, 0], T_OBS - 1, kd=KD[tr] if use_kd else None)
        a = m.act(Ztr.reshape(-1, G)); rng_ = (a.max(0).values - a.min(0).values).numpy()
        W = m.Wm().numpy(); S = np.abs(W) * rng_[None, :]
        res = {"W": W.round(4).tolist(), "strength": S.round(4).tolist(),
               "sign_accuracy_true_edges": float(np.mean([np.sign(W[i, j]) == s for i, j, s in EDGES])),
               "learned_kd_eff": {NAMES[t]: float(torch.sigmoid(m.kd_logit[t])) for t in TRAIN_KD + [HELDOUT_KD]} if use_kd else None}
        res["T2_beats_T1_as_regulator_of_g4_g7"] = float(np.mean([S[4, 2] > S[4, 1], S[7, 2] > S[7, 1]]))
        ob = torch.tensor(obs)
        wt_te = split & (kd == -1)
        P = m.decode(m.rollout(Xt[wt_te, 0], T_OBS - 1))
        res["WT_test_mse"] = float(F.mse_loss(P, Xt[wt_te][..., ob]))
        # held-out knockdown (T2): applied with the nominal strength 0.1 for every model (fair)
        ho = kd == HELDOUT_KD
        Pk = m.decode(m.rollout(Xt[ho, 0], T_OBS - 1, kd=KD[ho], eff_override=KD_STR)).numpy()
        Pw = m.decode(m.rollout(Xt[wt_te, 0], T_OBS - 1)).numpy()
        res["heldoutKD_traj_mse"] = float(((Pk - X[ho][..., obs]) ** 2).mean())
        pe = Pk.mean(0) - Pw.mean(0); te = X[ho][..., obs].mean(0) - X[wt_te][..., obs].mean(0)
        keep = [k for k, g in enumerate(obs) if g != HELDOUT_KD]
        pe, te = pe[:, keep], te[:, keep]
        res["heldoutKD_effect_corr"] = float(np.corrcoef(pe.ravel(), te.ravel())[0, 1])
        big = np.abs(te[-10:].mean(0)) > 0.1
        res["heldoutKD_effect_sign_acc"] = float((np.sign(pe[-10:].mean(0)[big]) == np.sign(te[-10:].mean(0)[big])).mean()) if big.any() else None
        res["heldoutKD_genes_with_effect"] = [NAMES[obs[keep[k]]] for k in np.nonzero(big)[0]]
        hid = [g for g in range(G) if g not in obs]
        if hid:
            Zt = m.rollout(Xt[split, 0], T_OBS - 1, kd=KD[split] if use_kd else None)[..., hid[0]].numpy()
            res["hidden_unit_corr_truth"] = float(np.corrcoef(Zt.ravel(), truth[split][..., hid[0]].ravel())[0, 1])
    return S, res


def run(cond, seed, variants):
    obs = CONDS[cond]
    truth, X, kd, split = make_data(seed)
    O, P, D, kinds = binding_prior(seed)
    A = adjacency(); IND = indirect_pairs(A)
    OUT.mkdir(parents=True, exist_ok=True)
    for v in variants:
        use_prior, use_kd, shuf = VARIANTS[v][:3]
        allkd = len(VARIANTS[v]) > 3
        sp = split.copy()
        if allkd:  # 3/4 of T2-knockdown cells move into training
            idx = np.nonzero(kd == HELDOUT_KD)[0]
            sp[np.random.default_rng(99 + seed).choice(idx, 3 * len(idx) // 4, replace=False)] = False
        Pv = P.copy()
        if shuf:
            r = np.random.default_rng(7 + seed)
            for j in TFS:
                idx = [i for i in range(G) if i != j]; Pv[idx, j] = Pv[r.permutation(idx), j]
        lam = LAM * ((1 - 0.9 * Pv) if use_prior else np.ones((G, G)))
        t0 = time.time()
        m = train(X[~sp], kd[~sp], obs, use_kd, lam, seed)
        S, res = evaluate(m, X, truth, kd, sp, obs, use_kd)
        if allkd:  # T2 knockdown was seen in training: relabel so it is not reported as held-out
            for k in [k for k in res if k.startswith("heldoutKD")]:
                res[k.replace("heldoutKD", "seenT2KD")] = res.pop(k)
        res.update(edge_metrics(S, A, IND, D)); res["seconds"] = time.time() - t0
        res.update(condition=cond, seed=seed, variant=v)
        json.dump(res, open(OUT / f"{cond}_s{seed}_{v}.json", "w"), indent=1)
        print(cond, seed, v, f"{res['seconds']:.0f}s AUROC={res['AUROC']:.3f} AUPRC={res['AUPRC']:.3f} P@9={res['precision_at_9']:.2f} "
              f"ind={res['indirect_FP_in_top9']} dec={res['decoy_FP_in_top9']} hoKDr={res.get('heldoutKD_effect_corr', res.get('seenT2KD_effect_corr')):.2f}", flush=True)


def baselines(cond, seed):
    from sklearn.linear_model import LassoCV
    obs = CONDS[cond]
    truth, X, kd, split = make_data(seed)
    O, P, D, kinds = binding_prior(seed)
    A = adjacency(); IND = indirect_pairs(A)
    tr = ~split; Xtr, kdtr = X[tr], kd[tr]
    out = {"prior_kinds": {f"{NAMES[j]}->{NAMES[i]}": k for (i, j), k in kinds.items()},
           "decoys": [f"{NAMES[j]}->{NAMES[i]}" for i, j in zip(*np.nonzero(D))]}
    S = {}
    wt = Xtr[kdtr == -1]; flat = wt.reshape(-1, G)
    C = np.zeros((G, G)); L = np.zeros((G, G)); LV = np.zeros((G, G)); MD = np.zeros((G, G))
    for i in obs:
        for j in obs:
            if i == j:
                continue
            C[i, j] = abs(np.corrcoef(flat[:, i], flat[:, j])[0, 1])
            L[i, j] = max(np.corrcoef(wt[:, :-l, j].ravel(), wt[:, l:, i].ravel())[0, 1] for l in range(1, 8))
    Z = Xtr[:, :-1, obs].reshape(-1, len(obs)); Z = (Z - Z.mean(0)) / Z.std(0)
    for i in obs:
        vel = ((Xtr[:, 1:, i] - Xtr[:, :-1, i]) / DT_OBS).ravel()
        coef = LassoCV(cv=5, random_state=0, n_alphas=30).fit(Z, vel).coef_
        for k, j in enumerate(obs):
            if j != i:
                LV[i, j] = abs(coef[k])
    for j in TRAIN_KD:  # total effect of knocking down j on each gene (includes indirect effects)
        kdm = Xtr[kdtr == j].mean(0); wtm = wt.mean(0)
        for i in obs:
            if i != j:
                MD[i, j] = np.abs(kdm[:, i] - wtm[:, i]).mean() / (wt[:, :, i].std() + 1e-9)
    S["|corr| (WT)"] = C; S["lagged corr"] = L; S["Lasso-VAR"] = LV
    S["KD mean-difference (total effect)"] = MD; S["binding prior only"] = O
    S["KD mean-diff + prior (product)"] = MD * (P + 0.05)
    out["scores"] = {k: edge_metrics(v, A, IND, D) for k, v in S.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{cond}_s{seed}_baselines.json", "w"), indent=1)
    for k, v in out["scores"].items():
        print(cond, seed, k, f"AUROC={v['AUROC']:.3f} AUPRC={v['AUPRC']:.3f} P@9={v['precision_at_9']:.2f} ind={v['indirect_FP_in_top9']} dec={v['decoy_FP_in_top9']}")


def report():
    ms = lambda xs: (f"{np.mean(xs):.3f} ± {np.std(xs):.3f}" if len(xs) else "–")
    rows = ["# BOND-D: perturbation-aware dynamics + binding prior (toy, numeric results)", "",
            f"Generated by `experiments/bond_dynamics.py report`. Seeds {SEEDS}; mean ± sd. 90 candidate directed edges, 9 true. "
            "Top-9 = the 9 highest-scoring edges. Training knockdowns: T0, T1, T3. Held-out knockdown: T2 (applied at test with nominal "
            "strength 0.1). '+T2KD' variants also train on T2 knockdowns, so their T2 metrics are 'seen', not held-out.", ""]
    hdr = ("| Method | AUROC | AUPRC | Precision@9 | Indirect FP in top-9 | Decoy FP in top-9 | Sign acc. (true edges) | "
           "T2 beats T1 as regulator of g4/g7 | WT test MSE | T2-KD traj MSE | T2-KD effect corr | T2-KD effect sign acc | Hidden T1 unit vs truth r |")
    for cond in CONDS:
        rows += [f"## Condition: {cond}", "", hdr, "|" + "---|" * 13]
        for v in VARIANTS:
            R = [json.load(open(OUT / f"{cond}_s{s}_{v}.json")) for s in SEEDS]
            g = lambda k: ms([r[k] for r in R if r.get(k) is not None])
            gk = lambda k: ms([r.get(f"heldoutKD_{k}", r.get(f"seenT2KD_{k}")) for r in R if r.get(f"heldoutKD_{k}", r.get(f"seenT2KD_{k}")) is not None])
            t2 = ms([np.mean([np.array(r["strength"])[4, 2] > np.array(r["strength"])[4, 1],
                              np.array(r["strength"])[7, 2] > np.array(r["strength"])[7, 1]]) for r in R])
            tag = " (T2 seen)" if "T2KD" in v else ""
            rows.append(f"| model: {v} | {g('AUROC')} | {g('AUPRC')} | {g('precision_at_9')} | {g('indirect_FP_in_top9')} | {g('decoy_FP_in_top9')} | "
                        f"{g('sign_accuracy_true_edges')} | {t2} | {g('WT_test_mse')} | {gk('traj_mse')}{tag} | {gk('effect_corr')}{tag} | "
                        f"{gk('effect_sign_acc')}{tag} | {g('hidden_unit_corr_truth')} |")
        B = [json.load(open(OUT / f"{cond}_s{s}_baselines.json")) for s in SEEDS]
        for k in B[0]["scores"]:
            g = lambda kk: ms([b["scores"][k][kk] for b in B])
            rows.append(f"| baseline: {k} | {g('AUROC')} | {g('AUPRC')} | {g('precision_at_9')} | {g('indirect_FP_in_top9')} | {g('decoy_FP_in_top9')} | – | – | – | – | – | – | – |")
        r0 = json.load(open(OUT / f"{cond}_s0_base.json"))
        rows += ["", f"Indirect pairs in truth: {r0['indirect_pairs_total']} (T0→T2, T1→g4, T1→g7, T0→g9); decoy pairs: {r0['decoys_total']}; "
                 f"genes with a real late T2-knockdown effect: {', '.join(r0['heldoutKD_genes_with_effect'])}.", ""]
        for v in ["prior+perturb", "prior+perturb+T2KD"]:
            R = [json.load(open(OUT / f"{cond}_s{s}_{v}.json")) for s in SEEDS]
            effs = {t: ms([r["learned_kd_eff"][t] for r in R if t in r["learned_kd_eff"]]) for t in R[-1]["learned_kd_eff"]}
            rows += [f"Learned knockdown efficiencies, {v} (true 0.1; T0/T3 have no regulated production, so theirs is unidentifiable): "
                     + ", ".join(f"{k}={v_}" for k, v_ in effs.items()), ""]
    (ROOT / "docs/05_bond_dynamics_results.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    torch.set_num_threads(3)
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), sys.argv[4].split(","))
    elif sys.argv[1] == "baselines":
        baselines(sys.argv[2], int(sys.argv[3]))
    else:
        report()
