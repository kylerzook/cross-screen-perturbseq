"""Real-data test: interventional GRN on Replogle 2022 K562 genome-wide Perturb-seq (pseudobulk),
scored on HELD-OUT knockdowns, with and without a binding prior (K562 ChIP-seq or JASPAR occupancy).

Setting (snapshot pseudobulk, so no dynamics): steady-state linear SEM on knockdown responses
    r_j = B r_j + d_j e_j     (r_j = z-scored pseudobulk profile of knockdown j over the gene set)
  * KD-aware nodewise regression (DCDI/inspre-style): gene i's row of B is fit on profiles of
    knockdowns that do NOT target i; held-out knockdowns are never used for fitting.
  * prior: adaptive-lasso penalty factor 1 - a * p_ik for TF regulators k (p = ChIP-bound indicator,
    or occupancy rank); non-TF regulators unchanged.
  * prediction for a held-out knockdown j: r = c (I-B)^-1 e_j (truncated Neumann series, 4 hops),
    c scaled so the predicted self-knockdown equals the observed z of the target gene.
Gene set: perturbed genes with strong knockdown (fold_expr < 0.5), measured, ranked by number of
responders |z| >= 1, top 200, plus all TFs passing the same filters that have K562 ChIP-seq.
Metrics (5-fold CV over knockdowns): Pearson r of predicted vs observed held-out profiles (target
excluded), sign accuracy on |z| >= 1 responders, top-20 responder overlap; CausalBench-style edge
evaluation on held-out regulators: mean |z| of the target under the regulator's knockdown for the
top-K predicted edges vs all pairs; ChIP enrichment of top TF->gene edges (no-prior model only).
Baselines: mean training profile (generic response), target-only (no propagation), KD-unaware model.
"""
import gzip, json, sys, time, urllib.request, urllib.parse
from pathlib import Path
import numpy as np, h5py
from sklearn.linear_model import LassoCV, Lasso

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"; OUT = ROOT / "results/replogle"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from bond.pwm import parse_jaspar, logodds, scan_energies
from bond.occupancy import sigmoid


def load_perturb():
    f = h5py.File(RAW / "K562_gwps_normalized_bulk_01.h5ad", "r")
    cats = [x.decode() if isinstance(x, bytes) else x for x in f["var/__categories/gene_name"][:]]
    var = np.array([cats[c] for c in f["var/gene_name"][:]])
    gt = [x.decode() for x in f["obs/gene_transcript"][:]]
    fe = f["obs/fold_expr"][:]
    X = np.nan_to_num(f["X"][:], nan=0.0, posinf=0.0, neginf=0.0)
    return var, np.array([s.split("_")[1] for s in gt]), fe, X


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
        nm = "_".join(acc.split("_")[:2]); sym = nm2sym.get(nm); chrom = loc.split(":")[0]
        seq = "".join(buf)
        if nm.startswith("NM_") and sym in keep and sym not in out and "_" not in chrom and len(seq) == 1000:
            a, b = loc.split(":")[1].split("-")
            out[sym] = (lut[np.frombuffer(seq.encode(), np.uint8)], chrom, int(a) - 1, int(b))
    with gzip.open(RAW / "upstream1000.fa.gz", "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                flush(); name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    flush()
    return out


def encode_chip_k562(tfs):
    q = [("type", "File"), ("status", "released"), ("assembly", "GRCh38"), ("format", "json"), ("limit", "all"),
         ("assay_title", "TF ChIP-seq"), ("biosample_ontology.term_name", "K562"), ("file_format", "bed"),
         ("field", "href"), ("field", "file_size"), ("field", "output_type"), ("field", "target.label")]
    best = {}
    for ot, rank in [("conservative IDR thresholded peaks", 0), ("IDR thresholded peaks", 1)]:
        url = "https://www.encodeproject.org/search/?" + urllib.parse.urlencode(q + [("output_type", ot)])
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=120) as r:
            for f in json.load(r)["@graph"]:
                t = (f.get("target") or {}).get("label", "").upper()
                if t in tfs and (t not in best or (rank, -f.get("file_size", 0)) < best[t][:2]):
                    best[t] = (rank, -f.get("file_size", 0), f["href"])
    peaks = {}
    for t, (_, _, href) in best.items():
        p = RAW / "encode" / f"K562_{t}.bed.gz"
        if not p.exists():
            urllib.request.urlretrieve("https://www.encodeproject.org" + href, p)
        d = {}
        with gzip.open(p, "rt") as fh:
            for line in fh:
                c, s, e = line.split("\t")[:3]; d.setdefault(c, []).append((int(s), int(e)))
        peaks[t] = {c: np.array(sorted(v)) for c, v in d.items()}
    return peaks


def bound(peaks_t, chrom, s, e):
    if chrom not in peaks_t:
        return False
    a = peaks_t[chrom]; j = np.searchsorted(a[:, 0], e) - 1
    return bool(j >= 0 and np.maximum.accumulate(a[:, 1])[j] > s)


def build():
    cache = OUT / "setup.npz"
    if cache.exists():
        return dict(np.load(cache, allow_pickle=True))
    var, pg, fe, X = load_perturb()
    col = {g: i for i, g in enumerate(var)}
    best = {}
    for r, g in enumerate(pg):
        if g in col and fe[r] == fe[r] and fe[r] < 0.5 and (g not in best or fe[r] < fe[best[g]]):
            best[g] = r
    cand = np.array(sorted(best))
    nresp = np.array([(np.abs(X[best[g]]) >= 1).sum() for g in cand])
    J = {m["name"].upper() for m in parse_jaspar(RAW / "jaspar2024_vert_nr.txt") if "::" not in m["name"]}
    tf_cand = [g for g, n in zip(cand, nresp) if g in J and n >= 5]
    peaks = encode_chip_k562(set(tf_cand))
    top = list(cand[np.argsort(-nresp)[:200]])
    genes = sorted(set(top) | set(peaks))
    prom = promoters(set(genes))
    genes = [g for g in genes if g in prom]
    idx = np.array([col[g] for g in genes])
    R = np.stack([X[best[g], idx] for g in genes])  # rows: knockdown of gene g; cols: genes
    n = len(genes); tfs = [k for k, g in enumerate(genes) if g in peaks]
    chip = np.zeros((n, n))
    for k in tfs:
        for i, g in enumerate(genes):
            if i != k:
                _, c, s, e = prom[g]; chip[i, k] = bound(peaks[genes[k]], c, s, e)
    # JASPAR occupancy prior (promoter, no accessibility)
    Jm = {m["name"].upper(): m for m in parse_jaspar(RAW / "jaspar2024_vert_nr.txt") if "::" not in m["name"]}
    seqs = np.stack([prom[g][0] for g in genes])
    bg = np.bincount(seqs.ravel(), minlength=5)[:4].astype(float); bg = (bg + bg[::-1]) / 2; bg /= bg.sum()
    occ = np.zeros((n, n))
    rng = np.random.default_rng(0)
    for k in tfs:
        W = logodds(Jm[genes[k]]["pfm"], bg)
        s = rng.choice(4, size=(3000, 100 + W.shape[0] - 1), p=bg).astype(np.int8)
        et = float(np.quantile(scan_energies(s, W)[0][:, 0, :].ravel(), 1e-4))
        occ[:, k] = sigmoid(et - 2 - scan_energies(seqs, W)[0]).sum(axis=(1, 2))
    np.fill_diagonal(occ, 0)
    d = dict(genes=np.array(genes), R=R, tfs=np.array(tfs), chip=chip, occ=occ,
             fold=np.array([fe[best[g]] for g in genes]))
    np.savez(cache, **d)
    return d


def rank_prior(M, tfs):
    n = M.shape[0]; P = np.zeros_like(M)
    for k in tfs:
        col = M[:, k].copy(); col[k] = -np.inf
        P[:, k] = col.argsort().argsort() / (n - 1); P[k, k] = 0
    return P


def fit_B(R, train_rows, P, alpha, kd_aware=True, lam=None):
    n = R.shape[1]; B = np.zeros((n, n)); pf_all = 1 - alpha * P; lams = {}
    for i in range(n):
        rows = [j for j in train_rows if (j != i or not kd_aware)]
        cols = [k for k in range(n) if k != i]
        F = R[np.ix_(rows, cols)]; y = R[rows, i]
        mu, sd = F.mean(0), F.std(0) + 1e-9
        Fs = (F - mu) / sd / pf_all[i, cols]
        if lam is None:
            m = LassoCV(cv=3, alphas=15, max_iter=3000, random_state=0).fit(Fs, y); lams[i] = m.alpha_
        else:
            m = Lasso(alpha=lam[i], max_iter=3000).fit(Fs, y)
        B[i, cols] = m.coef_ / pf_all[i, cols] / sd
    return B, lams


def predict(B, j, zself, hops=4):
    n = B.shape[0]; e = np.zeros(n); e[j] = 1.0; r = e.copy(); term = e.copy()
    for _ in range(hops):
        term = B @ term; r = r + term
    return r * (zself / r[j] if abs(r[j]) > 1e-9 else zself)


def main():
    t0 = time.time()
    d = build(); genes, R, tfs = list(d["genes"]), d["R"], list(d["tfs"]); n = len(genes)
    print(f"setup {time.time() - t0:.0f}s genes={n} TFs with ChIP={len(tfs)} chip-bound frac={d['chip'][:, tfs].mean():.3f}", flush=True)
    Pc = d["chip"].copy(); Po = rank_prior(d["occ"], tfs)
    rs = np.random.default_rng(1); Ps = Pc.copy()
    for k in tfs:
        idx = [i for i in range(n) if i != k]; Ps[idx, k] = Pc[rs.permutation(idx), k]
    variants = {"KD-aware, no prior": (np.zeros((n, n)), 0.0, True),
                "KD-aware + ChIP prior (a=0.8)": (Pc, 0.8, True),
                "KD-aware + JASPAR occupancy prior (a=0.8)": (Po, 0.8, True),
                "KD-aware + shuffled ChIP prior (a=0.8)": (Ps, 0.8, True),
                "KD-unaware, no prior": (np.zeros((n, n)), 0.0, False)}
    folds = np.array_split(np.random.default_rng(0).permutation(n), 5)
    res = {k: {"corr": [], "sign_acc": [], "top20_overlap": [], "edge_mean_absz_topK": [], "edge_chip_frac_topTF": []} for k in variants}
    base = {"mean training profile": {"corr": [], "sign_acc": [], "top20_overlap": []},
            "target only (no propagation)": {"corr": [], "sign_acc": [], "top20_overlap": []}}
    allpairs_absz = []
    for fi, fo in enumerate(folds):
        tr = [j for j in range(n) if j not in fo]
        lam = None
        for name, (P, a, aware) in variants.items():
            B, lams = fit_B(R, tr, P, a, aware, lam=None if name.startswith("KD-aware, no prior") or lam is None else lam)
            if name == "KD-aware, no prior":
                lam = lams  # same per-gene lambda reused for prior variants -> isolates the prior's effect
            S = np.abs(B)
            for j in fo:
                obs = R[j].copy(); keep = np.arange(n) != j
                pr = predict(B, j, R[j, j])
                res[name]["corr"].append(np.corrcoef(pr[keep], obs[keep])[0, 1] if pr[keep].std() > 1e-12 else 0.0)
                big = keep & (np.abs(obs) >= 1)
                res[name]["sign_acc"].append(float((np.sign(pr[big]) == np.sign(obs[big])).mean()) if big.any() else np.nan)
                t_o = set(np.argsort(-np.abs(np.where(keep, obs, 0)))[:20]); t_p = set(np.argsort(-np.abs(np.where(keep, pr, 0)))[:20])
                res[name]["top20_overlap"].append(len(t_o & t_p) / 20)
            # CausalBench-style: edges out of held-out regulators (their own knockdowns unseen in training)
            cols = list(fo); sub = S[:, cols]; K = 200
            flat = np.argsort(-sub.ravel())[:K]
            ii, kk = np.unravel_index(flat, sub.shape)
            res[name]["edge_mean_absz_topK"].append(float(np.mean([abs(R[cols[k], i]) for i, k in zip(ii, kk)])))
            if fi == 0 and name == "KD-aware, no prior":
                pass
            tsub = [k for k in tfs]
            if tsub:
                St = S[:, tsub].copy(); St[tsub, range(len(tsub))] = 0
                top = np.argsort(-St.ravel())[:200]; a_, b_ = np.unravel_index(top, St.shape)
                res[name]["edge_chip_frac_topTF"].append(float(d["chip"][a_, np.array(tsub)[b_]].mean()))
            print(f"fold {fi} {name}: corr={np.nanmean(res[name]['corr'][-len(fo):]):.3f} ({time.time() - t0:.0f}s)", flush=True)
        for j in fo:
            obs = R[j].copy(); keep = np.arange(n) != j
            allpairs_absz.append(np.abs(obs[keep]).mean())
            mp = R[tr].mean(0)
            for bname, pr in [("mean training profile", mp), ("target only (no propagation)", np.eye(n)[j] * R[j, j])]:
                c = np.corrcoef(pr[keep], obs[keep])[0, 1] if pr[keep].std() > 1e-12 else 0.0
                base[bname]["corr"].append(c)
                big = keep & (np.abs(obs) >= 1)
                base[bname]["sign_acc"].append(float((np.sign(pr[big]) == np.sign(obs[big])).mean()) if big.any() else np.nan)
                t_o = set(np.argsort(-np.abs(np.where(keep, obs, 0)))[:20]); t_p = set(np.argsort(-np.abs(np.where(keep, pr, 0)))[:20])
                base[bname]["top20_overlap"].append(len(t_o & t_p) / 20)
    summ = lambda v: [float(np.nanmean(v)), float(np.nanstd(v)), int(np.sum(~np.isnan(np.array(v, float))))]
    out = {"n_genes": n, "n_tfs_with_chip": len(tfs), "chip_bound_base_rate": float(d["chip"][:, tfs].sum() / (len(tfs) * (n - 1))),
           "mean_absz_all_pairs": float(np.mean(allpairs_absz)),
           "models": {k: {m: summ(v) for m, v in r.items()} for k, r in res.items()},
           "baselines": {k: {m: summ(v) for m, v in r.items()} for k, r in base.items()}}
    json.dump(out, open(OUT / "replogle_heldout.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
