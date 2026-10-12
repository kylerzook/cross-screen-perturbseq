"""F6: replicate 'context changes gain, not wiring' in two more datasets (scPerturb h5ad files).

Shifrut 2018 — primary human T cells, CRISPR knockout of 20 genes, cells with vs without TCR
  stimulation. Context = stimulation. For each knockout: effect = mean(KO) - mean(non-targeting) inside
  each condition (log CP10k), z from Welch SE.
Wessels 2023 — THP-1 monocytes, Cas13 RNA knockdown, single and dual perturbations. Context = a second
  knockdown: effect of A alone = mean(A) - mean(control); effect of A in C's context = mean(AC) - mean(C).

Metrics (same as the Norman analysis): among targets with |z| > 3 in both contexts, the fraction whose
sign flips; the same for a split-half null within one context; gain = slope of context-2 effect on
context-1 effect (total least squares on strong targets) and median |ratio|; Jaccard of significant sets;
correlation of effects on the union of the top-50 targets.

Usage: python experiments/final_context.py shifrut | wessels
"""
import json, sys, time
from pathlib import Path
import numpy as np, h5py, scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/scperturb"
OUT = ROOT / "results/final"; OUT.mkdir(parents=True, exist_ok=True)
Z = 3.0


def cat(f, k):
    g = f["obs"][k]
    cats = np.array([x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]], dtype=object)
    codes = g["codes"][:]
    return np.where(codes >= 0, cats[np.maximum(codes, 0)], None)


def load(name, n_genes=2000, exclude=()):
    f = h5py.File(RAW / f"{name}.h5ad", "r")
    Xg = f["X"]
    X = sp.csr_matrix((Xg["data"][:], Xg["indices"][:], Xg["indptr"][:]), shape=tuple(Xg.attrs["shape"]))
    var = f["var"]
    vk = "_index" if "_index" in var else list(var.keys())[0]
    genes = np.array([x.decode() if isinstance(x, bytes) else x for x in var[vk][:]])
    lib = np.asarray(X.sum(1)).ravel()
    Xn = sp.diags(1e4 / np.maximum(lib, 1)) @ X
    Xn.data = np.log1p(Xn.data)
    m = np.asarray(Xn.mean(0)).ravel(); m2 = np.asarray(Xn.multiply(Xn).mean(0)).ravel(); v = m2 - m ** 2
    ok = (m > 0.05) & ~np.isin(genes, list(exclude))
    disp = np.where(ok, v / (m + 1e-9), -1)
    keep = np.argsort(-disp)[:n_genes]
    return f, np.asarray(Xn[:, keep].todense(), dtype=np.float32), genes[keep]


def stats(X, idx):
    x = X[idx]; return x.mean(0), x.var(0, ddof=1) / len(idx)


def effect(X, a, b):
    ma, va = stats(X, a); mb, vb = stats(X, b)
    d = ma - mb; return d, d / np.sqrt(va + vb + 1e-12)


def compare(d1, z1, d2, z2):
    s1, s2 = np.abs(z1) > Z, np.abs(z2) > Z
    both = s1 & s2
    out = {"n_sig_ctx1": int(s1.sum()), "n_sig_ctx2": int(s2.sum()), "n_sig_both": int(both.sum()),
           "jaccard": float(both.sum() / max((s1 | s2).sum(), 1)),
           "sign_flips": int((np.sign(d1[both]) != np.sign(d2[both])).sum())}
    if both.sum() >= 5:
        x, y = d1[both], d2[both]
        C = np.cov(np.c_[x, y].T); w, V = np.linalg.eigh(C); vmax = V[:, -1]
        out["tls_slope_ctx2_vs_ctx1"] = float(vmax[1] / vmax[0])
        out["median_abs_ratio"] = float(np.median(np.abs(y) / np.abs(x)))
    top = np.argsort(-np.abs(z1))[:50]; top = np.union1d(top, np.argsort(-np.abs(z2))[:50])
    out["corr_top50_union"] = float(np.corrcoef(d1[top], d2[top])[0, 1])
    return out


def halves(rng, idx):
    p = rng.permutation(idx); return p[: len(p) // 2], p[len(p) // 2:]


def summarize(rows, key):
    R = [r[key] for r in rows if key in r]
    flips = sum(r["sign_flips"] for r in R); both = sum(r["n_sig_both"] for r in R)
    sl = [r["tls_slope_ctx2_vs_ctx1"] for r in R if "tls_slope_ctx2_vs_ctx1" in r]
    ra = [r["median_abs_ratio"] for r in R if "median_abs_ratio" in r]
    gain = [max(abs(s), 1 / max(abs(s), 1e-9)) for s in sl]
    return {"n_comparisons": len(R), "sign_flips": int(flips), "strong_shared_targets": int(both),
            "flip_fraction": float(flips / max(both, 1)),
            "jaccard_mean": float(np.mean([r["jaccard"] for r in R])),
            "corr_top50_median": float(np.nanmedian([r["corr_top50_union"] for r in R])),
            "tls_slope_median": float(np.median(sl)) if sl else None,
            "fold_gain_change_median (max(s,1/s))": float(np.median(gain)) if gain else None,
            "median_abs_ratio_median": float(np.median(ra)) if ra else None,
            "n_with_slope": len(sl)}


def shifrut():
    t0 = time.time()
    f0 = h5py.File(RAW / "ShifrutMarson2018.h5ad", "r")
    tgt = cat(f0, "target"); cond = cat(f0, "perturbation_2")
    targets = sorted({t for t in tgt if t not in (None, "NonTarget")})
    f, X, genes = load("ShifrutMarson2018", exclude=targets)
    rng = np.random.default_rng(0)
    rows = []
    for t in targets:
        r = {"target": t}
        idx = {c: np.nonzero((tgt == t) & (cond == c))[0] for c in ("NoStim", "Stim")}
        ctl = {c: np.nonzero((tgt == "NonTarget") & (cond == c))[0] for c in ("NoStim", "Stim")}
        r["n_cells"] = {c: int(len(v)) for c, v in idx.items()}
        if min(len(v) for v in idx.values()) < 50:
            continue
        e = {c: effect(X, idx[c], ctl[c]) for c in idx}
        r["context"] = compare(*e["NoStim"], *e["Stim"])
        for c in ("NoStim", "Stim"):  # split-half null within each condition
            a1, a2 = halves(rng, idx[c]); c1, c2 = halves(rng, ctl[c])
            r[f"null_{c}"] = compare(*effect(X, a1, c1), *effect(X, a2, c2))
        # power-matched context comparison: half the cells in each context, like the null
        a1, _ = halves(rng, idx["NoStim"]); c1, _ = halves(rng, ctl["NoStim"])
        a2, _ = halves(rng, idx["Stim"]); c2, _ = halves(rng, ctl["Stim"])
        r["context_halfcells"] = compare(*effect(X, a1, c1), *effect(X, a2, c2))
        rows.append(r)
    act = [r for r in rows if max(r["context"]["n_sig_ctx1"], r["context"]["n_sig_ctx2"]) >= 10]
    out = {"dataset": "Shifrut 2018 (primary T cells, CRISPR KO, NoStim vs Stim)", "n_genes_scored": int(X.shape[1]),
           "n_knockouts": len(rows), "n_knockouts_with_>=10_sig_targets": len(act),
           "context (NoStim -> Stim)": summarize(act, "context"),
           "context, half cells (power-matched to null)": summarize(act, "context_halfcells"),
           "null split-half NoStim": summarize(act, "null_NoStim"), "null split-half Stim": summarize(act, "null_Stim"),
           "per_knockout": rows, "seconds": time.time() - t0}
    json.dump(out, open(OUT / "f6_shifrut.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_knockout"}, indent=1))


def wessels():
    t0 = time.time()
    f0 = h5py.File(RAW / "WesselsSatija2023.h5ad", "r")
    pert = cat(f0, "perturbation")
    singles = sorted({p for p in pert if p and p != "control" and "_" not in p})
    f, X, genes = load("WesselsSatija2023", exclude=singles)
    ctl = np.nonzero(pert == "control")[0]
    cnt = {p: int((pert == p).sum()) for p in set(pert) if p}
    rng = np.random.default_rng(0)
    rows = []
    for p, n in cnt.items():
        if "_" not in p or n < 100:
            continue
        a, c = p.split("_")
        for A_, C_ in ((a, c), (c, a)):
            if cnt.get(A_, 0) < 60 or cnt.get(C_, 0) < 60:
                continue
            iA = np.nonzero(pert == A_)[0]; iC = np.nonzero(pert == C_)[0]; iAC = np.nonzero(pert == p)[0]
            r = {"A": A_, "context": C_, "n": [len(iA), len(iC), len(iAC)]}
            dA, zA = effect(X, iA, ctl); dAC, zAC = effect(X, iAC, iC)
            r["context"] = compare(dA, zA, dAC, zAC)
            a1, a2 = halves(rng, iA); c1, c2 = halves(rng, ctl)
            r["null"] = compare(*effect(X, a1, c1), *effect(X, a2, c2))
            rows.append(r)
    act = [r for r in rows if r["context"]["n_sig_ctx1"] >= 10]
    out = {"dataset": "Wessels 2023 (THP-1, Cas13 knockdown, single vs dual)", "n_genes_scored": int(X.shape[1]),
           "n_A_in_context_comparisons": len(rows), "n_with_>=10_sig_targets_alone": len(act),
           "A alone vs A in C's context": summarize(act, "context"),
           "null: A alone split-half": summarize(act, "null"),
           "per_comparison": rows, "seconds": time.time() - t0}
    json.dump(out, open(OUT / "f6_wessels.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_comparison"}, indent=1))


if __name__ == "__main__":
    {"shifrut": shifrut, "wessels": wessels}[sys.argv[1]]()
