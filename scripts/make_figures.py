"""Figures for the BOND manuscript (paper/figures/*.pdf). Earlier-experiment numbers are copied from
docs/03, 06, 07, 09, 10 (those docs are generated from results/*.json); final tests are read from
results/final/*.json."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper/figures"; FIG.mkdir(parents=True, exist_ok=True)
F = ROOT / "results/final"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 8.5})
# Okabe-Ito colour-blind-safe palette (grey kept only for neutral/reference elements)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "grey": "#999999", "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000"}
J = lambda p: json.load(open(p))


def lbl(ax, s):
    ax.text(-0.18, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold")


# ---------------------------------------------------------------- Fig 1: binding vs function
def fig1():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    ax = axs[0]
    names = ["PWM max", "Occupancy", "Occ.+ATAC", "ATAC only"]
    chip = [0.566, 0.568, 0.610, 0.577]; func = [0.513, 0.517, 0.492, 0.419]
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, chip, w, color=C["blue"], label="ChIP binding (197 TFs)")
    ax.bar(x + w / 2, func, w, color=C["orange"], label="KD response (21 TFs)")
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_ylim(0.38, 0.66)
    ax.set_ylabel("Mean AUROC (per TF)"); ax.legend(frameon=False, fontsize=6.5, loc="upper left"); lbl(ax, "a")
    ax.set_title("Occupancy predicts binding a little,\nfunction not at all")
    ax = axs[1]
    names = ["Hub prior", "Hub+ATAC", "+occupancy"]
    v = [0.760, 0.759, 0.768]
    ax.bar(names, v, color=[C["grey"], C["grey"], C["blue"]]); ax.set_ylim(0.5, 0.8)
    ax.set_ylabel("AUROC, held-out TFs"); ax.set_title("Most binding signal is 'which\ngenes are bound by anyone'"); lbl(ax, "b"); ax.set_xticks(range(3)); ax.set_xticklabels(names, rotation=25, ha="right")
    for i, t in enumerate(v):
        ax.text(i, t + 0.005, f"{t:.3f}", ha="center", fontsize=6.5)
    ax = axs[2]
    names = ["P(bound)", "P(bound | resp.)", "P(resp. | bound)", "P(resp. | unbound)"]
    v = [0.169, 0.216, 0.016, 0.013]
    ax.barh(names, v, color=[C["grey"], C["blue"], C["orange"], C["grey"]])
    ax.set_xlabel("Median over 21 TFs"); ax.set_title("Binding vs function (K562)\nodds ratio 1.12"); lbl(ax, "c")
    fig.tight_layout(); fig.savefig(FIG / "fig1_binding_function.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 2: medium synthetic
def fig2():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axs[0]
    m = [("BOND-V (full)", 18.638, 0.093), ("− prior", 15.483, 0.069), ("− KD-aware", 17.618, 0.094),
         ("− velocity (static)", 16.355, 0.009), ("DCDI-style", 14.423, 0.008), ("CellOracle-style", 8.915, 0.142),
         ("GENIE3-style", 5.726, 0.016), ("KD mean-diff.", 26.439, 0.356), ("Prior only", 5.400, 0.195)]
    y = np.arange(len(m))[::-1]
    ax.barh(y, [a for _, a, _ in m], color=[C["blue"]] * 5 + [C["grey"]] * 4)
    for yy, (_, a, ind) in zip(y, m):
        ax.text(a + 0.4, yy, f"ind. {ind:.2f}", va="center", fontsize=6)
    ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in m]); ax.set_xlabel("AUPRC ÷ base rate (direct edges)")
    ax.set_title("80-gene simulator, 10 seeds\n(ind. = indirect fraction of top-E edges)"); lbl(ax, "a"); ax.set_xlim(0, 33)
    ax = axs[1]
    k = [0, 1, 2, 4, 8, 12]
    cur = {"random": [14.132, 14.748, 14.993, 15.786, 16.517, 17.636], "degree": [14.132, 14.175, 14.269, 14.916, 15.920, 17.820],
           "active (disambiguation)": [14.132, 14.975, 15.631, 16.908, 17.877, 18.344]}
    for (n, v), c in zip(cur.items(), [C["grey"], C["orange"], C["blue"]]):
        ax.plot(k, v, "o-", color=c, label=n, ms=3)
    ax.set_xlabel("# training knockdowns"); ax.set_ylabel("AUPRC ÷ base rate"); ax.legend(frameon=False)
    ax.set_title("Experiment selection: active > random\nin 10/10 seeds (area +0.95 ± 0.67)"); lbl(ax, "b")
    fig.tight_layout(); fig.savefig(FIG / "fig2_medium_synthetic.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 3: response structure + F1 + F2
def fig3():
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.4), gridspec_kw={"width_ratios": [1, 1, 1.1, 1.5]})
    ax = axs[0]
    parts = [("noise", 0.737), ("generic", 0.263 * 0.224), ("pathway", 0.263 * 0.263), ("specific", 0.263 * 0.513)]
    bottom = 0; ty = [0.37, 0.70, 0.80, 0.92]
    for (n, v), c, t in zip(parts, [C["grey"], C["orange"], C["green"], C["blue"]], ty):
        ax.bar(0, v, bottom=bottom, color=c, width=0.6); ax.text(0.35, t, f"{n} {v:.2f}", va="center", fontsize=6); bottom += v
    ax.set_xlim(-0.5, 1.3); ax.set_xticks([]); ax.set_ylabel("Share of profile energy")
    ax.set_title("Replogle GW\n(4,507 KDs)"); lbl(ax, "a")
    ax = axs[1]
    n = ["raw", "pathway", "specific", "mismatched"]; v = [0.362, 0.931, 0.221, -0.003]
    ax.bar(n, v, color=[C["grey"], C["green"], C["blue"], C["red"]]); ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(n, rotation=35, ha="right"); ax.set_ylabel("Cross-screen r (median)")
    ax.set_title("Specific part\nreproduces"); lbl(ax, "b")
    ax = axs[2]
    f1 = J(F / "f1_domino.json")["by_K"]
    Ks = [5, 10, 20, 50]
    ax.plot(Ks, [f1[str(k)]["corr_mean_reproducible"] for k in Ks], "o-", color=C["blue"], label="cascade through\nresponders", ms=3)
    ax.plot(Ks, [f1[str(k)]["null_corr_mean_reproducible"] for k in Ks], "s--", color=C["grey"], label="random mediators", ms=3)
    ax.axhline(f1["5"]["reliability_mean_reproducible"], color=C["green"], lw=0.8, ls=":")
    ax.text(5, f1["5"]["reliability_mean_reproducible"] - 0.05, "reliability ceiling", fontsize=6, color=C["green"])
    ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels(Ks); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_ylim(-0.05, 0.5)
    ax.set_xlabel("# mediators K"); ax.set_ylabel("corr with independent screen"); ax.legend(frameon=False, fontsize=6)
    ax.set_title("F1: domino share ≈ 0\n(507 reproducible KDs)"); lbl(ax, "c")
    ax = axs[3]
    f2 = J(F / "f2_benchmark.json"); s = f2["scores_mean_corr"]; se = f2["scores_sem"]
    preds = ["mean training profile", "kNN raw (k=50)", "kNN raw (k=10)", "factorised (generic mean + pathway kNN50 + specific kNN10)"]
    short = ["mean profile", "kNN-50", "kNN-10", "factorised"]
    tg = ["gw raw (same screen)", "es raw (independent screen)", "es specific residual (independent screen)"]
    tshort = ["same-screen raw", "other-screen raw", "other-screen specific"]
    w = 0.2; x = np.arange(3)
    for i, (p, sn, c) in enumerate(zip(preds, short, [C["grey"], C["blue"], C["green"], C["orange"]])):
        ax.bar(x + (i - 1.5) * w, [s[p][t] for t in tg], w, yerr=[se[p][t] for t in tg], color=c, label=sn, error_kw={"lw": 0.6})
    ax.set_xticks(x); ax.set_xticklabels(tshort, rotation=15); ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("corr with held-out KD"); ax.set_ylim(-0.02, 0.5); ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper right")
    ax.set_title("F2: the target decides the winner\n(1,396 held-out KDs)"); lbl(ax, "d")
    fig.tight_layout(); fig.savefig(FIG / "fig3_response_structure.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 4: context (gain not wiring)
def fig4():
    w = J(F / "f6_wessels.json")
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    ax = axs[0]
    labels = ["Norman\nCRISPRa pair", "Norman\ncell cycle", "Wessels\nCas13 pair"]
    ctx = [0.015, 5 / 10223, w["A alone vs A in C's context"]["flip_fraction"]]
    nul = [0.000, np.nan, w["null: A alone split-half"]["flip_fraction"]]
    x = np.arange(3)
    ax.bar(x - 0.18, np.array(ctx) * 100, 0.36, color=C["blue"], label="across contexts")
    ax.bar(x + 0.18, np.nan_to_num(np.array(nul) * 100), 0.36, color=C["grey"], label="split-half null")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("% strong shared targets\nthat flip sign")
    ax.legend(frameon=False); ax.set_title("Sign flips are rare (≤1.5%), though above null"); lbl(ax, "a")
    ax = axs[1]
    sl = [r["context"]["tls_slope_ctx2_vs_ctx1"] for r in w["per_comparison"] if r["context"]["n_sig_ctx1"] >= 10 and "tls_slope_ctx2_vs_ctx1" in r["context"]]
    sn = [r["null"]["tls_slope_ctx2_vs_ctx1"] for r in w["per_comparison"] if r["context"]["n_sig_ctx1"] >= 10 and "tls_slope_ctx2_vs_ctx1" in r["null"]]
    bins = np.linspace(-1, 3, 33)
    ax.hist(np.clip(sl, -1, 3), bins, color=C["blue"], alpha=0.7, label=f"A in C's context (n={len(sl)})")
    ax.hist(np.clip(sn, -1, 3), bins, color=C["grey"], alpha=0.6, label=f"split-half null (n={len(sn)})")
    ax.axvline(1, color="k", lw=0.6, ls="--"); ax.set_xlabel("gain = slope of effect in context vs alone")
    ax.set_ylabel("# comparisons"); ax.legend(frameon=False, fontsize=6); ax.set_title("Wessels 2023: gain spread"); lbl(ax, "b")
    fig.tight_layout(); fig.savefig(FIG / "fig4_context.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 5: memory functions
def fig5():
    m = J(F / "f3_memory_main.json"); nz = J(F / "f3_memory_noise.json"); dt = J(F / "f3_memory_dt.json")
    mean = lambda L, k="auroc_memory": float(np.nanmean([x[k] for x in L if x.get(k) is not None]))
    sd = lambda L, k="auroc_memory": float(np.nanstd([x[k] for x in L if x.get(k) is not None]))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    ax = axs[0]
    items = [("memory\ngain", m["main_clean"], "auroc_memory"),
             ("control:\nnone hidden", m["control_all_observed_clean"], "auroc_memory"),
             ("snapshot\n1−R²", m["main_clean"], "auroc_snapshot_unexplained_var"),
             ("weak reg.\nfluct.", m["weak_regulator_noise_clean"], "auroc_memory")]
    ax.bar(range(4), [mean(L, k) for _, L, k in items], yerr=[sd(L, k) for _, L, k in items],
           color=[C["blue"], C["grey"], C["orange"], C["purple"]], error_kw={"lw": 0.6})
    ax.axhline(0.5, color="k", lw=0.6, ls="--"); ax.set_xticks(range(4)); ax.set_xticklabels([n for n, _, _ in items], fontsize=6.2)
    ax.set_ylabel("AUROC: gene has hidden regulator"); ax.set_ylim(0, 1); ax.set_title("Clean time courses (10 seeds)"); lbl(ax, "a")
    ax = axs[1]
    lv = ["0"] + list(nz["meas_sweep"].keys())
    vals = [mean(m["main_clean"])] + [mean(nz["meas_sweep"][k]) for k in nz["meas_sweep"]]
    sds = [sd(m["main_clean"])] + [sd(nz["meas_sweep"][k]) for k in nz["meas_sweep"]]
    ax.errorbar([float(v) for v in lv], vals, yerr=sds, fmt="o-", color=C["blue"], ms=3, lw=1)
    ax.axhline(0.5, color="k", lw=0.6, ls="--"); ax.set_xlabel("measurement noise (× fluctuation sd)"); ax.set_ylim(0.2, 1)
    ax.set_ylabel("AUROC"); ax.set_title("Measurement noise kills it"); lbl(ax, "b")
    ax = axs[2]
    ks = list(dt["dt_sweep_clean"].keys())
    ax.errorbar([float(k) for k in ks], [mean(dt["dt_sweep_clean"][k]) for k in ks], yerr=[sd(dt["dt_sweep_clean"][k]) for k in ks],
                fmt="o-", color=C["blue"], ms=3, lw=1)
    ax.set_xscale("log"); ax.axhline(0.5, color="k", lw=0.6, ls="--"); ax.set_ylim(0, 1)
    ax.set_xlabel("sampling interval (hidden decay time ≈ 0.7–7)"); ax.set_ylabel("AUROC"); ax.set_title("Needs dense sampling"); lbl(ax, "c")
    fig.tight_layout(); fig.savefig(FIG / "fig5_memory.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 6: attraction + prior forms + depth
def fig6():
    a = J(F / "f4_attraction.json"); f5 = [J(p) for p in sorted(glob.glob(str(F / "f5_s*.json")))]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.7), gridspec_kw={"width_ratios": [0.9, 1.1, 1.2], "wspace": 0.9})
    ax = axs[0]
    sw = a["three_gene"]["noise_sweep_observational"]
    se = [r["sigma_e"] for r in sw]
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, xs ** 2 / (xs ** 2 + 0.25), color=C["grey"], label="theory c·b·λ")
    ax.plot(se, [r["empirical"] for r in sw], "o", color=C["blue"], ms=4, label="simulation")
    ax.set_xlabel("measurement noise on B (σ_e)"); ax.set_ylabel("spurious A→C coefficient"); ax.legend(frameon=False, loc="lower right", fontsize=6)
    ax.set_title("A→B→C, B noisy\n(σ_B = 0.5)"); lbl(ax, "a")
    ax = axs[1]
    t = a["network_fp_table"]
    keys = [k for k in t if "longer" not in k]
    ax.barh(range(len(keys))[::-1], [t[k]["false_positive_rate"] * 100 for k in keys], color=[C["blue"] if "source perturbed" in k else C["grey"] for k in keys])
    ax.set_yticks(range(len(keys))[::-1]); ax.set_yticklabels([("src P" if "source perturbed" in k else "src U") + " / med " + {"unperturbed": "U", "perturbed": "P", "hidden": "hidden"}[k.split("mediator ")[1]] + f" ({t[k]['n_pairs']})" for k in keys], fontsize=6.5)
    ax.set_xlabel("% of 2-hop pairs called"); ax.set_title("Indirect false positives\n(P = perturbed, U = not)"); lbl(ax, "b")
    ax = axs[2]
    if f5:
        names = list(f5[0]["variants"].keys())
        short = ["none", "soft site", "soft identity", "hard identity", "hard TF-only", "hard binding", "identity+site"]
        v = [np.mean([r["variants"][n]["AUPRC_ratio"] for r in f5]) for n in names]
        e = [np.std([r["variants"][n]["AUPRC_ratio"] for r in f5]) for n in names]
        ceil = [np.mean([r["variants"][n]["recall_ceiling"] for r in f5]) for n in names]
        ax.bar(range(len(names)), v, yerr=e, color=[C["grey"], C["orange"], C["orange"], C["blue"], C["blue"], C["red"], C["green"]], error_kw={"lw": 0.6})
        ax.set_xticks(range(len(names))); ax.set_xticklabels([f"{a} ({c:.2f})" for a, c in zip(short, ceil)], rotation=50, ha="right", fontsize=6.5)
        ax.set_ylabel("AUPRC ÷ base rate"); ax.set_title(f"Prior forms, {len(f5)} seeds\n(recall ceiling in brackets)"); lbl(ax, "c")
    fig.tight_layout(); fig.savefig(FIG / "fig6_attraction_priors.pdf"); plt.close(fig)


def fig7():
    f5 = [J(p) for p in sorted(glob.glob(str(F / "f5_s*.json")))]
    if not f5:
        return
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    Ds = ["1", "2", "3", "4", "6", "8", "12"]
    for n, c, ls in zip(["no prior", "hard identity mask + soft site prior", "no prior, stabilised (rho=0.8)",
                         "hard identity mask + soft site prior, stabilised (rho=0.8)"], [C["grey"], C["blue"], C["grey"], C["blue"]], ["-", "-", "--", "--"]):
        v = [np.nanmean([r["depth"][n][d] for r in f5]) for d in Ds]
        ax.plot([int(d) for d in Ds], v, "o" + ls, color=c, ms=3, label=n.replace("hard identity mask + soft site prior", "identity mask + site prior").replace("rho", "ρ"))
    ax.set_xlabel("propagation depth D"); ax.set_ylabel("corr with held-out KD effect"); ax.legend(frameon=False, fontsize=6)
    dd = np.concatenate([r["true_downstream_depth_heldout"] for r in f5])
    ax.set_title(f"Truncated propagation (true depth median {np.median(dd):.0f})")
    fig.tight_layout(); fig.savefig(FIG / "fig7_depth.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, repr(e))
