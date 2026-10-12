"""Figures for revision 1 (paper/figures/r*.pdf) from results/revision and results/final."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper/figures"; FIG.mkdir(parents=True, exist_ok=True)
R = ROOT / "results/revision"; F = ROOT / "results/final"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "axes.titlesize": 8.5})
C = {"blue": "#3b6ea8", "orange": "#d9822b", "green": "#4c9a5f", "red": "#c0504d", "grey": "#8c8c8c", "purple": "#7b5ea7"}
J = lambda p: json.load(open(p))


def lbl(ax, s):
    ax.text(-0.2, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold")


def r1():
    d = J(R / "decomp_sensitivity.json")
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.4), gridspec_kw={"width_ratios": [0.7, 1.2, 1]})
    ax = axs[0]; x = d["K_sensitivity"]["3"]["pooled_share_of_total_energy"]
    parts = [("noise", x["noise"], C["grey"]), ("generic", x["generic"], C["orange"]), ("shared axes", x["pathway"], C["green"]),
             ("specific", x["specific (residual minus expected noise)"], C["blue"])]
    b = 0
    for n, v, c in parts:
        ax.bar(0, v, bottom=b, color=c, width=0.6); ax.text(0.36, b + v / 2, f"{n} {v:.2f}", va="center", fontsize=6.5); b += v
    ax.set_xlim(-0.5, 1.6); ax.set_xticks([]); ax.set_ylabel("Pooled share of total energy")
    ax.set_title(f"All knockdowns, K=3\n(median noise share {d['K_sensitivity']['3']['median_noise_share_per_knockdown']:.2f})"); lbl(ax, "a")
    ax = axs[1]
    Ks = [int(k) for k in d["K_sensitivity"]]
    for key, c, n in [("generic", C["orange"], "generic"), ("pathway", C["green"], "shared axes"), ("specific", C["blue"], "specific")]:
        ax.plot(Ks, [d["K_sensitivity"][str(k)]["pooled_share_of_signal"][key] for k in Ks], "o-", color=c, ms=3, label=n)
    ax.axvline(3, color="k", lw=0.5, ls=":"); ax.text(3.3, 0.85, "permutation\nnull K=3", fontsize=6)
    ax.set_xscale("symlog", linthresh=1); ax.set_xticks(Ks); ax.set_xticklabels(Ks); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("number of shared axes K"); ax.set_ylabel("pooled share of signal"); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("The split depends on K"); lbl(ax, "b")
    ax = axs[2]; a = d["axis_stability"]
    v = [a["cos(generic_gw, generic_es)"]] + a["principal_angle_cosines_pathway3"]
    nul = [np.nan] + a["random_subspace_cosines"]
    xs = np.arange(4)
    ax.bar(xs - 0.18, v, 0.36, color=C["blue"], label="GW vs essential screen")
    ax.bar(xs + 0.18, np.nan_to_num(nul), 0.36, color=C["grey"], label="random subspace")
    ax.set_xticks(xs); ax.set_xticklabels(["generic", "axis 1", "axis 2", "axis 3"]); ax.set_ylabel("|cosine| (principal angle)")
    ax.set_ylim(0, 1.3); ax.legend(frameon=False, fontsize=6.5, loc="upper right"); ax.set_title("Axes fitted separately per screen"); lbl(ax, "c")
    fig.tight_layout(); fig.savefig(FIG / "r1_decomposition.pdf"); plt.close(fig)


def r2():
    d = J(R / "f2_revised.json"); s = d["scores"]; rel = d["reliability"]
    tg = ["same screen, raw", "independent screen, raw", "independent screen, mean-centred", "independent screen, specific residual"]
    tl = ["same screen\nraw", "indep. screen\nraw", "indep. screen\nmean-centred", "indep. screen\nspecific"]
    preds = ["mean training profile", "kNN k=10", "kNN k=100", "factorised (generic mean + pathway kNN50 + specific kNN50)"]
    pl = ["mean profile", "kNN-10", "kNN-100", "factorised"]
    cols = [C["grey"], C["green"], C["blue"], C["orange"]]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, met, ttl in [(axs[0], "corr", "Correlation (bars: 95% CI)"), (axs[1], "R2_vs_zero", "Magnitude-sensitive: $R^2$ vs zero")]:
        x = np.arange(4); w = 0.2
        for i, (p, n, c) in enumerate(zip(preds, pl, cols)):
            vals, err = [], [[], []]
            for t in tg:
                v = s[p][t][met]
                if v is None or v[0] != v[0]:
                    vals.append(0); err[0].append(0); err[1].append(0)
                else:
                    vals.append(v[0]); err[0].append(v[0] - v[1]); err[1].append(v[2] - v[0])
            ax.bar(x + (i - 1.5) * w, vals, w, yerr=err if met == "corr" else None, color=c, label=n, error_kw={"lw": 0.6})
        if met == "corr":
            for k, t in enumerate(tg[1:], start=1):
                cv = rel[t]["ceiling_sqrt_r_mean (equal-reliability assumption)"]
                ax.plot([k - 0.45, k + 0.45], [cv, cv], color="k", lw=0.8, ls="--")
            ax.text(0.6, 0.79, "dashed: ceiling √r (equal reliability)", fontsize=6)
            ax.set_ylim(-0.05, 0.85)
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(x); ax.set_xticklabels(tl, fontsize=6.5); ax.set_title(ttl)
    axs[0].set_ylabel("score on held-out knockdowns"); axs[0].legend(frameon=False, fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(1.1, -0.22))
    lbl(axs[0], "a"); lbl(axs[1], "b")
    fig.tight_layout(); fig.savefig(FIG / "r2_benchmark.pdf"); plt.close(fig)


def r3():
    d = J(R / "f1_revised.json"); d2 = J(R / "f1_pc_round2.json")
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.5), gridspec_kw={"width_ratios": [0.8, 1.1, 1.3]})
    ax = axs[0]
    names = list(d["real"].keys()); short = ["specific\nresidual", "generic\nremoved", "raw"]
    for i, n in enumerate(names):
        x = d["real"][n]["10"]["diff_mean_ci"]
        ax.errorbar(i, x[0], yerr=[[x[0] - x[1]], [x[2] - x[0]]], fmt="o", color=C["blue"], capsize=3)
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels(short); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("cascade − random-mediator corr"); ax.set_title("Real data (507 KDs)"); lbl(ax, "a")
    real = d["real"]["specific residual (original)"]["10"]["diff_mean_ci"]
    xt = [0, 0.05, 0.1, 0.25, 0.5]
    def curve(ax, pc, cov, c, lab, ls="o-"):
        al = sorted({float(k.split("alpha=")[1]) for k in pc if k.startswith(f"coverage={cov} ")})
        v = [pc[f"coverage={cov} alpha={a}"]["diff_mean_ci"] for a in al]
        ax.errorbar(al, [x[0] for x in v], yerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]], fmt=ls, ms=3, color=c, label=lab)
    ax = axs[1]; base = d2["variants"]["base (orthogonal, strongest mediators, coordinates restored, equal reliability)"]
    for cov, c in [("1.0", C["blue"]), ("0.5", C["green"]), ("0.25", C["orange"])]:
        curve(ax, base, cov, c, f"coverage {cov}")
    ax.axhspan(real[1], real[2], color=C["red"], alpha=0.2, label="real data 95% CI")
    ax.set_xscale("symlog", linthresh=0.05); ax.set_xticks(xt); ax.set_xticklabels(["0", ".05", ".1", ".25", ".5"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("cascade share α (exact)"); ax.legend(frameon=False, fontsize=5.5); ax.set_title("Base generator"); lbl(ax, "b")
    ax = axs[2]
    sel = [("base (orthogonal, strongest mediators, coordinates restored, equal reliability)", "base", C["blue"]),
           ("heterogeneous mediator strengths (lognormal, sd 1)", "heterogeneous", C["green"]),
           ("screen-2 reliability halved", "screen-2 rel. ½", C["purple"]),
           ("mediators ranked at random (ranking errors)", "random ranking", C["orange"]),
           ("mediator coordinates not restored", "mediators not visible", C["red"]),
           ("all adverse: random ranking, no restore, heterogeneous, screen-2 halved", "all adverse", C["grey"])]
    for key, lab, c in sel:
        curve(ax, d2["variants"][key], "1.0", c, lab)
    ax.axhspan(real[1], real[2], color=C["red"], alpha=0.15)
    ax.set_xscale("symlog", linthresh=0.05); ax.set_xticks(xt); ax.set_xticklabels(["0", ".05", ".1", ".25", ".5"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("cascade share α (coverage 1)"); ax.legend(frameon=False, fontsize=5.5); ax.set_title("Generator robustness"); lbl(ax, "c")
    fig.tight_layout(); fig.savefig(FIG / "r3_domino.pdf"); plt.close(fig)


def r4():
    d = J(R / "binding.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axs[0]
    b = d["binding_auroc_mean_ci"]; f = d["function_auroc_mean_ci"]
    items = [("PWM max", b["PWM_max"], f["PWM_max"]), ("occupancy", b["occupancy (mu=Ethr-2)"], f["occupancy (mu=Ethr-2)"]),
             ("occ. × ATAC", b["occupancy x ATAC"], f["occupancy x ATAC"]), ("ChIP bound", None, f["ChIP bound (measured binding)"])]
    for i, (n, bb, ff) in enumerate(items):
        if bb:
            ax.errorbar(i - 0.12, bb[0], yerr=[[bb[0] - bb[1]], [bb[2] - bb[0]]], fmt="o", color=C["blue"], capsize=2, label="ChIP binding (197 TFs)" if i == 0 else None)
        ax.errorbar(i + 0.12, ff[0], yerr=[[ff[0] - ff[1]], [ff[2] - ff[0]]], fmt="s", color=C["orange"], capsize=2, label="KD responders (22 TFs)" if i == 0 else None)
    ax.axhline(0.5, color="k", lw=0.5, ls="--"); ax.set_xticks(range(4)); ax.set_xticklabels([x[0] for x in items])
    ax.set_ylabel("mean per-TF AUROC [95% CI]"); ax.legend(frameon=False, fontsize=6.5); ax.set_title("Promoters (1 kb upstream), K562"); lbl(ax, "a")
    ax = axs[1]; s = d["binding_auroc_within_strata"]
    keys = [("hub", "hub (leave-TF-out binding frequency)"), ("occ. × ATAC", "occupancy x ATAC"), ("PWM max", "PWM_max")]
    x = np.arange(3)
    allv = [b["hub (leave-TF-out binding frequency)"][0], b["occupancy x ATAC"][0], b["PWM_max"][0]]
    ax.bar(x - 0.25, allv, 0.25, color=C["grey"], label="all promoters")
    ax.bar(x, [s[f"{k} | within ATAC quintile"][0] for _, k in keys], 0.25, color=C["green"], label="within ATAC quintiles")
    ax.bar(x + 0.25, [s[f"{k} | within expression quintile"][0] for _, k in keys], 0.25, color=C["purple"], label="within expression quintiles")
    ax.set_xticks(x); ax.set_xticklabels([k for k, _ in keys]); ax.set_ylim(0.5, 0.8); ax.set_ylabel("binding AUROC")
    ax.legend(frameon=False, fontsize=6.5); ax.set_title("Hub prior is not explained by\naccessibility or expression"); lbl(ax, "b")
    fig.tight_layout(); fig.savefig(FIG / "r4_binding.pdf"); plt.close(fig)


def r5():
    d = J(R / "context.json")
    ds = [k for k in d if isinstance(d[k], dict) and "counts" in d[k]]
    keys = ["  of which: reversal", "  of which: same sign, different magnitude", "  of which: gained", "  of which: lost"]
    labs = ["reversed", "same sign,\nother size", "gained", "lost", "other\n(intermediate)"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, dn, L in zip(axs, ds, "ab"):
        c = d[dn]["counts"]; tot = c["significant context difference |z_diff|>3"]
        obs = [c[k]["observed"] for k in keys]; nul = [c[k]["null_mean"] for k in keys]
        obs.append(tot["observed"] - sum(obs)); nul.append(tot["null_mean"] - sum(nul))
        x = np.arange(5)
        ax.bar(x - 0.18, obs, 0.36, color=C["blue"], label="observed"); ax.bar(x + 0.18, nul, 0.36, color=C["grey"], label="matched null")
        shared = c["shared_same_sign"]["observed"] + c["reversed"]["observed"]
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.5); ax.set_ylabel("count, |z_diff| > 3")
        ax.set_title(f"{dn.split(' (')[0]}: {tot['observed']:,} changes (null {tot['null_mean']:,.0f})\n"
                     f"reversals: {100 * obs[0] / tot['observed']:.1f}% of changes, {100 * c['reversed']['observed'] / shared:.1f}% of strong shared")
        ax.legend(frameon=False, fontsize=6.5); lbl(ax, L)
    fig.tight_layout(); fig.savefig(FIG / "r5_context.pdf"); plt.close(fig)


def s_depth():
    Rs = [J(p) for p in sorted(glob.glob(str(F / "depthctrl2_s*.json")))]
    if not Rs:
        return
    recs = [r for R_ in Rs for r in R_["records"]]
    Ds = [1, 2, 3, 4, 6, 8]
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.2), sharey="row")
    for row, (tgt, tn) in enumerate([("clean", "noise-free target"), ("meas", "measured target")]):
        for ax, (nm, f) in zip(axs[row], [("depth 1", lambda x: x <= 1), ("depth 2", lambda x: x == 2), ("depth ≥ 3", lambda x: x >= 3)]):
            rr = [r for r in recs if f(r["depth"])]
            for op, c, ls, lab in [("true", C["green"], "-", "true operator"), ("est", C["grey"], "-", "estimated"), ("stab", C["blue"], "--", "estimated, stabilised")]:
                ax.plot(Ds, [np.nanmean([r[f"{op}|{tgt}|{D}"] for r in rr]) for D in Ds], "o" + ls, color=c, ms=3, label=lab)
            ax.set_title(f"{nm} (n={len(rr)}), {tn}", fontsize=7)
            if row == 1:
                ax.set_xlabel("propagation depth D")
        axs[row][0].set_ylabel("corr with KD effect (log scale)")
    axs[0][0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "s_depth_control.pdf"); plt.close(fig)


def s_masks():
    Rs = [J(p) for p in sorted(glob.glob(str(F / "f5masks_s*.json")))]
    if not Rs:
        return
    names = list(Rs[0]["variants"].keys())
    v = [np.mean([r["variants"][n]["AUPRC_ratio"] for r in Rs]) for n in names]
    e = [np.std([r["variants"][n]["AUPRC_ratio"] for r in Rs]) for n in names]
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.bar(range(len(names)), v, yerr=e, color=C["blue"], error_kw={"lw": 0.6})
    ax.axhline(15.5, color=C["grey"], ls="--", lw=0.8); ax.text(len(names) - 0.5, 15.8, "no prior", fontsize=6, ha="right")
    ax.axhline(18.6, color=C["orange"], ls="--", lw=0.8); ax.text(len(names) - 0.5, 18.9, "soft site prior", fontsize=6, ha="right")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("AUPRC ÷ base rate"); ax.set_title(f"Imperfect regulator masks ({len(Rs)} seeds)")
    fig.tight_layout(); fig.savefig(FIG / "s_masks.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in (r1, r2, r3, r4, r5, s_depth, s_masks):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, repr(e))
