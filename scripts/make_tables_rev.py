"""LaTeX tables for revision 1 (paper/tables/r_*.tex) from results/revision/*.json and results/final/*.json."""
import json, glob
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/revision"; F = ROOT / "results/final"; T = ROOT / "paper/tables"; T.mkdir(parents=True, exist_ok=True)
J = lambda p: json.load(open(p))
esc = lambda s: s.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&").replace(">", r"$>$").replace("|", r"$|$")


def ci(v, nd=3):
    if v is None:
        return "n/a"
    m, lo, hi = v[0], v[1], v[2]
    if m != m:
        return "n/a"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def w(name, body):
    (T / f"{name}.tex").write_text(body); print(name)


def t_decomp():
    d = J(R / "decomp_sensitivity.json")
    def row(lab, x):
        p = x["pooled_share_of_total_energy"]; s = x["pooled_share_of_signal"]
        return (f"{lab} & {p['noise']:.2f} & {p['generic']:.2f} & {p['pathway']:.2f} & {p['specific (residual minus expected noise)']:.2f} & "
                f"{s['generic']:.2f} & {s['pathway']:.2f} & {s['specific']:.2f} & {x['median_noise_share_per_knockdown']:.2f} \\\\")
    rows = [r"\multicolumn{9}{l}{\emph{Number of shared axes $K$ (all genes, 4{,}507 knockdowns)}} \\"]
    rows += [row(f"$K={k}$" + (" (permutation null)" if k == "3" else ""), v) for k, v in d["K_sensitivity"].items()]
    rows.append(r"\midrule\multicolumn{9}{l}{\emph{Gene panel ($K=3$)}} \\")
    rows += [row(esc(k), v) for k, v in d["gene_panel_sensitivity (K=3)"].items()]
    rows.append(r"\midrule\multicolumn{9}{l}{\emph{Cells per knockdown, tertiles ($K=3$; 1 = fewest)}} \\")
    rows += [row(k, v) for k, v in d["cell_count_tertiles (K=3)"].items()]
    rows.append(r"\midrule\multicolumn{9}{l}{\emph{Knockdown strength, tertiles ($K=3$; 1 = strongest)}} \\")
    rows += [row(k, v) for k, v in d["knockdown_strength_tertiles (K=3; tertile 1 = strongest)"].items()]
    w("r_decomp", r"""\begin{tabular}{lcccc|ccc|c}
\toprule
 & \multicolumn{4}{c|}{Pooled share of total energy} & \multicolumn{3}{c|}{Pooled share of signal} & Median noise \\
Setting & noise & generic & shared axes & specific & generic & shared axes & specific & share / KD \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
    n = d["noise_model_check_across_screens"]; a = d["axis_stability"]
    p = n["observed/predicted noise energy of (gw - es) difference, median [IQR]"]; c = n["non-targeting controls, random cross-screen pairs, median [IQR]"]
    w("r_noise_axes", r"""\begin{tabular}{lc}
\toprule
Check & Value \\
\midrule
""" + f"Observed / predicted noise energy, gw$-$es difference, same knockdown ({n['n_knockdowns']}) & {p[0]:.2f} [{p[1]:.2f}, {p[2]:.2f}] \\\\\n"
      f"Same, random pairs of non-targeting profiles (300) & {c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}] \\\\\n"
      f"$|\\cos|$ between generic directions fitted separately per screen & {a['cos(generic_gw, generic_es)']:.2f} \\\\\n"
      f"Principal-angle cosines, 3 shared axes fitted separately per screen & {', '.join(f'{x:.2f}' for x in a['principal_angle_cosines_pathway3'])} \\\\\n"
      f"Same for a random 3-dimensional subspace & {', '.join(f'{x:.2f}' for x in a['random_subspace_cosines'])} \\\\\n" + r"""\bottomrule
\end{tabular}""")
    g = a["top_loading_genes_gw"]
    w("r_axes_genes", "; ".join(f"axis {k[-1]}: {', '.join(v[:8])}" for k, v in g.items()) + ". Generic direction: " + ", ".join(a["top_loading_genes_generic"][:8]) + ".")


def t_f2():
    d = J(R / "f2_revised.json"); s = d["scores"]
    tg = ["same screen, raw", "independent screen, raw", "independent screen, mean-centred", "independent screen, specific residual"]
    rows = []
    for p, v in s.items():
        cells = []
        for t in tg:
            cells.append(f"{ci(v[t]['corr'])} & {v[t]['R2_vs_zero'][0]:.3f}")
        rows.append(esc(p) + " & " + " & ".join(cells) + r" \\")
    rel = d["reliability"]
    rr = "Cross-screen $r$ of the target (mean) & -- & -- & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{rel[t]['cross_screen_r_mean']:.2f}}}" for t in tg[1:]) + r" \\"
    rc = "Ceiling $\\sqrt{r}$ (mean; equal reliability) & -- & -- & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{rel[t]['ceiling_sqrt_r_mean (equal-reliability assumption)']:.2f}}}" for t in tg[1:]) + r" \\"
    w("r_f2", r"""\begin{tabular}{l cc cc cc cc}
\toprule
 & \multicolumn{2}{c}{Same screen, raw} & \multicolumn{2}{c}{Indep., raw} & \multicolumn{2}{c}{Indep., mean-centred} & \multicolumn{2}{c}{Indep., specific} \\
Predictor & corr [95\% CI] & $R^2_0$ & corr [95\% CI] & $R^2_0$ & corr [95\% CI] & $R^2_0$ & corr [95\% CI] & $R^2_0$ \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + rr + "\n" + rc + r"""
\bottomrule
\end{tabular}""")


def t_f1():
    d = J(R / "f1_revised.json")
    rows = []
    for v, kk in d["real"].items():
        x = kk["10"]
        rows.append(f"{esc(v)} & {ci(x['corr_mean_ci'])} & {ci(x['null_mean_ci'])} & {ci(x['diff_mean_ci'])} & {x['frac_beats_null']:.2f} \\\\")
    w("r_f1_real", r"""\begin{tabular}{lcccc}
\toprule
Profiles used (mediators and target) & corr, cascade [95\% CI] & corr, random mediators & difference & frac.\ beats null \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
    d2 = J(R / "f1_pc_round2.json")
    pc = d2["variants"]["base (orthogonal, strongest mediators, coordinates restored, equal reliability)"]
    covs = sorted({k.split()[0].split("=")[1] for k in pc}, reverse=True)
    alphas = sorted({float(k.split()[1].split("=")[1]) for k in pc})
    rows = []
    for a in alphas:
        cells = []
        for c in covs:
            x = pc[f"coverage={c} alpha={a}"]
            dd = x["diff_mean_ci"]
            cells.append(f"{dd[0]:.3f} [{dd[1]:.3f}, {dd[2]:.3f}]")
        rows.append(f"{a:g} & " + " & ".join(cells) + r" \\")
    real = d["real"]["specific residual (original)"]["10"]["diff_mean_ci"]
    w("r_f1_pc", r"""\begin{tabular}{l""" + "c" * len(covs) + r"""}
\toprule
True cascade share $\alpha$ & """ + " & ".join(f"coverage {c}" for c in covs) + r""" \\
\midrule
""" + "\n".join(rows) + f"\n\\midrule\nReal data (specific residual) & \\multicolumn{{{len(covs)}}}{{c}}{{{real[0]:.3f} [{real[1]:.3f}, {real[2]:.3f}]}} \\\\" + r"""
\bottomrule
\end{tabular}""")


def t_binding():
    d = J(R / "binding.json")
    rows = [f"{esc(k)} & {ci(v)} & {v[3]} \\\\" for k, v in d["binding_auroc_mean_ci"].items()]
    rows.append(r"\midrule \multicolumn{3}{l}{\emph{Within-stratum AUROC (averaged over 5 quintiles)}} \\")
    rows += [f"{esc(k)} & {ci(v)} & {v[3]} \\\\" for k, v in d["binding_auroc_within_strata"].items()]
    rows.append(r"\midrule \multicolumn{3}{l}{\emph{Predicting knockdown responders ($|z|\ge0.5$)}} \\")
    rows += [f"{esc(k)} & {ci(v)} & {v[3]} \\\\" for k, v in d["function_auroc_mean_ci"].items()]
    h = d["hub_characterisation"]
    w("r_binding", r"""\begin{tabular}{lcc}
\toprule
Score & mean per-TF AUROC [95\% CI] & TFs \\
\midrule
\multicolumn{3}{l}{\emph{Predicting ChIP-seq binding at promoters}} \\
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
    w("r_hub", f"Spearman correlation of the hub score with ATAC signal {h['spearman(hub, ATAC)']:.2f} and with log expression {h['spearman(hub, log expression)']:.2f}; "
      f"{100 * h['frac_promoters_bound_by_>=50%_of_TFs']:.1f}\\% of promoters are bound by at least half of the TFs.")


def t_context():
    d = J(R / "context.json")
    keys = ["strong_in_ctx1", "strong_in_ctx2", "shared_same_sign", "reversed", "gained (ns in ctx1, strong in ctx2)",
            "lost (strong in ctx1, ns in ctx2)", "significant context difference |z_diff|>3", "  of which: reversal",
            "  of which: same sign, different magnitude", "  of which: gained", "  of which: lost"]
    labels = ["strong alone ($|z|>3$)", "strong in context", "shared, same sign", "shared, reversed sign", "gained (alone $|z|<1.96$)",
              "lost (context $|z|<1.96$)", "significant change, $|z_\\mathrm{diff}|>3$", "\\quad of which reversal", "\\quad of which same sign, other size",
              "\\quad of which gained", "\\quad of which lost"]
    ds = [k for k in d if isinstance(d[k], dict) and "counts" in d[k]]
    rows = []
    sub = ["  of which: reversal", "  of which: same sign, different magnitude", "  of which: gained", "  of which: lost"]
    for dn in ds:
        c = d[dn]["counts"]; tot = c["significant context difference |z_diff|>3"]
        c["  of which: other / intermediate"] = {"observed": tot["observed"] - sum(c[k]["observed"] for k in sub),
                                                  "null_mean": tot["null_mean"] - sum(c[k]["null_mean"] for k in sub), "null_sd": float("nan")}
    keys = keys + ["  of which: other / intermediate"]
    labels = labels + ["\\quad of which other (intermediate strength)"]
    for k, lab in zip(keys, labels):
        cells = []
        for dn in ds:
            c = d[dn]["counts"][k]
            sd = "" if c["null_sd"] != c["null_sd"] else f" $\\pm$ {c['null_sd']:.0f}"
            cells.append(f"{c['observed']:,} & {c['null_mean']:,.0f}{sd}")
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{dn.split(' (')[0]} ({d[dn]['n_comparisons']} comparisons)}}" for dn in ds)
    w("r_context", r"""\begin{tabular}{l""" + "rr" * len(ds) + r"""}
\toprule
 & """ + head + r""" \\
Gene-level category (summed over comparisons) & """ + " & ".join(["observed & matched null"] * len(ds)) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")


def t_masks():
    Rs = [J(p) for p in sorted(glob.glob(str(F / "f5masks_s*.json")))]
    if not Rs:
        return
    rows = []
    for n in Rs[0]["variants"]:
        g = lambda k: [r["variants"][n][k] for r in Rs]
        rows.append(f"{esc(n)} & {np.mean(g('n_allowed_regulators')):.0f} & {np.mean(g('recall_ceiling')):.2f} & {np.mean(g('AUPRC_ratio')):.1f} $\\pm$ {np.std(g('AUPRC_ratio')):.1f} & "
                    f"{np.mean(g('precision_at_E')):.3f} & {np.mean(g('heldout_effect_corr')):.2f} \\\\")
    w("r_masks", r"""\begin{tabular}{lccccc}
\toprule
Regulator mask & allowed regulators & recall ceiling & AUPRC$\times$ & Precision@$E$ & held-out KD corr \\
\midrule
""" + "\n".join(rows) + f"\n\\midrule\nReference: no prior & -- & 1.00 & 15.5 $\\pm$ 2.2 & 0.382 & 0.33 \\\\\nReference: soft site prior only & -- & 1.00 & 18.6 $\\pm$ 2.4 & 0.467 & 0.42 \\\\" + r"""
\bottomrule
\end{tabular}""" + f"\n% {len(Rs)} seeds")


def t_depth():
    Rs = [J(p) for p in sorted(glob.glob(str(F / "depthctrl_s*.json")))]
    if not Rs:
        return
    Ds = ["1", "2", "3", "4", "6", "8"]
    strata = [("downstream depth 1", lambda d: d <= 1), ("depth 2", lambda d: d == 2), ("depth $\\ge3$", lambda d: d >= 3)]
    rows = []
    for nm, part, pre in [("true operator", "true", ""), ("estimated", "estimated", ""), ("estimated, stabilised", "estimated", "stab_")]:
        for sn, f in strata:
            recs = [r for R_ in Rs for r in R_[part] if f(r["depth"])]
            if not recs:
                continue
            vals = [np.nanmean([r[pre + D] for r in recs]) for D in Ds]
            rows.append(f"{nm} & {sn} ({len(recs)}) & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
    w("r_depth", r"""\begin{tabular}{llcccccc}
\toprule
Operator & held-out regulators (n) & $D{=}1$ & 2 & 3 & 4 & 6 & 8 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""" + f"\n% {len(Rs)} seeds; true operator spectral radius mean {np.mean([r['true_spectral_radius'] for r in Rs]):.2f}")


if __name__ == "__main__":
    for f in (t_decomp, t_f2, t_f1, t_binding, t_context, t_masks, t_depth):
        try:
            f()
        except Exception as e:
            print("FAILED", f.__name__, repr(e))


# ------------------------------------------------------------------------------- round 2
def t_f1pc2():
    d = J(R / "f1_pc_round2.json")
    alphas = ["0.0", "0.05", "0.1", "0.25", "0.5"]
    rows = []
    for v, x in d["variants"].items():
        for cov in ["1.0", "0.25"]:
            cells = []
            for a in alphas:
                k = f"coverage={cov} alpha={a}"
                if k not in x:
                    cells.append("--"); continue
                dd = x[k]["diff_mean_ci"]
                star = "$^\\ast$" if dd[1] > d["real_data_diff_ci"][2] else ""
                cells.append(f"{dd[0]:.3f}{star}")
            rows.append(f"{esc(v)} & {cov} & " + " & ".join(cells) + r" \\")
    real = d["real_data_diff_ci"]
    w("r_f1_pc2", r"""\begin{tabular}{p{6.2cm}c""" + "c" * len(alphas) + r"""}
\toprule
Generator & coverage & \multicolumn{5}{c}{cascade share $\alpha$ (exact energy fraction)} \\
 & & 0 & 0.05 & 0.1 & 0.25 & 0.5 \\
\midrule
""" + "\n".join(rows) + f"\n\\midrule\nReal data (specific residual) & -- & \\multicolumn{{5}}{{c}}{{{real[0]:.3f} [{real[1]:.3f}, {real[2]:.3f}]}} \\\\" + r"""
\bottomrule
\end{tabular}""")


def t_f2cal():
    d = J(R / "f2_calibrated.json"); s = d["scores"]
    tg = ["independent screen, raw", "independent screen, mean-centred", "independent screen, specific residual"]
    rows = []
    for p, v in s.items():
        lam = [np.mean(d["lambda_per_fold"][p][t]) for t in tg]
        rows.append(esc(p) + " & " + " & ".join(f"{v[t]['R2_vs_zero_uncalibrated'][0]:.3f} & {ci(v[t]['R2_vs_zero_calibrated'])} & {l:.2f}" for t, l in zip(tg, lam)) + r" \\")
    w("r_f2cal", r"""\begin{tabular}{l ccc ccc ccc}
\toprule
 & \multicolumn{3}{c}{Indep., raw} & \multicolumn{3}{c}{Indep., mean-centred} & \multicolumn{3}{c}{Indep., specific} \\
Predictor & $R^2_0$ & $R^2_0$ calibrated [95\% CI] & $\bar\lambda$ & $R^2_0$ & calibrated & $\bar\lambda$ & $R^2_0$ & calibrated & $\bar\lambda$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
    rows = []
    for k, v in d["paired_differences"].items():
        a, t = k.split(" | ")
        rows.append(f"{esc(a)} & {esc(t.replace('independent screen, ', ''))} & {ci(v['corr'])} & {ci(v['R2_calibrated'], 4)} \\\\")
    w("r_f2pairs", r"""\begin{tabular}{llcc}
\toprule
Paired comparison & target & $\Delta$ corr [95\% CI] & $\Delta R^2_0$ calibrated [95\% CI] \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")


def t_noise2():
    d = J(R / "noise_pcs.json")
    rows = [f"{esc(k)} & {v[0]:.3f} & {v[1]:.3f} \\\\" for k, v in d["estimators"].items()]
    b = d["bootstrap_over_controls"]
    rows.append(f"Bootstrap over 585 controls (95\\% interval, median estimator) & [{b['pooled'][0]:.3f}, {b['pooled'][1]:.3f}] & [{b['median'][0]:.3f}, {b['median'][1]:.3f}] \\\\")
    w("r_noise_est", r"""\begin{tabular}{lcc}
\toprule
Noise-constant estimator & pooled noise share & median per-knockdown noise share \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
    C = d["matched_pc_abs_cosine_gw_rows_es_cols"]
    w("r_pcmatch", r"""\begin{tabular}{lccc}
\toprule
GW axis $\backslash$ essential axis & 1 & 2 & 3 \\
\midrule
""" + "\n".join(f"{i + 1} & " + " & ".join(f"{x:.2f}" for x in row) + r" \\" for i, row in enumerate(C)) + r"""
\bottomrule
\end{tabular}""")
    g = d["canonical_direction_top_genes"]
    w("r_canon_genes", "; ".join(f"aligned direction {k[-1]}: {', '.join(v[:8])}" for k, v in list(g.items())[:2]) + ".")


def t_depth2():
    Rs = [J(p) for p in sorted(glob.glob(str(F / "depthctrl2_s*.json")))]
    if not Rs:
        return
    recs = [r for R_ in Rs for r in R_["records"]]
    Ds = ["1", "2", "3", "4", "6", "8"]
    strata = [("depth 1", lambda d: d <= 1), ("depth 2", lambda d: d == 2), ("depth $\\ge3$", lambda d: d >= 3)]
    rows = []
    for op, opn in [("true", "true operator"), ("est", "estimated"), ("stab", "estimated, stabilised")]:
        for tgt, tn in [("clean", "noise-free"), ("meas", "measured")]:
            for sn, f in strata:
                rr = [r for r in recs if f(r["depth"])]
                vals = [np.nanmean([r[f"{op}|{tgt}|{D}"] for r in rr]) for D in Ds]
                diffs = np.array([max(r[f"{op}|{tgt}|{D}"] for D in Ds[1:]) - r[f"{op}|{tgt}|1"] for r in rr])
                rows.append(f"{opn} & {tn} & {sn} ({len(rr)}) & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
    cm = np.nanmean([r["clean_vs_meas"] for r in recs])
    w("r_depth2", r"""\begin{tabular}{lllcccccc}
\toprule
Operator & target & regulators (n) & $D{=}1$ & 2 & 3 & 4 & 6 & 8 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""" + f"\n% corr(noise-free, measured targets) mean {cm:.2f}; {len(Rs)} seeds")
    # paired D=2..8 best vs D=1 with bootstrap CI, deep regulators
    out = {}
    for op in ("true", "est", "stab"):
        for tgt in ("clean", "meas"):
            rr = [r for r in recs if r["depth"] >= 3]
            dd = [np.nanmean([r[f"{op}|{tgt}|{D}"] for D in ("2", "3", "4")]) - r[f"{op}|{tgt}|1"] for r in rr]
            out[f"{op}|{tgt}"] = ci_boot(dd)
    (T / "r_depth2_diff.json").write_text(json.dumps(out, indent=1))


def ci_boot(v, n=2000):
    v = np.array([x for x in v if x == x]); r = np.random.default_rng(0)
    m = [r.choice(v, len(v)).mean() for _ in range(n)]
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(len(v))]


if __name__ == "__main__":
    for f in (t_f1pc2, t_f2cal, t_noise2, t_depth2):
        try:
            f()
        except Exception as e:
            print("FAILED", f.__name__, repr(e))
