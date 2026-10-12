"""LaTeX tables for the final tests (paper/tables/*.tex), generated from results/final/*.json."""
import json, glob
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "results/final"; T = ROOT / "paper/tables"; T.mkdir(parents=True, exist_ok=True)
J = lambda p: json.load(open(p))
ms = lambda v: f"{np.nanmean(v):.3f} $\\pm$ {np.nanstd(v):.3f}"


def w(name, body):
    (T / f"{name}.tex").write_text(body); print(name)


def t_f1():
    d = J(F / "f1_domino.json")["by_K"]
    rows = "\n".join(f"{K} & {d[K]['corr_mean_reproducible']:.3f} & {d[K]['null_corr_mean_reproducible']:.3f} & "
                     f"{d[K]['frac_beats_null_reproducible']:.2f} & {d[K]['domino_share_mean_reproducible']:.3f} & {d[K]['domino_share_median_reproducible']:.3f} \\\\"
                     for K in ["5", "10", "20", "50"])
    w("f1", r"""\begin{tabular}{rccccc}
\toprule
$K$ mediators & corr (cascade) & corr (random mediators) & frac.\ beats null & domino share (mean) & (median) \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}""")


def t_f2():
    d = J(F / "f2_benchmark.json"); s, se = d["scores_mean_corr"], d["scores_sem"]
    tg = ["gw raw (same screen)", "es raw (independent screen)", "es specific residual (independent screen)"]
    names = {"zero": "zero", "mean training profile": "mean training profile",
             "generic only (mean amplitude x generic direction)": "generic only",
             "kNN raw (k=50)": "similarity kNN ($k{=}50$)", "kNN raw (k=10)": "similarity kNN ($k{=}10$)",
             "factorised (generic mean + pathway kNN50 + specific kNN10)": "factorised (generic + pathway kNN50 + specific kNN10)"}
    rows = []
    for p, n in names.items():
        cells = []
        for t in tg:
            best = max(s[q][t] for q in names)
            v = f"{s[p][t]:.3f} $\\pm$ {se[p][t]:.3f}"
            cells.append(f"\\textbf{{{v}}}" if abs(s[p][t] - best) < 1e-12 else v)
        rows.append(f"{n} & " + " & ".join(cells) + r" \\")
    w("f2", r"""\begin{tabular}{lccc}
\toprule
Predictor & Same screen, raw & Independent screen, raw & Independent screen, specific \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")


def t_f3():
    m = J(F / "f3_memory_main.json"); n = J(F / "f3_memory_noise.json"); dt = J(F / "f3_memory_dt.json")
    g = lambda L, k="auroc_memory": [x[k] for x in L if x.get(k) is not None]
    rows = [
        ("Memory gain, 2 hidden regulators (clean)", m["main_clean"], "auroc_memory"),
        ("Same genes, nothing hidden (control)", m["control_all_observed_clean"], "auroc_memory"),
        ("Snapshot detector ($1-R^2$), hidden regulators", m["main_clean"], "auroc_snapshot_unexplained_var"),
        ("Memory gain, weak regulator fluctuations", m["weak_regulator_noise_clean"], "auroc_memory"),
    ]
    out = [f"{a} & {ms(g(L, k))} & {len(g(L, k))} \\\\" for a, L, k in rows]
    for k, L in n["meas_sweep"].items():
        out.append(f"Memory gain, measurement noise {k}$\\times$ fluct.\\ sd & {ms(g(L))} & {len(L)} \\\\")
    for k, L in n["meas_sweep_control"].items():
        out.append(f"\\quad control (nothing hidden), noise {k} & {ms(g(L))} & {len(L)} \\\\")
    out.append(f"Noise 0.5 + causal smoothing (attempted fix) & {ms(g(n['meas0.5_with_causal_smoothing']))} & 6 \\\\")
    for k, L in dt["dt_sweep_clean"].items():
        out.append(f"Sampling interval {k} (clean) & {ms(g(L))} & {len(L)} \\\\")
    gp = ms(g(m["main_clean"], "memory_gain_pos_median")); gn = ms(g(m["main_clean"], "memory_gain_neg_median"))
    op = ms(g(m["main_clean"], "oracle_gain_pos_median")); on = ms(g(m["main_clean"], "oracle_gain_neg_median"))
    w("f3", r"""\begin{tabular}{lcc}
\toprule
Detector / condition & AUROC (mean $\pm$ sd over seeds) & seeds \\
\midrule
""" + "\n".join(out) + r"""
\bottomrule
\end{tabular}""")
    w("f3_gain", f"Median memory gain (fraction of held-out error removed by the past), clean main condition: genes with a hidden regulator {gp}, other genes {gn}. "
      f"Oracle gain from adding the true hidden state: {op} vs {on}.")


def t_f4():
    a = J(F / "f4_attraction.json")
    rows = []
    for k, v in a["three_gene"].items():
        if k == "noise_sweep_observational":
            continue
        rows.append(f"{k} & {v['theory_coef_c_b_lambda']:.3f} & {v['empirical_coef_mean']:.3f} $\\pm$ {v['empirical_coef_sd']:.3f} & {v['mean_t_stat']:.1f} & {v['lasso_selects_spurious_A_to_C']:.2f} \\\\")
    w("f4_three", r"""\begin{tabular}{lcccc}
\toprule
Condition & theory $cb\lambda$ & simulated coefficient & mean $t$ & Lasso selects A$\to$C \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
    t = a["network_fp_table"]
    rows = [f"{k.split(' | ')[0]} & {k.split(' | ')[1]} & {v['n_pairs']} & {100 * v['false_positive_rate']:.1f} $\\pm$ {100 * v['seed_sd']:.1f} \\\\" for k, v in t.items()]
    w("f4_net", r"""\begin{tabular}{llcc}
\toprule
Source (regulator) & 2-hop mediator & indirect pairs & \% called (top-$E$), $\pm$ seed sd \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
    f = a["identifiability_flag"]
    w("f4_flag", r"""\begin{tabular}{lc}
\toprule
Quantity & Value \\
\midrule
""" + "\n".join(f"{k.replace('_', ' ')} & {v:.3f} \\\\" if isinstance(v, float) else f"{k.replace('_', ' ')} & {v} \\\\" for k, v in f.items()) + r"""
\bottomrule
\end{tabular}""")


def t_f5():
    R = [J(p) for p in sorted(glob.glob(str(F / "f5_s*.json")))]
    names = list(R[0]["variants"].keys())
    rows = []
    for n in names:
        g = lambda k: [r["variants"][n][k] for r in R]
        rows.append(f"{n.replace('%', chr(92) + '%')} & {np.mean(g('recall_ceiling')):.2f} & {ms(g('AUROC'))} & {ms(g('AUPRC_ratio'))} & {ms(g('precision_at_E'))} & {ms(g('indirect_frac_of_topE'))} & {ms(g('heldout_effect_corr'))} \\\\")
    w("f5", r"""\begin{tabular}{p{4.6cm}cccccc}
\toprule
Prior form & recall ceiling & AUROC & AUPRC$\times$ & Precision@$E$ & indirect frac. & held-out KD corr \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
    Ds = ["1", "2", "3", "4", "6", "8", "12", "inf"]
    keys = [k for k in R[0]["depth"]]
    rows = []
    for k in keys:
        if not all(k in r["depth"] for r in R):
            continue
        rows.append(k.replace("rho", "$\\rho$") + " & " + " & ".join(f"{np.nanmean([r['depth'][k][d] for r in R]):.2f}" for d in Ds) +
                    f" & {np.mean([r['depth'][k]['spectral_radius'] for r in R]):.2f} \\\\")
    w("f5_depth", r"""\begin{tabular}{p{4.2cm}ccccccccc}
\toprule
Model & $D{=}1$ & 2 & 3 & 4 & 6 & 8 & 12 & $(I-J)^{-1}$\textsuperscript{a} & $\rho(J)$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")


def t_f6():
    s = J(F / "f6_shifrut.json"); wv = J(F / "f6_wessels.json")
    def row(lab, d):
        sl = d["tls_slope_median"]; gg = d["fold_gain_change_median (max(s,1/s))"]
        return (f"{lab} & {d['n_comparisons']} & {d['sign_flips']}/{d['strong_shared_targets']} ({100 * d['flip_fraction']:.2f}\\%) & "
                f"{d['jaccard_mean']:.3f} & {d['corr_top50_median']:.2f} & {('%.2f' % sl) if sl is not None else '--'} & {('%.2f' % gg) if gg is not None else '--'} \\\\")
    rows = [r"\multicolumn{7}{l}{\emph{Norman 2019 (K562, CRISPRa; from earlier analysis)}} \\",
            r"A alone vs A in C's context & 260 & -- (1.5\%) & 0.185 & -- & -- & -- \\",
            r"A alone, split-half null & 251 & -- (0.0\%) & 0.430 & -- & -- & -- \\",
            r"Cell-cycle state bins & 152 & 5/10223 (0.05\%) & -- & 0.96 & -- & 1.26 \\",
            r"\midrule \multicolumn{7}{l}{\emph{Wessels 2023 (THP-1, Cas13 knockdown)}} \\",
            row("A alone vs A in C's context", wv["A alone vs A in C's context"]),
            row("A alone, split-half null", wv["null: A alone split-half"]),
            r"\midrule \multicolumn{7}{l}{\emph{Shifrut 2018 (primary T cells, CRISPR KO)}} \\",
            row("No stimulation vs stimulation", s["context (NoStim -> Stim)"]),
            row("split-half null (no stim.)", s["null split-half NoStim"]),
            row("split-half null (stim.)", s["null split-half Stim"])]
    w("f6", r"""\begin{tabular}{lcccccc}
\toprule
Comparison & $n$ & sign flips / strong shared & Jaccard & corr top-50 & gain slope (median) & fold gain change \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")


if __name__ == "__main__":
    for f in (t_f1, t_f2, t_f3, t_f4, t_f5, t_f6):
        try:
            f()
        except Exception as e:
            print("FAILED", f.__name__, repr(e))
