"""ProBound free-energy models from MotifCentral (Rube et al., Nat Biotechnol 2022).

MotifCentral v1.0.0 JSON: one mononucleotide binding mode per fit; coefficients are
ln(relative affinity) per (position, base), position-major in ACGT order (best base = 0).
Here W[j, b] = coefficient, so for a site S = sum_j W[j, s_j] = -ddG/RT and E = S_max - S
is a calibrated relative binding free energy in kT units (unlike PWM log-odds).

Selection rule (one model per gene symbol, fixed before looking at results):
single-factor fits only; prefer human (9606) > mouse (10090) > other species; then lowest fit_id.
"""
import json
import numpy as np
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "data/raw/probound/MotifCentral.v1.0.0.json"
_RANK = {9606: 0, 10090: 1}


def load_models(path=PATH):
    best = {}
    for d in json.load(open(path)):
        fac = d["metadata"]["factors"]
        if len(fac) != 1:
            continue
        sym = fac[0]["gene_symbol"].upper()
        key = (_RANK.get(fac[0]["tax_id"], 2), d["metadata"]["fit_id"])
        if sym not in best or key < best[sym][0]:
            mode = d["modelSettings"]["bindingModes"][0]
            assert d["modelSettings"]["letterOrder"] == "ACGT"
            c = np.array(d["coefficients"]["bindingModes"][0]["mononucleotide"], float)
            W = c.reshape(mode["size"], 4)
            W = W - W.max(1, keepdims=True)
            best[sym] = (key, dict(W=W, fit_id=d["metadata"]["fit_id"], tax_id=fac[0]["tax_id"],
                                   experiment=d["metadata"]["experiments"][0][0]))
    return {k: v[1] for k, v in best.items()}
