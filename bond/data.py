"""Shared loaders for the real-data tests (K562 / HepG2)."""
import csv
from pathlib import Path
import numpy as np
import h5py
from .seq import load_promoters, background
from .pwm import parse_jaspar, logodds, scan_energies

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
PERTURB = RAW / "K562_gwps_normalized_bulk_01.h5ad"


def perturb_var_genes():
    with h5py.File(PERTURB, "r") as f:
        cats = [x.decode() if isinstance(x, bytes) else x for x in f["var/__categories/gene_name"][:]]
        return np.array([cats[c] for c in f["var/gene_name"][:]])


def jaspar_by_name():
    out = {}
    for m in parse_jaspar(RAW / "jaspar2024_vert_nr.txt"):
        if "::" in m["name"]:
            continue
        out.setdefault(m["name"].upper(), m)
    return out


def manifest():
    return list(csv.DictReader(open(RAW / "encode_manifest.tsv"), delimiter="\t"))


def encode_path(row):
    sub = row["kind"]
    ext = row["href"].split("/")[-1]
    return RAW / "encode" / row["cell"] / sub / f"{row['target'] or row['kind']}__{ext}"


def chip_files(cell):
    J = jaspar_by_name()
    return {r["target"]: encode_path(r) for r in manifest()
            if r["kind"] == "chip" and r["cell"] == cell and r["target"] in J}


def other_file(cell, kind):
    return [encode_path(r) for r in manifest() if r["kind"] == kind and r["cell"] == cell][0]


def promoters(keep_genes=None):
    return load_promoters(RAW / "upstream1000.fa.gz", RAW / "refGene.txt.gz", keep_genes=keep_genes)


def energy_threshold(W, bg, pval=1e-4, n=400000, seed=0):
    rng = np.random.default_rng(seed)
    w = W.shape[0]
    seq = rng.choice(4, size=(n // 100, 100 + w - 1), p=bg).astype(np.int8)
    E, _, _ = scan_energies(seq, W)
    return float(np.quantile(E[:, 0, :].ravel(), pval))


def chip_files_all(cell):
    """All released TF ChIP peak files for a cell line (no motif-source filter)."""
    return {r["target"]: encode_path(r) for r in manifest() if r["kind"] == "chip" and r["cell"] == cell}


def motif_source(model):
    """Return (tf -> callable(bg) -> W) for 'jaspar' (PWM log-odds) or 'probound' (free energies)."""
    if model == "jaspar":
        from .pwm import logodds
        J = jaspar_by_name()
        return {t: (lambda bg, m=m: logodds(m["pfm"], bg)) for t, m in J.items()}
    if model == "probound":
        from .probound import load_models
        return {t: (lambda bg, m=m: m["W"]) for t, m in load_models().items()}
    raise ValueError(model)
