"""JASPAR parsing, log-odds conversion and vectorised scanning.

Energy convention (Berg-von Hippel approximation): for a site with log-odds score S,
relative binding energy in kT units is E = S_max - S >= 0 (consensus site E = 0).
This is the 'PWM -> approximate energy matrix' stand-in for a true ProBound free-energy model.
"""
import numpy as np


def parse_jaspar(path):
    motifs, cur, rows = [], None, []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur is not None:
                motifs.append((cur[0], cur[1], np.array(rows, float)))
            parts = line[1:].split()
            cur, rows = (parts[0], parts[1] if len(parts) > 1 else parts[0]), []
        else:
            nums = line.split("[")[1].split("]")[0].split()
            rows.append([float(x) for x in nums])
    if cur is not None:
        motifs.append((cur[0], cur[1], np.array(rows, float)))
    # each matrix is 4 x w (A,C,G,T rows)
    return [dict(id=i, name=n, pfm=m) for i, n, m in motifs]


def logodds(pfm, bg, pseudo=0.8):
    """pfm 4 x w counts -> w x 4 natural-log log-odds."""
    counts = pfm.T  # w x 4
    n = counts.sum(1, keepdims=True)
    p = (counts + pseudo * bg) / (n + pseudo)
    return np.log(p / bg)


def revcomp_matrix(W):
    return W[::-1, ::-1].copy()


def scan_energies(int_seqs, W):
    """Return E (G, 2, P) float32 relative energies (kT) on both strands and S_max/S_min.

    int_seqs: (G, L) int8 with 0..3 bases and 4 = N (scores 0 for N).
    """
    G, L = int_seqs.shape
    w = W.shape[0]
    P = L - w + 1
    out = np.empty((G, 2, P), dtype=np.float32)
    smax = W.max(1).sum()
    smin = W.min(1).sum()
    for k, M in enumerate((W, revcomp_matrix(W))):
        Mp = np.vstack([M.T, np.zeros((1, w))]).astype(np.float32)  # 5 x w (row 4 = N)
        S = np.zeros((G, P), dtype=np.float32)
        for j in range(w):
            S += Mp[int_seqs[:, j:j + P], j]
        out[:, k, :] = smax - S
    return out, float(smax), float(smin)
