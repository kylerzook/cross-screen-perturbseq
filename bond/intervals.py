"""BED interval helpers (overlap labels, per-position accessibility)."""
import gzip
import numpy as np
from collections import defaultdict


def read_bed(path, signal_col=None):
    """Return dict chrom -> (starts, ends, signal) sorted by start."""
    d = defaultdict(list)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t")
            sig = float(f[signal_col]) if signal_col is not None else 1.0
            d[f[0]].append((int(f[1]), int(f[2]), sig))
    out = {}
    for c, rows in d.items():
        a = np.array(rows, dtype=float)
        a = a[np.argsort(a[:, 0])]
        out[c] = (a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2])
    return out


def count_peaks(bed):
    return int(sum(len(v[0]) for v in bed.values()))


def overlap_any(bed, chrom, start, end):
    """Boolean per region: does any interval overlap [start, end)?"""
    res = np.zeros(len(chrom), dtype=bool)
    cache = {}
    for i, (c, s, e) in enumerate(zip(chrom, start, end)):
        if c not in bed:
            continue
        if c not in cache:
            st, en, _ = bed[c]
            cache[c] = (st, np.maximum.accumulate(en))
        st, pm = cache[c]
        j = np.searchsorted(st, e) - 1
        res[i] = j >= 0 and pm[j] > s
    return res


def position_signal(bed, chrom, start, end, strand, L, max_len=20000):
    """(G, L) per-position max signal in transcript orientation (0 outside intervals)."""
    out = np.zeros((len(chrom), L), dtype=np.float32)
    for i, (c, s, e, sd) in enumerate(zip(chrom, start, end, strand)):
        if c not in bed:
            continue
        st, en, sig = bed[c]
        lo = np.searchsorted(st, s - max_len)
        hi = np.searchsorted(st, e)
        for k in range(lo, hi):
            a, b = max(st[k], s), min(en[k], e)
            if b <= a:
                continue
            ga, gb = a - s, b - s  # genomic-offset within region
            if sd == "+":
                out[i, ga:gb] = np.maximum(out[i, ga:gb], sig[k])
            else:
                out[i, L - gb:L - ga] = np.maximum(out[i, L - gb:L - ga], sig[k])
    return out


def to_log_acc(pos_signal, w, floor=0.02, q=99):
    """Normalise signal to (floor, 1], take log, align to motif-start positions (centre)."""
    nz = pos_signal[pos_signal > 0]
    scale = np.percentile(nz, q) if nz.size else 1.0
    a = np.clip(pos_signal / scale, 0, 1)
    a = np.maximum(a, floor)
    la = np.log(a)
    P = pos_signal.shape[1] - w + 1
    c = w // 2
    return la[:, c:c + P]
