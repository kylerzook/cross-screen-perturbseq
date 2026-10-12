"""Sequence utilities: FASTA parsing, integer encoding, promoter loading."""
import gzip
import numpy as np

_LUT = np.full(256, 4, dtype=np.int8)  # 0=A 1=C 2=G 3=T 4=N
for i, c in enumerate("ACGT"):
    _LUT[ord(c)] = i
    _LUT[ord(c.lower())] = i


def encode(seq: str) -> np.ndarray:
    return _LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]


def read_fasta(path):
    op = gzip.open if str(path).endswith(".gz") else open
    name, buf = None, []
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name is not None:
        yield name, "".join(buf)


def background(int_seqs: np.ndarray) -> np.ndarray:
    """Base composition (A,C,G,T) of an int-encoded array, strand-symmetrised."""
    counts = np.bincount(int_seqs.ravel(), minlength=5)[:4].astype(float)
    counts = (counts + counts[::-1]) / 2.0
    return counts / counts.sum()


def load_promoters(upstream_fa, refgene_gz, keep_genes=None, length=1000):
    """One upstream-1kb promoter per gene symbol from UCSC upstream1000 + refGene.

    Returns dict with symbols, int-encoded seqs (G, length), and genomic coords
    (chrom, start0, end0, strand) so position i of the (transcript-oriented)
    sequence maps to genomic 0-based coordinate start0 + i ('+') or end0 - 1 - i ('-').
    """
    nm2sym = {}
    with gzip.open(refgene_gz, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            nm2sym.setdefault(f[1], f[12])
    seen, syms, seqs, chroms, starts, ends, strands = set(), [], [], [], [], [], []
    for name, s in read_fasta(upstream_fa):
        acc_part, loc = name.split()[0], name.split()[1]
        nm = "_".join(acc_part.split("_")[:2])
        if not nm.startswith("NM_"):
            continue
        sym = nm2sym.get(nm)
        chrom = loc.split(":")[0]
        if sym is None or sym in seen or "_" in chrom or len(s) != length:
            continue
        if keep_genes is not None and sym not in keep_genes:
            continue
        a, b = loc.split(":")[1].split("-")
        seen.add(sym)
        syms.append(sym); seqs.append(encode(s)); chroms.append(chrom)
        starts.append(int(a) - 1); ends.append(int(b))
        strands.append("+" if acc_part.endswith("_f") else "-")
    return dict(symbols=np.array(syms), seqs=np.stack(seqs), chrom=np.array(chroms),
                start=np.array(starts), end=np.array(ends), strand=np.array(strands))
