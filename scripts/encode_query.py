"""Query the ENCODE portal for the file manifests used by Tests 2-3.

Writes data/raw/encode_manifest.tsv (one row per selected file).
Selection rules (documented in docs/02_limitations.md):
  * TF ChIP-seq: GRCh38, released, bed narrowPeak, prefer 'conservative IDR thresholded peaks',
    fall back to 'IDR thresholded peaks'; one file per (cell line, target) - largest file kept.
  * ATAC-seq: GRCh38 'pseudoreplicated peaks' (fallback 'IDR thresholded peaks'), one per cell line.
  * RNA-seq: GRCh38 'gene quantifications' tsv from total RNA-seq, one per cell line.
"""
import json, sys, urllib.request, urllib.parse, csv
BASE = "https://www.encodeproject.org/search/?"
CELLS = ["K562", "HepG2"]
FIELDS = ["accession", "href", "file_size", "output_type", "target.label", "dataset",
          "biosample_ontology.term_name", "assay_title", "file_format_type", "preferred_default"]

def search(params):
    q = [("type", "File"), ("status", "released"), ("assembly", "GRCh38"),
         ("format", "json"), ("limit", "all")] + params + [("field", f) for f in FIELDS]
    url = BASE + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["@graph"]

rows = []
for cell in CELLS:
    # TF ChIP peaks
    best = {}
    for ot, rank in [("conservative IDR thresholded peaks", 0), ("IDR thresholded peaks", 1)]:
        for f in search([("assay_title", "TF ChIP-seq"), ("biosample_ontology.term_name", cell),
                         ("output_type", ot), ("file_format", "bed")]):
            tgt = (f.get("target") or {}).get("label")
            if not tgt:
                continue
            key = tgt.upper()
            cand = (rank, -f.get("file_size", 0), f)
            if key not in best or cand[:2] < best[key][:2]:
                best[key] = cand
    for tgt, (_, _, f) in sorted(best.items()):
        rows.append(dict(kind="chip", cell=cell, target=tgt, accession=f["accession"],
                         href=f["href"], output_type=f["output_type"], size=f.get("file_size", 0)))
    # ATAC peaks
    for ot in ["pseudoreplicated peaks", "IDR thresholded peaks"]:
        fs = search([("assay_title", "ATAC-seq"), ("biosample_ontology.term_name", cell),
                     ("output_type", ot), ("file_format", "bed")])
        if fs:
            f = max(fs, key=lambda x: x.get("file_size", 0))
            rows.append(dict(kind="atac", cell=cell, target="", accession=f["accession"],
                             href=f["href"], output_type=ot, size=f.get("file_size", 0)))
            break
    # RNA gene quantifications
    fs = search([("assay_title", "total RNA-seq"), ("biosample_ontology.term_name", cell),
                 ("output_type", "gene quantifications"), ("file_format", "tsv")])
    if fs:
        f = sorted(fs, key=lambda x: x["accession"])[0]
        rows.append(dict(kind="rna", cell=cell, target="", accession=f["accession"],
                         href=f["href"], output_type=f["output_type"], size=f.get("file_size", 0)))

with open("data/raw/encode_manifest.tsv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
    w.writeheader(); w.writerows(rows)
from collections import Counter
print(Counter((r["kind"], r["cell"]) for r in rows))
