"""Gene-set estimate of the contamination fraction with pysoupx, mirroring 06_geneset.R.

Usage: python 06_geneset.py <task_dir> <out_json> [nomt|all]
``nomt`` (default) leaves out mitochondrial genes and MALAT1 when choosing the five genes.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

import pysoupx

td, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
tod = sp.csc_matrix(scipy.io.mmread(gzip.open(td / "tod.mtx.gz")), dtype=np.float64)
genes = pd.read_csv(td / "genes.tsv", sep="\t")
droplets = (td / "droplets.tsv").read_text().split()
cl = pd.read_csv(td / "clusters.tsv", sep="\t", dtype=str)
sc = pysoupx.SoupChannel(tod, tod[:, pd.Index(droplets).get_indexer(cl.cell)], genes=genes.id.tolist(),
                         cells=cl.cell.tolist(), droplets=droplets)
sc = pysoupx.set_clusters(sc, dict(zip(cl.cell, cl.cluster)))
est = sc.soup_profile["est"]
# mitochondrial genes and MALAT1 are enriched in empty droplets relative to the ambient RNA in cells
skip = genes.symbol.str.contains(r"(?:^|_)(?:mt-|malat1$)", case=False, regex=True).to_numpy()
mode = sys.argv[3] if len(sys.argv) > 3 else "nomt"
pool = lambda g: genes.id[(genes.genome == g).to_numpy() & (~skip if mode == "nomt" else True)].to_numpy()
top = {g: est[pool(g)].sort_values(ascending=False, kind="mergesort").index[:5].tolist() for g in ("mouse", "human")}
use = pysoupx.estimate_non_expressing_cells(sc, top)
sc = pysoupx.calculate_contamination_fraction(sc, top, use, force_accept=True, verbose=False)
rho = float(np.unique(sc.meta_data["rho"])[0])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"mode": mode, "rho": rho, "n_use": {k: int(v) for k, v in use.sum().items()}, "genes": top}))
print(td.name, f"gene-set rho {rho:.4f}")
