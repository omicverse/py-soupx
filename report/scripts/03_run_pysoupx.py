"""pysoupx run on one task, mirroring 02_run_soupx.R.

Usage: python 03_run_pysoupx.py <task_dir> <out_dir>
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

import pysoupx


def main():
    td, od = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    od.mkdir(parents=True, exist_ok=True)
    tod = sp.csc_matrix(scipy.io.mmread(gzip.open(td / "tod.mtx.gz")), dtype=np.float64)
    genes = pd.read_csv(td / "genes.tsv", sep="\t")
    droplets = (td / "droplets.tsv").read_text().split()
    cl = pd.read_csv(td / "clusters.tsv", sep="\t", dtype=str)
    toc = tod[:, pd.Index(droplets).get_indexer(cl.cell)]
    t0 = time.perf_counter()
    sc = pysoupx.SoupChannel(tod, toc, genes=genes.id.tolist(), cells=cl.cell.tolist(), droplets=droplets)
    sc = pysoupx.set_clusters(sc, dict(zip(cl.cell, cl.cluster)))
    sc = pysoupx.auto_est_cont(sc, verbose=False)
    out = sp.csc_matrix(pysoupx.adjust_counts(sc, verbose=0))
    elapsed = time.perf_counter() - t0
    out.sort_indices()
    out.indptr.astype(np.int32).tofile(od / "adj_p.bin")
    out.indices.astype(np.int32).tofile(od / "adj_i.bin")
    out.data.astype(np.float64).tofile(od / "adj_x.bin")
    np.asarray(sc.soup_profile["est"], dtype=np.float64).tofile(od / "soup_est.bin")
    fit = sc.fit
    (od / "result.json").write_text(json.dumps({
        "engine": "pysoupx", "version": pysoupx.__version__, "python": sys.version.split()[0],
        "rho": float(fit["rhoEst"]), "rho_fwhm": [float(v) for v in np.ravel(fit["rhoFWHM"])],
        "markers": sorted(set(map(str, fit["markersUsed"]["gene"]))), "seconds": elapsed,
        "dim": list(out.shape)}))
    print(td.name, f"rho={float(fit['rhoEst']):.4f} {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
