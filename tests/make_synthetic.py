"""Generate the shared synthetic dataset used by the R-parity tests.

A small droplet scRNA-seq channel is simulated with a known
contamination fraction so that both R ``SoupX`` and ``pysoupx`` can be
run on *identical* input.  The matrices are written as plain TSV (genes
x columns) plus a clusters table; both the R driver and the Python tests
read these files.

Layout written to ``out_dir``:

* ``tod.tsv``       — table of droplets  (genes x droplets), raw matrix.
* ``toc.tsv``       — table of counts    (genes x cells),   filtered.
* ``clusters.tsv``  — cell -> cluster mapping.
* ``meta.tsv``      — true rho and other generation parameters.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

#: contamination fraction baked into the synthetic data.
RHO_TRUE = 0.10
N_GENES = 200
N_CELLS = 300
N_EMPTY = 5000
N_CELL_TYPES = 4
SEED = 0


def generate(out_dir: str) -> dict:
    """Write the synthetic dataset into ``out_dir``; return a summary."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # per-cell-type expression profiles: 40 shared low-expression genes
    # plus a private block of 20 strong marker genes each.
    profs = []
    for ct in range(N_CELL_TYPES):
        prof = np.zeros(N_GENES)
        prof[:40] = 0.5
        prof[40 + ct * 20: 40 + (ct + 1) * 20] = 10.0
        profs.append(prof / prof.sum())
    # the soup is a lysed mix of all cell types.
    soup_p = np.mean(profs, axis=0)

    toc = np.zeros((N_GENES, N_CELLS), dtype=int)
    clusters = []
    for j in range(N_CELLS):
        ct = j % N_CELL_TYPES
        clusters.append(f"c{ct}")
        n = 500
        endo = rng.multinomial(int(round(n * (1 - RHO_TRUE))), profs[ct])
        cont = rng.multinomial(int(round(n * RHO_TRUE)), soup_p)
        toc[:, j] = endo + cont

    empties = np.array(
        [rng.multinomial(30, soup_p) for _ in range(N_EMPTY)]
    ).T
    tod = np.hstack([toc, empties])

    genes = [f"g{i}" for i in range(N_GENES)]
    cells = [f"cell{i}" for i in range(N_CELLS)]
    droplets = cells + [f"d{i}" for i in range(N_EMPTY)]

    pd.DataFrame(tod, index=genes, columns=droplets).to_csv(
        os.path.join(out_dir, "tod.tsv"), sep="\t")
    pd.DataFrame(toc, index=genes, columns=cells).to_csv(
        os.path.join(out_dir, "toc.tsv"), sep="\t")
    pd.DataFrame({"cell": cells, "cluster": clusters}).to_csv(
        os.path.join(out_dir, "clusters.tsv"), sep="\t", index=False)
    pd.DataFrame({"key": ["rho_true", "n_genes", "n_cells", "n_empty"],
                  "value": [RHO_TRUE, N_GENES, N_CELLS, N_EMPTY]}).to_csv(
        os.path.join(out_dir, "meta.tsv"), sep="\t", index=False)
    return {"genes": genes, "cells": cells, "droplets": droplets,
            "rho_true": RHO_TRUE}


if __name__ == "__main__":  # pragma: no cover
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    generate(d)
    print(f"wrote synthetic dataset to {d}")
