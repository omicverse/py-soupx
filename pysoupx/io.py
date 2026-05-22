"""Loading helpers — 10x CellRanger directories and AnnData round-trips."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .soupchannel import SoupChannel

__all__ = ["load_10x", "make_soup_channel", "to_anndata"]


def _read_10x_mtx(path: str):
    """Read a 10x ``matrix.mtx(.gz)`` directory; return (mtx, genes, bcs).

    The matrix is returned as genes x cells (CellRanger convention).
    """
    from scipy.io import mmread

    def _find(*names):
        for n in names:
            p = os.path.join(path, n)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"None of {names} in {path}")

    mtx = mmread(_find("matrix.mtx", "matrix.mtx.gz")).tocsc()
    feat = _find("features.tsv", "features.tsv.gz", "genes.tsv",
                 "genes.tsv.gz")
    bc = _find("barcodes.tsv", "barcodes.tsv.gz")
    genes = pd.read_csv(feat, sep="\t", header=None)
    # gene symbol is column 1 if present, else column 0
    gene_names = (genes[1] if genes.shape[1] > 1 else genes[0]).astype(str)
    barcodes = pd.read_csv(bc, sep="\t", header=None)[0].astype(str)
    return mtx, list(gene_names), list(barcodes)


def load_10x(
    data_dir: str,
    cell_ranger_pipestance: bool = True,
    keep_droplets: bool = False,
) -> SoupChannel:
    """Load a 10x CellRanger output folder into a :class:`SoupChannel`.

    Looks for ``raw_feature_bc_matrix`` / ``raw_gene_bc_matrices`` (the
    table of droplets) and ``filtered_feature_bc_matrix`` /
    ``filtered_gene_bc_matrices`` (the table of counts).  Genes are
    aligned automatically.  Port of SoupX ``load10X``.
    """
    def _resolve(*names):
        for n in names:
            p = os.path.join(data_dir, n)
            if os.path.isdir(p):
                # descend through single-subdir genome folders
                while True:
                    if os.path.exists(os.path.join(p, "matrix.mtx")) or \
                       os.path.exists(os.path.join(p, "matrix.mtx.gz")):
                        return p
                    subs = [d for d in os.listdir(p)
                            if os.path.isdir(os.path.join(p, d))]
                    if len(subs) == 1:
                        p = os.path.join(p, subs[0])
                    else:
                        return p
        raise FileNotFoundError(f"None of {names} found in {data_dir}")

    raw_dir = _resolve("raw_feature_bc_matrix", "raw_gene_bc_matrices",
                       "raw_gene_bc_matrix")
    filt_dir = _resolve("filtered_feature_bc_matrix",
                        "filtered_gene_bc_matrices",
                        "filtered_gene_bc_matrix")
    tod, tod_genes, tod_bc = _read_10x_mtx(raw_dir)
    toc, toc_genes, toc_bc = _read_10x_mtx(filt_dir)
    if tod_genes != toc_genes:
        # align tod to toc gene order
        idx = {g: i for i, g in enumerate(tod_genes)}
        order = [idx[g] for g in toc_genes]
        tod = tod[order, :]
    return SoupChannel(tod, toc, genes=toc_genes, cells=toc_bc,
                       droplets=tod_bc, keep_droplets=keep_droplets)


def make_soup_channel(
    tod,
    toc,
    genes=None,
    cells=None,
    droplets=None,
    meta_data: Optional[pd.DataFrame] = None,
    calc_soup_profile: bool = True,
    **kwargs,
) -> SoupChannel:
    """Flexible :class:`SoupChannel` constructor.

    ``tod`` / ``toc`` may be numpy / scipy matrices, pandas DataFrames
    (index = genes, columns = droplets/cells) or AnnData objects
    (cells x genes).  Gene / cell names are inferred when possible.
    """
    def _coerce(m, axis_genes):
        if hasattr(m, "X") and hasattr(m, "var_names"):  # AnnData
            return (m.X.T, list(map(str, m.var_names)),
                    list(map(str, m.obs_names)))
        if isinstance(m, pd.DataFrame):
            return (sp.csc_matrix(m.values), list(map(str, m.index)),
                    list(map(str, m.columns)))
        return sp.csc_matrix(m), None, None

    tod_m, tod_g, tod_c = _coerce(tod, True)
    toc_m, toc_g, toc_c = _coerce(toc, True)
    g = genes or toc_g or tod_g
    c = cells or toc_c
    d = droplets or tod_c
    if g is None:
        g = [f"gene{i}" for i in range(toc_m.shape[0])]
    if c is None:
        c = [f"cell{i}" for i in range(toc_m.shape[1])]
    return SoupChannel(tod_m, toc_m, genes=g, cells=c, droplets=d,
                       meta_data=meta_data,
                       calc_soup_profile=calc_soup_profile, **kwargs)


def to_anndata(sc: SoupChannel, corrected: Optional[sp.spmatrix] = None):
    """Return an AnnData (cells x genes) from a :class:`SoupChannel`.

    The (optionally corrected) ``toc`` becomes ``X``; the soup profile
    is stored in ``var`` and ``rho`` / ``clusters`` in ``obs``.
    """
    import anndata as ad

    mat = sc.toc if corrected is None else corrected
    adata = ad.AnnData(
        X=sp.csr_matrix(mat.T),
        obs=sc.meta_data.copy(),
        var=pd.DataFrame(index=sc.genes),
    )
    if sc.soup_profile is not None:
        adata.var["soup_est"] = sc.soup_profile["est"].to_numpy()
        adata.var["soup_counts"] = sc.soup_profile["counts"].to_numpy()
    if corrected is not None:
        adata.layers["raw_counts"] = sp.csr_matrix(sc.toc.T)
    return adata
