"""The :class:`SoupChannel` container — a faithful port of SoupX's
``SoupChannel`` list object.

A :class:`SoupChannel` bundles, for a single 10x-style channel:

* ``tod`` — table of droplets (genes x droplets), the *raw* unfiltered
  matrix used to estimate the soup profile (dropped after estimation
  unless ``keep_droplets=True``);
* ``toc`` — table of counts (genes x cells), the filtered cell matrix;
* ``meta_data`` — a per-cell :class:`pandas.DataFrame` carrying at least
  ``nUMIs`` and, once estimated/set, ``clusters`` and ``rho``;
* ``soup_profile`` — per-gene soup fraction (``est``) and total
  ``counts``;
* ``genes`` / ``cells`` — row / column names.

All matrices are stored as :class:`scipy.sparse.csc_matrix` for speed.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

ArrayLike = Union[np.ndarray, sp.spmatrix]


def _as_csc(m: ArrayLike) -> sp.csc_matrix:
    """Coerce a dense / sparse matrix to ``csc_matrix`` (float64)."""
    if sp.issparse(m):
        return m.tocsc().astype(np.float64)
    return sp.csc_matrix(np.asarray(m, dtype=np.float64))


class SoupChannel:
    """A single channel of droplet scRNA-seq data.

    Parameters
    ----------
    tod
        Table of droplets — genes x droplets matrix (raw / unfiltered).
        May be the same as ``toc`` if a raw matrix is unavailable.
    toc
        Table of counts — genes x cells matrix (filtered).
    genes
        Gene (row) names.  Required.
    cells
        Cell (column) names of ``toc``.  Required.
    droplets
        Column names of ``tod``.  Optional; defaults to integer labels.
    meta_data
        Optional per-cell metadata; index must match ``cells``.
    calc_soup_profile
        If ``True`` (default) the soup profile is estimated on
        construction via :meth:`estimate_soup`.
    soup_range
        ``(low, high)`` UMI bounds (exclusive) for empty droplets.
    keep_droplets
        Keep ``tod`` after soup estimation (uses more memory).
    """

    def __init__(
        self,
        tod: ArrayLike,
        toc: ArrayLike,
        genes: Sequence[str],
        cells: Sequence[str],
        droplets: Optional[Sequence[str]] = None,
        meta_data: Optional[pd.DataFrame] = None,
        calc_soup_profile: bool = True,
        soup_range: Sequence[float] = (0, 100),
        keep_droplets: bool = False,
    ) -> None:
        toc = _as_csc(toc)
        tod = _as_csc(tod)
        genes = list(map(str, genes))
        cells = list(map(str, cells))
        if tod.shape[0] != toc.shape[0]:
            raise ValueError(
                "tod and toc have different numbers of genes; both must "
                "have the same genes in the same order."
            )
        if len(genes) != toc.shape[0]:
            raise ValueError("len(genes) must equal the number of rows.")
        if len(cells) != toc.shape[1]:
            raise ValueError("len(cells) must equal the number of toc cols.")
        if droplets is None:
            droplets = [f"drop{i}" for i in range(tod.shape[1])]
        droplets = list(map(str, droplets))

        self.tod: Optional[sp.csc_matrix] = tod
        self.toc: sp.csc_matrix = toc
        self.genes: list[str] = genes
        self.cells: list[str] = cells
        self.droplets: list[str] = droplets

        n_umis = np.asarray(toc.sum(axis=0)).ravel()
        self.meta_data = pd.DataFrame({"nUMIs": n_umis}, index=cells)
        if meta_data is not None:
            md = meta_data.copy()
            if not (sorted(md.index.astype(str)) == sorted(cells)):
                raise ValueError(
                    "meta_data index must match the cell names of toc."
                )
            md = md.reindex(cells)
            md = md.drop(columns=[c for c in md.columns if c == "nUMIs"])
            self.meta_data = pd.concat([self.meta_data, md], axis=1)

        self.n_drop_umis = np.asarray(tod.sum(axis=0)).ravel()
        self.soup_profile: Optional[pd.DataFrame] = None
        self.fit: Optional[dict] = None

        if calc_soup_profile:
            from .estimate import estimate_soup

            estimate_soup(self, soup_range=soup_range,
                          keep_droplets=keep_droplets)

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_anndata(
        cls,
        adata,
        raw=None,
        cluster_key: Optional[str] = None,
        **kwargs,
    ) -> "SoupChannel":
        """Build a :class:`SoupChannel` from AnnData objects.

        Parameters
        ----------
        adata
            Filtered AnnData (cells x genes) — becomes the ``toc``.
        raw
            Raw / unfiltered AnnData (droplets x genes) — becomes the
            ``tod``.  If ``None``, ``adata`` is reused (no real soup
            estimation possible until a soup profile is set).
        cluster_key
            If given, ``adata.obs[cluster_key]`` is loaded as clusters.
        kwargs
            Forwarded to :class:`SoupChannel`.
        """
        genes = list(map(str, adata.var_names))
        cells = list(map(str, adata.obs_names))
        toc = adata.X.T  # genes x cells
        if raw is None:
            tod, droplets = toc, cells
        else:
            if list(map(str, raw.var_names)) != genes:
                # align raw to the filtered gene order
                raw = raw[:, adata.var_names]
            tod = raw.X.T
            droplets = list(map(str, raw.obs_names))
        meta = None
        if cluster_key is not None and cluster_key in adata.obs:
            meta = pd.DataFrame(
                {"clusters": adata.obs[cluster_key].astype(str).values},
                index=cells,
            )
        sc = cls(tod, toc, genes=genes, cells=cells, droplets=droplets,
                 meta_data=meta, **kwargs)
        return sc

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    @property
    def n_genes(self) -> int:
        return self.toc.shape[0]

    @property
    def n_cells(self) -> int:
        return self.toc.shape[1]

    def toc_dataframe(self) -> pd.DataFrame:
        """Return ``toc`` as a dense genes x cells DataFrame."""
        return pd.DataFrame(
            np.asarray(self.toc.todense()), index=self.genes,
            columns=self.cells,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"SoupChannel: {self.n_genes} genes x {self.n_cells} cells"
                f"{'' if self.soup_profile is None else ' (soup estimated)'}")
