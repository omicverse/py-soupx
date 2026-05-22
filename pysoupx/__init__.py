"""pysoupx: a pure-Python port of the R package **SoupX**.

SoupX (Young & Behjati, *GigaScience* 2020, 9(12):giaa151) removes
ambient ("soup") mRNA contamination from droplet-based single-cell
RNA-seq data.  ``pysoupx`` is a faithful, dependency-light Python port
built on numpy / scipy / pandas / anndata.

Pipeline
--------
1. **Estimate the soup profile** — :func:`estimate_soup` uses very-low
   UMI "empty" droplets from the raw matrix; the per-gene soup fraction
   is ``sum(gene counts) / sum(total counts)`` across those droplets.
2. **Estimate the contamination fraction rho** — :func:`auto_est_cont`
   finds bimodal cluster-marker genes (tf-idf), assumes non-expressing
   clusters carry only soup, and takes the posterior mode as the global
   rho.  :func:`set_contamination_fraction` sets it manually.
3. **Remove the soup** — :func:`adjust_counts` performs a constrained
   multinomial subtraction yielding a corrected (integer-able) matrix.

Core object
-----------
* :class:`SoupChannel` — bundles the table of droplets / counts, the
  soup profile and per-cell metadata.  Build from matrices, a 10x
  directory (:func:`load_10x`) or an AnnData
  (:meth:`SoupChannel.from_anndata`).

Public API
----------
:class:`SoupChannel`, :func:`estimate_soup`, :func:`set_soup_profile`,
:func:`set_clusters`, :func:`set_contamination_fraction`,
:func:`quick_markers`, :func:`estimate_non_expressing_cells`,
:func:`auto_est_cont`, :func:`calculate_contamination_fraction`,
:func:`adjust_counts`, :func:`alloc`, :func:`expand_clusters`,
:func:`load_10x`, :func:`make_soup_channel`, :func:`to_anndata`,
:func:`plot_contamination_fraction`.
"""
from __future__ import annotations

from .soupchannel import SoupChannel
from .estimate import (
    estimate_soup,
    set_soup_profile,
    set_clusters,
    set_contamination_fraction,
    quick_markers,
    estimate_non_expressing_cells,
    auto_est_cont,
)
from .contamination import calculate_contamination_fraction
from .adjust import adjust_counts, alloc, expand_clusters
from .io import load_10x, make_soup_channel, to_anndata
from .plotting import plot_contamination_fraction

__version__ = "0.1.0"

__all__ = [
    "SoupChannel",
    "estimate_soup",
    "set_soup_profile",
    "set_clusters",
    "set_contamination_fraction",
    "quick_markers",
    "estimate_non_expressing_cells",
    "auto_est_cont",
    "calculate_contamination_fraction",
    "adjust_counts",
    "alloc",
    "expand_clusters",
    "load_10x",
    "make_soup_channel",
    "to_anndata",
    "plot_contamination_fraction",
    "__version__",
]
