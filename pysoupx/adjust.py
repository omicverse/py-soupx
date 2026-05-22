"""``adjust_counts`` and the helpers ``alloc`` / ``expand_clusters`` —
faithful ports of SoupX's soup-removal machinery.

``adjust_counts`` produces a corrected count matrix with the expected
ambient ("soup") signal removed.  Three methods are supported:

* ``subtraction`` (default) — distributes ``nUMIs * rho`` soup counts
  across genes proportional to the soup profile, capped by observed
  counts (the constrained ``alloc`` redistribution).
* ``soupOnly`` — a Poisson p-value procedure that removes whole genes
  judged to be pure contamination.
* ``multinomial`` — explicitly maximises the multinomial likelihood per
  cell (slow; nearly identical to ``subtraction``).

When clustering is available the correction is computed at the cluster
level and re-expanded to single cells (``expand_clusters``), exactly as
SoupX does.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2, poisson

from .soupchannel import SoupChannel

__all__ = ["adjust_counts", "alloc", "expand_clusters"]


# ----------------------------------------------------------------------
# alloc
# ----------------------------------------------------------------------
def alloc(
    tgt: float,
    bucket_lims: np.ndarray,
    ws: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Distribute ``tgt`` units across buckets, capped by ``bucket_lims``.

    Counts are spread proportional to weights ``ws``; any bucket that
    would overflow its cap is set to its cap and the residual is
    redistributed.  Bit-exact port of SoupX's ``alloc``.
    """
    bucket_lims = np.asarray(bucket_lims, dtype=np.float64)
    n = len(bucket_lims)
    if ws is None:
        ws = np.full(n, 1.0 / n)
    else:
        ws = np.asarray(ws, dtype=np.float64)
    s = ws.sum()
    if s == 0:
        return np.zeros(n)
    ws = ws / s
    # fast path: nothing overflows
    if np.all(tgt * ws <= bucket_lims):
        return tgt * ws
    # order by the point at which each bucket saturates
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ws > 0, bucket_lims / ws, np.inf)
    o = np.argsort(ratio, kind="stable")
    w = ws[o]
    y = bucket_lims[o]
    cw = np.concatenate([[0.0], np.cumsum(w[:-1])])
    cy = np.concatenate([[0.0], np.cumsum(y[:-1])])
    with np.errstate(divide="ignore", invalid="ignore"):
        k = y / w * (1 - cw) + cy
    k[w == 0] = np.inf
    b = k <= tgt
    resid = tgt - y[b].sum()
    denom = 1 - w[b].sum()
    w_renorm = w / denom if denom != 0 else np.zeros_like(w)
    out = np.where(b, y, resid * w_renorm)
    # reverse the sort
    inv = np.empty(n, dtype=int)
    inv[o] = np.arange(n)
    return out[inv]


# ----------------------------------------------------------------------
# expandClusters
# ----------------------------------------------------------------------
def expand_clusters(
    clust_soup_cnts: sp.spmatrix,
    cell_obs_cnts: sp.spmatrix,
    clusters: np.ndarray,
    cluster_names: Sequence[str],
    cell_weights: np.ndarray,
) -> sp.csc_matrix:
    """Expand cluster-level soup counts down to single cells.

    For each cluster the soup counts of every gene are distributed over
    its member cells (weighted by ``cell_weights``, capped by observed
    counts) using :func:`alloc`.  Port of SoupX ``expandClusters``.
    """
    clust_soup_cnts = clust_soup_cnts.tocsc()
    cell_obs_cnts = cell_obs_cnts.tocsc()
    clusters = np.asarray([str(c) for c in clusters])
    ws = np.asarray(cell_weights, dtype=np.float64)
    n_genes = cell_obs_cnts.shape[0]

    out = sp.lil_matrix((n_genes, cell_obs_cnts.shape[1]), dtype=np.float64)
    for j, cl in enumerate(cluster_names):
        w_cells = np.where(clusters == str(cl))[0]
        if w_cells.size == 0:
            continue
        wsum = ws[w_cells].sum()
        ww = ws[w_cells] / wsum if wsum > 0 else np.full(len(w_cells),
                                                         1.0 / len(w_cells))
        lims = cell_obs_cnts[:, w_cells].tocsr()
        n_soup = np.asarray(clust_soup_cnts[:, j].todense()).ravel()
        row_sums = np.asarray(lims.sum(axis=1)).ravel()
        # genes needing redistribution (some but not all counts are soup)
        w_genes = np.where((n_soup > 0) & (n_soup < row_sums))[0]
        for g in w_genes:
            row = lims.getrow(g)
            cols = row.indices
            vals = row.data
            alloc_vals = alloc(n_soup[g], vals, ww[cols])
            for c_local, v in zip(cols, alloc_vals):
                out[g, w_cells[c_local]] = v
        # genes where all counts are soup -> set to observed limits
        w_full = np.where((n_soup > 0) & (n_soup >= row_sums)
                          & (row_sums > 0))[0]
        for g in w_full:
            row = lims.getrow(g)
            for c_local, v in zip(row.indices, row.data):
                out[g, w_cells[c_local]] = v
    return out.tocsc()


# ----------------------------------------------------------------------
# adjustCounts
# ----------------------------------------------------------------------
def _stochastic_round(m: sp.coo_matrix, rng: np.random.Generator
                      ) -> sp.coo_matrix:
    """Floor each value, then bump up by 1 with prob = fractional part."""
    x = m.data
    fl = np.floor(x)
    frac = x - fl
    bump = rng.binomial(1, np.clip(frac, 0, 1))
    out = sp.coo_matrix((fl + bump, (m.row, m.col)), shape=m.shape)
    out.eliminate_zeros()
    return out


def adjust_counts(
    sc: SoupChannel,
    clusters: Optional[Union[Sequence, Mapping, bool]] = None,
    method: str = "subtraction",
    round_to_int: bool = False,
    tol: float = 1e-3,
    p_cut: float = 0.01,
    seed: Optional[int] = None,
    verbose: int = 1,
) -> sp.csc_matrix:
    """Return the corrected (soup-removed) count matrix.

    Parameters
    ----------
    sc
        A :class:`SoupChannel` with a soup profile and ``rho`` set.
    clusters
        Cluster labels (mapping / sequence).  ``None`` auto-loads from
        ``sc.meta_data``; ``False`` corrects each cell individually.
    method
        ``'subtraction'`` (default), ``'soupOnly'`` or ``'multinomial'``.
    round_to_int
        Stochastically round the result to integers.
    seed
        RNG seed for the multinomial tie-breaks / stochastic rounding.

    Returns
    -------
    scipy.sparse.csc_matrix
        Genes x cells corrected counts.
    """
    if method not in ("subtraction", "soupOnly", "multinomial"):
        raise ValueError(f"Unknown method '{method}'.")
    if "rho" not in sc.meta_data.columns:
        raise ValueError("Contamination fraction (rho) must be set first.")
    if sc.soup_profile is None:
        raise ValueError("Soup profile not estimated.")
    rng = np.random.default_rng(seed)

    if clusters is None:
        if "clusters" in sc.meta_data.columns:
            clusters = sc.meta_data["clusters"].astype(str).to_dict()
        else:
            clusters = False

    # ------------------------------------------------------------------
    # cluster path: correct at cluster level, then re-expand
    # ------------------------------------------------------------------
    if clusters is not False:
        if isinstance(clusters, Mapping):
            cl_map = {str(k): str(v) for k, v in clusters.items()}
        else:
            cl_map = {c: str(v) for c, v in zip(sc.cells, clusters)}
        if not set(sc.cells).issubset(cl_map):
            raise ValueError("clusters must cover every cell in toc.")
        cl_arr = np.array([cl_map[c] for c in sc.cells])
        cl_levels = sorted(set(cl_arr))

        toc = sc.toc.tocsc()
        n_umis = sc.meta_data["nUMIs"].to_numpy()
        rho = sc.meta_data["rho"].to_numpy()

        cluster_toc = np.zeros((sc.n_genes, len(cl_levels)))
        cl_numis = np.zeros(len(cl_levels))
        cl_rho = np.zeros(len(cl_levels))
        for j, cl in enumerate(cl_levels):
            cols = np.where(cl_arr == cl)[0]
            cluster_toc[:, j] = np.asarray(toc[:, cols].sum(axis=1)).ravel()
            cl_numis[j] = n_umis[cols].sum()
            cl_rho[j] = (np.sum(rho[cols] * n_umis[cols]) / n_umis[cols].sum()
                         if n_umis[cols].sum() > 0 else 0.0)

        tmp = SoupChannel.__new__(SoupChannel)
        tmp.tod = None
        tmp.toc = sp.csc_matrix(cluster_toc)
        tmp.genes = sc.genes
        tmp.cells = cl_levels
        tmp.droplets = []
        tmp.n_drop_umis = np.array([])
        tmp.soup_profile = sc.soup_profile
        tmp.fit = None
        tmp.meta_data = pd.DataFrame(
            {"nUMIs": cl_numis, "rho": cl_rho}, index=cl_levels
        )
        # corrected cluster matrix -> cluster soup counts
        corrected = adjust_counts(tmp, clusters=False, method=method,
                                  round_to_int=False, tol=tol, p_cut=p_cut,
                                  seed=seed, verbose=verbose)
        clust_soup = tmp.toc - corrected
        cell_soup = expand_clusters(
            clust_soup, sc.toc, cl_arr, cl_levels, n_umis * rho,
        )
        out = (sc.toc - cell_soup).tocsc()
        out.data = np.maximum(out.data, 0.0)
        out.eliminate_zeros()
    else:
        out = _adjust_single(sc, method, tol, p_cut, rng, verbose)

    if round_to_int:
        out = _stochastic_round(out.tocoo(), rng).tocsc()
    return out


def _adjust_single(sc, method, tol, p_cut, rng, verbose) -> sp.csc_matrix:
    """Cell-level (no clusters) correction for one of the three methods."""
    toc = sc.toc.tocsc()
    soup_frac = sc.soup_profile["est"].to_numpy()
    n_umis = sc.meta_data["nUMIs"].to_numpy()
    rho = sc.meta_data["rho"].to_numpy()
    exp_soup = n_umis * rho

    if method == "subtraction":
        cols = []
        for e in range(toc.shape[1]):
            col = toc.getcol(e)
            idx = col.indices
            vals = col.data
            if len(idx) == 0:
                cols.append(sp.csc_matrix((toc.shape[0], 1)))
                continue
            removed = alloc(exp_soup[e], vals, soup_frac[idx])
            new_vals = vals - removed
            new_vals = np.maximum(new_vals, 0.0)
            cols.append(sp.csc_matrix(
                (new_vals, idx, [0, len(idx)]), shape=(toc.shape[0], 1)))
        out = sp.hstack(cols).tocsc()
        out.eliminate_zeros()
        return out

    if method == "multinomial":
        # initialise with the subtraction solution
        sub = _adjust_single(sc, "subtraction", tol, p_cut, rng, verbose)
        sub_int = _stochastic_round(sub.tocoo(), rng).tocsc()
        fit_init = (toc - sub_int).tocsc()
        ps = soup_frac
        cols = []
        for i in range(toc.shape[1]):
            tgt_n = int(round(rho[i] * n_umis[i]))
            lims = np.asarray(toc.getcol(i).todense()).ravel()
            fit = np.asarray(fit_init.getcol(i).todense()).ravel()
            fit = _multinomial_cell(fit, lims, ps, tgt_n, rng)
            cols.append(sp.csc_matrix(fit.reshape(-1, 1)))
        soup = sp.hstack(cols).tocsc()
        out = (toc - soup).tocsc()
        out.data = np.maximum(out.data, 0.0)
        out.eliminate_zeros()
        return out

    # soupOnly
    coo = toc.tocoo()
    g = coo.row
    c = coo.col
    x = coo.data
    # p-value of each count being soup (Poisson upper tail)
    lam = n_umis[c] * soup_frac[g] * rho[c]
    p = poisson.sf(x - 1, lam)
    # order by cell, then by p-value descending
    order = np.lexsort((-p, c))
    rtot = np.empty_like(x)
    keep_mask = np.zeros(len(x), dtype=bool)
    # process per cell
    start = 0
    sorted_c = c[order]
    for cell in np.unique(sorted_c):
        sel = order[sorted_c == cell]
        csum = np.cumsum(x[sel])
        # P(running soup total exceeds target) Poisson upper tail
        p_soup = poisson.sf(csum - x[sel] - 1, n_umis[cell] * rho[cell])
        pp = p[sel] * p_soup
        with np.errstate(divide="ignore"):
            q = chi2.sf(-2 * np.log(np.clip(pp, 1e-300, None)), 4)
        keep = q < p_cut
        keep_mask[sel] = keep
    out = sp.coo_matrix((x[keep_mask], (g[keep_mask], c[keep_mask])),
                        shape=toc.shape).tocsc()
    out.eliminate_zeros()
    return out


def _multinomial_cell(fit, lims, ps, tgt_n, rng) -> np.ndarray:
    """Maximise the multinomial soup likelihood for one cell.

    ``fit`` is the running soup allocation (initialised from
    subtraction); it is iteratively perturbed so its total reaches
    ``tgt_n`` while maximising ``sum(fit * log ps)``.  Port of SoupX's
    inner multinomial loop in ``adjustCounts``.
    """
    fit = fit.astype(np.float64).copy()
    with np.errstate(divide="ignore"):
        log_ps = np.log(ps)
    while True:
        increasable = fit < lims
        decreasable = fit > 0
        if not increasable.any() or not decreasable.any():
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            del_inc = log_ps[increasable] - np.log(fit[increasable] + 1)
            del_dec = -log_ps[decreasable] + np.log(fit[decreasable])
        w_inc_all = np.where(increasable)[0][del_inc == del_inc.max()]
        w_dec_all = np.where(decreasable)[0][del_dec == del_dec.max()]
        w_inc = (rng.choice(w_inc_all) if len(w_inc_all) > 1
                 else w_inc_all[0])
        w_dec = (rng.choice(w_dec_all) if len(w_dec_all) > 1
                 else w_dec_all[0])
        cur_n = fit.sum()
        if cur_n < tgt_n:
            fit[w_inc] += 1
        elif cur_n > tgt_n:
            fit[w_dec] -= 1
        else:
            del_tot = del_inc.max() + del_dec.max()
            if del_tot == 0:
                fit[w_dec_all] -= 1
                zero_bucket = np.unique(np.concatenate([w_inc_all,
                                                        w_dec_all]))
                fit[zero_bucket] += len(w_dec_all) / len(zero_bucket)
                break
            elif del_tot < 0:
                break
            else:
                fit[w_inc] += 1
                fit[w_dec] -= 1
    return fit
