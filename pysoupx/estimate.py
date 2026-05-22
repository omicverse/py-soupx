"""Soup-profile estimation, clustering / contamination setters and the
automatic contamination estimator — faithful ports of SoupX's
``estimateSoup``, ``setSoupProfile``, ``setClusters``,
``setContaminationFraction``, ``quickMarkers``,
``estimateNonExpressingCells`` and ``autoEstCont``.
"""
from __future__ import annotations

import warnings
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import gamma as gamma_dist
from scipy.stats import poisson

from .soupchannel import SoupChannel

__all__ = [
    "estimate_soup",
    "set_soup_profile",
    "set_clusters",
    "set_contamination_fraction",
    "quick_markers",
    "estimate_non_expressing_cells",
    "auto_est_cont",
]


# ----------------------------------------------------------------------
# soup profile
# ----------------------------------------------------------------------
def estimate_soup(
    sc: SoupChannel,
    soup_range: Sequence[float] = (0, 100),
    keep_droplets: bool = False,
) -> SoupChannel:
    """Estimate the soup expression profile from empty droplets.

    Droplets with ``soup_range[0] < nUMIs < soup_range[1]`` (endpoints
    excluded) are taken to contain only ambient RNA.  The per-gene soup
    fraction is ``sum(gene counts) / sum(total counts)`` over those
    droplets — exactly SoupX's ``estimateSoup``.
    """
    if sc.tod is None:
        raise ValueError(
            "Table of droplets (tod) has been dropped; cannot estimate "
            "the soup.  Provide a soup profile via set_soup_profile()."
        )
    lo, hi = soup_range
    w = np.where((sc.n_drop_umis > lo) & (sc.n_drop_umis < hi))[0]
    if w.size == 0:
        raise ValueError(
            f"No droplets found with UMI counts in ({lo}, {hi}); cannot "
            "estimate the soup.  Widen soup_range."
        )
    sub = sc.tod[:, w]
    counts = np.asarray(sub.sum(axis=1)).ravel()
    total = counts.sum()
    est = counts / total if total > 0 else np.zeros_like(counts)
    sc.soup_profile = pd.DataFrame(
        {"est": est, "counts": counts}, index=sc.genes
    )
    if not keep_droplets:
        sc.tod = None
    return sc


def set_soup_profile(sc: SoupChannel, soup_profile: pd.DataFrame) -> SoupChannel:
    """Manually set / replace the soup profile.

    ``soup_profile`` must have columns ``est`` and ``counts`` and an
    index that is a subset of the channel's gene names.
    """
    if "est" not in soup_profile.columns:
        raise ValueError("'est' column missing from soup_profile")
    if "counts" not in soup_profile.columns:
        raise ValueError("'counts' column missing from soup_profile")
    if not set(map(str, soup_profile.index)).issuperset(set(sc.genes)):
        raise ValueError("soup_profile invalid: not all genes found.")
    sc.soup_profile = soup_profile.loc[sc.genes].copy()
    return sc


# ----------------------------------------------------------------------
# clusters / contamination
# ----------------------------------------------------------------------
def set_clusters(
    sc: SoupChannel,
    clusters: Union[Sequence, Mapping],
) -> SoupChannel:
    """Attach clustering information to ``sc.meta_data['clusters']``.

    ``clusters`` may be a mapping ``{cell: cluster}`` or a sequence in
    the same order as ``sc.meta_data``.
    """
    if isinstance(clusters, Mapping):
        names = {str(k) for k in clusters}
        if set(sc.cells).issubset(names):
            vals = [str(clusters[c]) for c in sc.cells]
        else:
            raise ValueError(
                "Invalid cluster specification: mapping does not cover "
                "all cells."
            )
    elif isinstance(clusters, pd.Series):
        if set(sc.cells).issubset(set(map(str, clusters.index))):
            vals = [str(clusters[c]) for c in sc.cells]
        elif len(clusters) == len(sc.meta_data):
            vals = [str(v) for v in clusters.values]
        else:
            raise ValueError("Invalid cluster specification. See help.")
    else:
        clusters = list(clusters)
        if len(clusters) != len(sc.meta_data):
            raise ValueError("Invalid cluster specification. See help.")
        vals = [str(v) for v in clusters]
    if any(v in ("nan", "None", "") for v in vals):
        raise ValueError("NAs found in cluster names.")
    sc.meta_data["clusters"] = vals
    return sc


def set_contamination_fraction(
    sc: SoupChannel,
    cont_frac: Union[float, Mapping, pd.Series],
    force_accept: bool = False,
) -> SoupChannel:
    """Manually set the contamination fraction ``rho``.

    ``cont_frac`` is either a scalar (applied to every cell) or a
    per-cell mapping / Series.  Mirrors SoupX's
    ``setContaminationFraction`` thresholds (>1 hard error, >0.5 error,
    >0.3 warning) which can be downgraded with ``force_accept``.
    """
    if np.isscalar(cont_frac):
        vals = np.full(len(sc.meta_data), float(cont_frac))
    else:
        if isinstance(cont_frac, Mapping):
            cont_frac = pd.Series(cont_frac)
        cont_frac.index = cont_frac.index.astype(str)
        if not set(cont_frac.index).issubset(set(sc.cells)):
            raise ValueError(
                "cont_frac must be a scalar or a mapping whose keys are "
                "cell names."
            )
        vals = (sc.meta_data.get("rho", pd.Series(np.nan, index=sc.cells))
                .astype(float).to_numpy().copy())
        for cell, v in cont_frac.items():
            vals[sc.cells.index(cell)] = float(v)

    mx = float(np.nanmax(vals))
    if mx > 1:
        raise ValueError(
            "Contamination fraction greater than 1 detected. This is "
            "impossible and likely a failure of the estimation procedure."
        )
    if mx > 0.5:
        msg = (f"Extremely high contamination estimated ({mx:.2g}). This "
               "likely represents a failure in estimating the "
               "contamination fraction. Set force_accept=True to proceed.")
        if force_accept:
            warnings.warn(msg)
        else:
            raise ValueError(msg)
    elif mx > 0.3:
        warnings.warn(f"Estimated contamination is very high ({mx:.2g}).")
    sc.meta_data["rho"] = vals
    return sc


# ----------------------------------------------------------------------
# quickMarkers
# ----------------------------------------------------------------------
def quick_markers(
    toc: sp.spmatrix,
    clusters: Sequence,
    genes: Sequence[str],
    cells: Sequence[str],
    N: Optional[int] = 10,
    FDR: float = 0.01,
    express_cut: float = 0.9,
) -> pd.DataFrame:
    """tf-idf cluster markers — a port of SoupX ``quickMarkers``.

    Returns a DataFrame with one row per (gene, cluster) marker, sorted
    within each cluster by descending tf-idf, carrying ``geneFrequency``,
    ``geneFrequencyGlobal``, ``tfidf``, ``idf`` and ``qval``.

    ``N=None`` returns every gene passing the per-cluster FDR.
    """
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import hypergeom

    toc = toc.tocsc()
    genes = np.asarray(list(map(str, genes)))
    clusters = np.asarray([str(c) for c in clusters])
    n_cells = toc.shape[1]
    coo = toc.tocoo()
    mask = coo.data > express_cut
    gi = coo.row[mask]          # gene index of each "expressed" entry
    cj = coo.col[mask]          # cell index

    cl_levels = np.array(sorted(set(clusters)))
    cl_to_idx = {c: i for i, c in enumerate(cl_levels)}
    cl_cnts = np.array([(clusters == c).sum() for c in cl_levels],
                       dtype=float)

    n_clust = len(cl_levels)
    n_gene = len(genes)
    # nObs[g, c] = #cells in cluster c expressing gene g
    n_obs = np.zeros((n_gene, n_clust), dtype=float)
    cell_cluster_idx = np.array([cl_to_idx[c] for c in clusters])
    np.add.at(n_obs, (gi, cell_cluster_idx[cj]), 1.0)

    n_tot = n_obs.sum(axis=1)                       # per gene
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = n_obs / cl_cnts[None, :]
        ntf = (n_tot[:, None] - n_obs) / (n_cells - cl_cnts)[None, :]
        idf = np.log(n_cells / n_tot)
    idf[~np.isfinite(idf)] = 0.0
    score = tf * idf[:, None]

    # hypergeometric q-values per cluster
    qvals = np.ones((n_gene, n_clust))
    for c in range(n_clust):
        # P(X >= nObs) with X ~ hypergeom(n_cells, n_tot, cl_cnts[c])
        pv = hypergeom.sf(n_obs[:, c] - 1, n_cells, n_tot, cl_cnts[c])
        pv = np.where(np.isfinite(pv), pv, 1.0)
        qvals[:, c] = multipletests(pv, method="fdr_bh")[1]

    # second-best cluster frequency / name
    snd_best = np.zeros((n_gene, n_clust))
    snd_name = np.empty((n_gene, n_clust), dtype=object)
    for c in range(n_clust):
        other = np.delete(np.arange(n_clust), c)
        sub = tf[:, other]
        amax = np.argmax(sub, axis=1)
        snd_best[:, c] = sub[np.arange(n_gene), amax]
        snd_name[:, c] = cl_levels[other][amax]

    rows = []
    for c in range(n_clust):
        order = np.argsort(-score[:, c], kind="stable")
        passed = qvals[:, c] < FDR
        if N is not None and passed.sum() >= N:
            sel = order[:N]
        else:
            sel = order[passed[order]]
        for g in sel:
            rows.append({
                "gene": genes[g],
                "cluster": cl_levels[c],
                "geneFrequency": tf[g, c],
                "geneFrequencyOutsideCluster": ntf[g, c],
                "geneFrequencySecondBest": snd_best[g, c],
                "geneFrequencyGlobal": n_tot[g] / n_cells,
                "secondBestClusterName": snd_name[g, c],
                "tfidf": score[g, c],
                "idf": idf[g],
                "qval": qvals[g, c],
            })
    return pd.DataFrame(rows, columns=[
        "gene", "cluster", "geneFrequency", "geneFrequencyOutsideCluster",
        "geneFrequencySecondBest", "geneFrequencyGlobal",
        "secondBestClusterName", "tfidf", "idf", "qval"])


# ----------------------------------------------------------------------
# estimateNonExpressingCells
# ----------------------------------------------------------------------
def estimate_non_expressing_cells(
    sc: SoupChannel,
    non_expressed_gene_list: Mapping[str, Sequence[str]],
    clusters: Optional[Union[Sequence, Mapping, bool]] = None,
    maximum_contamination: float = 1.0,
    FDR: float = 0.05,
) -> pd.DataFrame:
    """Identify cells that genuinely do *not* express each gene set.

    A Poisson test asks whether a cell's observed counts for a gene set
    exceed ``maximum_contamination`` times the soup expectation; any
    cluster containing a genuinely-expressing cell is excluded.  Returns
    a boolean DataFrame (cells x gene-sets) — SoupX
    ``estimateNonExpressingCells``.
    """
    from statsmodels.stats.multitest import multipletests

    if sc.soup_profile is None:
        raise ValueError("Soup profile not estimated.")
    if clusters is None:
        if "clusters" in sc.meta_data.columns:
            clusters = sc.meta_data["clusters"].astype(str).to_dict()
        else:
            clusters = {c: c for c in sc.cells}
    elif clusters is False:
        clusters = {c: c for c in sc.cells}
    elif isinstance(clusters, Mapping):
        clusters = {str(k): str(v) for k, v in clusters.items()}
    else:
        clusters = {c: str(v) for c, v in zip(sc.cells, clusters)}

    gene_idx = {g: i for i, g in enumerate(sc.genes)}
    soup = sc.soup_profile["est"].to_numpy()
    n_umis = sc.meta_data["nUMIs"].to_numpy()
    toc = sc.toc.tocsr()

    set_names = list(non_expressed_gene_list)
    cnts = np.zeros((len(set_names), sc.n_cells))
    exp = np.zeros((len(set_names), sc.n_cells))
    for i, name in enumerate(set_names):
        gl = non_expressed_gene_list[name]
        if isinstance(gl, str):
            gl = [gl]
        idxs = [gene_idx[g] for g in gl]
        cnts[i] = np.asarray(toc[idxs, :].sum(axis=0)).ravel()
        exp[i] = soup[idxs].sum() * n_umis * maximum_contamination

    # Poisson test: P(X > cnts-1) ; BH per gene set
    pv = poisson.sf(cnts - 1, exp)
    qv = np.vstack([multipletests(pv[i], method="fdr_bh")[1]
                    for i in range(pv.shape[0])])

    cl_arr = np.array([clusters[c] for c in sc.cells])
    cl_levels = sorted(set(cl_arr))
    clust_exp = np.zeros((len(cl_levels), len(set_names)), dtype=bool)
    for ci, cl in enumerate(cl_levels):
        cols = np.where(cl_arr == cl)[0]
        # cluster is non-expressing for a set iff min q over its cells >= FDR
        clust_exp[ci] = qv[:, cols].min(axis=1) >= FDR
    cl_to_row = {c: i for i, c in enumerate(cl_levels)}
    cell_mat = np.vstack([clust_exp[cl_to_row[clusters[c]]] for c in sc.cells])

    n_use = int(cell_mat.sum())
    if n_use == 0:
        warnings.warn("No non-expressing cells identified.")
    elif n_use < 100:
        warnings.warn(
            f"Fewer than 100 non-expressing cells identified ({n_use})."
        )
    return pd.DataFrame(cell_mat, index=sc.cells, columns=set_names)


# ----------------------------------------------------------------------
# autoEstCont
# ----------------------------------------------------------------------
def auto_est_cont(
    sc: SoupChannel,
    top_markers: Optional[pd.DataFrame] = None,
    tfidf_min: float = 1.0,
    soup_quantile: float = 0.90,
    max_markers: int = 100,
    contamination_range: Sequence[float] = (0.01, 0.8),
    rho_max_fdr: float = 0.2,
    prior_rho: float = 0.05,
    prior_rho_std_dev: float = 0.10,
    force_accept: bool = False,
    verbose: bool = True,
) -> SoupChannel:
    """Automatically estimate the contamination fraction ``rho``.

    Faithful port of SoupX ``autoEstCont``.  Bimodal cluster-marker
    genes (high tf-idf, abundant in the soup) provide per-marker rho
    estimates in clusters where they are confidently non-expressed; a
    gamma-prior posterior is aggregated over those estimates and its
    mode (within ``contamination_range``) is the final ``rho``.
    """
    from statsmodels.stats.multitest import multipletests

    if "clusters" not in sc.meta_data.columns:
        raise ValueError("Clustering information must be supplied; run "
                         "set_clusters first.")
    if sc.soup_profile is None:
        raise ValueError("Soup profile not estimated.")

    clusters = sc.meta_data["clusters"].astype(str)
    cl_levels = sorted(clusters.unique())
    # collapse toc by cluster
    toc = sc.toc.tocsc()
    cluster_toc = np.zeros((sc.n_genes, len(cl_levels)))
    for j, cl in enumerate(cl_levels):
        cols = np.where(clusters.to_numpy() == cl)[0]
        cluster_toc[:, j] = np.asarray(toc[:, cols].sum(axis=1)).ravel()
    ssc_numis = cluster_toc.sum(axis=0)

    soup_est = sc.soup_profile["est"]
    soup_prof = soup_est.sort_values(ascending=False)
    soup_min = np.quantile(soup_prof.to_numpy(), soup_quantile)

    # ---- markers -----------------------------------------------------
    if top_markers is None:
        mrks = quick_markers(sc.toc, clusters.to_numpy(), sc.genes, sc.cells,
                             N=None)
        # keep most specific entry per gene
        mrks = mrks.sort_values(["gene", "tfidf"], ascending=[True, False])
        mrks = mrks.drop_duplicates("gene", keep="first")
        mrks = mrks.sort_values("tfidf", ascending=False).reset_index(drop=True)
        mrks = mrks[mrks["tfidf"] > tfidf_min].reset_index(drop=True)
    else:
        mrks = top_markers.reset_index(drop=True)

    tgts_in_soup = set(soup_prof.index[soup_prof > soup_min])
    filt_pass = mrks[mrks["gene"].isin(tgts_in_soup)].reset_index(drop=True)
    tgts = list(filt_pass["gene"].head(max_markers))
    if verbose:
        print(f"{len(mrks)} genes passed tf-idf cut-off and "
              f"{len(filt_pass)} soup quantile filter. "
              f"Taking the top {len(tgts)}.")
    if len(tgts) == 0:
        raise ValueError(
            "No plausible marker genes found. Is the channel low "
            "complexity? If not, reduce tfidf_min or soup_quantile."
        )
    if len(tgts) < 10:
        warnings.warn("Fewer than 10 marker genes found.")

    # ---- non-expressing matrix (per gene, at cell level) -------------
    gene_list = {g: [g] for g in tgts}
    ute_cells = estimate_non_expressing_cells(
        sc, gene_list, maximum_contamination=max(contamination_range),
        FDR=rho_max_fdr,
    )
    # collapse to cluster level: one row per cluster
    cl_arr = clusters.to_numpy()
    ute = np.zeros((len(cl_levels), len(tgts)), dtype=bool)
    for j, cl in enumerate(cl_levels):
        rep = np.where(cl_arr == cl)[0][0]
        ute[j] = ute_cells.iloc[rep].to_numpy()

    # ---- observed / expected counts per cluster ----------------------
    gene_idx = {g: i for i, g in enumerate(sc.genes)}
    tgt_i = np.array([gene_idx[g] for g in tgts])
    exp_cnts = np.outer(soup_est.to_numpy()[tgt_i], ssc_numis)
    obs_cnts = cluster_toc[tgt_i, :]

    # FDR that observed is consistent with full contamination
    pp = poisson.cdf(obs_cnts, exp_cnts * max(contamination_range))
    qq = multipletests(pp.ravel(), method="fdr_bh")[1].reshape(pp.shape)

    with np.errstate(divide="ignore", invalid="ignore"):
        rhos = obs_cnts / exp_cnts
    # rank index within each gene row (1-based, ties broken by order)
    rho_idx = np.argsort(np.argsort(rhos, axis=1, kind="stable"),
                         axis=1, kind="stable") + 1

    n_g, n_c = rhos.shape
    dd = pd.DataFrame({
        "gene": np.repeat(tgts, n_c),
        "passNonExp": ute.T.ravel(),
        "rhoEst": rhos.ravel(),
        "rhoIdx": rho_idx.ravel(),
        "obsCnt": obs_cnts.ravel(),
        "expCnt": exp_cnts.ravel(),
        "isExpressedFDR": qq.ravel(),
    })
    mrk_gene_idx = {g: i for i, g in enumerate(mrks["gene"])}
    dd["tfidf"] = dd["gene"].map(lambda g: mrks["tfidf"].iloc[mrk_gene_idx[g]]
                                 if g in mrk_gene_idx else np.nan)
    soup_rank = {g: i for i, g in enumerate(soup_prof.index)}
    dd["soupExp"] = dd["gene"].map(lambda g: soup_prof.iloc[soup_rank[g]])
    dd["useEst"] = dd["passNonExp"]

    n_use = int(dd["useEst"].sum())
    if n_use < 10:
        warnings.warn("Fewer than 10 independent estimates; rho estimation "
                      "is likely unstable.")
    if verbose:
        print(f"Using {n_use} independent estimates of rho.")

    # ---- gamma-prior posterior aggregation ---------------------------
    v2 = (prior_rho_std_dev / prior_rho) ** 2
    k = 1 + v2 ** -2 / 2 * (1 + np.sqrt(1 + 4 * v2))
    theta = prior_rho / (k - 1)

    rho_probes = np.arange(0, 1.0 + 1e-9, 0.001)
    use = dd[dd["useEst"]]
    if len(use) == 0:
        raise ValueError("No usable rho estimates; consider lowering "
                         "tfidf_min or rho_max_fdr.")
    obs = use["obsCnt"].to_numpy()
    expc = use["expCnt"].to_numpy()
    # posterior density of rho ~ Gamma(k+obs, scale=theta/(1+theta*expCnt))
    shape = k + obs
    scale = theta / (1 + theta * expc)
    post = np.array([
        np.mean(gamma_dist.pdf(e, a=shape, scale=scale))
        for e in rho_probes
    ])
    prior_curve = gamma_dist.pdf(rho_probes, a=k, scale=theta)

    lo, hi = contamination_range
    w = np.where((rho_probes >= lo) & (rho_probes <= hi))[0]
    rho_est = float(rho_probes[w][np.argmax(post[w])])
    half = post[w] >= post[w].max() / 2
    rho_fwhm = (float(rho_probes[w][half].min()),
                float(rho_probes[w][half].max()))
    if verbose:
        print(f"Estimated global rho of {rho_est:.2f}")

    sc.fit = {
        "dd": dd,
        "priorRho": prior_rho,
        "priorRhoStdDev": prior_rho_std_dev,
        "posterior": post,
        "prior": prior_curve,
        "rhoProbes": rho_probes,
        "rhoEst": rho_est,
        "rhoFWHM": rho_fwhm,
        "markersUsed": mrks,
    }
    set_contamination_fraction(sc, rho_est, force_accept=force_accept)
    return sc
