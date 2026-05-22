"""``calculate_contamination_fraction`` — estimate rho from user-supplied
non-expressed gene sets, a faithful port of SoupX's
``calculateContaminationFraction``.

Given gene sets that are biologically absent from some cells (e.g.
haemoglobin genes outside erythrocytes) and a boolean matrix of which
cells genuinely do not express them (see
:func:`pysoupx.estimate_non_expressing_cells`), the contamination
fraction is the rate of a Poisson GLM ``counts ~ 1`` with offset
``log(expected soup counts)`` — i.e. ``rho = sum(observed) /
sum(expected)``.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .estimate import set_contamination_fraction
from .soupchannel import SoupChannel

__all__ = ["calculate_contamination_fraction"]


def calculate_contamination_fraction(
    sc: SoupChannel,
    non_expressed_gene_list: Mapping[str, Sequence[str]],
    use_to_est: pd.DataFrame,
    force_accept: bool = False,
    verbose: bool = True,
) -> SoupChannel:
    """Estimate a single global contamination fraction for the channel.

    Parameters
    ----------
    sc
        A :class:`SoupChannel` with an estimated soup profile.
    non_expressed_gene_list
        Mapping ``{set_name: [genes]}`` of gene sets assumed absent in
        some cells.
    use_to_est
        Boolean DataFrame (cells x gene-sets) marking, for each set, the
        cells to use for estimation — usually from
        :func:`pysoupx.estimate_non_expressing_cells`.
    force_accept
        Forwarded to :func:`set_contamination_fraction`.

    Returns
    -------
    SoupChannel
        ``sc`` with ``rho`` set in ``meta_data`` and a Poisson-GLM fit
        summary stored in ``sc.fit``.
    """
    if sc.soup_profile is None:
        raise ValueError("Soup profile not estimated.")
    if int(np.asarray(use_to_est).sum()) == 0:
        raise ValueError("use_to_est must not be all False.")

    gene_idx = {g: i for i, g in enumerate(sc.genes)}
    soup = sc.soup_profile["est"].to_numpy()
    n_umis = sc.meta_data["nUMIs"]
    toc = sc.toc.tocsr()

    obs_counts = []
    exp_soup = []
    for i, name in enumerate(non_expressed_gene_list):
        gl = non_expressed_gene_list[name]
        if isinstance(gl, str):
            gl = [gl]
        idxs = [gene_idx[g] for g in gl]
        s_frac = float(soup[idxs].sum())
        col = use_to_est[name] if name in use_to_est.columns \
            else use_to_est.iloc[:, i]
        cells = [c for c in sc.cells if bool(col[c])]
        if not cells:
            continue
        cell_pos = [sc.cells.index(c) for c in cells]
        cnts = np.asarray(toc[idxs][:, cell_pos].sum(axis=0)).ravel()
        obs_counts.append(cnts)
        exp_soup.append(n_umis.loc[cells].to_numpy() * s_frac)

    obs = np.concatenate(obs_counts)
    exp = np.concatenate(exp_soup)
    # Poisson GLM counts ~ 1 + offset(log(exp)) -> MLE rho = sum obs / sum exp
    rho = obs.sum() / exp.sum()

    # Wald 95% CI on the log-rate (matches glm confint closely)
    se = 1.0 / np.sqrt(obs.sum()) if obs.sum() > 0 else np.inf
    rho_low = float(rho * np.exp(-1.959963984540054 * se))
    rho_high = float(rho * np.exp(1.959963984540054 * se))

    sc.fit = {"method": "poisson_glm", "rhoEst": float(rho),
              "rhoLow": rho_low, "rhoHigh": rho_high,
              "nObs": int(obs.sum()), "expSoup": float(exp.sum())}
    sc = set_contamination_fraction(sc, float(rho), force_accept=force_accept)
    sc.meta_data["rhoLow"] = rho_low
    sc.meta_data["rhoHigh"] = rho_high
    if verbose:
        print(f"Estimated global contamination fraction of {100 * rho:.2f}%")
    return sc
