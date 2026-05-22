"""Self-contained smoke tests for pysoupx (no R required).

These check the internal consistency of every ported routine on the
synthetic dataset and on tiny hand-built examples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pysoupx as soup
from pysoupx.adjust import alloc


# ----------------------------------------------------------------------
# SoupChannel construction
# ----------------------------------------------------------------------
def test_soupchannel_construction(loaded):
    sc = soup.SoupChannel(
        loaded["tod"].values, loaded["toc"].values,
        genes=loaded["genes"], cells=loaded["cells"],
        droplets=loaded["droplets"],
    )
    assert sc.n_genes == len(loaded["genes"])
    assert sc.n_cells == len(loaded["cells"])
    assert sc.soup_profile is not None
    # nUMIs equals the column sums of toc
    np.testing.assert_allclose(
        sc.meta_data["nUMIs"].to_numpy(),
        loaded["toc"].values.sum(axis=0),
    )
    # soup profile is a probability vector
    assert abs(sc.soup_profile["est"].sum() - 1.0) < 1e-9
    assert (sc.soup_profile["est"] >= 0).all()


def test_soupchannel_gene_mismatch_errors(loaded):
    bad = loaded["toc"].values[:-1]
    with pytest.raises(ValueError):
        soup.SoupChannel(loaded["tod"].values, bad,
                         genes=loaded["genes"], cells=loaded["cells"])


def test_from_anndata_roundtrip(loaded):
    ad = pytest.importorskip("anndata")
    raw = ad.AnnData(
        X=sp.csr_matrix(loaded["tod"].values.T),
        var=pd.DataFrame(index=loaded["genes"]),
        obs=pd.DataFrame(index=loaded["droplets"]),
    )
    filt = ad.AnnData(
        X=sp.csr_matrix(loaded["toc"].values.T),
        var=pd.DataFrame(index=loaded["genes"]),
        obs=pd.DataFrame(
            {"leiden": loaded["clusters"]["cluster"].values},
            index=loaded["cells"]),
    )
    sc = soup.SoupChannel.from_anndata(filt, raw=raw, cluster_key="leiden")
    assert sc.soup_profile is not None
    assert "clusters" in sc.meta_data.columns
    out_ad = soup.to_anndata(sc)
    assert out_ad.shape == (sc.n_cells, sc.n_genes)


# ----------------------------------------------------------------------
# alloc — the constrained redistribution primitive
# ----------------------------------------------------------------------
def test_alloc_no_overflow():
    # target small relative to caps -> simple proportional split
    out = alloc(3.0, np.array([10.0, 10.0, 10.0]))
    np.testing.assert_allclose(out, [1.0, 1.0, 1.0])


def test_alloc_with_overflow():
    # one bucket too small -> it saturates, residual redistributed
    out = alloc(9.0, np.array([1.0, 100.0, 100.0]))
    assert abs(out[0] - 1.0) < 1e-12          # capped
    assert abs(out.sum() - 9.0) < 1e-12       # total preserved
    assert out[1] == out[2]                   # equal weights


def test_alloc_total_never_exceeds_caps():
    rng = np.random.default_rng(0)
    for _ in range(50):
        lims = rng.random(8) * 5
        out = alloc(lims.sum() * 0.6, lims)
        assert (out <= lims + 1e-9).all()
        assert abs(out.sum() - lims.sum() * 0.6) < 1e-9


# ----------------------------------------------------------------------
# soup profile / setters
# ----------------------------------------------------------------------
def test_estimate_soup_range(loaded):
    sc = soup.SoupChannel(
        loaded["tod"].values, loaded["toc"].values,
        genes=loaded["genes"], cells=loaded["cells"],
        droplets=loaded["droplets"], calc_soup_profile=False,
        keep_droplets=True,
    )
    assert sc.soup_profile is None
    soup.estimate_soup(sc, soup_range=(0, 100))
    assert sc.soup_profile is not None


def test_set_soup_profile_manual(loaded):
    sc = soup.SoupChannel(
        loaded["toc"].values, loaded["toc"].values,
        genes=loaded["genes"], cells=loaded["cells"],
        calc_soup_profile=False,
    )
    counts = loaded["toc"].values.sum(axis=1)
    prof = pd.DataFrame({"est": counts / counts.sum(), "counts": counts},
                        index=loaded["genes"])
    soup.set_soup_profile(sc, prof)
    assert abs(sc.soup_profile["est"].sum() - 1.0) < 1e-9


def test_set_contamination_fraction_thresholds(loaded):
    sc = soup.SoupChannel(loaded["tod"].values, loaded["toc"].values,
                          genes=loaded["genes"], cells=loaded["cells"],
                          droplets=loaded["droplets"])
    soup.set_contamination_fraction(sc, 0.1)
    assert (sc.meta_data["rho"] == 0.1).all()
    with pytest.raises(ValueError):           # > 1 impossible
        soup.set_contamination_fraction(sc, 1.5)
    with pytest.raises(ValueError):           # > 0.5 needs force_accept
        soup.set_contamination_fraction(sc, 0.7)
    soup.set_contamination_fraction(sc, 0.7, force_accept=True)
    assert (sc.meta_data["rho"] == 0.7).all()


# ----------------------------------------------------------------------
# quick_markers / autoEstCont
# ----------------------------------------------------------------------
def test_quick_markers_finds_planted_markers(loaded):
    mrks = soup.quick_markers(
        sp.csc_matrix(loaded["toc"].values),
        loaded["clusters"]["cluster"].values,
        loaded["genes"], loaded["cells"], N=10,
    )
    assert len(mrks) > 0
    # planted markers for cluster c0 live in genes g40..g59
    c0 = mrks[mrks["cluster"] == "c0"]
    planted = {f"g{i}" for i in range(40, 60)}
    assert len(set(c0["gene"]) & planted) >= 5


def test_auto_est_cont_recovers_rho(loaded):
    sc = soup.SoupChannel(loaded["tod"].values, loaded["toc"].values,
                          genes=loaded["genes"], cells=loaded["cells"],
                          droplets=loaded["droplets"])
    soup.set_clusters(sc, dict(zip(loaded["clusters"]["cell"],
                                   loaded["clusters"]["cluster"])))
    soup.auto_est_cont(sc, tfidf_min=0.5, verbose=False, force_accept=True)
    rho = sc.meta_data["rho"].iloc[0]
    # true contamination is 0.10
    assert 0.04 < rho < 0.16
    assert sc.fit is not None and "posterior" in sc.fit


# ----------------------------------------------------------------------
# adjust_counts
# ----------------------------------------------------------------------
def test_adjust_counts_removes_expected_soup(loaded):
    sc = soup.SoupChannel(loaded["tod"].values, loaded["toc"].values,
                          genes=loaded["genes"], cells=loaded["cells"],
                          droplets=loaded["droplets"])
    soup.set_clusters(sc, dict(zip(loaded["clusters"]["cell"],
                                   loaded["clusters"]["cluster"])))
    soup.set_contamination_fraction(sc, 0.10)
    out = soup.adjust_counts(sc, method="subtraction")
    total_in = loaded["toc"].values.sum()
    total_out = out.sum()
    removed = total_in - total_out
    # roughly 10% of counts removed, never negative, never above input
    assert 0.05 * total_in < removed < 0.15 * total_in
    assert (out.data >= 0).all()
    dense_out = np.asarray(out.todense())
    assert (dense_out <= loaded["toc"].values + 1e-9).all()


def test_adjust_counts_round_to_int(loaded):
    sc = soup.SoupChannel(loaded["tod"].values, loaded["toc"].values,
                          genes=loaded["genes"], cells=loaded["cells"],
                          droplets=loaded["droplets"])
    soup.set_clusters(sc, dict(zip(loaded["clusters"]["cell"],
                                   loaded["clusters"]["cluster"])))
    soup.set_contamination_fraction(sc, 0.10)
    out = soup.adjust_counts(sc, round_to_int=True, seed=0)
    vals = out.data
    assert np.allclose(vals, np.round(vals))   # all integers


def test_adjust_counts_methods_agree(loaded):
    """subtraction and multinomial give near-identical totals."""
    sc = soup.SoupChannel(loaded["tod"].values, loaded["toc"].values,
                          genes=loaded["genes"], cells=loaded["cells"],
                          droplets=loaded["droplets"])
    soup.set_contamination_fraction(sc, 0.10)
    sub = soup.adjust_counts(sc, clusters=False, method="subtraction")
    multi = soup.adjust_counts(sc, clusters=False, method="multinomial",
                               seed=0)
    # total removed should match closely
    assert abs(sub.sum() - multi.sum()) / sub.sum() < 0.02


def test_soup_only_method_runs(loaded):
    sc = soup.SoupChannel(loaded["tod"].values, loaded["toc"].values,
                          genes=loaded["genes"], cells=loaded["cells"],
                          droplets=loaded["droplets"])
    soup.set_contamination_fraction(sc, 0.10)
    out = soup.adjust_counts(sc, clusters=False, method="soupOnly")
    assert out.shape == (sc.n_genes, sc.n_cells)
    assert out.sum() <= loaded["toc"].values.sum()


# ----------------------------------------------------------------------
# calculate_contamination_fraction
# ----------------------------------------------------------------------
def test_calculate_contamination_fraction(loaded):
    sc = soup.SoupChannel(loaded["tod"].values, loaded["toc"].values,
                          genes=loaded["genes"], cells=loaded["cells"],
                          droplets=loaded["droplets"])
    soup.set_clusters(sc, dict(zip(loaded["clusters"]["cell"],
                                   loaded["clusters"]["cluster"])))
    # use cluster c0's marker genes as a non-expressed set for other cells
    gene_set = {"c0markers": [f"g{i}" for i in range(40, 60)]}
    ute = soup.estimate_non_expressing_cells(sc, gene_set)
    if ute.values.sum() == 0:
        pytest.skip("no non-expressing cells in synthetic data")
    soup.calculate_contamination_fraction(sc, gene_set, ute, verbose=False)
    assert "rho" in sc.meta_data.columns
    assert 0.0 < sc.meta_data["rho"].iloc[0] < 0.5
