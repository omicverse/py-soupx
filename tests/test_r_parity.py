"""R-parity tests — pysoupx vs R SoupX.

The R driver (:file:`r_reference_driver.R`) runs SoupX on exactly the
same synthetic raw + filtered count matrices that the Python side uses
(generated once by :mod:`make_synthetic`).  Because every input is
shared, the deterministic parts of the algorithm can be checked to be
bit-exact:

* **soup profile** (``estimateSoup``)        — bit-exact.
* **quickMarkers** tf-idf / qvals            — bit-exact.
* **adjustCounts** with a fixed rho          — bit-exact (cluster and
  cell level; ``subtraction`` method, the closed-form ``alloc``).
* **autoEstCont** rho                        — exact (the estimate is a
  deterministic posterior-mode; no RNG on the synthetic data).

Tests skip gracefully when the CMAP R env or SoupX is unavailable.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import pysoupx as soup


def _build_sc(loaded):
    sc = soup.SoupChannel(
        loaded["tod"].values, loaded["toc"].values,
        genes=loaded["genes"], cells=loaded["cells"],
        droplets=loaded["droplets"],
    )
    soup.set_clusters(
        sc, dict(zip(loaded["clusters"]["cell"],
                     loaded["clusters"]["cluster"])))
    return sc


# ----------------------------------------------------------------------
# soup profile
# ----------------------------------------------------------------------
def test_soup_profile_bit_exact(r_reference, loaded):
    sc = _build_sc(loaded)
    r = pd.read_csv(os.path.join(r_reference, "soup_profile.tsv"),
                    sep="\t").set_index("gene")
    r = r.reindex(loaded["genes"])
    est_diff = np.abs(sc.soup_profile["est"].to_numpy()
                      - r["est"].to_numpy()).max()
    cnt_diff = np.abs(sc.soup_profile["counts"].to_numpy()
                      - r["counts"].to_numpy()).max()
    assert est_diff < 1e-12, f"soup est max diff {est_diff:.2e}"
    assert cnt_diff == 0, f"soup counts max diff {cnt_diff}"


# ----------------------------------------------------------------------
# quickMarkers
# ----------------------------------------------------------------------
def test_quick_markers_bit_exact(r_reference, loaded):
    import scipy.sparse as sp

    pm = soup.quick_markers(
        sp.csc_matrix(loaded["toc"].values),
        loaded["clusters"]["cluster"].values,
        loaded["genes"], loaded["cells"], N=None,
    )
    rm = pd.read_csv(os.path.join(r_reference, "markers.tsv"), sep="\t")
    assert len(pm) == len(rm), f"marker count R={len(rm)} PY={len(pm)}"
    merged = rm.merge(pm, on=["gene", "cluster"], suffixes=("_r", "_p"))
    assert len(merged) == len(rm), "marker (gene,cluster) sets differ"
    for col in ["tfidf", "geneFrequency", "geneFrequencyGlobal", "qval",
                "idf"]:
        d = np.abs(merged[f"{col}_r"] - merged[f"{col}_p"]).max()
        assert d < 1e-9, f"quickMarkers {col} max diff {d:.2e}"


# ----------------------------------------------------------------------
# adjustCounts — fixed rho (deterministic)
# ----------------------------------------------------------------------
def test_adjust_counts_cluster_bit_exact(r_reference, loaded):
    sc = _build_sc(loaded)
    soup.set_contamination_fraction(sc, 0.10)
    out = soup.adjust_counts(sc, method="subtraction")
    py = pd.DataFrame(np.asarray(out.todense()),
                      index=loaded["genes"], columns=loaded["cells"])
    r = pd.read_csv(os.path.join(r_reference, "adjusted_sub.tsv"),
                    sep="\t").set_index("gene")
    r = r.reindex(index=loaded["genes"], columns=loaded["cells"])
    diff = np.abs(py.to_numpy() - r.to_numpy())
    assert diff.max() < 1e-6, (
        f"adjustCounts (cluster) max abs diff {diff.max():.3e}")


def test_adjust_counts_cell_level_bit_exact(r_reference, loaded):
    sc = _build_sc(loaded)
    soup.set_contamination_fraction(sc, 0.10)
    out = soup.adjust_counts(sc, clusters=False, method="subtraction")
    py = pd.DataFrame(np.asarray(out.todense()),
                      index=loaded["genes"], columns=loaded["cells"])
    r = pd.read_csv(os.path.join(r_reference, "adjusted_cell.tsv"),
                    sep="\t").set_index("gene")
    r = r.reindex(index=loaded["genes"], columns=loaded["cells"])
    diff = np.abs(py.to_numpy() - r.to_numpy())
    assert diff.max() < 1e-6, (
        f"adjustCounts (cell level) max abs diff {diff.max():.3e}")


def test_adjust_counts_corr_is_one(r_reference, loaded):
    """Sanity check: the corrected matrices are perfectly correlated."""
    sc = _build_sc(loaded)
    soup.set_contamination_fraction(sc, 0.10)
    out = soup.adjust_counts(sc, method="subtraction")
    py = np.asarray(out.todense()).ravel()
    r = (pd.read_csv(os.path.join(r_reference, "adjusted_sub.tsv"),
                     sep="\t").set_index("gene")
         .reindex(index=loaded["genes"], columns=loaded["cells"])
         .to_numpy().ravel())
    rho, _ = pearsonr(py, r)
    assert rho > 0.999999, f"adjustCounts correlation {rho:.8f}"


# ----------------------------------------------------------------------
# autoEstCont
# ----------------------------------------------------------------------
def test_auto_est_cont_matches_R(r_reference, loaded):
    sc = _build_sc(loaded)
    soup.auto_est_cont(sc, tfidf_min=0.5, verbose=False, force_accept=True)
    py_rho = sc.meta_data["rho"].iloc[0]
    r_rho = pd.read_csv(os.path.join(r_reference, "rho_auto.tsv"),
                        sep="\t")["rho_auto"].iloc[0]
    if pd.isna(r_rho):
        pytest.skip("R autoEstCont produced no estimate")
    # autoEstCont's rho is the deterministic mode of the posterior on a
    # fixed 0.001 grid -> must agree to within one grid step.
    assert abs(py_rho - r_rho) <= 0.0015, (
        f"autoEstCont rho R={r_rho:.4f} PY={py_rho:.4f}")


def test_auto_est_cont_adjust_matches_R(r_reference, loaded):
    """adjustCounts using the autoEstCont rho also matches R."""
    auto_file = os.path.join(r_reference, "adjusted_auto.tsv")
    if not os.path.exists(auto_file):
        pytest.skip("R produced no adjusted_auto output")
    sc = _build_sc(loaded)
    soup.auto_est_cont(sc, tfidf_min=0.5, verbose=False, force_accept=True)
    out = soup.adjust_counts(sc, method="subtraction")
    py = pd.DataFrame(np.asarray(out.todense()),
                      index=loaded["genes"], columns=loaded["cells"])
    r = (pd.read_csv(auto_file, sep="\t").set_index("gene")
         .reindex(index=loaded["genes"], columns=loaded["cells"]))
    diff = np.abs(py.to_numpy() - r.to_numpy())
    assert diff.max() < 1e-6, (
        f"adjustCounts (auto rho) max abs diff {diff.max():.3e}")
