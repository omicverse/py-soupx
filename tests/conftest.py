"""Shared fixtures and the R-reference build hook for pysoupx tests.

The synthetic dataset (``tests/make_synthetic.py``) is generated once and
fed *identically* to R ``SoupX`` (via ``r_reference_driver.R``) and to
``pysoupx`` so the two ports can be compared.  Tests that need the R
reference skip gracefully when the CMAP R environment or the ``SoupX``
package is unavailable.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
R_DRIVER = os.path.join(HERE, "r_reference_driver.R")
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # so `make_synthetic` is importable

# CMAP path-based R environment (override with PYSOUPX_RSCRIPT).
_RSCRIPT = os.environ.get(
    "PYSOUPX_RSCRIPT", "/scratch/users/steorra/env/CMAP/bin/Rscript")

_R_OUT_FILES = ["soup_profile.tsv", "markers.tsv", "rho_auto.tsv",
                "adjusted_sub.tsv", "adjusted_cell.tsv"]


def _rscript() -> str | None:
    if shutil.which(_RSCRIPT) or os.path.exists(_RSCRIPT):
        return _RSCRIPT
    if shutil.which("Rscript"):
        return "Rscript"
    return None


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """Generate the shared synthetic dataset; return its directory."""
    from make_synthetic import generate

    d = tmp_path_factory.mktemp("soupx_data")
    generate(str(d))
    return str(d)


@pytest.fixture(scope="session")
def r_reference(synthetic, tmp_path_factory):
    """Run R SoupX on the synthetic data; return the output directory.

    Skips all dependent tests if R / SoupX are not available.
    """
    rscript = _rscript()
    if rscript is None:
        pytest.skip("Rscript not found")
    out = tmp_path_factory.mktemp("soupx_R")
    env = dict(os.environ)
    gcc = "/share/software/user/open/gcc/14.2.0/bin"
    if os.path.isdir(gcc):
        env["PATH"] = gcc + os.pathsep + env.get("PATH", "")
    try:
        res = subprocess.run(
            [rscript, R_DRIVER, synthetic, str(out)],
            capture_output=True, text=True, timeout=900, env=env,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"R reference driver could not run: {e}")
    if res.returncode != 0 or not all(
        os.path.exists(os.path.join(out, f)) for f in _R_OUT_FILES
    ):
        pytest.skip(f"R SoupX reference unavailable:\n{res.stderr[-1500:]}")
    return str(out)


@pytest.fixture(scope="session")
def loaded(synthetic):
    """Load the synthetic matrices as plain DataFrames."""
    tod = pd.read_csv(os.path.join(synthetic, "tod.tsv"),
                      sep="\t", index_col=0)
    toc = pd.read_csv(os.path.join(synthetic, "toc.tsv"),
                      sep="\t", index_col=0)
    clu = pd.read_csv(os.path.join(synthetic, "clusters.tsv"), sep="\t")
    return {"tod": tod, "toc": toc, "clusters": clu,
            "genes": list(toc.index), "cells": list(toc.columns),
            "droplets": list(tod.columns)}
