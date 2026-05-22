"""Minimal end-to-end example — drop this into a Jupyter cell or run as a script.

Demonstrates the standalone pysoupx pipeline on the SoupX toyData (a real
10x CellRanger output that ships a raw + filtered matrix pair). SoupX needs
the raw unfiltered matrix because the soup profile is estimated from the
empty droplets.
"""
from __future__ import annotations

import os

import pandas as pd

import pysoupx as soup


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data", "toyData")

    # 1) Load the 10x folder (raw + filtered) into a SoupChannel.
    sc = soup.load_10x(data_dir, keep_droplets=True)
    print(f"SoupChannel: {sc.n_genes} genes x {sc.n_cells} cells")
    print(sc.soup_profile.sort_values("est", ascending=False).head())

    # 2) Attach clusters (from the bundled metaData) and auto-estimate rho.
    meta = pd.read_csv(os.path.join(data_dir, "metaData.tsv"),
                       sep="\t", index_col=0)
    soup.set_clusters(sc, dict(zip(meta.index, meta["res.1"].astype(str))))
    soup.auto_est_cont(sc, tfidf_min=0.5, verbose=False, force_accept=True)
    print(f"estimated contamination rho = {sc.meta_data['rho'].iloc[0]:.4f}")

    # 3) Remove the soup -> corrected (genes x cells) matrix.
    corrected = soup.adjust_counts(sc, method="subtraction", round_to_int=True)
    print(f"corrected matrix: {corrected.shape}, "
          f"counts {sc.toc.sum():.0f} -> {corrected.sum():.0f}")

    # Back to AnnData for the scanpy / omicverse ecosystem.
    adata = soup.to_anndata(sc, corrected=corrected)
    print(adata)


if __name__ == "__main__":
    main()
