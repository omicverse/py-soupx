"""Head-to-head speed benchmark: R SoupX vs pysoupx.

Runs both on the SoupX toyData 10x folder and reports wall time for the
two deterministic kernels that the parity tests cover:

  * estimateSoup + autoEstCont  — soup profile + contamination estimate
  * adjustCounts (subtraction)  — the constrained soup-subtraction

Both ports start from identical input (the same raw + filtered matrices),
so the comparison reflects the algorithm rather than data wrangling.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pysoupx as soup


HERE = Path(__file__).parent
DATA = HERE / "data" / "toyData"
WORK = HERE / "compare_out"
RSCRIPT = "/scratch/users/steorra/env/CMAP/bin/Rscript"
GCC_BIN = "/share/software/user/open/gcc/14.2.0/bin"


R_BENCH_SCRIPT = """
suppressPackageStartupMessages({library(SoupX); library(Matrix)})
args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[[1]]
sc <- load10X(data_dir)
meta <- read.table(file.path(data_dir, "metaData.tsv"), header = TRUE,
                   sep = "\\t", row.names = 1, check.names = FALSE)
clusters <- setNames(as.character(meta[colnames(sc$toc), "res.1"]),
                     colnames(sc$toc))
sc <- setClusters(sc, clusters)
t0 <- proc.time()[[3]]
sc <- autoEstCont(sc, tfidfMin = 0.5, doPlot = FALSE,
                  forceAccept = TRUE, verbose = FALSE)
t_est <- proc.time()[[3]] - t0
t1 <- proc.time()[[3]]
invisible(adjustCounts(sc, method = "subtraction", roundToInt = FALSE,
                       verbose = 0))
t_adj <- proc.time()[[3]] - t1
cat(sprintf("R_EST=%.4f\\nR_ADJ=%.4f\\n", t_est, t_adj))
"""


def time_r(runs: int = 3) -> tuple[float, float]:
    script = WORK / "_bench_r.R"
    script.parent.mkdir(exist_ok=True)
    script.write_text(R_BENCH_SCRIPT)
    env = os.environ.copy()
    if os.path.isdir(GCC_BIN):
        env["PATH"] = GCC_BIN + os.pathsep + env.get("PATH", "")
    est, adj = [], []
    for _ in range(runs):
        proc = subprocess.run([RSCRIPT, str(script), str(DATA)],
                              env=env, capture_output=True, text=True,
                              check=True)
        for line in proc.stdout.splitlines():
            if line.startswith("R_EST="):
                est.append(float(line.split("=")[1]))
            elif line.startswith("R_ADJ="):
                adj.append(float(line.split("=")[1]))
    return float(np.mean(est)), float(np.mean(adj))


def time_python(runs: int = 3) -> tuple[float, float]:
    meta = pd.read_csv(DATA / "metaData.tsv", sep="\t", index_col=0)
    cl = dict(zip(meta.index, meta["res.1"].astype(str)))
    est, adj = [], []
    for _ in range(runs):
        sc = soup.load_10x(str(DATA), keep_droplets=True)
        soup.set_clusters(sc, cl)
        t0 = time.perf_counter()
        soup.auto_est_cont(sc, tfidf_min=0.5, verbose=False,
                           force_accept=True)
        est.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        soup.adjust_counts(sc, method="subtraction")
        adj.append(time.perf_counter() - t1)
    return float(np.mean(est)), float(np.mean(adj))


def main() -> None:
    print("Dataset: SoupX toyData — 226 genes x 62 cells")

    print("\n[R] timing (3 runs)…")
    r_est, r_adj = time_r(runs=3)
    print(f"  estimateSoup+autoEstCont:  {r_est*1000:8.1f} ms")
    print(f"  adjustCounts:              {r_adj*1000:8.1f} ms")

    print("\n[py] timing (3 runs)…")
    p_est, p_adj = time_python(runs=3)
    print(f"  estimateSoup+autoEstCont:  {p_est*1000:8.1f} ms")
    print(f"  adjustCounts:              {p_adj*1000:8.1f} ms")

    print("\nSpeed-ups (R time / Python time):")
    print(f"  estimate:     {r_est/p_est:5.2f}x")
    print(f"  adjustCounts: {r_adj/p_adj:5.2f}x")
    print(f"  total:        {(r_est+r_adj)/(p_est+p_adj):5.2f}x")


if __name__ == "__main__":
    main()
