#!/usr/bin/env Rscript
# Run the original R SoupX end-to-end on a 10x CellRanger output folder
# (the SoupX toyData ships a real raw + filtered matrix pair).  Outputs
# everything the comparison notebook needs to overlay on the Python-side
# pysoupx results: the soup profile, the autoEstCont rho, and the
# corrected count matrix.
#
# Usage:
#   Rscript r_driver_soupx.R <toy10x_dir> <outdir>
#
# <toy10x_dir> must be a CellRanger-style folder with raw_gene_bc_matrices/
# and filtered_gene_bc_matrices/ plus a metaData.tsv (the SoupX toyData).

suppressPackageStartupMessages({
  library(SoupX)
  library(Matrix)
})

args     <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[[1]] else "data/toyData"
outdir   <- if (length(args) >= 2) args[[2]] else "compare_out/r_out"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("[R] loading 10x data from %s\n", data_dir))
sc <- load10X(data_dir)

# --- soup profile (estimateSoup) -------------------------------------
sp_df <- data.frame(gene   = rownames(sc$soupProfile),
                    est    = sc$soupProfile$est,
                    counts = sc$soupProfile$counts)
write.table(sp_df, file.path(outdir, "soup_profile.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- clusters from the bundled metaData ------------------------------
meta <- read.table(file.path(data_dir, "metaData.tsv"),
                   header = TRUE, sep = "\t", row.names = 1,
                   check.names = FALSE)
clusters <- setNames(as.character(meta[colnames(sc$toc), "res.1"]),
                     colnames(sc$toc))
sc <- setClusters(sc, clusters)

# --- automatic contamination estimate (autoEstCont) ------------------
rho_auto <- NA
ok <- tryCatch({
  sc_auto  <- autoEstCont(sc, tfidfMin = 0.5, doPlot = FALSE,
                          forceAccept = TRUE, verbose = FALSE)
  rho_auto <- sc_auto$metaData$rho[1]
  TRUE
}, error = function(e) {
  message("autoEstCont failed: ", conditionMessage(e)); FALSE
})
write.table(data.frame(rho_auto = rho_auto),
            file.path(outdir, "rho_auto.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- adjustCounts with rho fixed at 0.10 -----------------------------
sc_fix <- setContaminationFraction(sc, 0.10)
adj    <- as.matrix(adjustCounts(sc_fix, method = "subtraction",
                                 roundToInt = FALSE, verbose = 0))
write.table(data.frame(gene = rownames(adj), adj, check.names = FALSE),
            file.path(outdir, "adjusted_sub.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- adjustCounts using the autoEstCont rho --------------------------
if (ok) {
  adj_a <- as.matrix(adjustCounts(sc_auto, method = "subtraction",
                                  roundToInt = FALSE, verbose = 0))
  write.table(data.frame(gene = rownames(adj_a), adj_a, check.names = FALSE),
              file.path(outdir, "adjusted_auto.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

cat(sprintf("[R] wrote outputs to %s (n_genes=%d, n_cells=%d, rho_auto=%.4f)\n",
            outdir, nrow(sc$toc), ncol(sc$toc), rho_auto))
