#!/usr/bin/env Rscript
# Run R SoupX on the shared synthetic dataset so the Python port
# (pysoupx) can be checked against it.
#
# Usage:
#   Rscript r_reference_driver.R <data_dir> <out_dir>
#
# <data_dir> must contain tod.tsv / toc.tsv / clusters.tsv produced by
# tests/make_synthetic.py.  Outputs (TSV files in <out_dir>):
#   soup_profile.tsv   estimateSoup() per-gene soup fraction + counts
#   markers.tsv        quickMarkers() tf-idf cluster markers
#   rho_auto.tsv       autoEstCont() estimated global rho
#   adjusted_sub.tsv   adjustCounts() with rho fixed at 0.10 (subtraction)
#   adjusted_auto.tsv  adjustCounts() with the autoEstCont rho

suppressPackageStartupMessages({
  library(SoupX)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[[1]] else "synthetic"
out_dir  <- if (length(args) >= 2) args[[2]] else "R_out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

tod <- as.matrix(read.table(file.path(data_dir, "tod.tsv"),
                            header = TRUE, sep = "\t", row.names = 1,
                            check.names = FALSE))
toc <- as.matrix(read.table(file.path(data_dir, "toc.tsv"),
                            header = TRUE, sep = "\t", row.names = 1,
                            check.names = FALSE))
clu <- read.table(file.path(data_dir, "clusters.tsv"),
                  header = TRUE, sep = "\t", stringsAsFactors = FALSE)

tod <- Matrix(tod, sparse = TRUE)
toc <- Matrix(toc, sparse = TRUE)

# --- SoupChannel + soup profile -------------------------------------
sc <- SoupChannel(tod, toc, calcSoupProfile = FALSE)
sc <- estimateSoup(sc, soupRange = c(0, 100))
sp_df <- data.frame(gene = rownames(sc$soupProfile),
                    est = sc$soupProfile$est,
                    counts = sc$soupProfile$counts)
write.table(sp_df, file.path(out_dir, "soup_profile.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- clusters + quickMarkers ----------------------------------------
clusters <- setNames(clu$cluster, clu$cell)
sc <- setClusters(sc, clusters)
mrks <- quickMarkers(sc$toc, sc$metaData$clusters, N = Inf)
write.table(mrks, file.path(out_dir, "markers.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- autoEstCont ----------------------------------------------------
rho_auto <- NA
ok <- tryCatch({
  sc_auto <- autoEstCont(sc, tfidfMin = 0.5, doPlot = FALSE,
                         forceAccept = TRUE, verbose = FALSE)
  rho_auto <- sc_auto$metaData$rho[1]
  TRUE
}, error = function(e) { message("autoEstCont failed: ", conditionMessage(e)); FALSE })
write.table(data.frame(rho_auto = rho_auto),
            file.path(out_dir, "rho_auto.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- adjustCounts with rho fixed at 0.10 (subtraction) --------------
sc_fix <- setContaminationFraction(sc, 0.10)
adj <- adjustCounts(sc_fix, method = "subtraction", roundToInt = FALSE,
                    verbose = 0)
adj <- as.matrix(adj)
write.table(data.frame(gene = rownames(adj), adj, check.names = FALSE),
            file.path(out_dir, "adjusted_sub.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- adjustCounts at cell level (clusters = FALSE) ------------------
adj_cell <- adjustCounts(sc_fix, clusters = FALSE, method = "subtraction",
                         roundToInt = FALSE, verbose = 0)
adj_cell <- as.matrix(adj_cell)
write.table(data.frame(gene = rownames(adj_cell), adj_cell,
                       check.names = FALSE),
            file.path(out_dir, "adjusted_cell.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- adjustCounts with the autoEstCont rho --------------------------
if (ok) {
  adj_a <- adjustCounts(sc_auto, method = "subtraction", roundToInt = FALSE,
                        verbose = 0)
  adj_a <- as.matrix(adj_a)
  write.table(data.frame(gene = rownames(adj_a), adj_a, check.names = FALSE),
              file.path(out_dir, "adjusted_auto.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

cat("R SoupX reference written to", out_dir, "\n")
