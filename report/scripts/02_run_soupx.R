# Reference run: SoupX 1.6.2 on one task, the workflow of the SoupX vignette with fixed clusters.
# Usage: Rscript 02_run_soupx.R <task_dir> <out_dir>
suppressMessages({library(Matrix); library(SoupX); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
td <- args[1]; od <- args[2]
dir.create(od, recursive = TRUE, showWarnings = FALSE)
tod <- as(readMM(gzfile(file.path(td, "tod.mtx.gz"))), "CsparseMatrix")
genes <- read.delim(file.path(td, "genes.tsv"))
rownames(tod) <- genes$id; colnames(tod) <- readLines(file.path(td, "droplets.tsv"))
cl <- read.delim(file.path(td, "clusters.tsv"), colClasses = "character")
toc <- tod[, cl$cell]
t0 <- proc.time()[["elapsed"]]
sc <- SoupChannel(tod, toc)
sc <- setClusters(sc, setNames(cl$cluster, cl$cell))
sc <- autoEstCont(sc, doPlot = FALSE, verbose = FALSE)
out <- adjustCounts(sc, verbose = 0)
elapsed <- proc.time()[["elapsed"]] - t0
out <- as(out, "CsparseMatrix")
writeBin(out@p, file.path(od, "adj_p.bin"), size = 4); writeBin(out@i, file.path(od, "adj_i.bin"), size = 4)
writeBin(out@x, file.path(od, "adj_x.bin"), size = 8)
writeBin(sc$soupProfile$est, file.path(od, "soup_est.bin"), size = 8)
write_json(list(engine = "SoupX", version = as.character(packageVersion("SoupX")), R = R.version.string,
                rho = sc$fit$rhoEst, rho_fwhm = sc$fit$rhoFWHM, n_markers = length(sc$fit$markersUsed$gene),
                markers = unique(as.character(sc$fit$markersUsed$gene)), seconds = elapsed,
                dim = dim(out)), file.path(od, "result.json"), auto_unbox = TRUE, digits = NA)
cat(basename(td), sprintf("rho=%.4f %.1fs", sc$fit$rhoEst, elapsed), "\n")
