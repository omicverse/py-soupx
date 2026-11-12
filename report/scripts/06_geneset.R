# Gene-set estimate of the contamination fraction with SoupX 1.6.2: the five genes of each species most abundant
# in the soup are taken as genes that cells of the other species do not express.
# Usage: Rscript 06_geneset.R <task_dir> <out_json> [nomt|all]
suppressMessages({library(Matrix); library(SoupX); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE); td <- args[1]
tod <- as(readMM(gzfile(file.path(td, "tod.mtx.gz"))), "CsparseMatrix"); genes <- read.delim(file.path(td, "genes.tsv"))
rownames(tod) <- genes$id; colnames(tod) <- readLines(file.path(td, "droplets.tsv"))
cl <- read.delim(file.path(td, "clusters.tsv"), colClasses = "character")
sc <- SoupChannel(tod, tod[, cl$cell]); sc <- setClusters(sc, setNames(cl$cluster, cl$cell))
est <- sc$soupProfile$est; names(est) <- rownames(sc$soupProfile)
# mitochondrial genes and MALAT1 are enriched in empty droplets relative to the ambient RNA in cells
mode <- if (length(args) > 2) args[3] else "nomt"
skip <- grepl("(^|_)(mt-|malat1$)", genes$symbol, ignore.case = TRUE)
top <- function(g) names(sort(est[genes$id[genes$genome == g & (if (mode == "nomt") !skip else TRUE)]], decreasing = TRUE))[1:5]
gl <- list(mouse = top("mouse"), human = top("human"))
use <- estimateNonExpressingCells(sc, nonExpressedGeneList = gl)
sc <- calculateContaminationFraction(sc, gl, useToEst = use, forceAccept = TRUE, verbose = FALSE)
write_json(list(mode = mode, rho = unique(sc$metaData$rho), n_use = colSums(use), genes = gl), args[2], auto_unbox = TRUE, digits = NA)
cat(basename(td), sprintf("gene-set rho %.4f", unique(sc$metaData$rho)), "\n")
