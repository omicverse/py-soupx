# Benchmark: pysoupx vs R SoupX 1.6.2

Five 10x Genomics human–mouse mixtures (hgmm 1k/5k/10k v3, 6k/12k v2; raw and filtered matrices from
https://www.10xgenomics.com/datasets). Counts of mouse genes in human cells, and of human genes in mouse
cells, can only come from ambient RNA, which gives a ground truth for the correction.

```
python  scripts/01_prepare.py <dir with extracted 10x archives> <tasks_dir>   # one clustering for both
python  scripts/run_all.py <tasks_dir> <runs_dir> <Rscript> <python> 10 1,2   # idle cores, 1 thread each
Rscript scripts/06_geneset.R <task_dir> <out.json> [nomt|all]  ;  python scripts/06_geneset.py <task_dir> <out.json> [nomt|all]
python  scripts/05_figures.py <runs_dir> <tasks_dir> figures <geneset_dir>
```

Both run `SoupChannel(tod, toc)`, `setClusters`, `autoEstCont` and `adjustCounts` with defaults.
R: SoupX 1.6.2 on R 4.5.3 (conda-forge). `figures/comparison_per_dataset.tsv` holds the per-dataset numbers.
