# py-soupx

Pure-Python port of the R package
**[SoupX](https://github.com/constantAmateur/SoupX)** (Young & Behjati,
*"SoupX removes ambient RNA contamination from droplet-based single-cell
RNA sequencing data"*, **GigaScience** 2020, 9(12):giaa151).

`pysoupx` removes ambient ("soup") mRNA contamination from droplet-based
single-cell RNA-seq data. It is a standalone, dependency-light
re-implementation that does **not** require R or `rpy2`, and is designed
to be AnnData-friendly.

| | |
|---|---|
| PyPI / import name | `pysoupx` |
| Repository | `omicverse/py-soupx` |
| License | Apache-2.0 |
| Upstream | SoupX 1.6.x (GPL-2, R) |
| Numerical parity | deterministic parts **bit-exact** vs R SoupX |

## Install

```bash
pip install pysoupx              # once published
# or, from a checkout:
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pandas`, `anndata`, `statsmodels`
(and `matplotlib` for the optional diagnostic plot).

## How it works

SoupX corrects for ambient RNA in three steps:

1. **Estimate the soup profile** — from very-low-UMI "empty" droplets in
   the raw unfiltered matrix. The per-gene soup fraction is
   `sum(gene counts) / sum(total counts)` over those droplets.
2. **Estimate the contamination fraction `rho`** — `auto_est_cont` finds
   bimodally-expressed cluster-marker genes (sharply on in one cluster,
   off elsewhere) via tf-idf, assumes non-expressing clusters carry only
   soup, and takes the posterior mode as the global `rho`. `rho` can
   also be set manually or from user-supplied non-expressed gene sets.
3. **Remove the soup** (`adjust_counts`) — a constrained multinomial
   subtraction that yields an integer-able corrected matrix (not naive
   subtraction).

## Quick start

```python
import pysoupx as soup

# --- from a 10x CellRanger output folder -------------------------
sc = soup.load_10x("path/to/cellranger/outs")

# --- or from AnnData objects -------------------------------------
# filtered = cells x genes ; raw = droplets x genes
sc = soup.SoupChannel.from_anndata(filtered, raw=raw,
                                   cluster_key="leiden")

# --- or directly from count matrices -----------------------------
sc = soup.SoupChannel(tod, toc, genes=genes, cells=cells,
                      droplets=droplets)         # tod/toc: genes x cols

# 1. soup profile is estimated automatically on construction
sc.soup_profile.head()

# 2. clusters + automatic contamination estimate
sc = soup.set_clusters(sc, cell_to_cluster)      # dict or sequence
sc = soup.auto_est_cont(sc)                      # sets meta_data['rho']
#   ...or set it manually:
# sc = soup.set_contamination_fraction(sc, 0.10)

# 3. corrected count matrix (genes x cells, scipy sparse)
corrected = soup.adjust_counts(sc, round_to_int=True)

# back to AnnData
adata_corrected = soup.to_anndata(sc, corrected=corrected)
```

## API

| Function | SoupX equivalent |
|---|---|
| `SoupChannel` / `SoupChannel.from_anndata` | `SoupChannel` |
| `estimate_soup` | `estimateSoup` |
| `set_soup_profile` | `setSoupProfile` |
| `set_clusters` | `setClusters` |
| `set_contamination_fraction` | `setContaminationFraction` |
| `quick_markers` | `quickMarkers` |
| `estimate_non_expressing_cells` | `estimateNonExpressingCells` |
| `calculate_contamination_fraction` | `calculateContaminationFraction` |
| `auto_est_cont` | `autoEstCont` |
| `adjust_counts` | `adjustCounts` |
| `alloc` / `expand_clusters` | `alloc` / `expandClusters` |
| `load_10x` | `load10X` |
| `to_anndata` / `make_soup_channel` | (AnnData helpers) |
| `plot_contamination_fraction` | `autoEstCont(doPlot=TRUE)` |

`adjust_counts` supports all three SoupX methods: `subtraction`
(default), `soupOnly` and `multinomial`.

## R-parity testing

`tests/` runs the **same** synthetic raw + filtered count matrices
(empty droplets, four cell types, a known 10% contamination) through both
R `SoupX` and `pysoupx` and asserts they agree:

| Quantity | Result |
|---|---|
| Soup profile (`estimateSoup`) | **bit-exact** (max abs diff ~1e-16) |
| `quickMarkers` tf-idf / qvals / idf | **bit-exact** (max abs diff ~1e-16) |
| `adjustCounts` cluster-level, fixed rho | **bit-exact** (max abs diff 0) |
| `adjustCounts` cell-level, fixed rho | **bit-exact** (max abs diff ~1e-13) |
| `auto_est_cont` rho | **exact** (identical posterior-mode) |

The contamination estimate is the deterministic mode of a gamma-prior
posterior on a fixed 0.001 grid, so it is reproducible to within one
grid step. The only intrinsically stochastic steps are the optional
integer rounding (`round_to_int`) and the `multinomial` method's
tie-breaking, both seedable via the `seed` argument.

Run the suite (the R driver auto-runs if the CMAP R env is present, and
skips gracefully otherwise):

```bash
pytest tests/ -q
```

## Relationship to upstream

This is part of [omicverse](https://github.com/Starlitnightly/omicverse)'s
`py-X` port program. The algorithm is a faithful re-implementation of
SoupX; see the upstream paper and repository for the underlying method.

## Citation

If you use `pysoupx`, please cite the original SoupX paper:

> Young, M.D. & Behjati, S. SoupX removes ambient RNA contamination from
> droplet-based single-cell RNA sequencing data. *GigaScience* **9**,
> giaa151 (2020).
