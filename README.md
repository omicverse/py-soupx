# pysoupx

A **pure-Python re-implementation of [SoupX](https://github.com/constantAmateur/SoupX)** (Young & Behjati, *GigaScience* 2020, 9(12):giaa151) for removing ambient ("soup") mRNA contamination from droplet-based single-cell RNA-seq data.

- AnnData-native — drop-in for the scanpy ecosystem (`load_10x`, `SoupChannel.from_anndata`, `to_anndata`)
- **No `rpy2`**, no R install — soup-profile estimation, the tf-idf marker search, the `autoEstCont` posterior, and the constrained `adjustCounts` subtraction are all implemented directly in NumPy/SciPy
- Same function surface as the R workflow (`estimateSoup` → `setClusters` → `autoEstCont` → `adjustCounts`)
- Same results as R SoupX 1.6.2: same soup profile, marker genes and contamination fraction, corrected counts within ~2e-11, and the same integers as R for `round_to_int=True` with the same seed (see [Relationship to R SoupX](#relationship-to-r-soupx))

> omicverse uses this package for `ov.pp.ambient.remove_ambient(adata, method='soupx', raw=raw)`.

## Install

```bash
pip install pysoupx
```

Dependencies: `numpy`, `scipy`, `pandas`, `anndata`, `statsmodels` (and `matplotlib` for the optional diagnostic plot).

## Quick-start

SoupX needs the **raw unfiltered** droplet matrix — the soup profile is estimated from the empty droplets — plus the filtered cell matrix.

```python
import pysoupx as soup

# --- from a 10x CellRanger output folder -------------------------
sc = soup.load_10x("path/to/cellranger/outs")   # raw + filtered

# --- or from AnnData objects -------------------------------------
# filtered = cells x genes ; raw = droplets x genes
sc = soup.SoupChannel.from_anndata(filtered, raw=raw, cluster_key="leiden")

# 1) soup profile is estimated automatically on construction
sc.soup_profile.head()

# 2) clusters + automatic contamination estimate
sc = soup.set_clusters(sc, cell_to_cluster)      # dict or sequence
sc = soup.auto_est_cont(sc)                      # sets meta_data['rho']

# 3) corrected count matrix (genes x cells, scipy sparse)
corrected = soup.adjust_counts(sc, round_to_int=True)

adata_corrected = soup.to_anndata(sc, corrected=corrected)
```

## Low-level functional API (mirrors R one-to-one)

```python
from pysoupx import (
    estimate_soup, set_soup_profile, set_clusters,
    set_contamination_fraction, quick_markers,
    estimate_non_expressing_cells, calculate_contamination_fraction,
    auto_est_cont, adjust_counts, alloc, expand_clusters,
)

# Manual contamination fraction instead of autoEstCont
sc = set_contamination_fraction(sc, 0.10)

# Estimate rho from a user-supplied non-expressed gene set
ute = estimate_non_expressing_cells(sc, gene_set)
calculate_contamination_fraction(sc, gene_set, ute)
```

## What's included

| Python | R counterpart | Purpose |
|---|---|---|
| `SoupChannel` / `SoupChannel.from_anndata` | `SoupChannel` | bundles droplets / counts / soup profile / metadata |
| `estimate_soup` | `estimateSoup` | per-gene soup fraction from empty droplets |
| `set_soup_profile` | `setSoupProfile` | set a soup profile manually |
| `set_clusters` | `setClusters` | attach a cell→cluster mapping |
| `set_contamination_fraction` | `setContaminationFraction` | set `rho` manually |
| `quick_markers` | `quickMarkers` | tf-idf cluster-marker genes |
| `estimate_non_expressing_cells` | `estimateNonExpressingCells` | which cells truly lack a gene set |
| `calculate_contamination_fraction` | `calculateContaminationFraction` | `rho` from non-expressed gene sets |
| `auto_est_cont` | `autoEstCont` | fully automatic `rho` estimate |
| `adjust_counts` | `adjustCounts` | soup-subtracted corrected matrix |
| `alloc` / `expand_clusters` | `alloc` / `expandClusters` | the constrained redistribution primitives |
| `load_10x` | `load10X` | read a 10x CellRanger folder |
| `to_anndata` / `make_soup_channel` | (AnnData helpers) | round-trip with the scanpy ecosystem |
| `plot_contamination_fraction` | `autoEstCont(doPlot=TRUE)` | diagnostic posterior plot |

`adjust_counts` supports all three SoupX methods: `subtraction` (default), `soupOnly` and `multinomial`.

## Relationship to R SoupX

pysoupx follows SoupX 1.6.2 step by step. With the same raw and filtered matrices and clusters:

| Step | Agreement with SoupX 1.6.2 |
|---|---|
| Soup profile (`estimateSoup`) | identical |
| `quickMarkers` (tf-idf, hypergeometric q-values as R's `phyper`) | same markers |
| `autoEstCont` | same marker genes, same contamination fraction |
| `estimateNonExpressingCells` + `calculateContaminationFraction` | same cells, same rho |
| `adjustCounts` (subtraction, soupOnly, multinomial) | corrected counts within ~2e-11 |
| `adjustCounts(roundToInt = TRUE)` | same integers for the same seed (R's random numbers, `pysoupx.rrng`) |

`adjust_counts(round_to_int=True, seed=s)` reproduces `set.seed(s); adjustCounts(sc, roundToInt = TRUE)`:
the rounding draws R's `rbinom(n, 1, p)` from R's Mersenne-Twister stream.

`tests/test_r_reference.py` checks R's random numbers and `phyper` against stored values (no R needed);
`tests/test_r_parity.py` runs SoupX through `Rscript` (set `PYSOUPX_RSCRIPT`) and compares every step.
The benchmark on 10x human–mouse mixtures is in `report/`.

## Use in omicverse

`ov.pp.ambient.remove_ambient(adata, method='soupx', raw=raw_adata)` runs pysoupx on a filtered, clustered
AnnData and its raw droplet matrix.

## Citation

If you use this package, please cite the original SoupX paper:

> Young, M.D. & Behjati, S. **SoupX removes ambient RNA contamination from droplet-based single-cell RNA sequencing data.** *GigaScience* **9**, giaa151 (2020).

and acknowledge omicverse / this repo for the Python port.

## License

Apache-2.0.
