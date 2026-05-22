# pysoupx

A **pure-Python re-implementation of [SoupX](https://github.com/constantAmateur/SoupX)** (Young & Behjati, *GigaScience* 2020, 9(12):giaa151) for removing ambient ("soup") mRNA contamination from droplet-based single-cell RNA-seq data.

- AnnData-native — drop-in for the scanpy ecosystem (`load_10x`, `SoupChannel.from_anndata`, `to_anndata`)
- **No `rpy2`**, no R install — soup-profile estimation, the tf-idf marker search, the `autoEstCont` posterior, and the constrained `adjustCounts` subtraction are all implemented directly in NumPy/SciPy
- Same function surface as the R workflow (`estimateSoup` → `setClusters` → `autoEstCont` → `adjustCounts`)
- Bit-for-bit reproducibility against the R reference on the deterministic kernels (see `tests/test_r_parity.py`)

> This is a **standalone mirror** of the canonical implementation that lives in [`omicverse`](https://github.com/Starlitnightly/omicverse). All algorithmic work is developed upstream in omicverse and synced here for users who want SoupX without the full omicverse stack.

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

## Reproducing R results exactly

SoupX's core kernels are deterministic, so feeding both ports identical raw + filtered matrices yields bit-for-bit agreement:

| Quantity | Result |
|---|---|
| Soup profile (`estimateSoup`) | **bit-exact** (max abs diff ~1e-16) |
| `quickMarkers` tf-idf / qvals / idf | **bit-exact** (max abs diff ~1e-16) |
| `adjustCounts` cluster-level, fixed rho | **bit-exact** (max abs diff 0) |
| `adjustCounts` cell-level, fixed rho | **bit-exact** (max abs diff ~1e-13) |
| `autoEstCont` rho | **exact** (identical posterior mode) |

`tests/test_r_parity.py` runs the R reference (`r_reference_driver.R`) inside the `CMAP` R env on the same synthetic raw + filtered matrices the Python side uses, and checks the soup profile, marker table, `rho` and corrected matrices match. `examples/compare_R_vs_Python.ipynb` does the same on the real SoupX **toyData** 10x dataset and visualises the agreement with `omicverse`.

The only intrinsically stochastic steps are the optional integer rounding (`round_to_int`) and the `multinomial` method's tie-breaking, both seedable via the `seed` argument.

## Relationship to omicverse

Developed **upstream** in [`omicverse`](https://github.com/Starlitnightly/omicverse):

- Canonical implementation lives in omicverse
- Standalone mirror (this repo): same code, same API, minus the omicverse packaging

## Citation

If you use this package, please cite the original SoupX paper:

> Young, M.D. & Behjati, S. **SoupX removes ambient RNA contamination from droplet-based single-cell RNA sequencing data.** *GigaScience* **9**, giaa151 (2020).

and acknowledge omicverse / this repo for the Python port.

## License

Apache-2.0.
