"""Figure, table and the numbers quoted in the pysoupx technical report.

Usage: python 05_figures.py <runs_dir> <tasks_dir> <out_dir> [geneset_dir]
Compares SoupX 1.6.2 and pysoupx on five 10x Genomics human-mouse mixtures. In a mixture, counts of mouse
genes in human cells (and of human genes in mouse cells) can only come from ambient RNA, which gives a
ground truth for the contamination fraction and for the counts the correction should remove.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
import scipy.io
import scipy.sparse as sp
from scipy.stats import spearmanr

RUNS, TASKS, OUT = map(pathlib.Path, sys.argv[1:4])
GENESET = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else RUNS.parent / "geneset"
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
C_R, C_PY, C_TRUE = "#eb6834", "#2a78d6", "#52514e"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})
ENG = {"R": "SoupX", "py": "pysoupx"}
LABEL = {"hgmm_1k_v3": "hgmm 1k (v3)", "hgmm_5k_v3": "hgmm 5k (v3)", "hgmm_6k": "hgmm 6k (v2)",
         "hgmm_10k_v3": "hgmm 10k (v3)", "hgmm_12k": "hgmm 12k (v2)"}
EXAMPLE = "hgmm_12k"


def result(engine, task, run=1):
    d = RUNS / ENG[engine] / task / f"run{run}"
    r = json.load(open(d / "result.json"))
    adj = sp.csc_matrix((np.fromfile(d / "adj_x.bin"), np.fromfile(d / "adj_i.bin", np.int32),
                         np.fromfile(d / "adj_p.bin", np.int32)), shape=tuple(r["dim"]))
    return r, adj, np.fromfile(d / "soup_est.bin")


rows, cellwise, umap_frac = [], {}, {}
for t in sorted(LABEL, key=list(LABEL).index):
    td = TASKS / t
    genes = pd.read_csv(td / "genes.tsv", sep="\t")
    species = pd.read_csv(td / "species.tsv", sep="\t")
    droplets = (td / "droplets.tsv").read_text().split()
    tod = sp.csc_matrix(scipy.io.mmread(gzip.open(td / "tod.mtx.gz")))
    toc = tod[:, pd.Index(droplets).get_indexer(species.cell)].tocsc()
    mouse_gene = (genes.genome == "mouse").to_numpy()
    res = {e: result(e, t) for e in ENG}
    soup_mouse = res["R"][2][mouse_gene].sum()
    hum, mou = (species.species == "human").to_numpy(), (species.species == "mouse").to_numpy()

    def split(m):
        """Per cell: counts of the other species' genes and of the own species' genes."""
        m_cnt = np.asarray(m[mouse_gene].sum(axis=0)).ravel()
        h_cnt = np.asarray(m[~mouse_gene].sum(axis=0)).ravel()
        other = np.where(hum, m_cnt, np.where(mou, h_cnt, np.nan))
        own = np.where(hum, h_cnt, np.where(mou, m_cnt, np.nan))
        return other, own

    o0, s0 = split(toc)
    n = o0 + s0
    pure = hum | mou
    expected = np.where(hum, soup_mouse, 1 - soup_mouse)
    row = dict(dataset=t, label=LABEL[t], droplets=tod.shape[1], cells=toc.shape[1], human=int(hum.sum()),
               mouse=int(mou.sum()), true_rho=o0[pure].sum() / (n[pure] * expected[pure]).sum())
    # per species and per cell: contamination differs between cells
    row["true_human"] = o0[hum].sum() / (n[hum].sum() * soup_mouse)
    row["true_mouse"] = o0[mou].sum() / (n[mou].sum() * (1 - soup_mouse))
    cell_true = o0[pure] / (n[pure] * expected[pure])
    row["rho_cell_q1"], row["rho_cell_q3"] = np.percentile(cell_true, [25, 75])
    row["spearman_rho_numi"] = spearmanr(cell_true, n[pure])[0]
    # gene-set estimates (06_geneset): five most abundant soup genes, with and without MT genes and MALAT1
    for mode in ("all", "nomt"):
        gs = {e: json.load(open(GENESET / ENG[e] / f"{t}_{mode}.json")) for e in ENG}
        row[f"gs_{mode}_R"], row[f"gs_{mode}_py"] = gs["R"]["rho"], gs["py"]["rho"]
        if mode == "all":                                         # how over-represented these genes are in the soup
            ids = pd.Index(genes.id)
            ratios = []
            for sp_, cells_, gl in [("mouse", hum, gs["R"]["genes"]["mouse"]), ("human", mou, gs["R"]["genes"]["human"])]:
                own = (genes.genome == sp_).to_numpy()
                amb = np.asarray(toc[:, cells_].sum(axis=1)).ravel() * own
                soup_sp = res["R"][2] * own
                gi = ids.get_indexer(gl)
                ratios.append((soup_sp[gi].sum() / soup_sp.sum()) / (amb[gi].sum() / amb.sum()))
            row["top5_soup_overrep"] = float(np.mean(ratios))
    for e in ENG:
        r, adj, soup = res[e]
        o1, s1 = split(adj)
        row[f"rho_{e}"] = r["rho"]
        row[f"removed_{e}"] = 1 - o1[pure].sum() / o0[pure].sum()
        row[f"kept_{e}"] = s1[pure].sum() / s0[pure].sum()
        row[f"removed_human_{e}"] = 1 - o1[hum].sum() / o0[hum].sum()
        row[f"removed_mouse_{e}"] = 1 - o1[mou].sum() / o0[mou].sum()
        row[f"sec_{e}"] = min(json.load(open(RUNS / ENG[e] / t / f"run{k}" / "result.json"))["seconds"]
                              for k in (1, 2))                    # faster of the two runs
        if t == EXAMPLE:
            cellwise[e] = (o0[pure] / n[pure], o1[pure] / (o1[pure] + s1[pure]))
            umap_frac[e] = np.where(pure, o1 / (o1 + s1), np.nan)
    d = abs(res["py"][1] - res["R"][1])
    row["maxdiff"] = d.max()
    row["maxval"] = abs(res["R"][1]).max()
    row["soup_maxdiff"] = np.abs(res["py"][2] - res["R"][2]).max()
    row["same_markers"] = set(res["py"][0]["markers"]) == set(res["R"][0]["markers"])
    row["n_markers"] = len(res["R"][0]["markers"])
    row["entries"] = res["R"][1].nnz
    if t == EXAMPLE:                                              # UMAP of the cells, for display only
        umap_frac["before"] = np.where(pure, o0 / n, np.nan)
        umap_species = species.species.to_numpy()
        ad = scanpy.AnnData(sp.csr_matrix(toc.T).astype(np.float32))
        scanpy.pp.normalize_total(ad, target_sum=1e4)
        scanpy.pp.log1p(ad)
        scanpy.pp.highly_variable_genes(ad, n_top_genes=2000, subset=True)
        scanpy.pp.pca(ad, n_comps=30, random_state=0)
        scanpy.pp.neighbors(ad, n_neighbors=15, random_state=0)
        scanpy.tl.umap(ad, random_state=0)
        xy = ad.obsm["X_umap"]
    if t == EXAMPLE:                                              # 200,000 random nonzero entries of R's matrix
        xc = res["R"][1].tocoo()
        pick = np.random.default_rng(0).choice(xc.nnz, min(200_000, xc.nnz), replace=False)
        r_, c_ = xc.row[pick], xc.col[pick]
        pair = np.column_stack([xc.data[pick], np.asarray(res["py"][1].tocsr()[r_, c_]).ravel()])
        pair = pair[(pair > 0).all(axis=1)]
    rows.append(row)
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "comparison_per_dataset.tsv", sep="\t", index=False)

# ======================= Figure 1 =======================
fig = plt.figure(figsize=(7.2, 6.5))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.95], hspace=0.55)


def panel(ax, s, dx=-0.36):
    ax.text(dx, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def crop(img):
    ink = (img[..., 3] > 0.04) & (img[..., :3].min(-1) < 0.96)
    ys, xs = np.where(ink)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


top = gs[0].subgridspec(1, 4, width_ratios=[0.62, 1, 1, 1], wspace=0.55)
logos = [crop(plt.imread(ASSETS / f)) for f in ["pysoupx_logo.png", "omicverse_logo.png"]]
x = np.arange(len(tab))

axa = fig.add_subplot(top[1])
axa.scatter(x, tab.true_rho * 100, s=30, marker="_", color=C_TRUE, linewidth=1.2, label="cross-species truth", zorder=3)
axa.scatter(x - 0.12, tab.rho_R * 100, s=12, color=C_R, edgecolor="white", linewidth=0.3, label="SoupX (R)")
axa.scatter(x + 0.12, tab.rho_py * 100, s=12, color=C_PY, edgecolor="white", linewidth=0.3, label="pysoupx")
axa.set_xticks(x, tab.label, rotation=40, ha="right", rotation_mode="anchor", fontsize=5.5)
axa.set_ylim(0, max(tab.rho_R.max(), tab.true_rho.max()) * 130)
axa.set_ylabel("contamination (%)")
axa.set_title("Contamination fraction")
axa.grid(axis="y", color=GRID, lw=0.5)
axa.set_axisbelow(True)
axa.legend(fontsize=5, loc="upper left", handletextpad=0.1, borderaxespad=0.1)
panel(axa, "a")

axb = fig.add_subplot(top[2])
w = 0.36
axb.bar(x - w / 2, tab.removed_R * 100, w * 0.9, color=C_R, label="SoupX (R)")
axb.bar(x + w / 2, tab.removed_py * 100, w * 0.9, color=C_PY, label="pysoupx")
axb.set_xticks(x, tab.label, rotation=40, ha="right", rotation_mode="anchor", fontsize=5.5)
axb.set_ylim(0, 112)
axb.set_yticks([0, 20, 40, 60, 80, 100])
axb.set_ylabel("removed (%)")
axb.set_title("Cross-species counts removed")
axb.grid(axis="y", color=GRID, lw=0.5)
axb.set_axisbelow(True)
axb.legend(fontsize=5, loc="upper center", ncol=2, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1)
panel(axb, "b")

axc = fig.add_subplot(top[3])
lim = pair.max() * 1.05
axc.plot([0.1, lim], [0.1, lim], color=INK2, lw=0.6, zorder=0)
axc.scatter(pair[:, 0], pair[:, 1], s=0.5, color=C_PY, lw=0, rasterized=True)
axc.set_xscale("log")
axc.set_yscale("log")
axc.set_xlim(0.1, lim)
axc.set_ylim(0.1, lim)
axc.set_aspect("equal")
axc.set_xlabel("SoupX (R)")
axc.set_ylabel("pysoupx")
ex = tab.set_index("dataset").loc[EXAMPLE]
axc.set_title(f"Corrected counts ({LABEL[EXAMPLE].split(' (')[0]})")
axc.text(0.04, 0.96, f"{int(ex.entries):,} entries\nmax |difference| {ex.maxdiff:.0e}", transform=axc.transAxes,
         va="top", fontsize=5, color=INK2)
panel(axc, "c")

bot = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.3)
axd = fig.add_subplot(bot[0])
axd.scatter(tab.droplets, tab.sec_R, s=12, color=C_R, edgecolor="white", linewidth=0.3, label="SoupX 1.6.2 (R)")
axd.scatter(tab.droplets, tab.sec_py, s=12, color=C_PY, edgecolor="white", linewidth=0.3, label="pysoupx")
for _, r in tab.iterrows():
    axd.plot([r.droplets] * 2, [r.sec_py, r.sec_R], color=GRID, lw=0.8, zorder=0)
axd.set_xscale("log")
axd.set_yscale("log")
axd.set_xlabel("droplets in the raw matrix")
axd.set_yticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
axd.minorticks_off()
axd.set_xticks([4e5, 7e5, 1e6, 2e6], ["0.4M", "0.7M", "1M", "2M"])
axd.set_ylabel("wall time, 1 thread (s)")
axd.set_title("Run time, SoupChannel to adjustCounts")
axd.grid(color=GRID, lw=0.5)
axd.set_axisbelow(True)
axd.legend(fontsize=5.5, loc="upper left", handletextpad=0.1)
panel(axd, "d", dx=-0.2)

axe = fig.add_subplot(bot[1])
before, after_R = cellwise["R"]
_, after_py = cellwise["py"]
axe.scatter(before * 100, after_R * 100, s=1.5, color=C_R, lw=0, rasterized=True, label="SoupX (R)")
axe.scatter(before * 100, after_py * 100, s=0.4, color=C_PY, lw=0, rasterized=True, label="pysoupx")
top_ = np.nanpercentile(before * 100, 99.5)
axe.plot([0, top_], [0, top_], color=INK2, lw=0.6, zorder=0)
axe.set_xlim(0, top_)
axe.set_ylim(0, top_)
axe.set_xlabel("other-species counts before correction (% of cell)")
axe.set_ylabel("after correction (%)")
axe.set_title(f"Per cell, {LABEL[EXAMPLE]}")
axe.grid(color=GRID, lw=0.5)
axe.set_axisbelow(True)
leg = axe.legend(fontsize=5.5, loc="upper left", handletextpad=0.1)
for h in leg.legend_handles:
    h.set_sizes([10])
panel(axe, "e", dx=-0.2)


row3 = gs[2].subgridspec(1, 4, wspace=0.06)


def umap_ax(k, title, letter):
    ax = fig.add_subplot(row3[k])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title(title)
    ax.set_aspect("equal")
    panel(ax, letter, dx=-0.04)
    return ax


ax = umap_ax(0, f"{LABEL[EXAMPLE]}: species", "f")
for name, col in [("human", "#2f7f6f"), ("mouse", "#b5a642"), ("mixed", "#2a2a2a")]:
    m = umap_species == name
    ax.scatter(xy[m, 0], xy[m, 1], s=0.6 if name != "mixed" else 1.2, c=col, lw=0, rasterized=True,
               label=f"{name} ({m.sum():,})")
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=5.5, handletextpad=0.1,
                columnspacing=0.6)
for h in leg.legend_handles:
    h.set_sizes([10])
vmax = np.nanpercentile(umap_frac["before"] * 100, 99)
for k, (key, title, letter) in enumerate([("before", "Before correction", "g"), ("R", "After, SoupX (R)", "h"),
                                          ("py", "After, pysoupx", "i")]):
    ax = umap_ax(k + 1, title, letter)
    v = umap_frac[key] * 100
    nan = np.isnan(v)
    ax.scatter(xy[nan, 0], xy[nan, 1], s=0.6, c="#d9d8d3", lw=0, rasterized=True)
    o = np.flatnonzero(~nan)[np.argsort(v[~nan], kind="stable")]
    im = ax.scatter(xy[o, 0], xy[o, 1], c=v[o], s=0.6, cmap="magma_r", vmin=0, vmax=vmax, lw=0, rasterized=True)
cax = fig.axes[-2].inset_axes([0.05, -0.14, 0.9, 0.035])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.5, length=1.5)
cax.set_title("other-species counts (% of cell; mixed cells grey)", fontsize=5.5, color=INK2, pad=2)

pa, pd_ = axa.get_position(), axd.get_position()
x0, x1 = pd_.x0 - 0.2 * pd_.width, pa.x0 - 0.075
y0, y1 = pa.y0 - 0.02, pa.y1 + 0.035
h_ov = (x1 - x0) * 7.2 / 6.5 * logos[1].shape[0] / logos[1].shape[1]
for img, rect, anchor in [(logos[0], [x0, y0 + h_ov + 0.012, x1 - x0, y1 - y0 - h_ov - 0.012], "N"),
                          (logos[1], [x0, y0, x1 - x0, h_ov], "S")]:
    axl = fig.add_axes(rect)
    axl.imshow(img)
    axl.set_anchor(anchor)
    axl.set_axis_off()
fig.savefig(OUT / "fig1.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig1.png", bbox_inches="tight", pad_inches=0.02, dpi=200)

# ======================= numbers =======================
speed = tab.sec_R / tab.sec_py
big = tab.iloc[tab.droplets.idxmax()]
macros = {
    "nDatasets": len(tab), "nCellsTotal": f"{int(tab.cells.sum()):,}",
    "nCellsMin": f"{int(tab.cells.min()):,}", "nCellsMax": f"{int(tab.cells.max()):,}",
    "nDropMin": f"{tab.droplets.min() / 1e6:.2f}", "nDropMax": f"{tab.droplets.max() / 1e6:.2f}",
    "nSameRho": int((tab.rho_R == tab.rho_py).sum()), "nSameMarkers": int(tab.same_markers.sum()),
    "rhoMin": f"{tab.rho_R.min() * 100:.1f}", "rhoMax": f"{tab.rho_R.max() * 100:.1f}",
    "trueMin": f"{tab.true_rho.min() * 100:.1f}", "trueMax": f"{tab.true_rho.max() * 100:.1f}",
    "removedMin": f"{tab.removed_R.min() * 100:.0f}", "removedMax": f"{tab.removed_R.max() * 100:.0f}",
    "keptMin": f"{tab.kept_R.min() * 100:.1f}", "keptMax": f"{tab.kept_R.max() * 100:.1f}",
    "removedDiffMax": f"{(tab.removed_py - tab.removed_R).abs().max():.0e}",
    "maxDiff": f"{tab.maxdiff.max():.0e}", "soupMaxDiff": f"{tab.soup_maxdiff.max():.0e}",
    "speedMin": f"{speed.min():.1f}", "speedMax": f"{speed.max():.1f}", "speedMed": f"{speed.median():.1f}",
    "tBigR": f"{big.sec_R:.0f}", "tBigPy": f"{big.sec_py:.0f}", "bigName": big.label,
    "exampleName": LABEL[EXAMPLE], "exampleEntries": f"{int(ex.entries):,}", "nSample": "200,000",
    "spearMin": f"{tab.spearman_rho_numi.min():.2f}", "spearMax": f"{tab.spearman_rho_numi.max():.2f}",
    "exTrueHuman": f"{ex.true_human * 100:.1f}", "exTrueMouse": f"{ex.true_mouse * 100:.1f}",
    "exRemHuman": f"{ex.removed_human_R * 100:.0f}", "exRemMouse": f"{ex.removed_mouse_R * 100:.0f}",
    "nSameGeneset": int(((tab.gs_all_py / tab.gs_all_R - 1).abs().lt(1e-8)
                         & (tab.gs_nomt_py / tab.gs_nomt_R - 1).abs().lt(1e-8)).sum()),
    "gsRelDiffMax": f"{max((tab.gs_all_py / tab.gs_all_R - 1).abs().max(), (tab.gs_nomt_py / tab.gs_nomt_R - 1).abs().max()):.0e}",
    "gsAllMin": f"{tab.gs_all_R.min() * 100:.1f}", "gsAllMax": f"{tab.gs_all_R.max() * 100:.1f}",
    "gsNomtMin": f"{tab.gs_nomt_R.min() * 100:.1f}", "gsNomtMax": f"{tab.gs_nomt_R.max() * 100:.1f}",
    "overrepMin": f"{tab.top5_soup_overrep.min():.1f}", "overrepMax": f"{tab.top5_soup_overrep.max():.1f}",
}
for key, name in [("OneK", "hgmm_1k_v3"), ("FiveK", "hgmm_5k_v3"), ("SixK", "hgmm_6k"), ("TenK", "hgmm_10k_v3"),
                  ("TwelveK", "hgmm_12k")]:
    r_ = tab.set_index("dataset").loc[name]
    macros[f"rho{key}"] = f"{r_.rho_R * 100:.1f}"
    macros[f"true{key}"] = f"{r_.true_rho * 100:.1f}"
    macros[f"gsNomt{key}"] = f"{r_.gs_nomt_R * 100:.1f}"
with open(OUT / "numbers.tex", "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open(OUT / "numbers.tex").read())
cols = ["label", "true_rho", "true_human", "true_mouse", "rho_R", "gs_all_R", "gs_nomt_R", "removed_human_R",
        "removed_mouse_R", "spearman_rho_numi", "top5_soup_overrep"]
tab[cols].to_csv(OUT / "table_s1.tsv", sep="\t", index=False)
print(tab.drop(columns=["label"]).round(5).to_string())
