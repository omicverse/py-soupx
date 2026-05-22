"""Light-weight diagnostic plots — the contamination posterior density."""
from __future__ import annotations

from .soupchannel import SoupChannel

__all__ = ["plot_contamination_fraction"]


def plot_contamination_fraction(sc: SoupChannel, ax=None):
    """Plot the contamination-fraction posterior produced by
    :func:`pysoupx.auto_est_cont`.

    Reproduces SoupX's ``autoEstCont(doPlot=TRUE)`` figure: the
    aggregated posterior density, the gamma prior (dashed) and the
    estimated rho (red vertical line).
    """
    import matplotlib.pyplot as plt

    if sc.fit is None or "posterior" not in sc.fit:
        raise ValueError("Run auto_est_cont() first.")
    fit = sc.fit
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    x = fit["rhoProbes"]
    ax.plot(x, fit["posterior"], "-", color="black", label="posterior")
    ax.plot(x, fit["prior"], "--", color="grey", label="prior")
    ax.axvline(fit["rhoEst"], color="red",
               label=f"rho = {fit['rhoEst']:.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Contamination Fraction")
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ax
