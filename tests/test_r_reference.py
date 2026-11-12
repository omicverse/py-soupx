"""Against values computed with R 4.5.3 / SoupX 1.6.2 (no R needed)."""
from __future__ import annotations

import numpy as np

from pysoupx.estimate import _phyper_upper
from pysoupx.rrng import RRandom


def test_rbinom_bernoulli_follows_r():
    # set.seed(1); rbinom(10, 1, p); runif(1)
    p = np.array([0, 0.3, 0.7, 1, 0.5, 0.05, 0.95, 0.999, 0.25, 0])
    r = RRandom(1)
    np.testing.assert_array_equal(r.rbinom1(p), [0, 0, 1, 1, 1, 0, 1, 1, 1, 0])
    np.testing.assert_allclose(r.unif_rand(1), [0.6607978], atol=5e-8)   # p = 0 and 1 draw nothing
    # set.seed(42); x <- runif(2000); sum(rbinom(2000, 1, x))
    r = RRandom(42)
    assert r.rbinom1(r.unif_rand(2000)).sum() == 1017


def test_phyper_upper_tail():
    # phyper(q, NR, NB, 600, lower.tail = FALSE)
    ref = [float.fromhex(v) for v in
           "0x1p+0 0x1.ffffaddcb9edp-1 0x1.ffffffffec50ap-1 0x1.9f8279224332p-3 0x1p+0".split()]
    got = _phyper_upper(np.array([-1, 0, 5, 40, 200.]), np.array([100, 100, 300, 300, 3000.]),
                        np.array([4900, 4900, 4700, 4700, 2000.]), 600)
    np.testing.assert_allclose(got, ref, rtol=1e-10)
