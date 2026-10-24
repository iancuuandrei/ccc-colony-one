"""Policy-target constructions from one root search (arrays aligned to the legal list).

All inputs are 1-D float arrays over the root's legal actions:
  prior  P(a)  softmax of the network logits over legal actions (search-time prior)
  N      root visit counts
  Q      root mean values W/N from the root team's view (undefined where N == 0)
  v_net  network value of the root position (root team's view)
"""
from __future__ import annotations
import numpy as np

TINY = np.finfo(np.float64).tiny


def softmax(x):
    x = np.asarray(x, np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def visits(N):
    N = np.asarray(N, np.float64)
    return N / max(N.sum(), 1.0)


def mixed_value(prior, N, Q, v_net, mode="prior"):
    """Gumbel MuZero v_mix (mctx `_compute_mixed_value`) or a visit-weighted variant."""
    N = np.asarray(N, np.float64)
    vis = N > 0
    s = N.sum()
    if not vis.any():
        return float(v_net)
    if mode == "prior":
        p = np.maximum(np.asarray(prior, np.float64), TINY)
        wq = float((p[vis] * Q[vis]).sum() / p[vis].sum())
    else:  # visit-weighted: (v_net + sum W) / (1 + sum N)
        wq = float((N[vis] * Q[vis]).sum() / s)
    return (float(v_net) + s * wq) / (s + 1.0)


def completed_q(prior, N, Q, v_net, mode="prior"):
    N = np.asarray(N, np.float64)
    vm = mixed_value(prior, N, Q, v_net, mode)
    return np.where(N > 0, np.nan_to_num(Q, nan=vm), vm)


def gumbel(prior, N, Q, v_net, c_visit=50.0, c_scale=0.1, mode="prior"):
    """pi' = softmax(log prior + sigma(completedQ)), Q rescaled to [0,1] over legal actions."""
    N = np.asarray(N, np.float64)
    cq = completed_q(prior, N, Q, v_net, mode)
    lo, hi = cq.min(), cq.max()
    cq01 = (cq - lo) / max(hi - lo, 1e-8)
    sigma = (c_visit + N.max()) * c_scale * cq01
    return softmax(np.log(np.maximum(np.asarray(prior, np.float64), TINY)) + sigma)


def softq(N, Q, tau):
    """softmax(Q/tau) over visited actions only (raw Q in value units, [-1,1])."""
    N = np.asarray(N, np.float64)
    vis = N > 0
    out = np.zeros(len(N))
    if vis.any():
        out[vis] = softmax(np.asarray(Q, np.float64)[vis] / tau)
    return out


def all_targets(prior, N, Q, v_net):
    a = visits(N)
    b = gumbel(prior, N, Q, v_net)
    t = {
        "prior": np.asarray(prior, np.float64) / np.sum(prior),
        "visits": a,
        "gumbel": b,
        "gumbel_vw": gumbel(prior, N, Q, v_net, mode="visit"),
        "gumbel_c1": gumbel(prior, N, Q, v_net, c_scale=1.0),
        "softq_0.05": softq(N, Q, 0.05),
        "softq_0.1": softq(N, Q, 0.1),
        "softq_0.2": softq(N, Q, 0.2),
        "mix_visits_gumbel": 0.5 * a + 0.5 * b,
    }
    return t
