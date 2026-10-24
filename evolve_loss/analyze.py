"""Profile brute-force losses by simple features (al, own pieces, guards, attackers)."""

from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geom as G  # noqa: E402


def features(obs):
    own, en, locked, al, shrink_due, mmag = G.unpack(obs)
    open_ = ~locked
    occ = own.any(1) | en.any(1)
    K = own[:, 5]
    own_nk = own[:, :5].any(1)
    adjK = G.dilate(K)
    g = G.cnt(adjK & own_nk)
    n_att = sum(G.anyb(en[:, t] & G.attackers_of(K, t, occ, open_)) for t in range(6))
    return dict(al=al, n_own=G.cnt(own_nk), n_en=G.cnt(en[:, :5].any(1)), g=g, n_att=n_att,
                gate=(mmag & shrink_due), cyan_shrink=(~mmag & shrink_due))


def main():
    z = np.load(sys.argv[1])
    obs, code = z["obs"], z["code"]
    lost = code == 1
    f = features(obs.astype(np.float32))
    print("positions", len(code), "lost", lost.sum(), "stuck", (code == -1).sum())
    for k in ("al", "n_own", "n_en", "g", "n_att", "gate", "cyan_shrink"):
        print(k, "lost:", sorted(Counter(f[k][lost].tolist()).items()),
              " all:", sorted(Counter(f[k].tolist()).items()))
    key = Counter(zip(f["al"][lost].tolist(), (f["n_own"][lost] > 0).tolist(),
                      f["g"][lost].tolist(), f["n_att"][lost].tolist()))
    print("(al, has_own, g, n_att) among lost:")
    for k, v in key.most_common(40):
        print(" ", k, v)


if __name__ == "__main__":
    main()
