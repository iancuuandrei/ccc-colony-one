"""Check that the dense fixed-shape cert (no subsetting) equals the pruned one.

Usage: python dense_check.py LABELS.npz N_LOST N_OTHER
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cert as C  # noqa: E402


def main():
    z = np.load(sys.argv[1])
    obs, code = z["obs"], z["code"]
    rng = np.random.default_rng(5)
    pick = np.concatenate([rng.permutation(np.flatnonzero(code == 1))[:int(sys.argv[2])],
                           rng.permutation(np.flatnonzero(code != 1))[:int(sys.argv[3])]])
    o = obs[pick].astype(np.float32)
    t0 = time.time()
    a = C.cert_loss(o, two_ply=True, dense=False)
    t1 = time.time()
    b = C.cert_loss(o, two_ply=True, dense=True)
    t2 = time.time()
    c = C.cert_loss(o, two_ply=False, dense=True)
    d = C.cert_loss(o, two_ply=False, dense=False)
    print(f"n={len(o)} pruned={a.sum()} dense={b.sum()} mismatches={int((a != b).sum())} "
          f"cheap mismatches={int((c != d).sum())} viol={int((b & (code[pick] != 1)).sum())} "
          f"t_pruned={t1 - t0:.1f}s t_dense={t2 - t1:.1f}s")


if __name__ == "__main__":
    main()
