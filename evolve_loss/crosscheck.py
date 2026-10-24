"""Cross-check bf.label against the independent env.step reference (naive.label).

Usage: python crosscheck.py LABELS.npz N_LOST N_OTHER [workers]
Takes up to N_LOST positions labelled lost and N_OTHER random others.
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _work(args):
    import naive
    i, o = args
    return i, naive.label(o)


def main():
    path, n_lost, n_other = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    workers = min(4, int(sys.argv[4]) if len(sys.argv) > 4 else 4)
    z = np.load(path)
    obs, code = z["obs"], z["code"]
    rng = np.random.default_rng(7)
    lost = np.flatnonzero(code == 1)
    other = np.flatnonzero(code != 1)
    pick = np.concatenate([rng.permutation(lost)[:n_lost], rng.permutation(other)[:n_other]])
    t0 = time.time()
    mism = []
    with Pool(workers) as pool:
        for i, c in pool.imap_unordered(_work, ((int(i), obs[i].astype(np.float32)) for i in pick)):
            if c != code[i]:
                mism.append((i, int(code[i]), c))
    print(f"checked {len(pick)} (lost {min(n_lost, len(lost))}) mismatches {len(mism)} "
          f"{time.time() - t0:.0f}s")
    for m in mism[:20]:
        print("mismatch idx", m)


if __name__ == "__main__":
    main()
