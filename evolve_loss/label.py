"""Label positions with exact ground truth (bf.label), <= 4 processes.

Usage: python label.py OUT.npz GLOB [GLOB ...] [--max N] [--workers W]
Saves obs (float16, deduplicated), lost code (1/0/-1), nodes, source index.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _work(args):
    import bf
    i, o = args
    code, nodes = bf.label(o)
    return i, code, nodes


def load_unique(patterns):
    arrs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            arrs.append(np.load(f)["obs"])
    obs = np.concatenate(arrs).astype(np.float16)
    flat = obs.reshape(len(obs), -1)
    # dedupe on planes 0..15 (all planes; scalar planes included)
    _, idx = np.unique(flat.view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))),
                       return_index=True)
    idx = np.sort(idx)
    return obs[idx]


def main():
    args = sys.argv[1:]
    out = args.pop(0)
    max_n, workers = None, 4
    pats = []
    while args:
        a = args.pop(0)
        if a == "--max":
            max_n = int(args.pop(0))
        elif a == "--workers":
            workers = min(4, int(args.pop(0)))
        else:
            pats.append(a)
    obs = load_unique(pats)
    rng = np.random.default_rng(12345)
    perm = rng.permutation(len(obs))
    if max_n:
        perm = perm[:max_n]
    obs = obs[perm]
    n = len(obs)
    print("unique positions", n, flush=True)
    code = np.full(n, -2, np.int8)
    nodes = np.zeros(n, np.int64)
    t0 = time.time()
    with Pool(workers) as pool:
        done = 0
        for i, c, nd in pool.imap_unordered(_work, ((i, obs[i].astype(np.float32)) for i in range(n)),
                                            chunksize=64):
            code[i] = c
            nodes[i] = nd
            done += 1
            if done % 20000 == 0:
                print(done, "lost", int((code == 1).sum()), f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out, obs=obs, code=code, nodes=nodes)
    print("done", n, "lost", int((code == 1).sum()), "stuck", int((code == -1).sum()),
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
