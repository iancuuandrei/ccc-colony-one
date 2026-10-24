"""Random endgame-like positions to stress certificate soundness off the self-play
distribution (locks, shrink timing, side to move, sparse armies, enemy pieces
clustered near our Kernel). Writes OUT.npz with key 'obs' (N,16,8,8) float16.

Usage: python synth.py OUT.npz N SEED
"""

from __future__ import annotations

import sys

import numpy as np

RING = np.array([[int(min(r, c, 7 - r, 7 - c)) for c in range(8)] for r in range(8)])  # 0 = edge


def main():
    out, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(seed)
    obs = np.zeros((n, 16, 8, 8), np.float16)
    for i in range(n):
        L = rng.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.25, 0.05])
        open_ = RING >= L
        sq = [tuple(x) for x in np.argwhere(open_)]
        free = set(sq)
        o = obs[i]
        o[12] = ~open_
        k = sq[rng.integers(len(sq))]
        free.discard(k)
        o[5][k] = 1
        n_own = rng.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
        n_en = rng.choice([0, 1, 2, 3, 4], p=[0.05, 0.3, 0.3, 0.2, 0.15])
        near = [s for s in free if max(abs(s[0] - k[0]), abs(s[1] - k[1])) <= 3]
        # enemy kernel + enemy pieces, mostly near our Kernel
        for plane, cnt in ((11, 1),):
            pool = list(free)
            s = pool[rng.integers(len(pool))]
            free.discard(s)
            o[plane][s] = 1
        for t in rng.permutation(5)[:n_en]:
            pool = [s for s in near if s in free] if rng.random() < 0.8 else list(free)
            if not pool:
                pool = list(free)
            if not pool:
                break
            s = pool[rng.integers(len(pool))]
            free.discard(s)
            o[6 + t][s] = 1
        for t in rng.permutation(5)[:n_own]:
            pool = list(free)
            if not pool:
                break
            s = pool[rng.integers(len(pool))]
            free.discard(s)
            o[t][s] = 1
        o[13] = rng.choice([0.5, 1.0])
        o[14] = rng.choice([1.0, 0.5], p=[0.35, 0.65])
        o[15] = rng.integers(2)
    np.savez_compressed(out, obs=obs)
    print("wrote", out, n)


if __name__ == "__main__":
    main()
