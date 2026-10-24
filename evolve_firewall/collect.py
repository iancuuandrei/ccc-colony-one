"""Collect positions for the firewall study.

data/*  (self-play chunks, key 'obs'): every position is decoded (round, mover, actions left);
we keep unique positions and record per-round frequencies. Output:
  positions_selfplay.npz : obs (uint8 16x8x8) of unique positions in shrink rounds (urgency == 1),
                           plus 'pre' set = unique positions of the round before a shrink
  freq.json              : how often shrink-round / ring situations occur in self-play
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def ringmap():
    r = np.arange(8)
    return np.minimum(np.minimum(r[:, None], r[None, :]), np.minimum(7 - r[:, None], 7 - r[None, :]))


RING = ringmap()


def main():
    folders = sorted(glob.glob(os.path.join(ROOT, "data", "sp_*"))) + sorted(glob.glob(os.path.join(ROOT, "data", "vps_*")))
    seen = set()
    keep = []
    total = 0
    stats = dict(total=0, shrink_round=0, shrink_round_unique=0)
    by_round = {}
    for f in folders:
        for ch in sorted(glob.glob(os.path.join(f, "chunk*.npz"))):
            obs = np.load(ch)["obs"]
            x = obs.astype(np.float32)
            total += len(x)
            nl = np.rint(x[:, 12].sum((1, 2))).astype(int)
            c = np.select([nl == 0, nl == 28, nl == 48, nl == 60], [0, 1, 2, 3], -1)
            u = x[:, 14, 0, 0]
            nxt = np.array([8, 16, 24, 26, 99])[c]
            rnd = np.rint(nxt - 8 * (1 - u)).astype(int)
            for r in np.unique(rnd):
                by_round[int(r)] = by_round.get(int(r), 0) + int((rnd == r).sum())
            m = u > 0.999
            stats["shrink_round"] += int(m.sum())
            for o in obs[m]:
                key = (o > 0.5).astype(np.uint8)
                # plane 13/14/15 are scalars: keep exact values (0.5/1.0)
                k8 = np.concatenate([key[:13].reshape(-1), [int(round(2 * float(o[13, 0, 0]))), int(round(8 * float(o[14, 0, 0]))), int(o[15, 0, 0] > 0.5)]]).astype(np.uint8)
                bk = k8.tobytes()
                if bk in seen:
                    continue
                seen.add(bk)
                keep.append(k8)
    stats["total"] = total
    stats["shrink_round_unique"] = len(keep)
    stats["by_round"] = dict(sorted(by_round.items()))
    np.savez_compressed(os.path.join(HERE, "positions_selfplay.npz"), packed=np.stack(keep))
    with open(os.path.join(HERE, "freq_raw.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "by_round"}))


def unpack(packed):
    """packed (N, 13*64+3) uint8 -> obs float32 (N,16,8,8)."""
    n = len(packed)
    obs = np.zeros((n, 16, 8, 8), np.float32)
    obs[:, :13] = packed[:, :13 * 64].reshape(n, 13, 8, 8)
    obs[:, 13] = (packed[:, 13 * 64] / 2.0)[:, None, None]
    obs[:, 14] = (packed[:, 13 * 64 + 1] / 8.0)[:, None, None]
    obs[:, 15] = packed[:, 13 * 64 + 2][:, None, None]
    return obs


if __name__ == "__main__":
    main()
