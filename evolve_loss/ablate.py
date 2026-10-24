"""Cost/coverage ablations of the enemy-kill test inside losing_final.

Usage: python ablate.py LABELS.npz [LABELS2.npz ...]
Evaluates each variant on all lost positions (coverage) plus 60k random others
(violations) per label file, with the cheap structure (one-ply for actions_left
== 1, Kernel tree for Kernel-only actions_left == 2) and the full structure.
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def ew_r3cheap(own, en, open_, level=None):
    """R1e | R2e | R3e' | R4e where R3e' replaces the segment test by: Q has >= 2
    relocation squares (at most one can lie on the <= 2-square P-K segment besides
    Q's own square), or K has one guard and Q can capture it."""
    import cert as C
    import geom as G
    from geom import anyb, cnt, dilate
    base = C._EW_ORIG(own, en, open_, level=1)
    K = own[:, 5]
    occ = own.any(1) | en.any(1)
    own_nk = own[:, :5].any(1)
    guards = dilate(K) & own_nk
    g = cnt(guards)
    gmask = (g == 0)[:, None, None] | ((g == 1)[:, None, None] & guards)
    res = base
    for tp in (G.PING, G.DAEMON, G.TRACE, G.ICE):
        P = en[:, tp]
        for tq in range(6):
            if tq == tp:
                continue
            Q = en[:, tq]
            att = anyb(P & G.attackers_of(K, tp, occ & ~Q, open_))
            if not att.any():
                continue
            n = cnt(G.relocations(Q, tq, occ, own_nk, open_) & gmask)
            res |= att & ((n >= 2) | ((g == 1) & (n >= 1)))
    # R4e unchanged (same code as cert.EW level 3)
    empty = ~occ & open_
    for tp in (G.PING, G.DAEMON, G.TRACE, G.ICE):
        P = en[:, tp]
        if not P.any():
            continue
        for tb in range(5):
            B = own[:, tb]
            att = anyb(P & G.attackers_of(K, tp, occ & ~B, open_))
            if not att.any():
                continue
            gok = (g == 0) | ((g == 1) & anyb(B & guards))
            shot = (anyb(en[:, G.DAEMON] & dilate(B)) |
                    anyb(en[:, G.TRACE] & G.ray_set(B, empty, open_, G.DIAG, 2, 3)))
            res |= att & gok & shot
    return res


VARIANTS = {
    "ew1": dict(level=1, ebk=False),
    "ew1+ebk": dict(level=1, ebk=True),
    "ew2+ebk": dict(level=2, ebk=True),
    "ew3": dict(level=3, ebk=False),
    "ew3_r3cheap+ebk": dict(level="cheap", ebk=True),
    "ew3+ebk": dict(level=3, ebk=True),
}


def _run(args):
    path, idx, var, two_ply = args
    import cert as C
    if not hasattr(C, "_EW_ORIG"):
        C._EW_ORIG = C.EW
        C._EBK_ORIG = C.EBK
    v = VARIANTS[var]
    if v["level"] == "cheap":
        C.EW = ew_r3cheap
    else:
        C.EW = C._EW_ORIG
        C.LEVEL["ew"] = v["level"]
    C.EBK = C._EBK_ORIG if v["ebk"] else (lambda own, en, open_, mmag: np.zeros(len(mmag), bool))
    z = np.load(path)
    return var, two_ply, idx, C.cert_loss(z["obs"][idx].astype(np.float32), two_ply=two_ply)


def main():
    out = {}
    for path in sys.argv[1:]:
        z = np.load(path)
        code = z["code"]
        rng = np.random.default_rng(11)
        lost = np.flatnonzero(code == 1)
        other = rng.permutation(np.flatnonzero(code != 1))[:60000]
        sel = np.sort(np.concatenate([lost, other]))
        chunks = np.array_split(sel, 16)
        jobs = [(path, c, var, tp) for var in VARIANTS for tp in (False, True) for c in chunks]
        res = {}
        t0 = time.time()
        with Pool(4) as pool:
            for var, tp, idx, f in pool.imap_unordered(_run, jobs):
                res.setdefault((var, tp), []).append((idx, f))
        for (var, tp), parts in res.items():
            idx = np.concatenate([p[0] for p in parts])
            f = np.concatenate([p[1] for p in parts])
            c = code[idx]
            key = f"{os.path.basename(path)}|{var}|{'full' if tp else 'cheap'}"
            out[key] = dict(lost=int((c == 1).sum()), flagged_lost=int((f & (c == 1)).sum()),
                            coverage=round(float((f & (c == 1)).sum() / (c == 1).sum()), 4),
                            others_checked=int((c != 1).sum()), violations=int((f & (c != 1)).sum()))
            print(key, out[key], flush=True)
        print(f"{path}: {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(HERE, "ablations.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
