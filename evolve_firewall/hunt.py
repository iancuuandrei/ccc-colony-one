"""Counter-example hunt: generate many synthetic positions, keep those a certificate FLAGS (cheap numpy),
and brute-force only those with the engine oracle (4 processes). Any flagged position whose value
differs from the claim is a violation.

usage: python hunt.py <n_per_mode> <seed> [gens=gen1,gen7]
Writes hunt_<seed>.json with per-generation / per-certificate checked & violations, and
hunt_<seed>_violations.npz with the counter-examples (obs) if any.
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


def work(o):
    from oracle import solve
    r = solve(o, node_limit=2_000_000)
    return 9 if r["value"] is None else r["value"]


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    gens = (sys.argv[3] if len(sys.argv) > 3 else "gen1,gen7").split(",")
    from candidates import GENERATIONS
    from synth import generate
    t0 = time.time()
    xs = [generate(n, seed * 10 + i, mode) for i, mode in enumerate(("shrink", "boxed", "sparse"))]
    x = np.concatenate(xs).astype(np.float32)
    flags = {}
    for gname in gens:
        out = {}
        for i in range(0, len(x), 8192):
            for k, (claim, f) in GENERATIONS[gname](x[i:i + 8192]).items():
                out.setdefault(k, (claim, []))[1].append(f)
        flags[gname] = {k: (cl, np.concatenate(fs)) for k, (cl, fs) in out.items()}
    need = np.zeros(len(x), bool)
    for d in flags.values():
        for cl, f in d.values():
            need |= f
    idx = np.where(need)[0]
    t1 = time.time()
    with Pool(4) as pool:
        vals = pool.map(work, list(x[idx]), chunksize=16)
    value = np.full(len(x), 99)
    value[idx] = vals
    rep = dict(generated=len(x), flagged_any=int(len(idx)), seconds_gen=round(t1 - t0, 1),
               seconds_oracle=round(time.time() - t1, 1), gens={})
    viol = []
    for gname, d in flags.items():
        rep["gens"][gname] = {}
        for k, (cl, f) in d.items():
            ok = f & (value != 9)
            bad = ok & (value != cl)
            rep["gens"][gname][k] = dict(flagged=int(f.sum()), checked=int(ok.sum()), violations=int(bad.sum()))
            viol.extend(np.where(bad)[0].tolist())
    print(json.dumps(rep))
    json.dump(rep, open(os.path.join(HERE, f"hunt_{seed}.json"), "w"), indent=1)
    if viol:
        v = sorted(set(viol))
        np.savez_compressed(os.path.join(HERE, f"hunt_{seed}_violations.npz"), obs=x[v], value=value[v])


if __name__ == "__main__":
    main()
