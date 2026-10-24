"""Build brute-force ground truth (engine) for a position set, 4 processes.

usage: python ground.py selfplay            # positions_selfplay.npz (shrink rounds, unique)
       python ground.py synth <n> <seed>    # synthetic stress set
Output: ground_<name>.npz with obs (float16), value (-1/0/1, 9 = stuck/timeout), nodes,
        kill / esc / quiet (own-turn exhaustive facts).
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def work(o):
    from oracle import solve, own_turn_facts
    r = solve(o, node_limit=2_000_000)
    f = own_turn_facts(o)
    v = 9 if r["value"] is None else r["value"]
    r0 = solve(o, node_limit=2_000_000, fw=False)
    v0 = 9 if r0["value"] is None else r0["value"]
    return v, r["nodes"], int(f["kill"]), int(f["esc"]), int(f["quiet"]), v0


def main():
    name = sys.argv[1]
    if name == "selfplay":
        from collect import unpack
        obs = unpack(np.load(os.path.join(HERE, "positions_selfplay.npz"))["packed"])
        out = "selfplay"
    else:
        from synth import generate
        n, seed = int(sys.argv[2]), int(sys.argv[3])
        mode = sys.argv[4] if len(sys.argv) > 4 else "shrink"
        obs = generate(n, seed, mode)
        out = f"synth_{mode}_{seed}"
    t0 = time.time()
    with Pool(4) as pool:
        res = pool.map(work, list(obs), chunksize=32)
    res = np.array(res, dtype=np.int64)
    np.savez_compressed(os.path.join(HERE, f"ground_{out}.npz"), obs=obs.astype(np.float16),
                        value=res[:, 0], nodes=res[:, 1], kill=res[:, 2], esc=res[:, 3], quiet=res[:, 4],
                        value_nofw=res[:, 5])
    print(out, len(obs), "positions", round(time.time() - t0, 1), "s",
          "values", {int(k): int((res[:, 0] == k).sum()) for k in (-1, 0, 1, 9)})


if __name__ == "__main__":
    main()
