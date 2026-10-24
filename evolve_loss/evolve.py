"""Evaluate certificate generations against the brute-force labels and log them.

Usage: python evolve.py LABELS.npz GEN_NAME [--log]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bf  # noqa: E402
import gens  # noqa: E402
from colony_one.board import Board  # noqa: E402


def _chunk(args):
    name, path, i, j = args
    z = np.load(path)
    return i, gens.GENS[name][0](z["obs"][i:j].astype(np.float32))


def evaluate(path, name, log=False, chunk=10000, show=3, workers=4):
    from multiprocessing import Pool
    z = np.load(path)
    obs, code = z["obs"], z["code"]
    fn, text = gens.GENS[name]
    t0 = time.time()
    flag = np.zeros(len(obs), bool)
    jobs = [(name, path, i, min(len(obs), i + chunk)) for i in range(0, len(obs), chunk)]
    with Pool(min(4, workers)) as pool:
        for i, f in pool.imap_unordered(_chunk, jobs):
            flag[i:i + len(f)] = f
    sec = time.time() - t0
    lost = code == 1
    viol = flag & ~lost
    res = dict(generation=name, rule=text, labels=os.path.basename(path),
               positions=int(len(obs)), bruteforce_lost=int(lost.sum()),
               flagged=int(flag.sum()), violations=int(viol.sum()),
               coverage=round(float((flag & lost).sum() / max(1, lost.sum())), 4),
               seconds=round(sec, 1))
    al = np.rint(obs[:, 13, 0, 0].astype(np.float32) * 2).astype(int)
    res["by_actions_left"] = {int(a): dict(lost=int((lost & (al == a)).sum()),
                                           flagged=int((flag & (al == a)).sum()))
                              for a in (1, 2)}
    print(json.dumps(res, indent=1))
    for i in np.flatnonzero(viol)[:show]:
        print("VIOLATION", i, bf.obs_context(obs[i]))
        print(bf.board_str(Board.from_observation(obs[i].astype(np.float32), bf.M)))
    if log:
        lp = os.path.join(HERE, "generations.json")
        hist = json.load(open(lp)) if os.path.exists(lp) else []
        hist.append(res)
        json.dump(hist, open(lp, "w"), indent=1)
    return flag, code


if __name__ == "__main__":
    flag, _ = evaluate(sys.argv[1], sys.argv[2], log="--log" in sys.argv)
    np.save(os.path.join(HERE, f"flag_{sys.argv[2]}_{os.path.basename(sys.argv[1])[:-4]}.npy"), flag)
