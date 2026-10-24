"""Check the fixed-slot action generator (cert.apply_slot / cert.step_K) against the
official engine: same legal (from, to) set, same successor boards, same wins."""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bf  # noqa: E402
import cert as C  # noqa: E402
import geom as G  # noqa: E402
from colony_one import rules  # noqa: E402
from colony_one.board import Board  # noqa: E402
from colony_one.encoding import encode_board  # noqa: E402


def planes(board):
    o = encode_board(board, bf.M, 2, 1)
    return o[:12] > 0.5


def main():
    z = np.load(sys.argv[1])
    n = int(sys.argv[2])
    rng = np.random.default_rng(3)
    obs = z["obs"][rng.choice(len(z["obs"]), n, replace=False)].astype(np.float32)
    own, en, locked, al, sd, mm = G.unpack(obs)
    open_ = ~locked
    ge, rem = C.guard_removal(en)
    succ = [C.apply_slot(own, en, open_, s, ge, rem) for s in C.SLOTS]
    ksucc = {d: C.step_K(own, en, open_, d) for d in G.ALL8}
    bad = 0
    nact = 0
    for i in range(n):
        board = Board.from_observation(obs[i], bf.M)
        eng = {}
        for a in rules.legal_actions_for_team(board, bf.M):
            rec = board.apply_action(a)
            eng[(a.from_pos, a.to_pos)] = (planes(board), bf.is_kill(rec))
            board.undo_action(rec)
        mine = {}
        for s, (o2, e2, l, w) in zip(C.SLOTS, succ):
            if not l[i]:
                continue
            t, d, k, _, _ = s
            fr = tuple(int(x) for x in np.argwhere(own[i, t])[0])
            kk = 1 if k == 0 else k
            to = (fr[0] + d[0] * kk, fr[1] + d[1] * kk)
            assert (fr, to) not in mine
            mine[(fr, to)] = (np.concatenate([o2[i], e2[i]]), bool(w[i]))
        nact += len(eng)
        if set(eng) != set(mine):
            bad += 1
            print("legal set mismatch", i, set(eng) ^ set(mine))
            continue
        for key in eng:
            pe, we = eng[key]
            pm, wm = mine[key]
            if we != wm or (not we and not np.array_equal(pe, pm)):
                bad += 1
                print("successor mismatch", i, key, we, wm)
        # Kernel steps
        kp = tuple(int(x) for x in np.argwhere(own[i, 5])[0])
        for d, (o2, e2, l, w) in ksucc.items():
            key = (kp, (kp[0] + d[0], kp[1] + d[1]))
            if bool(l[i]) != (key in eng):
                bad += 1
                print("K legal mismatch", i, key)
            elif l[i]:
                pe, we = eng[key]
                if we != bool(w[i]) or (not we and not np.array_equal(pe, np.concatenate([o2[i], e2[i]]))):
                    bad += 1
                    print("K successor mismatch", i, key)
    print(f"positions {n} engine actions {nact} mismatches {bad}")


if __name__ == "__main__":
    main()
