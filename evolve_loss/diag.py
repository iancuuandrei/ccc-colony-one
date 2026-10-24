"""Diagnose misses: for an al==1 lost position, list mover actions whose successor
losing_final rejects, with the enemy's winning line found by brute force."""

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


def enemy_line(board, ctx):
    km = bf.kpos(board, bf.M)
    for b1 in rules.legal_actions_for_team(board, bf.E):
        r1 = board.apply_action(b1)
        if bf.is_kill(r1):
            board.undo_action(r1)
            return (b1,)
        for b2 in rules.legal_actions_for_team(board, bf.E):
            r2 = board.apply_action(b2)
            k = bf.is_kill(r2)
            board.undo_action(r2)
            if k:
                board.undo_action(r1)
                return (b1, b2)
        board.undo_action(r1)
    return None


def main():
    z = np.load(sys.argv[1])
    flag = np.load(sys.argv[2])
    want_al = int(sys.argv[3])
    nshow = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    obs, code = z["obs"], z["code"]
    al = np.rint(obs[:, 13, 0, 0].astype(np.float32) * 2).astype(int)
    miss = np.flatnonzero((code == 1) & ~flag & (al == want_al))
    print("misses", len(miss))
    C.LEVEL["ew"] = 3
    for i in miss[:nshow]:
        o = obs[i].astype(np.float32)
        board = Board.from_observation(o, bf.M)
        alx, sd, mm = bf.obs_context(o)
        print("=" * 30, i, "al", alx, "shrink_due", sd, "mover_magenta", mm)
        print(bf.board_str(board))
        ctx = bf.Ctx(mm, sd)
        for a in rules.legal_actions_for_team(board, bf.M):
            rec = board.apply_action(a)
            p = encode_board(board, bf.M, 2, 1)[None]
            own, en, locked, _, _, _ = G.unpack(p)
            lf = C.losing_final(own, en, ~locked, np.array([mm]), np.array([sd]))[0]
            if alx == 1 and not lf:
                print(" action", a.from_pos, a.to_pos, a.kind, "-> losing_final False; enemy line:",
                      enemy_line(board, ctx))
                print("  " + bf.board_str(board).replace("\n", "\n  "))
            board.undo_action(rec)


if __name__ == "__main__":
    main()
