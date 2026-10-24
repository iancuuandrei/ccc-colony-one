"""Cross-check the apply/undo oracle against the official ColonyOneEnv.step semantics.

Independent search: alpha-beta over env.clone() + env.step(Action) (the env's own turn change,
round counter, bare-Kernels rule and firewall). Values must equal oracle.solve on every position.
usage: python validate_oracle.py <n_selfplay> <n_synth>
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oracle import decode, horizon_for, solve  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "env"))
from colony_one import constants  # noqa: E402
from colony_one.board import Board  # noqa: E402
from colony_one.env import ColonyOneEnv  # noqa: E402


def env_from_obs(o):
    team, a, rnd, c = decode(o)
    env = ColonyOneEnv()
    env.board = Board.from_observation(np.asarray(o, np.float32), team)
    env.current_team = team
    env.actions_left = a
    env.round_number = rnd
    env.terminated = False
    env.winner = None
    return env, horizon_for(rnd)


def env_search(env, H, alpha, beta):
    team = env.current_team
    best = -2
    for ac in env.legal_actions():
        e2 = env.clone()
        _, _, term, _ = e2.step(ac)
        if term:
            v = 1 if e2.winner == team else -1
        elif e2.round_number > H:
            v = 0
        elif e2.current_team == team:
            v = env_search(e2, H, alpha, beta)
        else:
            v = -env_search(e2, H, -beta, -alpha)
        if v > best:
            best = v
            alpha = max(alpha, best)
            if alpha >= beta:
                break
    return best


def main():
    n_sp, n_sy = int(sys.argv[1]), int(sys.argv[2])
    from collect import unpack
    from synth import generate
    sp = unpack(np.load(os.path.join(HERE, "positions_selfplay.npz"))["packed"])
    rng = np.random.default_rng(7)
    sets = [sp[rng.choice(len(sp), n_sp, replace=False)], generate(n_sy, 99),
            generate(n_sy, 98, "boxed"), generate(n_sy, 97, "sparse")]
    mism = 0
    tot = 0
    t0 = time.time()
    counts = {}
    for obs in sets:
        for o in obs:
            env, H = env_from_obs(o)
            ve = env_search(env, H, -1, 1)
            vo = solve(o)["value"]
            tot += 1
            counts[ve] = counts.get(ve, 0) + 1
            if ve != vo:
                mism += 1
                print("MISMATCH", decode(o), ve, vo)
    print(f"checked {tot} positions, mismatches {mism}, env values {counts}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
