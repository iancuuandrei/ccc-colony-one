"""Synthetic stress positions near the firewall, encoded with the official encode_board.

Bias: Kernels on / next to the ring that locks next, other pieces crowded around the Kernels
(boxed Kernels, guards, blockers, one-blocker rays), few-piece endgames (bare-Kernel corner
cases), all four shrinks including the last 2x2 one (round 26).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "env"))

from colony_one import constants  # noqa: E402
from colony_one.board import Board  # noqa: E402
from colony_one.encoding import encode_board  # noqa: E402
from colony_one.pieces import Piece, PieceType  # noqa: E402

TYPES = list(PieceType)
NONK = [t for t in TYPES if t is not PieceType.KERNEL]


def rho(r, c):
    return min(r, c, 7 - r, 7 - c)


def make(rng, round_mode="shrink"):
    c = int(rng.choice(4, p=[0.35, 0.3, 0.25, 0.10]))
    rnd = constants.SHRINK_ROUNDS[c]
    if round_mode == "any":  # any round of the stage (lemma tests); "boxed": shrink round
        lo = 1 if c == 0 else constants.SHRINK_ROUNDS[c - 1] + 1
        rnd = int(rng.integers(lo, constants.SHRINK_ROUNDS[c] + 1))
    b = object.__new__(Board)
    b.grid = [[None] * 8 for _ in range(8)]
    b.locked = [[rho(r, col) < c for col in range(8)] for r in range(8)]
    open_sq = [(r, col) for r in range(8) for col in range(8) if rho(r, col) >= c]
    ring = [s for s in open_sq if rho(*s) == c]
    inner = [s for s in open_sq if rho(*s) > c]
    free = set(open_sq)

    def put(sq, piece):
        b.grid[sq[0]][sq[1]] = piece
        free.discard(sq)

    def pick(pool):
        pool = [s for s in pool if s in free]
        if not pool:
            pool = list(free)
        return pool[int(rng.integers(len(pool)))]

    kpos = {}
    for team in (constants.CYAN, constants.MAGENTA):
        p_ring = 0.65
        sq = pick(ring) if (rng.random() < p_ring or not inner) else pick(inner)
        put(sq, Piece(PieceType.KERNEL, team))
        kpos[team] = sq
    dens = rng.choice([0.15, 0.35, 0.6, 0.9])
    if round_mode == "boxed":
        # box one Kernel on the ring with its own pieces (inner neighbours first, then ring neighbours)
        boxed = constants.CYAN if rng.random() < 0.5 else constants.MAGENTA
        k = kpos[boxed]
        if rho(*k) != c:
            b.grid[k[0]][k[1]] = None
            free.add(k)
            k = pick(ring)
            put(k, Piece(PieceType.KERNEL, boxed))
            kpos[boxed] = k
        nb = [s for s in free if max(abs(s[0] - k[0]), abs(s[1] - k[1])) == 1]
        nb.sort(key=lambda s: (-rho(*s), rng.random()))
        types = list(NONK)
        rng.shuffle(types)
        for sq in nb:
            if not types or rng.random() < 0.08:
                continue
            put(sq, Piece(types.pop(), boxed))
        for t in types:  # leftovers somewhere near (often blocking the blockers)
            if rng.random() < 0.6 and free:
                near = [s for s in free if max(abs(s[0] - k[0]), abs(s[1] - k[1])) <= 2]
                put(pick(near) if near else pick(list(free)), Piece(t, boxed))
        other = constants.MAGENTA if boxed == constants.CYAN else constants.CYAN
        teams = (other,)
    else:
        teams = (constants.CYAN, constants.MAGENTA)
    for team in teams:
        if round_mode == "sparse":  # few-piece endgames: lone Kernels, squeezes, bare-Kernel edges
            dens = rng.choice([0.0, 0.1, 0.2, 0.35])
        for t in NONK:
            if rng.random() >= dens or not free:
                continue
            mode = rng.random()
            if mode < 0.45:
                k = kpos[team]
            elif mode < 0.8:
                k = kpos[constants.MAGENTA if team == constants.CYAN else constants.CYAN]
            else:
                k = None
            if k is not None:
                rad = 1 if rng.random() < 0.6 else 3
                near = [s for s in free if max(abs(s[0] - k[0]), abs(s[1] - k[1])) <= rad]
                sq = pick(near) if near else pick(list(free))
            else:
                sq = pick(list(free))
            put(sq, Piece(t, team))
    if b.only_kernels_remain():
        return None
    team = constants.CYAN if rng.random() < 0.5 else constants.MAGENTA
    a = 1 if rng.random() < 0.5 else 2
    return encode_board(b, team, a, rnd)


def generate(n, seed, round_mode="shrink"):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        o = make(rng, round_mode)
        if o is not None:
            out.append(o)
    return np.stack(out)
