"""Batched board geometry on (N,8,8) bool masks with fixed shifts only.

Mirrors rules.py: rays stop at pieces and locked squares; attack geometry is
symmetric, so "squares from which type t attacks T" is a ray out of T.
"""

from __future__ import annotations

import numpy as np

PING, DAEMON, EXPLOIT, TRACE, ICE, KERNEL = range(6)
ORTH = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL8 = ORTH + DIAG
KNIGHT = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def sh(a, dr, dc):
    """out[..., r+dr, c+dc] = a[..., r, c] (zero fill)."""
    out = np.zeros_like(a)
    rs = slice(max(dr, 0), 8 + min(dr, 0)); rd = slice(max(-dr, 0), 8 + min(-dr, 0))
    cs = slice(max(dc, 0), 8 + min(dc, 0)); cd = slice(max(-dc, 0), 8 + min(-dc, 0))
    out[..., rs, cs] = a[..., rd, cd]
    return out


def ray_set(src, empty, open_, dirs, dmin, dmax):
    """Squares at distance dmin..dmax from src along dirs; intermediates empty
    (and unlocked); the endpoint must be unlocked."""
    out = np.zeros_like(src)
    for dr, dc in dirs:
        cur = src.copy()
        for d in range(1, dmax + 1):
            cur = sh(cur, dr, dc) & open_
            if d >= dmin:
                out |= cur
            cur = cur & empty
    return out


def dilate(a):
    out = np.zeros_like(a)
    for dr, dc in ALL8:
        out |= sh(a, dr, dc)
    return out


def anyb(a):
    return a.any(axis=(-2, -1))


def cnt(a):
    return a.sum(axis=(-2, -1))


def attackers_of(T, typ, occ, open_):
    """Squares from which a piece of type typ attacks square(s) T (T occupied)."""
    empty = ~occ & open_
    if typ == PING:
        return ray_set(T, empty, open_, DIAG, 1, 2)
    if typ == DAEMON:
        return ray_set(T, empty, open_, ORTH, 2, 3) | ray_set(T, empty, open_, ALL8, 1, 1)
    if typ == EXPLOIT:
        out = np.zeros_like(T)
        for dr, dc in KNIGHT:
            out |= sh(T, dr, dc) & open_
        return out
    if typ == TRACE:
        return ray_set(T, empty, open_, DIAG, 2, 3)
    if typ == ICE:
        return ray_set(T, empty, open_, ALL8, 1, 2)
    if typ == KERNEL:
        return ray_set(T, empty, open_, ALL8, 1, 1)
    raise ValueError(typ)


def attack_map(src, typ, occ, open_):
    """Squares attacked (capturable if an opponent stands there) by a piece of
    type typ on src. Same geometry as attackers_of (symmetric)."""
    return attackers_of(src, typ, occ, open_)


def relocations(src, typ, occ, capturable, open_):
    """Squares a piece of type typ on src can RELOCATE to: empty squares, or
    squares in `capturable` (opponent non-Kernel pieces) by a capturing move."""
    empty = ~occ & open_
    if typ == PING:
        return (ray_set(src, empty, open_, ORTH, 1, 2) & empty) | \
            (ray_set(src, empty, open_, DIAG, 1, 2) & capturable)
    if typ == DAEMON:
        return (ray_set(src, empty, open_, ORTH, 1, 3) & empty) | \
            (ray_set(src, empty, open_, ORTH, 2, 3) & capturable)
    if typ == EXPLOIT:
        out = np.zeros_like(src)
        for dr, dc in KNIGHT:
            out |= sh(src, dr, dc)
        return out & open_ & (empty | capturable)
    if typ == TRACE:
        return ray_set(src, empty, open_, ORTH, 1, 1) & empty
    if typ == ICE:
        return ray_set(src, empty, open_, ALL8, 1, 2) & (empty | capturable)
    if typ == KERNEL:
        return ray_set(src, empty, open_, ALL8, 1, 1) & (empty | capturable)
    raise ValueError(typ)


def unpack(obs):
    obs = np.asarray(obs)
    P = obs[:, :12] > 0.5
    own, en = P[:, :6], P[:, 6:12]
    locked = obs[:, 12] > 0.5
    al = np.rint(obs[:, 13, 0, 0].astype(np.float32) * 2).astype(int)
    shrink_due = obs[:, 14, 0, 0].astype(np.float32) > 0.999
    mover_magenta = obs[:, 15, 0, 0].astype(np.float32) > 0.5
    return own, en, locked, al, shrink_due, mover_magenta
