"""Certificate generations (each: obs (N,16,8,8) float32 -> bool (N,))."""

from __future__ import annotations

import numpy as np

import cert as C
import geom as G


def _subset(mask, fn, *arrays):
    out = np.zeros(len(mask), bool)
    idx = np.flatnonzero(mask)
    if len(idx):
        out[idx] = fn(*[a[idx] for a in arrays])
    return out


def g1_konly(obs):
    C.LEVEL["ew"] = 1
    return _konly(obs)


def _konly(obs):
    own, en, locked, al, shrink_due, mmag = G.unpack(obs)
    open_ = ~locked
    konly = G.cnt(own[:, :5].any(1)) == 0
    return _subset(konly, C.kernel_tree_ok, own, en, open_, al, mmag, shrink_due)


def g2_konly_ew3(obs):
    C.LEVEL["ew"] = 3
    return _konly(obs)


def g3_konly_plus_oneply(obs):
    C.LEVEL["ew"] = 3
    own, en, locked, al, shrink_due, mmag = G.unpack(obs)
    open_ = ~locked
    konly = G.cnt(own[:, :5].any(1)) == 0
    a = _subset(konly, C.kernel_tree_ok, own, en, open_, al, mmag, shrink_due)
    b = _subset((al == 1) & ~konly, C.one_ply_ok, own, en, open_, mmag, shrink_due)
    return a | b


def g4_full(obs):
    C.LEVEL["ew"] = 3
    own, en, locked, al, shrink_due, mmag = G.unpack(obs)
    open_ = ~locked
    konly = G.cnt(own[:, :5].any(1)) == 0
    a = _subset(konly, C.kernel_tree_ok, own, en, open_, al, mmag, shrink_due)
    b = _subset((al == 1) & ~konly, C.one_ply_ok, own, en, open_, mmag, shrink_due)
    c = _subset((al == 2) & ~konly, C.two_ply_ok, own, en, open_, mmag, shrink_due)
    return a | b | c


def g6_cheap(obs):
    C.LEVEL["ew"] = 3
    return C.cert_loss(obs, two_ply=False)


def g7_final(obs):
    C.LEVEL["ew"] = 3
    return C.cert_loss(obs, two_ply=True)


GENS = {
    "g1_konly": (g1_konly,
                 "Mover has only its Kernel. Enumerate all Kernel lines of the remaining "
                 "actions (8 or 8x8, incl. attack-redirect on a guarded enemy Kernel); no "
                 "line captures the enemy Kernel, at least one line exists, and every line "
                 "ends in losing_final = bare-Kernel resolution lost | shrink-after-our-turn "
                 "kills our K (not theirs) | enemy R1/R2 kill on the (shrunk) board | "
                 "Cyan mover with K on the next ring and enemy K able to shuffle."),
    "g2_konly_ew3": (g2_konly_ew3,
                     "g1 with the enemy-kill test EW extended: R3e discovered attack (an enemy "
                     "blocker relocates off the slider-Kernel segment, K unguarded after) and "
                     "R4e (an enemy Daemon/Trace shot removes our only blocker of a slider line; "
                     "K unguarded after)."),
    "g3_konly_plus_oneply": (g3_konly_plus_oneply,
                             "g2, plus for actions_left == 1 with own non-Kernel pieces: enumerate all "
                             "76 fixed (piece type, offset) action slots exactly as rules.py; no slot "
                             "wins, some slot is legal, and every legal slot's successor satisfies "
                             "losing_final (EW level 3)."),
    "g4_full": (g4_full,
                "g3, plus for actions_left == 2 with own non-Kernel pieces: for every legal "
                "first action slot (none winning, at least one legal) the successor satisfies "
                "the actions_left == 1 one-ply certificate (76 x 76 slots)."),
    "g5_full_ebk": (g4_full,
                    "g4, with losing_final also accepting EBK: the enemy has only its Kernel, "
                    "we have <= 1 non-Kernel piece, and the enemy Kernel's two actions (capturing "
                    "that piece if any) end where the bare-Kernels resolution at the end of the "
                    "enemy's turn goes to the enemy (Chebyshev > 2 and smaller ring, or equal "
                    "ring with the enemy real Magenta)."),
    "g6_cheap": (g6_cheap,
                 "Port-friendly subset of g5 via cert.cert_loss(two_ply=False): actions_left == 1 "
                 "-> 76-slot one-ply; actions_left == 2 -> only Kernel-only movers (8x8 Kernel "
                 "tree). losing_final = bare-Kernels | shrink | EW level 3 | EBK."),
    "g7_final": (g7_final,
                 "Final cert.cert_loss(two_ply=True): actions_left == 1 -> 76-slot one-ply; "
                 "actions_left == 2 -> Kernel-only 8x8 tree, else 76x76 two-ply. Same logic as g5."),
    "g8_final": (g7_final,
                 "g7 with the Cyan-mover shrink clause relaxed: our Kernel on the next ring "
                 "at the end of our turn loses if ANY enemy piece has a quiet move to an empty "
                 "square (it can move there and back, so the enemy completes its turn and the "
                 "ring locks), instead of requiring a free square next to the enemy Kernel."),
}
