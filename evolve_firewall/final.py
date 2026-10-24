"""Final firewall certificates for Colony: One (numpy reference, batched).

certify(x) with x (B,16,8,8) in the mover's canonical view returns boolean arrays (B,):
  WIN   the mover certainly wins  (brute-force minimax value +1 up to the shrink)
  LOSS  the mover certainly loses (value -1)
and the parts:
  LOSS_M  Magenta to move: own Kernel stuck on the locking ring, enemy Kernel off it      [tier 1]
  WIN_M   Magenta to move: Cyan's Kernel on the locking ring, a line avoiding bare-Kernel loss [tier 1]
  LOSS_C  Cyan to move: Kernel stuck on the ring (core, tier 1) or every Cyan action sequence ends
          in burn / sound Magenta kill / bare-Kernel loss (squeeze, enumerated: a=1 static 6x64
          options, a=2 gathered pairs)
  WIN_C   Cyan to move: a pass or Kernel-step plan leaves Magenta's Kernel doomed (rare)
Only shrink rounds (plane 14 == 1) are ever certified; positions with only Kernels are left to the
bare-Kernel endgame wrapper. tier1=True returns the pure fixed-shift-operation subset.
"""
from __future__ import annotations

import numpy as np

import certs as C
import certs6 as C6


def certify(x, tier1=False):
    x = np.asarray(x, np.float32)
    own, en, L, a, u, mag, c, R, I = C.split(x)
    occ = sum(own) + sum(en)
    empty = np.clip(1.0 - occ - L, 0.0, 1.0)
    SR = np.abs(u - 1.0) < 1e-3                       # shrink at the end of this round
    K, E = own[5], en[5]
    nk_own = sum(own[:5]).sum(1)
    nk_en = sum(en[:5]).sum(1)
    nonbare = (nk_own + nk_en) > 0
    Kon = (K * R).sum(1) > 0
    Eon = (E * R).sum(1) > 0
    NK = C.no_kill(own, en, L, a)                     # mover cannot capture the enemy Kernel this turn
    ESC = C6.escape_x(own, en, L, R, I, a)            # mover's Kernel can leave R this turn (exact | NK)
    e_free = ((E @ C.ADJ) * empty).sum(1)
    SAFE = (nk_own == 0) | (nk_own >= 3) | (e_free >= a + 1)   # Magenta has a non-losing reply

    doomed = SR & nonbare & Kon & ~ESC & NK
    LOSS_M = mag & doomed & ~Eon
    WIN_M = mag & SR & C6.win_m_x(own, en, L, R, a)
    LOSS_C = ~mag & doomed & SAFE
    WIN_C = np.zeros(len(x), bool)
    if not tier1:
        cy = np.where(~mag & SR & nonbare & ~LOSS_C)[0]
        if len(cy):
            P, Q = [p[cy] for p in own], [q[cy] for q in en]
            sq = np.zeros(len(cy), bool)
            a_ = a[cy]
            for k, fn in ((1, C6.cyan_squeeze_a1), (2, C6.cyan_squeeze_a2_kernel)):
                i = np.where(a_ == k)[0]
                if len(i):
                    sq[i] |= fn([p[i] for p in P], [q[i] for q in Q], L[cy][i], R[cy][i])
            i = np.where((a_ == 2) & ~sq)[0]
            if len(i):
                sq[i] |= C6.cyan_squeeze_a2_full([p[i] for p in P], [q[i] for q in Q], L[cy][i], R[cy][i])
            LOSS_C[cy] |= sq
        from candidates import win_c_x
        WIN_C = win_c_x(x)
    return dict(WIN=WIN_M | WIN_C, LOSS=LOSS_M | LOSS_C, WIN_M=WIN_M, WIN_C=WIN_C, LOSS_M=LOSS_M,
                LOSS_C=LOSS_C)
