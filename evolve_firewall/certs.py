"""Firewall certificates for Colony: One -- numpy reference (batched, fixed 64x64 shift matrices).

Input: observations x (B,16,8,8) in the mover's canonical view (any float dtype).
Every quantity is a fixed matrix product / elementwise op / reduction over the 64 squares,
like net.py Features, so it ports 1:1 to an ONNX graph.

Notation (per position):
  own[t], en[t]  t = 0..5 Ping, Daemon, Exploit, Trace, Ice, Kernel      (planes 0-5, 6-11)
  K = own[5], E = en[5]; L = locked (plane 12); occ = sum own + sum en; empty = 1 - occ - L
  a = round(2 * plane13) in {1,2};  mag = plane15;  u = plane14
  c = completed shrinks = #{28,48,60 <= sum L} ... (sum L in {0,28,48,60} -> 0,1,2,3)
  rho(s) = min(r, col, 7-r, 7-col)  ring index of square s (rotation invariant)
  R = [rho == c]  (the ring that locks next),  I = [rho > c]  (squares that survive it)
  SR = [u == 1]   (this is a shrink round: R locks at the end of the current round)
"""
from __future__ import annotations

import numpy as np

ORTH = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL8 = ORTH + DIAG
KNIGHT = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def shift(dr, dc):
    m = np.zeros((64, 64), np.float32)
    for p in range(64):
        r, c = divmod(p, 8)
        rr, cc = r + dr, c + dc
        if 0 <= rr < 8 and 0 <= cc < 8:
            m[p, rr * 8 + cc] = 1
    return m


S = {d: shift(*d) for d in ALL8}
ADJ = sum(S[d] for d in ALL8)
KN = sum(shift(*d) for d in KNIGHT)
_r = np.arange(64) // 8
_c = np.arange(64) % 8
RHO = np.minimum(np.minimum(_r, _c), np.minimum(7 - _r, 7 - _c)).astype(np.float32)


def rays(src, dirs, lo, hi, empty):
    """Squares at distance lo..hi from src along dirs, through empty intermediates."""
    out = np.zeros_like(src)
    for d in dirs:
        cur = src @ S[d]
        for k in range(1, hi + 1):
            if k >= lo:
                out = out + cur
            if k < hi:
                cur = (cur * empty) @ S[d]
    return out


def split(x):
    f = np.asarray(x, np.float32).reshape(-1, 16, 64)
    own = [f[:, i] for i in range(6)]
    en = [f[:, 6 + i] for i in range(6)]
    L = f[:, 12]
    a = np.rint(2 * f[:, 13, 0]).astype(np.int64)
    u = f[:, 14, 0]
    mag = f[:, 15, 0] > 0.5
    nl = L.sum(1)
    c = (nl > 1).astype(np.int64) + (nl > 30) + (nl > 50)
    R = (RHO[None, :] == c[:, None]).astype(np.float32)
    I = (RHO[None, :] > c[:, None]).astype(np.float32)
    return own, en, L, a, u, mag, c, R, I


def attack_maps(P, empty):
    """Exact attack squares of the pieces P[0..5] (one plane per type), rules.py semantics."""
    return [
        rays(P[0], DIAG, 1, 2, empty),
        P[1] @ ADJ + rays(P[1], ORTH, 2, 3, empty),
        P[2] @ KN,
        rays(P[3], DIAG, 2, 3, empty),
        rays(P[4], ALL8, 1, 2, empty),
        P[5] @ ADJ,
    ]


def relocations(P, Q, empty):
    """Squares each own piece P[t] can relocate to (moves to empty squares or move-captures of a
    non-Kernel enemy).  Q = enemy planes."""
    qnk = sum(Q[:5])
    land = empty + qnk
    return [
        rays(P[0], ORTH, 1, 2, empty) * empty + rays(P[0], DIAG, 1, 2, empty) * qnk,
        rays(P[1], ORTH, 1, 3, empty) * empty + rays(P[1], ORTH, 2, 3, empty) * qnk,
        (P[2] @ KN) * land,
        rays(P[3], ORTH, 1, 1, empty) * empty,
        rays(P[4], ALL8, 1, 2, empty) * land,
        (P[5] @ ADJ) * land,
    ]


def quiet_moves(P, empty):
    """Legal non-capturing moves (to empty squares) of the pieces P."""
    return [
        rays(P[0], ORTH, 1, 2, empty) * empty,
        rays(P[1], ORTH, 1, 3, empty) * empty,
        (P[2] @ KN) * empty,
        rays(P[3], ORTH, 1, 1, empty) * empty,
        rays(P[4], ALL8, 1, 2, empty) * empty,
        (P[5] @ ADJ) * empty,
    ]


def no_kill(P, Q, L, a):
    """Sufficient condition: side P (to move, `a` actions left) can NOT capture Q's Kernel this turn.
    a=1 exact:  no attack on the Kernel, or it has a guard.
    a=2:        >=2 guards, or (no attack now, no relocate-then-attack [R2], no one-blocker ray [R3x])."""
    occ = sum(P) + sum(Q)
    empty = np.clip(1.0 - occ - L, 0.0, 1.0)
    E = Q[5]
    qnk = sum(Q[:5])
    g = ((E @ ADJ) * qnk).sum(1)
    att = attack_maps(P, empty)
    hit = (sum(att) * E).sum(1) > 0
    # R2 (net.py Features): one own piece relocates to a square attacking the Kernel
    dest = relocations(P, Q, empty)
    guard_sq = (E @ ADJ) * qnk
    ok = (g == 0)[:, None] * 1.0 + (g == 1)[:, None] * guard_sq
    r2 = np.zeros(len(E))
    for t in range(6):
        e2 = np.clip(empty + P[t], 0.0, 1.0)
        rev = [
            lambda: rays(E, DIAG, 1, 2, e2),
            lambda: E @ ADJ + rays(E, ORTH, 2, 3, e2),
            lambda: E @ KN,
            lambda: rays(E, DIAG, 2, 3, e2),
            lambda: rays(E, ALL8, 1, 2, e2),
            lambda: E @ ADJ,
        ][t]()
        r2 = r2 + (dest[t] * rev * ok).sum(1)
    # R3x: a slider whose ray onto the Kernel has exactly one occupied, unlocked-only intermediate
    r3 = np.zeros(len(E))
    for t, dirs, ks in ((0, DIAG, (2,)), (1, ORTH, (2, 3)), (3, DIAG, (2, 3)), (4, ALL8, (2,))):
        for d in dirs:
            steps = [E]
            for _ in range(3):
                steps.append(steps[-1] @ S[d])
            for k in ks:
                n_occ = sum((steps[j] * occ).sum(1) for j in range(1, k))
                n_lck = sum((steps[j] * L).sum(1) for j in range(1, k))
                at = (steps[k] * P[t]).sum(1)
                r3 = r3 + at * (n_occ == 1) * (n_lck == 0)
    nk1 = (~hit) | (g >= 1)
    nk2 = (g >= 2) | ((~hit) & (r2 <= 0) & (r3 <= 0))
    return np.where(a >= 2, nk2, nk1)


def escape(P, Q, L, R, I, a):
    """Exact: side P's Kernel (on ring R) can end its turn on an I square within `a` actions.
    (Enemy-Kernel squares are counted as enterable -- conservative for the loss certificates.)"""
    occ = sum(P) + sum(Q)
    empty = np.clip(1.0 - occ - L, 0.0, 1.0)
    own_all = sum(P)
    nadj = P[5] @ ADJ
    e1 = (nadj * I * (1 - own_all)).sum(1) > 0
    dest = relocations(P, Q, empty)
    e2 = sum((nadj * I * P[t]).sum(1) * (dest[t].sum(1) > 0) for t in range(5)) > 0
    M = nadj * R * (1 - own_all)
    e3 = ((M @ ADJ) * I * (1 - own_all)).sum(1) > 0
    return np.where(a >= 2, e1 | e2 | e3, e1)


def magenta_doomed_after(own, en, L, R, I):
    """Board with Magenta (= en) to move, 2 actions, shrink after: Magenta certainly loses if its
    Kernel is on R, cannot escape in 2, cannot kill our Kernel in 2, and our Kernel is off R."""
    a2 = np.full(len(own[5]), 2)
    Kon = (own[5] * R).sum(1) > 0
    Eon = (en[5] * R).sum(1) > 0
    return Eon & ~Kon & ~escape(en, own, L, R, I, a2) & no_kill(en, own, L, a2)


def win_c_plans(own, en, L, R, I, a):
    """Cyan to move with a actions: some explicit plan leaves Magenta doomed.
    a=2: pass (quiet move and back -> same board), or a Kernel step to an empty square followed
         by a second Kernel step to an empty square (all 8x8 direction pairs, evaluated in parallel);
    a=1: a Kernel step to an empty square (8 directions)."""
    occ = sum(own) + sum(en)
    empty = np.clip(1.0 - occ - L, 0.0, 1.0)
    K = own[5]
    res = np.zeros(len(K), bool)
    qm = sum(quiet_moves(own, empty)).sum(1) > 0
    res |= (a == 2) & qm & magenta_doomed_after(own, en, L, R, I)

    def step(P, d):
        Kd = P[5] @ S[d]
        e = np.clip(1.0 - (sum(P) + sum(en)) - L, 0.0, 1.0)
        valid = (Kd * e).sum(1) > 0
        return P[:5] + [Kd], valid

    for d1 in ALL8:
        P1, v1 = step(own, d1)
        res |= (a == 1) & v1 & magenta_doomed_after(P1, en, L, R, I)
        for d2 in ALL8:
            P2, v2 = step(P1, d2)
            res |= (a == 2) & v1 & v2 & magenta_doomed_after(P2, en, L, R, I)
    return res


def kill_sound(P, Q, L, a):
    """Sufficient condition that side P (to move, `a` actions) DOES capture Q's Kernel this turn:
    net.py rules R1 (attack and guards < actions) and R2 (relocate-then-attack, a=2)."""
    occ = sum(P) + sum(Q)
    empty = np.clip(1.0 - occ - L, 0.0, 1.0)
    E = Q[5]
    qnk = sum(Q[:5])
    g = ((E @ ADJ) * qnk).sum(1)
    hit = (sum(attack_maps(P, empty)) * E).sum(1) > 0
    r1 = hit & (a - g >= 1)
    dest = relocations(P, Q, empty)
    guard_sq = (E @ ADJ) * qnk
    ok = (g == 0)[:, None] * 1.0 + (g == 1)[:, None] * guard_sq
    r2 = np.zeros(len(E))
    for t in range(6):
        e2 = np.clip(empty + P[t], 0.0, 1.0)
        rev = [lambda: rays(E, DIAG, 1, 2, e2), lambda: E @ ADJ + rays(E, ORTH, 2, 3, e2),
               lambda: E @ KN, lambda: rays(E, DIAG, 2, 3, e2), lambda: rays(E, ALL8, 1, 2, e2),
               lambda: E @ ADJ][t]()
        r2 = r2 + (dest[t] * rev * ok).sum(1)
    return r1 | ((r2 > 0) & (a >= 2))


_ROW = (np.arange(64) // 8).astype(np.float32)
_COL = (np.arange(64) % 8).astype(np.float32)


def cheb(A, B):
    """Chebyshev distance between the single squares of planes A and B."""
    return np.maximum(np.abs((A * _ROW).sum(1) - (B * _ROW).sum(1)),
                      np.abs((A * _COL).sum(1) - (B * _COL).sum(1)))


def kernel_squeeze(own, en, L, R, I, a):
    """Cyan to move whose only legal actions are Kernel steps (a=1: no non-Kernel piece has a legal
    action; a=2: no non-Kernel pieces at all), Kernels more than `a` apart. Every Kernel plan
    (8 single steps, or 8x8 step pairs) ends in a certain loss:
      bare Kernels at the end of Cyan's turn  -> lost unless dist > 2 and rho(K') > rho(E),
      else Kernel on the locking ring R         -> burns,
      else Magenta kills it in 2 (R1/R2 from Magenta's view, sound).
    Returns (applicable, all_plans_lose)."""
    occ = sum(own) + sum(en)
    empty = np.clip(1.0 - occ - L, 0.0, 1.0)
    open_ = 1.0 - L
    K, E = own[5], en[5]
    nk_own = sum(own[:5]).sum(1)
    en_all = sum(en)
    q = quiet_moves(own, empty)
    att = attack_maps(own, empty)
    nonk_actions = sum((q[t] + att[t] * en_all).sum(1) for t in range(5))
    applicable = np.where(a >= 2, nk_own == 0, nonk_actions == 0) & (cheb(K, E) > a)

    def outcome_loss(Kp, enp):
        nk_en = sum(enp[:5]).sum(1)
        bare = (nk_own == 0) & (nk_en == 0)
        rk = (Kp * RHO).sum(1)
        re = (enp[5] * RHO).sum(1)
        bare_loss = (cheb(Kp, enp[5]) <= 2) | (rk <= re)
        on_r = (Kp * R).sum(1) > 0
        killm = kill_sound(enp, own[:5] + [Kp], L, np.full(len(Kp), 2))
        return np.where(bare, bare_loss, on_r | killm)

    def steps(Kc, enc):
        for d in ALL8:
            Kn = Kc @ S[d]
            own_all = sum(own[:5]) + Kc
            legal = (Kn * open_ * (1 - own_all)).sum(1) > 0
            enn = [e * (1 - Kn) for e in enc[:5]] + [enc[5]]
            yield Kn, enn, legal

    all1 = np.ones(len(K), bool)
    any1 = np.zeros(len(K), bool)
    all2 = np.ones(len(K), bool)
    any2 = np.zeros(len(K), bool)
    for K1, en1, l1 in steps(K, en):
        any1 |= l1
        all1 &= ~l1 | outcome_loss(K1, en1)
        if np.any((a >= 2) & l1 & applicable):
            some2 = np.zeros(len(K), bool)
            for K2, en2, l2 in steps(K1, en1):
                some2 |= l2
                all2 &= ~(l1 & l2) | outcome_loss(K2, en2)
            all2 &= ~l1 | some2  # a stuck continuation is not certified
            any2 |= l1
    lose = np.where(a >= 2, all2 & any2, all1 & any1)
    return applicable, lose


def certificates(x):
    """Returns dict of boolean arrays (B,):
      LOSS_C  Cyan to move, certified loss      LOSS_M  Magenta to move, certified loss
      WIN_M   Magenta to move, certified win    WIN_C2  Cyan to move with 2 actions, certified win
      LOSS, WIN (unions)."""
    own, en, L, a, u, mag, c, R, I = split(x)
    occ = sum(own) + sum(en)
    empty = np.clip(1.0 - occ - L, 0.0, 1.0)
    SR = np.abs(u - 1.0) < 1e-3
    K, E = own[5], en[5]
    nk_own = sum(own[:5]).sum(1)
    nk_en = sum(en[:5]).sum(1)
    nonbare = (nk_own + nk_en) > 0
    Kon = (K * R).sum(1) > 0
    Eon = (E * R).sum(1) > 0
    esc_own = escape(own, en, L, R, I, a)
    nk_own_turn = no_kill(own, en, L, a)
    qm_own = sum(quiet_moves(own, empty)).sum(1) > 0
    a2 = np.full_like(a, 2)
    esc_en2 = escape(en, own, L, R, I, a2)
    nk_en2 = no_kill(en, own, L, a2)
    e_free = ((E @ ADJ) * empty).sum(1)

    doomed_own = SR & Kon & ~esc_own & nk_own_turn & nonbare
    safe_reply = (nk_own == 0) | (nk_own >= 3) | (e_free >= a + 1)
    LOSS_C = ~mag & doomed_own & safe_reply
    LOSS_M = mag & doomed_own & ~Eon
    WIN_M = mag & SR & Eon & nonbare & ((nk_en == 0) | (nk_en > a) | qm_own)
    WIN_C2 = ~mag & SR & (a == 2) & Eon & ~Kon & nonbare & qm_own & ~esc_en2 & nk_en2
    return dict(LOSS_C=LOSS_C, LOSS_M=LOSS_M, WIN_M=WIN_M, WIN_C2=WIN_C2,
                LOSS=LOSS_C | LOSS_M, WIN=WIN_M | WIN_C2,
                # sub-lemmas exposed for verification
                _esc=esc_own, _nokill=nk_own_turn, _quiet=qm_own, _Kon=Kon, _Eon=Eon, _SR=SR)
