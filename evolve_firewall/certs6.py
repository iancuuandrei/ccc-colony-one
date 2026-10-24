"""Generation-6 primitives (numpy reference, batched, fixed shift matrices from certs.py).

The option enumerations (cyan_single_actions) gather only the legal (position, action) pairs for
speed; the static equivalent evaluates all 6 types x 64 targets with the same masks.
"""
from __future__ import annotations

import numpy as np

from certs import (ADJ, ALL8, DIAG, KN, ORTH, RHO, attack_maps, cheb, kill_sound, quiet_moves, rays,
                   relocations)

GUARD_ORDER = (0, 2, 3, 1, 4)  # sacrifice order Ping, Exploit, Trace, Daemon, Ice (plane indices)


def escape_x(P, Q, L, R, I, a):
    """escape() without the enemy-Kernel squares: under no-kill (NK) an action onto the enemy Kernel
    never moves our Kernel (a guard dies, the attacker stays), so this is exact given NK."""
    occ = sum(P) + sum(Q)
    empty = np.clip(1.0 - occ - L, 0.0, 1.0)
    enterable = 1.0 - sum(P) - Q[5]
    nadj = P[5] @ ADJ
    e1 = (nadj * I * enterable).sum(1) > 0
    dest = relocations(P, Q, empty)
    e2 = sum((nadj * I * P[t]).sum(1) * (dest[t].sum(1) > 0) for t in range(5)) > 0
    M = nadj * R * enterable
    e3 = ((M @ ADJ) * I * enterable).sum(1) > 0
    return np.where(a >= 2, e1 | e2 | e3, e1)


def remove_guard(Q):
    """Q's pieces after an attack on Q's guarded Kernel: the first adjacent guard in sacrifice order dies."""
    adj = Q[5] @ ADJ
    out = list(Q)
    done = np.zeros(len(Q[5]), bool)
    for t in GUARD_ORDER:
        here = (adj * Q[t]).sum(1) > 0
        hit = here & ~done
        out[t] = Q[t] * (1 - adj * hit[:, None])
        done |= here
    return out


def after_cyan_loss(P, Q, L, R):
    """Cyan (P) has just ended its turn on this board; Magenta (Q) has 2 actions, then ring R locks.
    Sufficient for a certain Cyan loss:
      bare Kernels now            -> Cyan (acting) loses unless dist > 2 and rho(K) > rho(E);
      Magenta kill (R1/R2, sound) -> loss;
      Cyan Kernel on R and Magenta has a non-losing reply (Cyan non-Kernels not in {1,2}, or the
      Magenta Kernel has an empty neighbour to step to and back) -> it burns: loss."""
    occ = sum(P) + sum(Q)
    empty = np.clip(1.0 - occ - L, 0.0, 1.0)
    nkp = sum(P[:5]).sum(1)
    nkq = sum(Q[:5]).sum(1)
    bare = (nkp == 0) & (nkq == 0)
    rk = (P[5] * RHO).sum(1)
    re = (Q[5] * RHO).sum(1)
    bare_loss = (cheb(P[5], Q[5]) <= 2) | (rk <= re)
    killm = kill_sound(Q, P, L, np.full(len(nkp), 2))
    on_r = (P[5] * R).sum(1) > 0
    safe = (nkp == 0) | (nkp >= 3) | (((Q[5] @ ADJ) * empty).sum(1) >= 1)
    return np.where(bare, bare_loss, killm | (on_r & safe))


def action_targets(P, Q, L):
    """Per own piece type: (move-to-empty, move-capture, shoot, attack-on-enemy-Kernel) target masks.
    Together they are exactly the legal actions of rules.py."""
    occ = sum(P) + sum(Q)
    empty = np.clip(1.0 - occ - L, 0.0, 1.0)
    qnk = sum(Q[:5])
    z = np.zeros_like(P[0])
    quiet = quiet_moves(P, empty)
    cap = [rays(P[0], DIAG, 1, 2, empty) * qnk, rays(P[1], ORTH, 2, 3, empty) * qnk, (P[2] @ KN) * qnk,
           z, rays(P[4], ALL8, 1, 2, empty) * qnk, (P[5] @ ADJ) * qnk]
    shot = [z, (P[1] @ ADJ) * qnk, z, rays(P[3], DIAG, 2, 3, empty) * qnk, z, z]
    att = attack_maps(P, empty)
    katt = [att[t] * Q[5] for t in range(6)]
    return quiet, cap, shot, katt


_EYE = np.eye(64, dtype=np.float32)


def single_actions(P, Q, L):
    """Every legal single action of side P, gathered: (row, P', Q', kills) where kills marks an
    action that captures Q's (unguarded) Kernel."""
    quiet, cap, shot, katt = action_targets(P, Q, L)
    g = ((Q[5] @ ADJ) * sum(Q[:5])).sum(1)
    rows, Ps, Qs, kills = [], [], [], []
    for t in range(6):
        for kind, M in (("move", quiet[t] + cap[t]), ("shot", shot[t]), ("katt", katt[t])):
            b, s = np.nonzero(M > 0)
            if len(b) == 0:
                continue
            onehot = _EYE[s]
            Pn = [p[b] for p in P]
            Qn = [q[b] for q in Q]
            if kind == "move":
                Pn[t] = onehot  # one piece per type and team
                Qn = [q * (1 - onehot) for q in Qn]
                k = np.zeros(len(b), bool)
            elif kind == "shot":
                Qn = [q * (1 - onehot) for q in Qn]
                k = np.zeros(len(b), bool)
            else:
                k = g[b] == 0
                Qn = remove_guard(Qn)
            rows.append(b); Ps.append(Pn); Qs.append(Qn); kills.append(k)
    if not rows:
        return np.zeros(0, int), None, None, np.zeros(0, bool)
    rows = np.concatenate(rows)
    Pc = [np.concatenate([p[t] for p in Ps]) for t in range(6)]
    Qc = [np.concatenate([q[t] for q in Qs]) for t in range(6)]
    return rows, Pc, Qc, np.concatenate(kills)


def all_options_lose(n, rows, loss):
    """Per position: at least one option and every option loses."""
    bad = np.zeros(n, bool)
    np.logical_or.at(bad, rows, ~loss)
    has = np.zeros(n, bool)
    has[rows] = True
    return has & ~bad


def cyan_squeeze_a1(P, Q, L, R):
    """Cyan to move with ONE action: every legal action leads to a certain loss (after_cyan_loss)
    and none captures Magenta's Kernel."""
    n = len(P[5])
    rows, Pc, Qc, kills = single_actions(P, Q, L)
    if len(rows) == 0:
        return np.zeros(n, bool)
    loss = ~kills & after_cyan_loss(Pc, Qc, L[rows], R[rows])
    return all_options_lose(n, rows, loss)


def cyan_squeeze_a2_kernel(P, Q, L, R):
    """Cyan to move with TWO actions whose only legal actions are Kernel actions, now and after any
    Kernel action (no non-Kernel action even with the Kernel's square vacated): every Kernel plan
    (step/capture/attack twice) ends in a certain loss and none captures Magenta's Kernel."""
    n = len(P[5])
    occ = sum(P) + sum(Q)
    # superset of the empty squares after any first Kernel action: the Kernel's square vacated and
    # every guard of Magenta's Kernel removed (a redirected attack kills one of them)
    guards = (Q[5] @ ADJ) * sum(Q[:5])
    empty_x = np.clip(1.0 - (occ - P[5] - guards) - L, 0.0, 1.0)
    q = quiet_moves(P, empty_x)
    att = attack_maps(P, empty_x)
    qall = sum(Q)
    nonk = sum((q[t] + att[t] * qall).sum(1) for t in range(5))
    idx = np.where(nonk == 0)[0]
    out = np.zeros(n, bool)
    if len(idx) == 0:
        return out
    P0 = [p[idx] for p in P]; Q0 = [x[idx] for x in Q]; L0 = L[idx]; R0 = R[idx]

    def kernel_actions(Pp, Qp, Lp):
        z = np.zeros_like(Pp[0])
        # non-Kernel pieces still block: pass them as locked-free occupancy via the enemy-free mask
        rows, Pc, Qc, kills = single_actions([z] * 5 + [Pp[5]], Qp, Lp + sum(Pp[:5]))
        if len(rows):
            Pc = [Pp[t][rows] for t in range(5)] + [Pc[5]]
        return rows, Pc, Qc, kills

    r1, P1, Q1, k1 = kernel_actions(P0, Q0, L0)
    if len(r1) == 0:
        return out
    L1, R1 = L0[r1], R0[r1]
    r2, P2, Q2, k2 = kernel_actions(P1, Q1, L1)
    if len(r2):
        loss2 = ~k2 & after_cyan_loss(P2, Q2, L1[r2], R1[r2])
    else:
        loss2 = np.zeros(0, bool)
    line_ok = ~k1 & all_options_lose(len(r1), r2, loss2)  # a first action with no continuation fails
    out[idx] = all_options_lose(len(idx), r1, line_ok)
    return out


def win_m_x(P, Q, L, R, a):
    """Magenta (P) to move in the shrink round, Cyan's Kernel on R: Magenta certainly wins if it has
    a line that does not end in bare Kernels within distance 2 (its only way to lose):
      Cyan non-Kernels 0 or > a | a quiet move (and back) | a sound kill (R1/R2) |
      a=2: a move-capture by Daemon/Exploit/Ice/Kernel and the move back, if Cyan has >= 2
           non-Kernels or the Kernels are > 2 apart (the Kernel ends where it started) |
      a=1: Kernels > 2 apart and any non-Kernel action."""
    nkq = sum(Q[:5]).sum(1)
    nkp = sum(P[:5]).sum(1)
    Eon = (Q[5] * R).sum(1) > 0
    quiet, cap, shot, katt = action_targets(P, Q, L)
    qm = sum(quiet).sum(1) > 0
    ks = kill_sound(P, Q, L, a)
    cr = sum(cap[t].sum(1) for t in (1, 2, 4, 5)) > 0
    far = cheb(P[5], Q[5]) > 2
    nonk_any = sum((quiet[t] + cap[t] + shot[t] + katt[t]).sum(1) for t in range(5)) > 0
    line = (nkq == 0) | (nkq > a) | qm | ks | ((a >= 2) & cr & ((nkq >= 2) | far)) | ((a == 1) & far & nonk_any)
    return Eon & ((nkp + nkq) > 0) & line


def _subset(Xs, idx):
    return [x[idx] for x in Xs]


def kernel_plans_lose(P, Q, L, R):
    """Necessary condition for the full a=2 squeeze: every pair of Kernel actions (other own pieces
    standing still, as blockers) ends in a certain loss."""
    n = len(P[5])
    out = np.zeros(n, bool)
    z = np.zeros_like(P[0])

    def kernel_actions(Pp, Qp, Lp):
        zz = np.zeros_like(Pp[0])
        rows, Pc, Qc, kills = single_actions([zz] * 5 + [Pp[5]], Qp, Lp + sum(Pp[:5]))
        if len(rows):
            Pc = [Pp[t][rows] for t in range(5)] + [Pc[5]]
        return rows, Pc, Qc, kills

    r1, P1, Q1, k1 = kernel_actions(P, Q, L)
    if len(r1) == 0:
        return out
    L1, R1 = L[r1], R[r1]
    r2, P2, Q2, k2 = kernel_actions(P1, Q1, L1)
    loss2 = (~k2 & after_cyan_loss(P2, Q2, L1[r2], R1[r2])) if len(r2) else np.zeros(0, bool)
    line_ok = ~k1 & all_options_lose(len(r1), r2, loss2)
    return all_options_lose(n, r1, line_ok)


def cyan_squeeze_a2_full(P, Q, L, R, chunk=128):
    """Cyan to move with TWO actions: every legal pair of actions (all pieces, all action kinds, the
    second generated on the board after the first) ends in a certain loss and never captures
    Magenta's Kernel. Prefiltered by kernel_plans_lose (a necessary condition)."""
    n = len(P[5])
    out = np.zeros(n, bool)
    cand = np.where(kernel_plans_lose(P, Q, L, R))[0]
    for i in range(0, len(cand), chunk):
        idx = cand[i:i + chunk]
        P0, Q0, L0, R0 = _subset(P, idx), _subset(Q, idx), L[idx], R[idx]
        r1, P1, Q1, k1 = single_actions(P0, Q0, L0)
        if len(r1) == 0:
            continue
        L1, R1 = L0[r1], R0[r1]
        r2, P2, Q2, k2 = single_actions(P1, Q1, L1)
        loss2 = (~k2 & after_cyan_loss(P2, Q2, L1[r2], R1[r2])) if len(r2) else np.zeros(0, bool)
        line_ok = ~k1 & all_options_lose(len(r1), r2, loss2)
        out[idx] = all_options_lose(len(idx), r1, line_ok)
    return out
