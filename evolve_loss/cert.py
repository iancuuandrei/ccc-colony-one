"""Forced-loss certificates for the side to move (numpy reference, fixed shift ops).

All functions are batched over N positions in the canonical view (planes 0-5
own = mover, 6-11 enemy). Building blocks:

  EW(own, en, open)      enemy (to move, 2 actions) has a verified kill:
                         R1e: some enemy piece attacks our Kernel square and our
                              guards g <= 1;
                         R2e: some enemy piece relocates (empty square or capture
                              of one of our non-Kernel pieces) to a square from
                              which it attacks our Kernel (origin vacated), and our
                              Kernel is unguarded after it (g == 0, or g == 1 and
                              the relocation captures that guard).
  step_K(state, d)       our Kernel's action toward direction d: move / capture /
                         attack on the enemy Kernel (win if unguarded, otherwise
                         the bodyguard by sacrifice order dies and K stays).
  losing_final(state)    end of our turn, exact env._end_turn order: bare-Kernels
                         resolution, firewall shrink after real Magenta's turn,
                         then EW on the (possibly shrunk) board; plus the shrink
                         that follows the enemy's turn when we are Cyan.
"""

from __future__ import annotations

import numpy as np

import geom as G
from geom import ALL8, anyb, cnt, dilate, sh

GUARD_ORDER = [G.PING, G.EXPLOIT, G.TRACE, G.DAEMON, G.ICE]
RING = np.array([[int(max(abs(r - 3.5), abs(c - 3.5))) for c in range(8)] for r in range(8)])


def dilate2(a):
    return dilate(dilate(a)) | a


def next_ring(open_):
    inner = open_.copy()
    for dr, dc in ALL8:
        inner &= sh(open_, dr, dc)
    return open_ & ~inner


def strict_ray(X, d, empty):
    """Squares at distance 1..3 from X along d, reached through and being empty."""
    out = np.zeros_like(X)
    cur = X
    for _ in range(3):
        cur = sh(cur, *d) & empty
        out |= cur
    return out


def between(K, P, empty):
    out = np.zeros_like(K)
    for d in ALL8:
        out |= strict_ray(K, d, empty) & strict_ray(P, (-d[0], -d[1]), empty)
    return out


LEVEL = {"ew": 3}


def EW(own, en, open_, level=None):
    """Enemy (to move, 2 actions) kill certificate vs our Kernel. level 1: R1e|R2e;
    2: + R3e (discovered attack: an enemy blocker leaves the line); 3: + R4e (an enemy
    shot removes our only blocker)."""
    level = LEVEL["ew"] if level is None else level
    K = own[:, 5]
    occ = own.any(1) | en.any(1)
    own_nk = own[:, :5].any(1)
    guards = dilate(K) & own_nk
    g = cnt(guards)
    r1 = np.zeros(len(K), bool)
    r2 = np.zeros(len(K), bool)
    gmask = (g == 0)[:, None, None] | ((g == 1)[:, None, None] & guards)
    D = []
    for t in range(6):
        src = en[:, t]
        r1 |= anyb(src & G.attackers_of(K, t, occ, open_))
        Dt = G.relocations(src, t, occ, own_nk, open_)
        D.append(Dt)
        A = G.attackers_of(K, t, occ & ~src, open_)
        r2 |= anyb(Dt & A & gmask)
    res = (r1 & (g <= 1)) | r2
    if level >= 2:
        # R3e: P (slider) attacks K once Q's square is empty; Q relocates off the P-K segment.
        for tp in (G.PING, G.DAEMON, G.TRACE, G.ICE):
            P = en[:, tp]
            for tq in range(6):
                if tq == tp:
                    continue
                Q = en[:, tq]
                occ_nq = occ & ~Q
                att = anyb(P & G.attackers_of(K, tp, occ_nq, open_))
                if not att.any():
                    continue
                btw = between(K, P, ~occ_nq & open_)
                res |= att & anyb(D[tq] & ~btw & gmask)
    if level >= 3:
        # R4e: our piece B is the only blocker of P's line to K; an enemy shooter S
        # (Daemon adjacent / Trace diagonal 2-3) removes B; K unguarded afterwards.
        empty = ~occ & open_
        for tp in (G.PING, G.DAEMON, G.TRACE, G.ICE):
            P = en[:, tp]
            if not P.any():
                continue
            for tb in range(5):
                B = own[:, tb]
                occ_nb = occ & ~B
                att = anyb(P & G.attackers_of(K, tp, occ_nb, open_))
                if not att.any():
                    continue
                isguard = anyb(B & guards)
                gok = (g == 0) | ((g == 1) & isguard)
                shot = (anyb(en[:, G.DAEMON] & dilate(B)) |
                        anyb(en[:, G.TRACE] & G.ray_set(B, empty, open_, G.DIAG, 2, 3)))
                res |= att & gok & shot
    return res


def step_K(own, en, open_, d):
    K = own[:, 5]
    s = sh(K, *d) & open_ & ~own.any(1)
    legal = anyb(s)
    EK = en[:, 5]
    isEK = anyb(s & EK)
    adjEK = dilate(EK)
    eg = [en[:, t] & adjEK for t in range(5)]
    ge = cnt(np.any(np.stack(eg, 1), 1))
    win = legal & isEK & (ge == 0)
    redirect = legal & isEK & (ge >= 1)
    move = legal & ~isEK
    rem = np.zeros_like(K)
    taken = np.zeros(len(K), bool)
    for t in GUARD_ORDER:
        present = anyb(eg[t]) & ~taken
        rem |= eg[t] & present[:, None, None]
        taken |= present
    own2 = own.copy()
    own2[:, 5] = np.where(move[:, None, None], s, K)
    kill = np.where(move[:, None, None], s, np.where(redirect[:, None, None], rem, False))
    en2 = en & ~kill[:, None]
    return own2, en2, legal, win


def EBK(own, en, open_, mmag):
    """Enemy (only its Kernel left) wins the bare-Kernels resolution at the end of its
    own turn: our side has <= 1 non-Kernel piece B, the enemy Kernel's 2 actions remove
    B (if any) and end on f with Chebyshev(f, K) > 2 and ring(f) < ring(K), or equal
    rings with the enemy being real Magenta (mover Cyan)."""
    K, EK = own[:, 5], en[:, 5]
    B = own[:, :5].any(1)
    nB = cnt(B)
    en_nk = cnt(en[:, :5].any(1))
    occ = own.any(1) | en.any(1)
    empty = ~occ & open_
    rk = (K * RING).sum((1, 2))[:, None, None]
    good = ~dilate2(K) & ((RING[None] < rk) | ((RING[None] == rk) & ~mmag[:, None, None])) & open_
    adjE = dilate(EK)
    # no own non-Kernel piece: EK walks two steps (second may return to its origin)
    r0 = anyb(dilate(adjE & empty) & (empty | EK) & good)
    # one piece B: capture first then step, or step then capture
    c1 = anyb(B & adjE) & anyb(dilate(B) & (empty | EK) & good)
    c2 = anyb(adjE & empty & dilate(B)) & anyb(B & good)
    return (en_nk == 0) & np.where(nB == 0, r0, (nB == 1) & (c1 | c2))


def losing_final(own, en, open_, mmag, shrink_due):
    K, EK = own[:, 5], en[:, 5]
    own_nk_n = cnt(own[:, :5].any(1))
    en_nk_n = cnt(en[:, :5].any(1))
    bare = (own_nk_n == 0) & (en_nk_n == 0)
    near2 = anyb(K & dilate2(EK))
    rK = (K * RING).sum((1, 2))
    rE = (EK * RING).sum((1, 2))
    bare_loss = near2 | (rE < rK) | ((rE == rK) & ~mmag)
    gate = mmag & shrink_due
    ring = next_ring(open_)
    K_on, EK_on = anyb(K & ring), anyb(EK & ring)
    # effective board for the enemy's turn (shrunk if the shrink fires after our turn)
    g3 = gate[:, None, None]
    open_e = np.where(g3, open_ & ~ring, open_)
    own_e = np.where(g3[:, None], own & ~ring[:, None], own)
    en_e = np.where(g3[:, None], en & ~ring[:, None], en)
    ew = EW(own_e, en_e, open_e) | EBK(own_e, en_e, open_e, mmag)
    occ = own.any(1) | en.any(1)
    # the enemy can play two quiet actions (a move to an empty square and back)
    empty = ~occ & open_
    nobody = np.zeros_like(empty)
    quiet = np.zeros(len(K), bool)
    for t in range(6):
        quiet |= anyb(G.relocations(en[:, t], t, occ, nobody, open_) & empty)
    cyan_ring_loss = ~mmag & shrink_due & K_on & quiet
    gate_res = ~EK_on & (K_on | ew)
    return np.where(bare, bare_loss, np.where(gate, gate_res, ew | cyan_ring_loss))


def mover_hits_and_guards(own, en, open_):
    occ = own.any(1) | en.any(1)
    EK = en[:, 5]
    hit_nk = np.zeros(len(EK), bool)
    for t in range(5):
        hit_nk |= anyb(own[:, t] & G.attackers_of(EK, t, occ, open_))
    hit_k = anyb(own[:, 5] & dilate(EK))
    ge = cnt(dilate(EK) & en[:, :5].any(1))
    return hit_nk, hit_k, ge


def kernel_tree_ok(own, en, open_, al, mmag, shrink_due):
    """For positions where every mover action is a Kernel action (no own non-Kernel
    piece can act -- here: no own non-Kernel piece at all): all Kernel lines of the
    remaining actions end in losing_final. Exact enumeration of 8 / 8x8 lines."""
    n = len(al)
    ok = np.ones(n, bool)
    any_legal = np.zeros(n, bool)
    for d1 in ALL8:
        o1, e1, l1, w1 = step_K(own, en, open_, d1)
        any_legal |= l1
        if True:
            lose1 = losing_final(o1, e1, open_, mmag, shrink_due)
            ok1_al1 = ~w1 & lose1
            ok1_al2 = ~w1.copy()
            any2 = np.zeros(n, bool)
            for d2 in ALL8:
                o2, e2, l2, w2 = step_K(o1, e1, open_, d2)
                any2 |= l2
                lose2 = losing_final(o2, e2, open_, mmag, shrink_due)
                ok1_al2 &= ~l2 | (~w2 & lose2)
            ok1_al2 &= any2
            ok1 = np.where(al == 1, ok1_al1, ok1_al2)
        ok &= ~l1 | ok1
    return ok & any_legal


# ---------------------------------------------------------------------------
# One-ply enumeration of ALL mover actions by fixed (piece type, offset) slots.
# Each slot is one candidate action; occupancy decides move / capture / shot /
# attack on the enemy Kernel, exactly as rules.py. 76 slots in total.
# ---------------------------------------------------------------------------
def _slots():
    S = []  # (type, direction, distance, empty_ok, enemy_kind) enemy_kind: 'move'|'shoot'|None
    for d in G.ORTH:
        for k in (1, 2):
            S.append((G.PING, d, k, True, None))
    for d in G.DIAG:
        for k in (1, 2):
            S.append((G.PING, d, k, False, "move"))
    for d in G.ORTH:
        S.append((G.DAEMON, d, 1, True, "shoot"))
        for k in (2, 3):
            S.append((G.DAEMON, d, k, True, "move"))
    for d in G.DIAG:
        S.append((G.DAEMON, d, 1, False, "shoot"))
    for d in G.KNIGHT:
        S.append((G.EXPLOIT, d, 0, True, "move"))  # jump, no path
    for d in G.ORTH:
        S.append((G.TRACE, d, 1, True, None))
    for d in G.DIAG:
        for k in (2, 3):
            S.append((G.TRACE, d, k, False, "shoot"))
    for d in G.ALL8:
        for k in (1, 2):
            S.append((G.ICE, d, k, True, "move"))
    for d in G.ALL8:
        S.append((G.KERNEL, d, 1, True, "move"))
    return S


SLOTS = _slots()


def guard_removal(en):
    EK = en[:, 5]
    adjEK = dilate(EK)
    eg = [en[:, t] & adjEK for t in range(5)]
    ge = cnt(np.any(np.stack(eg, 1), 1))
    rem = np.zeros_like(EK)
    taken = np.zeros(len(EK), bool)
    for t in GUARD_ORDER:
        present = anyb(eg[t]) & ~taken
        rem |= eg[t] & present[:, None, None]
        taken |= present
    return ge, rem


def apply_slot(own, en, open_, slot, ge=None, rem=None):
    """Returns own2, en2, legal, win for one action slot."""
    t, d, k, empty_ok, enemy_kind = slot
    if ge is None:
        ge, rem = guard_removal(en)
    occ = own.any(1) | en.any(1)
    empty = ~occ & open_
    enemy = en.any(1)
    src = own[:, t]
    if k == 0:
        tgt = sh(src, *d) & open_
    else:
        cur = src
        for _ in range(k - 1):
            cur = sh(cur, *d) & empty
        tgt = sh(cur, *d) & open_
    tgt_empty = tgt & empty if empty_ok else np.zeros_like(tgt)
    tgt_enemy = tgt & enemy if enemy_kind is not None else np.zeros_like(tgt)
    l_empty, l_enemy = anyb(tgt_empty), anyb(tgt_enemy)
    legal = l_empty | l_enemy
    isEK = anyb(tgt_enemy & en[:, 5])
    win = isEK & (ge == 0)
    redirect = isEK & (ge >= 1)
    relocate = l_empty | (l_enemy & ~isEK & (enemy_kind == "move"))
    own2 = own.copy()
    own2[:, t] = np.where(relocate[:, None, None], tgt, src)
    killed = np.where((l_enemy & ~isEK)[:, None, None], tgt_enemy,
                      np.where(redirect[:, None, None], rem, False))
    en2 = en & ~killed[:, None]
    return own2, en2, legal, win


def one_ply_ok(own, en, open_, mmag, shrink_due):
    """al == 1: no action wins, some action exists, every action ends in losing_final."""
    n = len(mmag)
    ge, rem = guard_removal(en)
    ok = np.ones(n, bool)
    any_legal = np.zeros(n, bool)
    for slot in SLOTS:
        o2, e2, l, w = apply_slot(own, en, open_, slot, ge, rem)
        if not l.any():
            continue
        any_legal |= l
        idx = np.flatnonzero(l & ok)
        bad = np.zeros(n, bool)
        bad[l & w] = True
        if len(idx):
            lf = losing_final(o2[idx], e2[idx], open_[idx], mmag[idx], shrink_due[idx])
            bad[idx[~lf]] = True
        ok &= ~bad
    return ok & any_legal


def two_ply_ok(own, en, open_, mmag, shrink_due):
    """al == 2: no first action wins, some first action exists, and after every legal
    first action the al == 1 certificate (one_ply_ok) holds. 76 x 76 fixed slots;
    the numpy reference prunes positions as soon as one first action escapes."""
    n = len(mmag)
    ge, rem = guard_removal(en)
    alive = np.ones(n, bool)
    any_legal = np.zeros(n, bool)
    for slot in SLOTS:
        o1, e1, l1, w1 = apply_slot(own, en, open_, slot, ge, rem)
        any_legal |= l1
        alive &= ~(l1 & w1)
        idx = np.flatnonzero(l1 & alive)
        if len(idx):
            ok1 = one_ply_ok(o1[idx], e1[idx], open_[idx], mmag[idx], shrink_due[idx])
            alive[idx[~ok1]] = False
    return alive & any_legal


# ---------------------------------------------------------------------------
# Dense (fixed-shape) reference: no subsetting, every slot evaluated for every
# position. Mathematically identical to the pruned functions above; this is the
# form to port (successor slots stacked into the batch axis, EW evaluated once on
# the expanded batch, AND/OR-reduced over the slot axis).
# ---------------------------------------------------------------------------
def one_ply_dense(own, en, open_, mmag, shrink_due):
    ge, rem = guard_removal(en)
    ok = np.ones(len(mmag), bool)
    any_legal = np.zeros(len(mmag), bool)
    for slot in SLOTS:
        o2, e2, l, w = apply_slot(own, en, open_, slot, ge, rem)
        lf = losing_final(o2, e2, open_, mmag, shrink_due)
        ok &= ~l | (~w & lf)
        any_legal |= l
    return ok & any_legal


def two_ply_dense(own, en, open_, mmag, shrink_due):
    ge, rem = guard_removal(en)
    ok = np.ones(len(mmag), bool)
    any_legal = np.zeros(len(mmag), bool)
    for slot in SLOTS:
        o1, e1, l1, w1 = apply_slot(own, en, open_, slot, ge, rem)
        ok &= ~l1 | (~w1 & one_ply_dense(o1, e1, open_, mmag, shrink_due))
        any_legal |= l1
    return ok & any_legal


def cert_loss(obs, two_ply=True, dense=False):
    """Forced-loss certificate for the side to move. obs: (N,16,8,8) canonical
    observations. True => every line of the mover's remaining actions lets the enemy
    win by the end of its next turn (verified 0 violations vs brute force).
    two_ply=False drops the 76x76 part (actions_left == 2 with own non-Kernel pieces
    is then only certified when the mover has no such piece, via the 8x8 Kernel tree).
    Assumes at most one piece per type per team (true for this game)."""
    own, en, locked, al, shrink_due, mmag = G.unpack(np.asarray(obs, np.float32))
    open_ = ~locked
    konly = cnt(own[:, :5].any(1)) == 0
    n = len(al)
    out = np.zeros(n, bool)
    if dense:
        op = one_ply_dense(own, en, open_, mmag, shrink_due)
        kt = kernel_tree_ok(own, en, open_, np.full(n, 2), mmag, shrink_due)
        tp = two_ply_dense(own, en, open_, mmag, shrink_due) if two_ply else np.zeros(n, bool)
        return np.where(al == 1, op, np.where(konly, kt, tp))

    def sub(mask, fn, *extra):
        idx = np.flatnonzero(mask)
        if len(idx):
            out[idx] = fn(own[idx], en[idx], open_[idx], *[e[idx] for e in extra])

    sub(al == 1, lambda o, e, op_, m, s: one_ply_ok(o, e, op_, m, s), mmag, shrink_due)
    sub((al == 2) & konly, lambda o, e, op_, a, m, s: kernel_tree_ok(o, e, op_, a, m, s),
        al, mmag, shrink_due)
    if two_ply:
        sub((al == 2) & ~konly, lambda o, e, op_, m, s: two_ply_ok(o, e, op_, m, s), mmag, shrink_due)
    return out
