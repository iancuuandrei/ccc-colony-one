"""Certificate generations (proposals). Each gen(x) -> {name: (claim, flags)}; claim = +1 win / -1 loss
from the mover's perspective. Built from the primitives in certs.py."""
from __future__ import annotations

import numpy as np

import certs as C
import certs6 as C6


_CACHE = {}


def base(x):
    """C.certificates(x), memoised per chunk (evaluate.py calls every generation on the same chunk)."""
    k = id(x)
    if k not in _CACHE:
        _CACHE.clear()
        _CACHE[k] = C.certificates(x)
    return _CACHE[k]


def on_subset(mask, fn, x):
    """Evaluate a heavy per-row predicate only where `mask` holds (exactly equivalent, faster)."""
    out = np.zeros(len(x), bool)
    idx = np.where(mask)[0]
    if len(idx):
        out[idx] = fn(x[idx])
    return out


def _common(x):
    own, en, L, a, u, mag, c, R, I = C.split(x)
    occ = sum(own) + sum(en)
    empty = np.clip(1.0 - occ - L, 0.0, 1.0)
    SR = np.abs(u - 1.0) < 1e-3
    K, E = own[5], en[5]
    Kon = (K * R).sum(1) > 0
    Eon = (E * R).sum(1) > 0
    return own, en, L, a, u, mag, c, R, I, empty, SR, K, E, Kon, Eon


def r1r2(own, en, L, a):
    """net.py rules R1 / R2 only (own-turn Kernel kill detector, known to miss discovered attacks)."""
    occ = sum(own) + sum(en)
    empty = np.clip(1.0 - occ - L, 0.0, 1.0)
    E = en[5]
    qnk = sum(en[:5])
    g = ((E @ C.ADJ) * qnk).sum(1)
    hit = (sum(C.attack_maps(own, empty)) * E).sum(1) > 0
    r1 = hit & (a - g >= 1)
    dest = C.relocations(own, en, empty)
    guard_sq = (E @ C.ADJ) * qnk
    ok = (g == 0)[:, None] * 1.0 + (g == 1)[:, None] * guard_sq
    r2 = np.zeros(len(E))
    for t in range(6):
        e2 = np.clip(empty + own[t], 0.0, 1.0)
        rev = [lambda: C.rays(E, C.DIAG, 1, 2, e2), lambda: E @ C.ADJ + C.rays(E, C.ORTH, 2, 3, e2),
               lambda: E @ C.KN, lambda: C.rays(E, C.DIAG, 2, 3, e2), lambda: C.rays(E, C.ALL8, 1, 2, e2),
               lambda: E @ C.ADJ][t]()
        r2 = r2 + (dest[t] * rev * ok).sum(1)
    return r1 | ((r2 > 0) & (a == 2))


def gen0(x):
    """Naive sketch: 'own Kernel on the locking ring with no free inner neighbour, no R1/R2 kill,
    enemy Kernel off the ring -> loss'; 'enemy Kernel on the ring, ours off -> win'. Colour-blind."""
    own, en, L, a, u, mag, c, R, I, empty, SR, K, E, Kon, Eon = _common(x)
    own_all = sum(own)
    e1 = ((K @ C.ADJ) * I * (1 - own_all)).sum(1) > 0
    loss = SR & Kon & ~e1 & ~r1r2(own, en, L, a) & ~Eon
    win = SR & Eon & ~Kon
    return {"LOSS": (-1, loss), "WIN": (1, win)}


def gen1(x):
    """+ exact escape with the actions left (E1; E2 vacate-then-step; E3 ring-step-then-in),
    + colour asymmetry: shrink after Magenta's turn, both burning -> Magenta.
      LOSS_C ignores where Magenta's Kernel is; WIN only for Magenta to move (Cyan gets no turn)."""
    own, en, L, a, u, mag, c, R, I, empty, SR, K, E, Kon, Eon = _common(x)
    esc = C.escape(own, en, L, R, I, a)
    nk = ~r1r2(own, en, L, a)
    loss_c = ~mag & SR & Kon & ~esc & nk
    loss_m = mag & SR & Kon & ~esc & nk & ~Eon
    win_m = mag & SR & Eon
    return {"LOSS_C": (-1, loss_c), "LOSS_M": (-1, loss_m), "WIN_M": (1, win_m)}


def gen2(x):
    """+ sound own-turn no-kill (adds R3x one-blocker rays, >=2 guards), + bare-Kernel safety:
      LOSS_C needs a non-losing Magenta reply (Cyan non-Kernels not in {1,2}, or Magenta Kernel keeps
      >= a+1 empty neighbours -> step out and back), WIN_M needs a capture-free line (quiet move, or
      Cyan non-Kernels not in [1,a]); positions with only Kernels are left to the endgame wrapper."""
    d = base(x)
    return {"LOSS_C": (-1, d["LOSS_C"]), "LOSS_M": (-1, d["LOSS_M"]), "WIN_M": (1, d["WIN_M"])}


def gen3(x):
    """+ WIN_C2: Cyan to move with 2 actions 'passes' (a quiet move and back restores the board), then
    Magenta's Kernel is doomed exactly like LOSS_M seen from Magenta (no escape in 2, no kill in 2,
    Cyan's Kernel off the ring)."""
    d = base(x)
    return {"LOSS_C": (-1, d["LOSS_C"]), "LOSS_M": (-1, d["LOSS_M"]), "WIN_M": (1, d["WIN_M"]),
            "WIN_C2": (1, d["WIN_C2"])}


def gen4(x):
    """+ WIN_C: Cyan to move (1 or 2 actions) wins when an explicit plan -- pass (2 actions), one
    Kernel step (1 action) or two Kernel steps (2 actions) onto empty squares -- leaves Magenta's
    Kernel doomed (on the ring, no escape in 2, no kill in 2, Cyan's Kernel off the ring)."""
    d = base(x)
    return {"LOSS_C": (-1, d["LOSS_C"]), "LOSS_M": (-1, d["LOSS_M"]), "WIN_M": (1, d["WIN_M"]),
            "WIN_C": (1, win_c(x))}


def win_c(x):
    own, en, L, a, u, mag, c, R, I, empty, SR, K, E, Kon, Eon = _common(x)
    nonbare = (sum(own[:5]).sum(1) + sum(en[:5]).sum(1)) > 0
    pre = ~mag & SR & nonbare & Eon  # necessary for every plan

    def heavy(xs):
        o, e, Ls, a_, u_, m_, c_, Rs, Is = C.split(xs)
        return C.win_c_plans(o, e, Ls, Rs, Is, a_)
    return on_subset(pre, heavy, x)


def loss_ck(x):
    own, en, L, a, u, mag, c, R, I, empty, SR, K, E, Kon, Eon = _common(x)
    nk_own = sum(own[:5]).sum(1)
    nonbare = (nk_own + sum(en[:5]).sum(1)) > 0
    e_free = ((E @ C.ADJ) * empty).sum(1)
    safe_reply = (nk_own == 0) | (nk_own >= 3) | (e_free >= a + 1)
    pre = ~mag & SR & nonbare & safe_reply

    def heavy(xs):
        o, e, Ls, a_, u_, m_, c_, Rs, Is = C.split(xs)
        app, lose = C.kernel_squeeze(o, e, Ls, Rs, Is, a_)
        return app & lose
    return on_subset(pre, heavy, x)


def gen5(x):
    """+ LOSS_CK (Kernel squeeze): Cyan to move whose only legal actions are Kernel steps; every
    Kernel plan ends on the locking ring (burns), or in a bare-Kernel loss, or on a square where
    Magenta's sound kill rules R1/R2 fire -> certified loss (firewall + tactics)."""
    d = base(x)
    return {"LOSS_C": (-1, d["LOSS_C"]), "LOSS_M": (-1, d["LOSS_M"]), "WIN_M": (1, d["WIN_M"]),
            "WIN_C": (1, win_c(x)), "LOSS_CK": (-1, loss_ck(x))}


def magenta_doomed_x(P, Q, L, R, I):
    a2 = np.full(len(P[5]), 2)
    Kon = (P[5] * R).sum(1) > 0
    Eon = (Q[5] * R).sum(1) > 0
    return Eon & ~Kon & ~C6.escape_x(Q, P, L, R, I, a2) & C.no_kill(Q, P, L, a2)


def win_c_x(x):
    """Cyan plans as in gen4 (pass / Kernel step / two Kernel steps), Magenta doom with escape_x."""
    own, en, L, a, u, mag, c, R, I, empty, SR, K, E, Kon, Eon = _common(x)
    nonbare = (sum(own[:5]).sum(1) + sum(en[:5]).sum(1)) > 0
    pre = ~mag & SR & nonbare & Eon

    def heavy(xs):
        P, Q, Ls, a_, u_, m_, c_, Rs, Is = C.split(xs)
        occ = sum(P) + sum(Q)
        emp = np.clip(1.0 - occ - Ls, 0.0, 1.0)
        res = (a_ == 2) & (sum(C.quiet_moves(P, emp)).sum(1) > 0) & magenta_doomed_x(P, Q, Ls, Rs, Is)

        def step(PP, d):
            Kd = PP[5] @ C.S[d]
            e = np.clip(1.0 - (sum(PP) + sum(Q)) - Ls, 0.0, 1.0)
            return PP[:5] + [Kd], (Kd * e).sum(1) > 0
        for d1 in C.ALL8:
            P1, v1 = step(P, d1)
            res |= (a_ == 1) & v1 & magenta_doomed_x(P1, Q, Ls, Rs, Is)
            for d2 in C.ALL8:
                P2, v2 = step(P1, d2)
                res |= (a_ == 2) & v1 & v2 & magenta_doomed_x(P2, Q, Ls, Rs, Is)
        return res
    return on_subset(pre, heavy, x)


def gen6(x):
    """Exactness pass. escape_x: under no-kill an action onto the enemy Kernel never moves ours
    (exact escape). LOSS_C6 = LOSS_C(escape_x) | a=1 full squeeze (every legal Cyan action -> bare
    loss, sound Magenta kill R1/R2, or Kernel burns with a non-losing Magenta reply) | a=2 Kernel-only
    squeeze with redirected attacks (guard removal in sacrifice order). LOSS_M6 with escape_x.
    WIN_M6: any Magenta line avoiding bare Kernels within distance 2 (quiet, capture-and-return,
    sound kill, count, far Kernels). WIN_C6: plans with escape_x."""
    own, en, L, a, u, mag, c, R, I, empty, SR, K, E, Kon, Eon = _common(x)
    nk_own = sum(own[:5]).sum(1)
    nonbare = (nk_own + sum(en[:5]).sum(1)) > 0
    NK = C.no_kill(own, en, L, a)
    esc = C6.escape_x(own, en, L, R, I, a)
    e_free = ((E @ C.ADJ) * empty).sum(1)
    safe_reply = (nk_own == 0) | (nk_own >= 3) | (e_free >= a + 1)
    loss_c = ~mag & SR & nonbare & Kon & ~esc & NK & safe_reply
    pre = ~mag & SR & nonbare & ~loss_c

    def heavy(xs):
        P, Q, Ls, a_, u_, m_, c_, Rs, Is = C.split(xs)
        out = np.zeros(len(xs), bool)
        i1 = np.where(a_ == 1)[0]
        if len(i1):
            out[i1] = C6.cyan_squeeze_a1([p[i1] for p in P], [q[i1] for q in Q], Ls[i1], Rs[i1])
        i2 = np.where(a_ == 2)[0]
        if len(i2):
            out[i2] = C6.cyan_squeeze_a2_kernel([p[i2] for p in P], [q[i2] for q in Q], Ls[i2], Rs[i2])
        return out
    squeeze = on_subset(pre, heavy, x)
    loss_m = mag & SR & nonbare & Kon & ~esc & NK & ~Eon
    win_m = mag & SR & C6.win_m_x(own, en, L, R, a)
    return {"LOSS_C6": (-1, loss_c | squeeze), "LOSS_M6": (-1, loss_m), "WIN_M6": (1, win_m),
            "WIN_C6": (1, win_c_x(x))}


def gen7(x):
    """+ full two-action squeeze for Cyan: every legal pair of Cyan actions (any pieces; the second
    action generated on the board after the first) ends in a bare-Kernel loss, a sound Magenta kill
    (R1/R2), or the Cyan Kernel burning with a non-losing Magenta reply. Prefilter: all Kernel-only
    plans lose (necessary)."""
    r = gen6(x)
    own, en, L, a, u, mag, c, R, I, empty, SR, K, E, Kon, Eon = _common(x)
    nonbare = (sum(own[:5]).sum(1) + sum(en[:5]).sum(1)) > 0
    pre = ~mag & SR & nonbare & (a == 2) & ~r["LOSS_C6"][1]

    def heavy(xs):
        P, Q, Ls, a_, u_, m_, c_, Rs, Is = C.split(xs)
        return C6.cyan_squeeze_a2_full(P, Q, Ls, Rs)
    extra = on_subset(pre, heavy, x)
    r["LOSS_C6"] = (-1, r["LOSS_C6"][1] | extra)
    return {"LOSS_C7": r["LOSS_C6"], "LOSS_M6": r["LOSS_M6"], "WIN_M6": r["WIN_M6"], "WIN_C6": r["WIN_C6"]}


def gen8_tier1(x):
    """Tier 1 only (pure fixed shift operations, no enumeration): LOSS_C core with escape_x, LOSS_M,
    WIN_M of gen6."""
    import final
    d = final.certify(x, tier1=True)
    return {"LOSS_C": (-1, d["LOSS_C"]), "LOSS_M": (-1, d["LOSS_M"]), "WIN_M": (1, d["WIN_M"]),
            "LOSS": (-1, d["LOSS"]), "WIN": (1, d["WIN"])}


def final_cert(x):
    """Final = final.certify (gen7 logic): unions WIN / LOSS and parts."""
    import final
    d = final.certify(x)
    return {"LOSS_C": (-1, d["LOSS_C"]), "LOSS_M": (-1, d["LOSS_M"]), "WIN_M": (1, d["WIN_M"]),
            "WIN_C": (1, d["WIN_C"]), "LOSS": (-1, d["LOSS"]), "WIN": (1, d["WIN"])}


GENERATIONS = {"gen0": gen0, "gen1": gen1, "gen2": gen2, "gen3": gen3, "gen4": gen4, "gen5": gen5, "gen6": gen6, "gen7": gen7, "gen8_tier1": gen8_tier1, "final": final_cert}
