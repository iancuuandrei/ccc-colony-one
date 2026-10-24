"""Verify the in-turn Kernel-kill rule for Colony: One against brute force.

Ground truth: exhaustive search over the mover's legal action sequences within
the remaining actions of the turn (1 or 2), using the official engine
(Board.apply_action / undo_action). Win = some action's record.captured_piece
is a Kernel.

Rule R1 (claim):  hit and g < actions_left
    hit = some legal action has to_pos == enemy Kernel square
    g   = enemy non-Kernel pieces 8-adjacent to the enemy Kernel
Rule R2 (extra, actions_left == 2, computed with fixed shift operations only):
    some own piece of type T (one per type per team) can RELOCATE (legal move,
    empty square or capture of a non-Kernel enemy) to a square s from which a
    T attacks the enemy Kernel on the occupancy with T's origin removed, and
    the Kernel is unguarded after the move (g == 0, or g == 1 and s is that
    single guard's square).

Usage: python verify_win.py [max_positions] [workers]
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(os.path.dirname(HERE), "env")
DATA = os.path.join(os.path.dirname(HERE), "data", "gen00")
sys.path.insert(0, ENV)

from colony_one import constants, rules  # noqa: E402
from colony_one.board import Board  # noqa: E402
from colony_one.pieces import PieceType  # noqa: E402

PING, DAEMON, EXPLOIT, TRACE, ICE, KERNEL = range(6)
ME, EN = constants.CYAN, constants.MAGENTA

# ---------------------------------------------------------------------------
# Brute force (engine)
# ---------------------------------------------------------------------------

def _enemy_kernel(board):
    for r in range(8):
        for c in range(8):
            p = board.grid[r][c]
            if p is not None and p.team == EN and p.piece_type is PieceType.KERNEL:
                return (r, c)
    return None


def _is_kill(rec):
    cp = rec.captured_piece
    return cp is not None and cp.piece_type is PieceType.KERNEL


def brute_force(board, actions_left):
    """Return (win, list of winning lines). A line is a tuple of (action, record-info)."""
    lines = []
    for a1 in rules.legal_actions_for_team(board, ME):
        r1 = board.apply_action(a1)
        if _is_kill(r1):
            lines.append(((a1, None),))
        elif actions_left == 2:
            info1 = (r1.moved, r1.captured_piece.piece_type.value if r1.captured_piece else None,
                     r1.actual_target_pos)
            for a2 in rules.legal_actions_for_team(board, ME):
                r2 = board.apply_action(a2)
                if _is_kill(r2):
                    lines.append(((a1, info1), (a2, None)))
                board.undo_action(r2)
        board.undo_action(r1)
    return bool(lines), lines


def legal_hit(board, k):
    return any(a.to_pos == k for a in rules.legal_actions_for_team(board, ME))


# ---------------------------------------------------------------------------
# Observation-only rules (numpy, batched, shift operations)
# ---------------------------------------------------------------------------
ORTH = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL8 = ORTH + DIAG
KNIGHT = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def sh(a, dr, dc):
    """Shift a (...,8,8) bool array by (dr, dc); out[r+dr, c+dc] = a[r, c]."""
    out = np.zeros_like(a)
    rs = slice(max(dr, 0), 8 + min(dr, 0)); rd = slice(max(-dr, 0), 8 + min(-dr, 0))
    cs = slice(max(dc, 0), 8 + min(dc, 0)); cd = slice(max(-dc, 0), 8 + min(-dc, 0))
    out[..., rs, cs] = a[..., rd, cd]
    return out


def ray_set(src, empty, open_, dirs, dmin, dmax):
    """Squares at distance dmin..dmax from src along dirs whose intermediates are
    empty and unlocked; the endpoint itself must be unlocked (open_)."""
    out = np.zeros_like(src)
    for dr, dc in dirs:
        cur = src.copy()
        for d in range(1, dmax + 1):
            cur = sh(cur, dr, dc) & open_
            if d >= dmin:
                out |= cur
            cur = cur & empty
    return out


def attackers_of(K, typ, occ, open_):
    """Squares from which a piece of type `typ` attacks square K (bool mask),
    given occupancy `occ` (K counts as occupied). Symmetric geometry, so we ray out of K."""
    empty = ~occ & open_
    if typ == PING:      # diagonal capture-move, range 1..2
        return ray_set(K, empty, open_, DIAG, 1, 2)
    if typ == DAEMON:    # orthogonal capture-move at 2..3, or adjacent shot (8 dirs)
        return ray_set(K, empty, open_, ORTH, 2, 3) | ray_set(K, empty, open_, ALL8, 1, 1)
    if typ == EXPLOIT:
        out = np.zeros_like(K)
        for dr, dc in KNIGHT:
            out |= sh(K, dr, dc) & open_
        return out
    if typ == TRACE:     # diagonal shot, range 2..3
        return ray_set(K, empty, open_, DIAG, 2, 3)
    if typ == ICE:       # 8 dirs, range 1..2
        return ray_set(K, empty, open_, ALL8, 1, 2)
    if typ == KERNEL:
        return ray_set(K, empty, open_, ALL8, 1, 1)
    raise ValueError


def relocations(src, typ, occ, enemy_nk, open_):
    """Squares a piece of type `typ` on src can RELOCATE to (move actions that
    actually move), excluding moves onto the enemy Kernel."""
    empty = ~occ & open_
    if typ == PING:      # orth empty slide 2; diag capture-move 1..2
        return ray_set(src, empty, open_, ORTH, 1, 2) & empty | \
            ray_set(src, empty, open_, DIAG, 1, 2) & enemy_nk
    if typ == DAEMON:    # orth slide 3 to empty; capture at distance 2..3
        return ray_set(src, empty, open_, ORTH, 1, 3) & empty | \
            ray_set(src, empty, open_, ORTH, 2, 3) & enemy_nk
    if typ == EXPLOIT:
        out = np.zeros_like(src)
        for dr, dc in KNIGHT:
            out |= sh(src, dr, dc)
        return out & open_ & (empty | enemy_nk)
    if typ == TRACE:     # orth step to empty
        return ray_set(src, empty, open_, ORTH, 1, 1) & empty
    if typ == ICE:       # 8 dirs range 2, capture on contact
        return ray_set(src, empty, open_, ALL8, 1, 2) & (empty | enemy_nk)
    if typ == KERNEL:
        return ray_set(src, empty, open_, ALL8, 1, 1) & (empty | enemy_nk)
    raise ValueError


def obs_rules(obs):
    """obs: (N,16,8,8). Returns dict of per-position arrays (computed from obs only)."""
    P = obs[:, :12] > 0.5
    own, en = P[:, :6], P[:, 6:12]
    locked = obs[:, 12] > 0.5
    open_ = ~locked
    al = np.rint(obs[:, 13, 0, 0] * 2).astype(int)
    occ = own.any(1) | en.any(1)
    K = en[:, KERNEL]
    has_k = K.any((1, 2))
    enemy_nk = en[:, :KERNEL].any(1)
    adjK = ray_set(K, np.ones_like(K), np.ones_like(K), ALL8, 1, 1)
    guards = adjK & enemy_nk
    g = guards.sum((1, 2))

    hit = np.zeros(len(obs), bool)
    for t in range(6):
        hit |= (own[:, t] & attackers_of(K, t, occ, open_)).any((1, 2))
    r1 = has_k & hit & (g < al)

    # R2: relocate-then-attack by the same piece (actions_left == 2)
    gmask = np.where((g == 0)[:, None, None], True, (g == 1)[:, None, None] & guards)
    r2 = np.zeros(len(obs), bool)
    for t in range(6):
        src = own[:, t]
        occ_wo = occ & ~src                    # occupancy with T's origin removed
        A = attackers_of(K, t, occ_wo, open_)
        D = relocations(src, t, occ, enemy_nk & ~K, open_)
        r2 |= (D & A & gmask).any((1, 2))
    r2 &= has_k & (al == 2)
    return dict(al=al, g=g, hit=hit, has_k=has_k, r1=r1, r2=r2)


# ---------------------------------------------------------------------------
# Line categorisation
# ---------------------------------------------------------------------------

def categorise(lines, board):
    """Categories of 2-action winning lines (first action a1 by X, kill by Y)."""
    cats = set()
    for line in lines:
        if len(line) == 1:
            cats.add("direct")
            continue
        (a1, (moved, cap, tgt)), (a2, _) = line
        X_after = a1.to_pos if moved else a1.from_pos
        same = a2.from_pos == X_after
        if same:
            if moved and cap is None:
                cats.add("self:approach")
            elif moved:
                cats.add("self:approach-capture")
            else:
                cats.add("self:shoot-then-attack")
        else:
            if cap is None:
                cats.add("other:discovered(vacate-ray)")
            else:
                cats.add("other:capture-unblock/unguard")
    return cats


PRIORITY = ["direct", "self:approach", "self:approach-capture", "self:shoot-then-attack",
            "other:discovered(vacate-ray)", "other:capture-unblock/unguard"]


def board_str(board, k=None):
    rows = []
    for r in range(8):
        s = []
        for c in range(8):
            p = board.grid[r][c]
            if board.locked[r][c]:
                s.append("##")
            elif p is None:
                s.append(" .")
            else:
                s.append((p.abbr.upper() if p.team == ME else p.abbr.lower()))
        rows.append(f"{r} " + " ".join(s))
    rows.append("   " + "  ".join(str(c) for c in range(8)))
    return "\n".join(rows)


def work(args):
    idx, obs = args
    board = Board.from_observation(obs, ME)
    al = int(round(float(obs[13, 0, 0]) * 2))
    k = _enemy_kernel(board)
    if k is None:
        return idx, None
    win, lines = brute_force(board, al)
    cats = categorise(lines, board) if win else set()
    return idx, dict(win=win, hit_engine=legal_hit(board, k), cats=sorted(cats),
                     line=str(lines[0]) if win else None)


def main():
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    obs = np.concatenate([np.load(f)["obs"] for f in sorted(glob.glob(os.path.join(DATA, "chunk*.npz")))])
    rng = np.random.default_rng(0)
    if max_n < len(obs):
        obs = obs[np.sort(rng.choice(len(obs), max_n, replace=False))]
    obs = obs.astype(np.float32)
    N = len(obs)
    t0 = time.time()
    R = obs_rules(obs)
    t_rules = time.time() - t0

    t0 = time.time()
    res = [None] * N
    with Pool(workers) as pool:
        for i, r in pool.imap_unordered(work, ((i, obs[i]) for i in range(N)), chunksize=256):
            res[i] = r
    t_bf = time.time() - t0

    valid = np.array([r is not None for r in res])
    win = np.array([bool(r and r["win"]) for r in res])
    hit_eng = np.array([bool(r and r["hit_engine"]) for r in res])
    al, r1, r2, hit = R["al"], R["r1"], R["r2"], R["hit"]
    cats = [set(r["cats"]) if r else set() for r in res]

    out = dict(positions_total=int(N), positions_checked=int(valid.sum()),
               no_enemy_kernel=int((~valid).sum()),
               actions_left_counts={int(a): int(((al == a) & valid).sum()) for a in (1, 2)},
               seconds_rules_numpy=round(t_rules, 2), seconds_bruteforce=round(t_bf, 1))
    out["hit_geometry_vs_engine_mismatch"] = int((valid & (hit != hit_eng)).sum())
    out["bruteforce_wins"] = int((valid & win).sum())
    out["R1_flag"] = int((valid & r1).sum())
    out["R1_violations"] = int((valid & r1 & ~win).sum())
    out["R1_misses"] = int((valid & ~r1 & win).sum())
    out["R1_misses_by_actions_left"] = {int(a): int((valid & ~r1 & win & (al == a)).sum()) for a in (1, 2)}
    # categories of misses
    prim, anyc = Counter(), Counter()
    for i in np.flatnonzero(valid & ~r1 & win):
        c = cats[i]
        anyc.update(c)
        prim[next(p for p in PRIORITY if p in c)] += 1
    out["R1_misses_primary_category"] = dict(prim)
    out["R1_misses_category_any"] = dict(anyc)
    # R2 checks
    selfreloc = np.array([("self:approach" in c) or ("self:approach-capture" in c) for c in cats])
    out["R2_flag"] = int((valid & r2).sum())
    out["R2_violations"] = int((valid & r2 & ~win).sum())
    out["R2_vs_bruteforce_selfrelocation_mismatch"] = int((valid & (r2 != selfreloc)).sum())
    out["R1_or_R2_flag"] = int((valid & (r1 | r2)).sum())
    out["R1_or_R2_violations"] = int((valid & (r1 | r2) & ~win).sum())
    rem = valid & win & ~(r1 | r2)
    out["R1_or_R2_misses"] = int(rem.sum())
    pr = Counter()
    for i in np.flatnonzero(rem):
        pr[next(p for p in PRIORITY if p in cats[i])] += 1
    out["R1_or_R2_misses_primary_category"] = dict(pr)

    # examples
    ex = {}
    for name, mask in [("R1_violation", valid & r1 & ~win), ("R2_violation", valid & r2 & ~win),
                       ("R2_mismatch", valid & (r2 != selfreloc))]:
        ex[name] = []
        for i in np.flatnonzero(mask)[:5]:
            b = Board.from_observation(obs[i], ME)
            ex[name].append(dict(i=int(i), al=int(al[i]), g=int(R["g"][i]), board=board_str(b), res=res[i]))
    for cat in PRIORITY[1:]:
        for i in np.flatnonzero(valid & ~r1 & win):
            if next(p for p in PRIORITY if p in cats[i]) == cat:
                b = Board.from_observation(obs[i], ME)
                ex["miss_" + cat] = dict(i=int(i), al=int(al[i]), g=int(R["g"][i]),
                                         board=board_str(b), line=res[i]["line"])
                break
    ex["R1_or_R2_residual_misses"] = []
    for i in np.flatnonzero(rem)[:20]:
        b = Board.from_observation(obs[i], ME)
        ex["R1_or_R2_residual_misses"].append(dict(i=int(i), al=int(al[i]), g=int(R["g"][i]),
                                                   board=board_str(b), res=res[i]))
    out["examples"] = ex
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))
    for name in ("R1_violation", "R2_violation", "R2_mismatch", "R1_or_R2_residual_misses"):
        for e in ex[name]:
            print(name, e["i"], e["al"], e["g"], e["res"]); print(e["board"])


if __name__ == "__main__":
    main()
