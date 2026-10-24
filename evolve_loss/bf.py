"""Exact ground truth for forced-loss certificates (Colony: One, official engine).

Question answered for a canonical observation (side to move = "mover", labelled
CYAN on the rebuilt board; the real colour comes from plane 15):

    LOST  <=>  for EVERY legal sequence of the mover's remaining actions (1 or 2),
               the game is won by the enemy no later than the end of the enemy's
               next turn (2 actions), the enemy choosing its actions.

Everything that can end the game inside that window is modelled exactly, as in
env.ColonyOneEnv.step/_end_turn:
  * Kernel capture (record.captured_piece is a Kernel) ends the game at once;
  * bare-Kernels resolution at the end of every turn (distance <= 2: acting team
    loses; else smaller ring wins; equal rings: real Magenta);
  * firewall shrink at the end of real Magenta's turn when plane 14 == 1.0
    (shrink round), both-Kernels-burn -> real Magenta.

Conservative conventions (they can only make LOST rarer, never wrongly true):
  * a mover line that gets stuck (no legal action left in the turn) counts as
    an escape; a mover with no legal action at the root is reported as stuck;
  * an enemy line that gets stuck counts as no win for the enemy.

Pruning is exact: early exit on the first escaping mover line; the enemy search
stops at its first winning line; enemy kill-in-2 after its first action b1 is
detected as "our Kernel has no guard and some enemy action targets its square"
(that action then captures the Kernel), using rules.legal_actions per piece.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(os.path.dirname(HERE), "env")
if ENV not in sys.path:
    sys.path.insert(0, ENV)

import numpy as np  # noqa: E402

from colony_one import constants, rules  # noqa: E402
from colony_one.board import Board  # noqa: E402
from colony_one.pieces import PieceType  # noqa: E402

M, E = constants.CYAN, constants.MAGENTA  # canonical labels: mover / enemy
KER = PieceType.KERNEL
ADJ8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def other(t):
    return E if t == M else M


def kpos(board, team):
    g = board.grid
    for r in range(8):
        row = g[r]
        for c in range(8):
            p = row[c]
            if p is not None and p.team == team and p.piece_type is KER:
                return (r, c)
    return None


def n_guards(board, k, team):
    r0, c0 = k
    n = 0
    g = board.grid
    for dr, dc in ADJ8:
        r, c = r0 + dr, c0 + dc
        if 0 <= r < 8 and 0 <= c < 8:
            p = g[r][c]
            if p is not None and p.team == team and p.piece_type is not KER:
                n += 1
    return n


def targets_square(board, sq, by_team):
    """Some legal action of by_team has to_pos == sq (official per-piece generator).
    Every attack has Chebyshev reach <= 3, so farther pieces are skipped."""
    r0, c0 = sq
    g = board.grid
    for r in range(max(0, r0 - 3), min(8, r0 + 4)):
        for c in range(max(0, c0 - 3), min(8, c0 + 4)):
            p = g[r][c]
            if p is not None and p.team == by_team:
                for a in rules.legal_actions(board, (r, c)):
                    if a.to_pos == sq:
                        return True
    return False


def is_kill(rec):
    cp = rec.captured_piece
    return cp is not None and cp.piece_type is KER


def ring_of(p):
    return int(max(abs(p[0] - 3.5), abs(p[1] - 3.5)))


class Ctx:
    __slots__ = ("mover_magenta", "shrink_due", "cache", "killers", "nodes")

    def __init__(self, mover_magenta, shrink_due):
        self.mover_magenta = mover_magenta
        self.shrink_due = shrink_due
        self.cache = {}
        self.killers = []
        self.nodes = 0

    def magenta_side(self):
        return M if self.mover_magenta else E

    def is_magenta(self, team):
        return (team == M) == self.mover_magenta


def end_of_turn(board, acting, ctx):
    """env._end_turn: returns (winner|None, firewall_record|None). Caller undoes the record."""
    if board.only_kernels_remain():
        km, ke = kpos(board, M), kpos(board, E)
        if max(abs(km[0] - ke[0]), abs(km[1] - ke[1])) <= 2:
            return other(acting), None
        rm, re_ = ring_of(km), ring_of(ke)
        if rm < re_:
            return M, None
        if re_ < rm:
            return E, None
        return ctx.magenta_side(), None
    if ctx.shrink_due and ctx.is_magenta(acting):
        rec = board.apply_firewall_shrink()
        if rec is None:
            return None, None
        dead = [p.team for _, _, p in rec.killed if p.piece_type is KER]
        if len(dead) == 2:
            return ctx.magenta_side(), rec
        if len(dead) == 1:
            return other(dead[0]), rec
        return None, rec
    return None, None


def board_key(board):
    g = board.grid
    return (tuple(None if p is None else (p.team, p.piece_type) for row in g for p in row),
            sum(map(sum, board.locked)))


def enemy_wins(board, ctx):
    """Exists an enemy line (its full 2-action turn) that wins by the end of that turn."""
    key = board_key(board)
    hit = ctx.cache.get(key)
    if hit is not None:
        return hit
    res = _enemy_wins(board, ctx)
    ctx.cache[key] = res
    return res


def _try_b1(board, b1, km, ctx):
    ctx.nodes += 1
    r = board.apply_action(b1)
    win = is_kill(r) or (n_guards(board, km, M) == 0 and targets_square(board, km, E))
    board.undo_action(r)
    return win


def _enemy_wins(board, ctx):
    km = kpos(board, M)
    # killer heuristic: re-try first actions that won elsewhere (legality re-checked)
    for b1 in ctx.killers:
        p = board.grid[b1.from_pos[0]][b1.from_pos[1]]
        if p is None or p.team != E:
            continue
        if b1 in rules.legal_actions(board, b1.from_pos):
            if _try_b1(board, b1, km, ctx):
                return True
    acts = rules.legal_actions_for_team(board, E)
    for b1 in acts:
        if _try_b1(board, b1, km, ctx):
            ctx.killers.insert(0, b1)
            del ctx.killers[6:]
            return True
    # wins that only appear at the end of the enemy's turn
    need = False
    if ctx.shrink_due and ctx.is_magenta(E):
        open_ring = board.next_open_ring()
        if open_ring >= 0 and board.is_on_ring(km[0], km[1], open_ring):
            need = True
    if not need:
        enemy_nk = sum(1 for row in board.grid for p in row
                       if p is not None and p.team == E and p.piece_type is not KER)
        mover_nk = sum(1 for row in board.grid for p in row
                       if p is not None and p.team == M and p.piece_type is not KER)
        need = enemy_nk == 0 and mover_nk <= 2
    if not need:
        return False
    for b1 in acts:
        r1 = board.apply_action(b1)
        for b2 in rules.legal_actions_for_team(board, E):
            ctx.nodes += 1
            r2 = board.apply_action(b2)
            if is_kill(r2):
                w, rec = E, None
            else:
                w, rec = end_of_turn(board, E, ctx)
            if rec is not None:
                board.undo_firewall_shrink(rec)
            board.undo_action(r2)
            if w == E:
                board.undo_action(r1)
                return True
        board.undo_action(r1)
    return False


def after_mover_turn(board, ctx):
    """True if the enemy wins by the end of its next turn from here (mover's turn just ended)."""
    w, rec = end_of_turn(board, M, ctx)
    if w is not None:
        res = (w == E)
    else:
        res = enemy_wins(board, ctx)
    if rec is not None:
        board.undo_firewall_shrink(rec)
    return res


def mover_lost(board, al, ctx):
    """True / False, or None when the mover has no legal action at all."""
    acts = rules.legal_actions_for_team(board, M)
    if not acts:
        return None
    for a1 in acts:
        r1 = board.apply_action(a1)
        if is_kill(r1):
            board.undo_action(r1)
            return False
        if al == 1:
            lost = after_mover_turn(board, ctx)
        else:
            acts2 = rules.legal_actions_for_team(board, M)
            lost = bool(acts2)
            for a2 in acts2:
                r2 = board.apply_action(a2)
                line_lost = (not is_kill(r2)) and after_mover_turn(board, ctx)
                board.undo_action(r2)
                if not line_lost:
                    lost = False
                    break
        board.undo_action(r1)
        if not lost:
            return False
    return True


def obs_context(obs):
    al = int(round(float(obs[13, 0, 0]) * 2))
    shrink_due = float(obs[14, 0, 0]) > 0.999
    mover_magenta = float(obs[15, 0, 0]) > 0.5
    return al, shrink_due, mover_magenta


def label(obs):
    """Returns (code, nodes): code 1 lost, 0 not lost, -1 stuck/invalid."""
    obs = np.asarray(obs, dtype=np.float32)
    board = Board.from_observation(obs, M)
    al, shrink_due, mover_magenta = obs_context(obs)
    if kpos(board, M) is None or kpos(board, E) is None:
        return -1, 0
    ctx = Ctx(mover_magenta, shrink_due)
    res = mover_lost(board, al, ctx)
    if res is None:
        return -1, ctx.nodes
    return int(res), ctx.nodes


def enemy_threat_now(obs):
    """Diagnostic: enemy would win from here if it moved now (mover passes)."""
    obs = np.asarray(obs, dtype=np.float32)
    board = Board.from_observation(obs, M)
    al, shrink_due, mover_magenta = obs_context(obs)
    ctx = Ctx(mover_magenta, shrink_due)
    return enemy_wins(board, ctx)


def board_str(board):
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
                s.append(p.abbr.upper() if p.team == M else p.abbr.lower())
        rows.append(f"{r} " + " ".join(s))
    rows.append("   " + "  ".join(str(c) for c in range(8)))
    return "\n".join(rows)
