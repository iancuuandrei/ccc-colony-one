"""Exact firewall-horizon oracle for Colony: One, built on the official engine.

solve(obs) -> value in {-1, 0, +1} from the MOVER's perspective:
  +1  the mover can force a win before / at the next firewall shrink resolves,
  -1  the opponent can force the mover's loss within that horizon,
   0  the game is not decided by the end of the shrink round (it continues).

The horizon is the end of Magenta's turn in the next shrink round H >= current round
(H in SHRINK_ROUNDS). Everything that can end the game inside it is resolved exactly as
ColonyOneEnv.step / _end_turn does: Kernel capture ends the game at once; at each turn end
the bare-Kernels rule is checked first, then (Magenta's turn of a shrink round only) the
outer ring locks and destroyed Kernels decide (both -> Magenta).

Search: fail-soft alpha-beta on the 3-valued game (window (-1, 1) is exact because the values
are bounded), Board.apply_action / undo_action and apply_firewall_shrink / undo_firewall_shrink
from the engine, legal moves from rules.legal_actions_for_team.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "env"))

from colony_one import constants  # noqa: E402
from colony_one.board import Board  # noqa: E402
from colony_one.pieces import PieceType  # noqa: E402
from colony_one.rules import legal_actions_for_team  # noqa: E402

CYAN, MAGENTA = constants.CYAN, constants.MAGENTA
SHRINK = constants.SHRINK_ROUNDS
KERNEL = PieceType.KERNEL


class Stuck(Exception):
    """A side to move has no legal action (engine would raise)."""


def opp(t):
    return MAGENTA if t == CYAN else CYAN


def decode(obs):
    """obs (16,8,8) -> (team, actions_left, round, completed_shrinks)."""
    obs = np.asarray(obs, dtype=np.float32)
    team = MAGENTA if obs[15, 0, 0] > 0.5 else CYAN
    a = int(round(2 * float(obs[13, 0, 0])))
    nl = int(round(float(obs[12].sum())))
    c = {0: 0, 28: 1, 48: 2, 60: 3}[nl]
    u = float(obs[14, 0, 0])
    nxt = SHRINK[c]
    rnd = int(round(nxt - 8 * (1 - u)))
    return team, a, rnd, c


class Solver:
    def __init__(self, board, horizon_round, node_limit=None, fw=True):
        self.fw = fw  # fw=False: same horizon, firewall disabled (to tell firewall-decided outcomes)
        self.b = board
        self.H = horizon_round
        self.nodes = 0
        self.node_limit = node_limit

    # value for `team` that has `a` actions left in round `rnd`
    def search(self, team, a, rnd, alpha, beta):
        self.nodes += 1
        if self.node_limit and self.nodes > self.node_limit:
            raise TimeoutError
        b = self.b
        acts = legal_actions_for_team(b, team)
        if not acts:
            raise Stuck
        # immediate Kernel capture available? (cheap pre-check, exact via engine apply)
        kern, rest = [], []
        for ac in acts:
            tgt = b.grid[ac.to_pos[0]][ac.to_pos[1]]
            if tgt is not None and tgt.piece_type is KERNEL and tgt.team != team:
                kern.append(ac)
            elif b.grid[ac.from_pos[0]][ac.from_pos[1]].piece_type is KERNEL:
                kern.append(ac)  # own Kernel moves early: escapes / approaches matter most
            else:
                rest.append(ac)
        best = -2
        for ac in kern + rest:
            rec = b.apply_action(ac)
            try:
                cp = rec.captured_piece
                if cp is not None and cp.piece_type is KERNEL:
                    v = 1
                elif a > 1:
                    v = self.search(team, a - 1, rnd, alpha, beta)
                else:
                    v = self.end_turn(team, rnd, alpha, beta)
            finally:
                b.undo_action(rec)
            if v > best:
                best = v
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
        return best

    def end_turn(self, team, rnd, alpha, beta):
        """`team` has just finished its turn in round `rnd`; value from `team`'s view."""
        b = self.b
        if b.only_kernels_remain():
            w = b.kernel_endgame_winner(acting_team=team)
            return 1 if w == team else -1
        fr = None
        if team == MAGENTA and rnd in SHRINK and self.fw:
            fr = b.apply_firewall_shrink()
        try:
            if fr is not None:
                kk = [p for _, _, p in fr.killed if p.piece_type is KERNEL]
                if kk:
                    w = MAGENTA if len(kk) == 2 else opp(kk[0].team)
                    return 1 if w == team else -1
            if team == MAGENTA and rnd >= self.H:
                return 0  # horizon: game continues
            nrnd = rnd + 1 if team == MAGENTA else rnd
            return -self.search(opp(team), constants.MAX_ACTION_PER_TURN, nrnd, -beta, -alpha)
        finally:
            if fr is not None:
                b.undo_firewall_shrink(fr)


def horizon_for(rnd):
    for s in SHRINK:
        if s >= rnd:
            return s
    return None


def solve(obs, node_limit=None, fw=True):
    """Returns dict(value, nodes, team, a, rnd, c) or value None if stuck/timeout."""
    team, a, rnd, c = decode(obs)
    board = Board.from_observation(np.asarray(obs, dtype=np.float32), team)
    H = horizon_for(rnd)
    s = Solver(board, H, node_limit, fw)
    try:
        v = s.search(team, a, rnd, -1, 1)
        status = "ok"
    except Stuck:
        v, status = None, "stuck"
    except TimeoutError:
        v, status = None, "timeout"
    return dict(value=v, nodes=s.nodes, team=team, a=a, rnd=rnd, c=c, status=status)


def own_turn_facts(obs):
    """Exhaustive over the mover's remaining actions THIS turn (engine):
    kill  = some line captures the enemy Kernel,
    esc   = some line ends the turn with the own Kernel off the ring that locks next
            (Kernel standing on an open square with ring index > c),
    quiet = some legal action captures nothing (a move to an empty square)."""
    team, a, rnd, c = decode(obs)
    b = Board.from_observation(np.asarray(obs, dtype=np.float32), team)

    def ring(r, col):
        return min(r, col, 7 - r, 7 - col)

    def kpos():
        for r in range(8):
            for col in range(8):
                p = b.grid[r][col]
                if p is not None and p.team == team and p.piece_type is KERNEL:
                    return r, col
        return None

    kill = esc = False
    quiet = False
    acts1 = legal_actions_for_team(b, team)
    for ac in acts1:
        if ac.kind == "move" and b.grid[ac.to_pos[0]][ac.to_pos[1]] is None:
            quiet = True
    for ac in acts1:
        rec = b.apply_action(ac)
        try:
            cp = rec.captured_piece
            if cp is not None and cp.piece_type is KERNEL:
                kill = True
                continue
            if a == 1:
                k = kpos()
                if k is not None and ring(*k) > c:
                    esc = True
            else:
                for ac2 in legal_actions_for_team(b, team):
                    rec2 = b.apply_action(ac2)
                    try:
                        cp2 = rec2.captured_piece
                        if cp2 is not None and cp2.piece_type is KERNEL:
                            kill = True
                        else:
                            k = kpos()
                            if k is not None and ring(*k) > c:
                                esc = True
                    finally:
                        b.undo_action(rec2)
        finally:
            b.undo_action(rec)
    return dict(kill=kill, esc=esc, quiet=quiet, a=a, c=c, team=team, rnd=rnd)
