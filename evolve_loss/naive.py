"""Independent, slow reference for bf.label: drives the official ColonyOneEnv.step
on deep copies (real colours, real firewall timing), no shortcuts except exact
forall/exists early exits. Used only to cross-check bf.py on samples."""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(os.path.dirname(HERE), "env")
if ENV not in sys.path:
    sys.path.insert(0, ENV)

import copy  # noqa: E402

import numpy as np  # noqa: E402

from colony_one import constants  # noqa: E402
from colony_one.board import Board  # noqa: E402
from colony_one.env import ColonyOneEnv  # noqa: E402


def make_env(obs):
    obs = np.asarray(obs, dtype=np.float32)
    team = constants.MAGENTA if obs[15, 0, 0] > 0.5 else constants.CYAN
    al = int(round(float(obs[13, 0, 0]) * 2))
    urg = float(obs[14, 0, 0])
    remaining = int(round(8 * (1 - urg)))
    rnd = 8 if remaining == 0 else 8 - remaining  # shrink round iff urgency == 1
    env = object.__new__(ColonyOneEnv)
    env._rng = np.random.default_rng(0)
    env._scene = None
    env.board = Board.from_observation(obs, team)  # real coordinates and colours
    env.current_team = team
    env.actions_left = al
    env.round_number = rnd
    env.action = 1
    env.terminated = False
    env.winner = None
    return env, team


def enemy_can_win(env, enemy):
    """env: enemy to move with 2 actions. Exists a line winning within this turn."""
    assert env.current_team == enemy and env.actions_left == 2
    for a1 in env.legal_actions():
        e1 = copy.deepcopy(env)
        e1.step(a1)
        if e1.terminated:
            if e1.winner == enemy:
                return True
            continue
        for a2 in e1.legal_actions():
            e2 = copy.deepcopy(e1)
            e2.step(a2)
            if e2.terminated and e2.winner == enemy:
                return True
    return False


def label(obs):
    env, me = make_env(obs)
    enemy = constants.MAGENTA if me == constants.CYAN else constants.CYAN
    acts = env.legal_actions()
    if not acts:
        return -1

    def line_lost(e):
        # e: after an action of mine
        if e.terminated:
            return e.winner == enemy
        if e.current_team == me:  # still my turn
            acts2 = e.legal_actions()
            if not acts2:
                return False
            for a in acts2:
                e2 = copy.deepcopy(e)
                e2.step(a)
                if not line_lost(e2):
                    return False
            return True
        return enemy_can_win(e, enemy)

    for a in acts:
        e = copy.deepcopy(env)
        e.step(a)
        if not line_lost(e):
            return 0
    return 1
