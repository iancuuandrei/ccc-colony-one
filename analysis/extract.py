"""Extract per-position features and per-game metadata from data/gen00.

Read-only on the data; writes analysis/features.npz and analysis/final_positions.json.
Uses at most 2 worker processes.

python analysis/extract.py
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / 'env'
DATA = ROOT / 'data' / 'gen00'
OUT = Path(__file__).resolve().parent

VALUES = np.array([1.0, 3.0, 2.2, 2.2, 3.2], np.float32)  # Ping Daemon Exploit Trace Ice
NAMES = ['ping', 'daemon', 'exploit', 'trace', 'ice', 'kernel']


def round_from_obs(obs: np.ndarray) -> np.ndarray:
    """Round number from locked-square count (stage) and urgency plane."""
    locked = obs[:, 12].reshape(len(obs), -1).sum(1).round().astype(int)
    u = obs[:, 14, 0, 0].astype(np.float64)
    stage = np.select([locked == 0, locked == 28, locked == 48, locked == 60], [0, 1, 2, 3], -1)
    assert (stage >= 0).all(), np.unique(locked)
    r = np.where(stage < 3, stage * 8 + np.rint(8 * u), np.where(u > 0.95, 26, 25))
    return r.astype(int), stage


def ring_c(rc):
    """Ring distance to centre (0 = central 2x2, 3 = edge)."""
    r, c = rc
    return int(max(abs(r - 3.5), abs(c - 3.5)))


def vector_features(obs: np.ndarray) -> dict:
    n = len(obs)
    f = {}
    cnt = obs[:, :12].reshape(n, 12, -1).sum(2)
    own, en = cnt[:, :6], cnt[:, 6:12]
    for i, nm in enumerate(NAMES[:5]):
        f[f'own_{nm}'] = own[:, i]
        f[f'en_{nm}'] = en[:, i]
    f['own_kernel_present'] = own[:, 5]
    f['en_kernel_present'] = en[:, 5]
    f['mat_count_diff'] = own[:, :5].sum(1) - en[:, :5].sum(1)
    f['mat_value_diff'] = own[:, :5] @ VALUES - en[:, :5] @ VALUES
    f['own_pieces'] = own[:, :5].sum(1)
    f['en_pieces'] = en[:, :5].sum(1)
    # guards + kernel rings
    own_nk = obs[:, 0:5].sum(1) > 0.5
    en_nk = obs[:, 6:11].sum(1) > 0.5
    pad = lambda a: np.pad(a, ((0, 0), (1, 1), (1, 1)))
    def neigh_count(mask):
        p = pad(mask.astype(np.int32))
        s = np.zeros_like(mask, dtype=np.int32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    s += p[:, 1 + dr:9 + dr, 1 + dc:9 + dc]
        return s
    own_k = obs[:, 5] > 0.5
    en_k = obs[:, 11] > 0.5
    f['own_guards'] = (neigh_count(own_nk) * own_k).reshape(n, -1).sum(1)
    f['en_guards'] = (neigh_count(en_nk) * en_k).reshape(n, -1).sum(1)
    rr, cc = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
    ringmap = np.maximum(np.abs(rr - 3.5), np.abs(cc - 3.5)).astype(int)
    f['own_kring'] = (ringmap * own_k).reshape(n, -1).sum(1)
    f['en_kring'] = (ringmap * en_k).reshape(n, -1).sum(1)
    f['magenta_to_move'] = obs[:, 15, 0, 0].astype(np.float32)
    f['actions_left'] = np.rint(obs[:, 13, 0, 0] * 2).astype(int)
    rnd, stage = round_from_obs(obs)
    f['round'] = rnd
    f['stage'] = stage
    # next ring to lock (edge index) == stage; kernel edge index = 3 - kring
    f['own_k_on_next_lock'] = ((3 - f['own_kring']) == stage).astype(int)
    f['en_k_on_next_lock'] = ((3 - f['en_kring']) == stage).astype(int)
    nxt = np.array([8, 16, 24, 26])[np.clip(stage, 0, 3)]
    f['rounds_to_lock'] = nxt - rnd
    return f


def _rule_worker(args):
    obs_list, = args
    sys.path.insert(0, str(ENV))
    from colony_one.board import Board
    from colony_one.pieces import PieceType
    from colony_one.rules import legal_actions_for_team

    def kernel_pos(b, team):
        for r in range(8):
            for c in range(8):
                p = b.grid[r][c]
                if p is not None and p.team == team and p.piece_type is PieceType.KERNEL:
                    return (r, c)
        return None

    def win_in1(b, team, enemy):
        kp = kernel_pos(b, enemy)
        if kp is None:
            return False, 0
        acts = [a for a in legal_actions_for_team(b, team) if a.to_pos == kp]
        if not acts:
            return False, 0
        return b._pick_guard(kp, enemy) is None, len(acts)

    def win_in2(b, team, enemy):
        w, _ = win_in1(b, team, enemy)
        if w:
            return True
        for a in legal_actions_for_team(b, team):
            rec = b.apply_action(a)
            try:
                if rec.captured_piece is not None and rec.captured_piece.piece_type is PieceType.KERNEL:
                    return True
                if win_in1(b, team, enemy)[0]:
                    return True
            finally:
                b.undo_action(rec)
        return False

    out = []
    for o in obs_list:
        o = o.astype(np.float32)
        team = 'MAGENTA' if o[15, 0, 0] > 0.5 else 'CYAN'
        enemy = 'CYAN' if team == 'MAGENTA' else 'MAGENTA'
        al = int(round(o[13, 0, 0] * 2))
        b = Board.from_observation(o, team)
        mob_own = len(legal_actions_for_team(b, team))
        mob_en = len(legal_actions_for_team(b, enemy))
        w1, natk_en = win_in1(b, team, enemy)
        _, natk_own = win_in1(b, enemy, team)
        own_win_turn = win_in2(b, team, enemy) if al == 2 else w1
        en_win_2 = win_in2(b, enemy, team)  # static: enemy's next full turn on the current board
        out.append((mob_own, mob_en, natk_en, natk_own, int(w1), int(own_win_turn), int(en_win_2)))
    return np.array(out, np.int32)


def main():
    files = sorted(DATA.glob('chunk*.npz'))
    obs_all, z_all, q_all, left_all, pol_all, pidx_all, pval_all, gid_all = [], [], [], [], [], [], [], []
    g0 = 0
    games_per_chunk = []
    for fi in files:
        d = np.load(fi)
        left = d['left']
        ends = np.where(left == 1)[0]
        starts = np.r_[0, ends[:-1] + 1]
        gid = np.empty(len(left), np.int32)
        for k, (s, e) in enumerate(zip(starts, ends)):
            gid[s:e + 1] = g0 + k
            assert left[s] == e - s + 1
        g0 += len(ends)
        games_per_chunk.append(len(ends))
        obs_all.append(d['obs']); z_all.append(d['z']); q_all.append(d['q']); left_all.append(left)
        pol_all.append(d['has_pol']); gid_all.append(gid); pidx_all.append(d['pidx']); pval_all.append(d['pval'])
    obs = np.concatenate(obs_all)
    z = np.concatenate(z_all); q = np.concatenate(q_all); left = np.concatenate(left_all)
    gid = np.concatenate(gid_all); has_pol = np.concatenate(pol_all)
    pidx = np.concatenate(pidx_all); pval = np.concatenate(pval_all)
    print('positions', len(obs), 'games', g0, 'per chunk', set(games_per_chunk), flush=True)

    feats = vector_features(obs.astype(np.float32))

    # rule-based tactical features, 2 processes
    parts = np.array_split(np.arange(len(obs)), 40)
    with Pool(2) as pool:
        res = pool.map(_rule_worker, [(obs[p],) for p in parts])
    tac = np.concatenate(res)
    for i, nm in enumerate(['mob_own', 'mob_en', 'n_atk_en_kernel', 'n_atk_own_kernel',
                            'own_win_now', 'own_win_this_turn', 'en_win_next_turn_static']):
        feats[nm] = tac[:, i]
    print('tactical done', flush=True)

    np.savez_compressed(OUT / 'features.npz', z=z, q=q, left=left, gid=gid, has_pol=has_pol,
                        **{k: np.asarray(v) for k, v in feats.items()})

    # final-position classification by replaying every legal action of the last position
    sys.path.insert(0, str(ENV))
    from colony_one import encoding
    from colony_one.board import Board
    from colony_one.env import ColonyOneEnv
    from colony_one.pieces import PieceType
    finals = []
    last_idx = np.where(left == 1)[0]
    base = ColonyOneEnv()
    for i in last_idx:
        o = obs[i].astype(np.float32)
        team = 'MAGENTA' if o[15, 0, 0] > 0.5 else 'CYAN'
        al = int(feats['actions_left'][i]); rnd = int(feats['round'][i])
        mover_won = z[i] > 0
        cats = {}
        pol = {int(a): float(v) for a, v in zip(pidx[i], pval[i]) if a >= 0}
        env = base.clone()
        env.board = Board.from_observation(o, team)
        env.current_team = team; env.actions_left = al; env.round_number = rnd
        env.terminated = False; env.winner = None
        for act in env.legal_actions():
            e2 = env.clone()
            locked_before = sum(map(sum, e2.board.locked))
            _, _, term, info = e2.step(act)
            if not term:
                continue
            won = info['winner'] == team
            if won != mover_won:
                continue
            if e2.board.only_kernels_remain() and sum(map(sum, e2.board.locked)) == locked_before:
                cat = 'bare_kernels'
            elif sum(map(sum, e2.board.locked)) > locked_before:
                cat = 'firewall'
            else:
                cat = 'kernel_capture'
            idx = encoding.action_index(act, team)
            cats[cat] = cats.get(cat, 0.0) + pol.get(idx, 0.0) + 1e-6
        cat = max(cats, key=cats.get) if cats else 'unexplained'
        finals.append(dict(game=int(gid[i]), team=team, actions_left=al, round=rnd,
                           mover_won=bool(mover_won), category=cat, candidates=cats))
    (OUT / 'final_positions.json').write_text(json.dumps(finals))
    from collections import Counter
    print(Counter(f['category'] for f in finals))


if __name__ == '__main__':
    main()
