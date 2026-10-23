"""Self-play data generation under the referee MCTS (expert iteration).

python selfplay.py --model best.onnx --games 2000 --out data/gen01 [--pool a.onnx b.onnx]

Playout-cap randomisation (KataGo): a fraction of moves gets a full search and
becomes a policy target; the rest use a cheap search and only feed value
targets. Every position stores z (final result, side-to-move view) and the
root Q of its search.
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'env'))
K = 128  # max stored policy entries per position
_ev = {}

def evaluator(path):
    if path not in _ev:
        from colony_one.onnx_runner import OnnxEvaluator
        _ev[path] = OnnxEvaluator(path)
    return _ev[path]

def round_of(ob):
    locked = int(ob[12].sum())
    done = {0: 0, 28: 1, 48: 2, 60: 3}.get(locked, 0)
    nxt = (8, 16, 24, 26)[min(done, 3)]
    return max(1, int(round(nxt - 8 * (1 - float(ob[14, 0, 0])))))

_restart = {}

def restart_state(path, rng):
    if path not in _restart:
        _restart[path] = np.load(path)['obs']
    arr = _restart[path]
    return arr[int(rng.integers(len(arr)))].astype(np.float32)

def play(job):
    from colony_one import constants, encoding, mcts
    from colony_one.board import Board
    from colony_one.env import ColonyOneEnv
    learner, opponent, seed, o = job
    rng = np.random.default_rng(seed)
    learner_is_cyan = bool(seed % 2)
    ev_l = evaluator(learner)
    ev_o = evaluator(opponent) if opponent else ev_l
    # Root noise only applies inside the warmup window; widen it to 16 actions for self-play.
    warm = dict(dirichlet_eps=0.25, warmup_temperatures=(1.0, 1.0, 0.5, 0.5), warmup_step_actions=4)
    full = mcts.MCTSConfig(simulations=o['full_sims'], **warm)
    fast = mcts.MCTSConfig(simulations=o['fast_sims'], **warm)
    env = ColonyOneEnv()
    restarted = False
    if o.get('restart_file') and rng.random() < o.get('restart_prob', 0.0):
        # Search control (Go-Exploit / RGSC style): continue from an archived hard state.
        ob = restart_state(o['restart_file'], rng)
        team = constants.MAGENTA if ob[15, 0, 0] > 0.5 else constants.CYAN
        env.board = Board.from_observation(ob, team)
        env.current_team = team
        env.actions_left = int(round(float(ob[13, 0, 0]) * 2))
        env.round_number = round_of(ob)
        env.action = 100  # past the opening window
        restarted = True
    rec = []
    n = 0
    while not env.terminated:
        team = env.current_team
        is_learner = opponent is None or ((team == constants.CYAN) == learner_is_cyan)
        is_full = rng.random() < o['full_prob']
        cfg = full if is_full else fast
        planes = encoding.encode_board(env.board, team, env.actions_left, env.round_number)
        root = mcts.run(env.board, team, env.actions_left, env.round_number,
                        ev_l if is_learner else ev_o, cfg, rng=rng)
        idx, cnt = mcts.visit_distribution(root)
        cnt = np.asarray(cnt, np.float64)
        tot = cnt.sum()
        # Gumbel completed-Q improved policy (evolve_targets study: 50/50 mix with visits cut regret ~1/3)
        gpol = None
        if is_full and is_learner and len(idx) > 1:
            from policy_targets import gumbel
            prior = np.array([root.P[i] for i in idx]); Qa = np.array([root.W[i] / root.N[i] if root.N[i] else np.nan for i in idx])
            v_net = float((ev_l if is_learner else ev_o)(planes[None])[1][0])
            gpol = gumbel(prior, cnt, Qa, v_net)
        q = float(sum(root.W.values()) / max(1, sum(root.N.values())))
        if restarted:
            temp = 0.7 if n < 2 else (0.3 if n < 4 else 0.0)
        else:
            temp = 1.0 if n < o['temp1_actions'] else (0.5 if n < o['temp2_actions'] else 0.0)
        a = mcts.select_action_index(root, temp, rng=rng)
        if len(idx) > K:  # keep the most visited
            order = np.argsort(-cnt)[:K]
            idx = [idx[i] for i in order]; cnt = cnt[order]
            gpol = gpol[order] if gpol is not None else None
        rec.append((planes, team, idx, cnt / max(tot, 1), q, bool(is_full and is_learner), env.round_number, is_learner,
                    gpol if gpol is not None else cnt / max(tot, 1)))
        env.step(encoding.decode_action(a, team, env.board))
        n += 1
    m = len(rec)
    obs = np.stack([r[0] for r in rec]).astype(np.float16)
    pidx = np.full((m, K), -1, np.int16); pval = np.zeros((m, K), np.float16)
    pval_g = np.zeros((m, K), np.float16)
    for i, r in enumerate(rec):
        pidx[i, :len(r[2])] = r[2]; pval[i, :len(r[2])] = r[3]; pval_g[i, :len(r[2])] = r[8]
    z = np.array([1.0 if r[1] == env.winner else -1.0 for r in rec], np.float32)
    q = np.array([r[4] for r in rec], np.float32)
    has_pol = np.array([r[5] for r in rec], bool)
    q_valid = np.array([r[7] for r in rec], bool)  # root Q only from the learner's own search
    left = np.arange(m, 0, -1, dtype=np.float32)  # actions until game end
    learner_team = constants.CYAN if learner_is_cyan else constants.MAGENTA
    return dict(obs=obs, pidx=pidx, pval=pval, pval_g=pval_g, z=z, q=q, has_pol=has_pol, q_valid=q_valid, left=left,
                meta=(seed, env.winner, m, opponent is None or env.winner == learner_team, restarted))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--pool', nargs='*', default=[])
    p.add_argument('--pool-prob', type=float, default=0.2)
    p.add_argument('--games', type=int, default=1000)
    p.add_argument('--seed0', type=int, default=0)
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--full-sims', type=int, default=400)
    p.add_argument('--fast-sims', type=int, default=100)
    p.add_argument('--full-prob', type=float, default=0.35)
    p.add_argument('--temp1-actions', type=int, default=8)
    p.add_argument('--temp2-actions', type=int, default=16)
    p.add_argument('--out', required=True)
    p.add_argument('--chunk', type=int, default=200)
    p.add_argument('--restart-file', default=None, help='npz with obs of archived hard states')
    p.add_argument('--restart-prob', type=float, default=0.0)
    a = p.parse_args()
    o = dict(full_sims=a.full_sims, fast_sims=a.fast_sims, full_prob=a.full_prob,
             temp1_actions=a.temp1_actions, temp2_actions=a.temp2_actions,
             restart_file=str(Path(a.restart_file).resolve()) if a.restart_file and Path(a.restart_file).exists() else None,
             restart_prob=a.restart_prob)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = str(Path(a.model).resolve()); pool = [str(Path(x).resolve()) for x in a.pool]
    rng = np.random.default_rng(a.seed0 + 12345)
    jobs = []
    for g in range(a.games):
        opp = pool[int(rng.integers(len(pool)))] if pool and rng.random() < a.pool_prob else None
        jobs.append((model, opp, a.seed0 + g, o))
    t = time.time(); buf = []; ci = 0; stats = []
    def flush():
        nonlocal buf, ci
        if not buf: return
        cat = {k: np.concatenate([b[k] for b in buf]) for k in buf[0] if k != 'meta'}
        np.savez_compressed(out / f'chunk{ci:04d}.npz', **cat)
        ci += 1; buf = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(play, jobs, chunksize=1)):
            buf.append(r); stats.append(r['meta'])
            if len(buf) >= a.chunk: flush()
            if (i + 1) % 100 == 0:
                print(json.dumps({'games': i + 1, 'seconds': round(time.time() - t, 1)}), flush=True)
    flush()
    cyan = sum(1 for s in stats if s[1] == 'CYAN')
    summary = {'games': len(stats), 'cyan_wins': cyan, 'restarted': int(sum(1 for s in stats if s[4])), 'positions': int(sum(s[2] for s in stats)),
               'mean_len': float(np.mean([s[2] for s in stats])), 'seconds': time.time() - t,
               'model': a.model, 'pool': a.pool, 'opts': o}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))

if __name__ == '__main__':
    main()
