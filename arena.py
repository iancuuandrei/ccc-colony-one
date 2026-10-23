"""Parallel referee-MCTS matches between two ONNX models on the Level 4 scene.

python arena.py A.onnx B.onnx --games 64 --sims 100 --workers 30
Each seed is played twice (A as CYAN, then A as MAGENTA).
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'env'))

_agents = {}

def _evaluator(path):
    if path not in _agents:
        import onnxruntime as ort
        from colony_one.onnx_runner import OnnxEvaluator
        _agents[path] = OnnxEvaluator(path)
    return _agents[path]

def play(args):
    a, b, seed, a_is_cyan, sims = args
    from colony_one import constants, mcts
    from colony_one.env import ColonyOneEnv
    ea, eb = _evaluator(a), _evaluator(b)
    cfg = mcts.MCTSConfig(simulations=sims)
    rng = np.random.default_rng(seed * 2 + int(a_is_cyan))
    env = ColonyOneEnv()
    hist = []
    while not env.terminated:
        a_turn = (env.current_team == constants.CYAN) == a_is_cyan
        ev = ea if a_turn else eb
        act = mcts.search_action(env.board, env.current_team, env.actions_left, env.round_number, ev, cfg, rng=rng)
        hist.append([env.current_team[0], list(act.from_pos), list(act.to_pos)])
        env.step(act)
    a_team = constants.CYAN if a_is_cyan else constants.MAGENTA
    return {'seed': seed, 'a_cyan': a_is_cyan, 'a_won': env.winner == a_team,
            'winner': env.winner, 'actions': env.action, 'round': env.round_number, 'history': hist}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('a'); p.add_argument('b')
    p.add_argument('--games', type=int, default=32, help='seeds; each played both colours')
    p.add_argument('--sims', type=int, default=100)
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--out')
    o = p.parse_args()
    a, b = str(Path(o.a).resolve()), str(Path(o.b).resolve())
    jobs = [(a, b, s, c, o.sims) for s in range(o.start, o.start + o.games) for c in (True, False)]
    t = time.time()
    with ProcessPoolExecutor(o.workers) as ex:
        res = list(ex.map(play, jobs))
    wins = sum(r['a_won'] for r in res)
    cy = [r for r in res if r['a_cyan']]; mg = [r for r in res if not r['a_cyan']]
    summary = {'a': o.a, 'b': o.b, 'games': len(res), 'a_wins': wins, 'a_rate': wins / len(res),
               'a_cyan_wins': sum(r['a_won'] for r in cy), 'a_magenta_wins': sum(r['a_won'] for r in mg),
               'mean_actions': float(np.mean([r['actions'] for r in res])), 'seconds': time.time() - t}
    print(json.dumps(summary))
    if o.out:
        Path(o.out).write_text(json.dumps({'summary': summary, 'games': res}))

if __name__ == '__main__':
    main()
