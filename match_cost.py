"""CPU cost of one side in Level 4 matches (100 referee sims), single-threaded ONNX.

python match_cost.py model.onnx [opponent.onnx] [--games 3]
Reports per-side process CPU split into evaluator time and search overhead.
"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'env'))
from colony_one import constants, mcts
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator

args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--games"]
games = int(sys.argv[sys.argv.index('--games') + 1]) if '--games' in sys.argv else 3
me = OnnxEvaluator(args[0], num_threads=1)
opp = OnnxEvaluator(args[1] if len(args) > 1 else args[0], num_threads=1)
ev_cpu = 0.0
def timed(planes):
    global ev_cpu
    t = time.process_time(); r = me(planes); ev_cpu += time.process_time() - t
    return r
cfg = mcts.MCTSConfig(simulations=100)
for g in range(games):
    env = ColonyOneEnv(); rng = np.random.default_rng(g); side_cpu = 0.0; ev_cpu = 0.0; n = 0
    me_cyan = g % 2 == 0
    while not env.terminated:
        mine = (env.current_team == constants.CYAN) == me_cyan
        t = time.process_time()
        a = mcts.search_action(env.board, env.current_team, env.actions_left, env.round_number,
                               timed if mine else opp, cfg, rng=rng)
        if mine: side_cpu += time.process_time() - t; n += 1
        env.step(a)
    print(f'game {g}: my actions {n}, side CPU {side_cpu:.2f}s, evaluator CPU {ev_cpu:.2f}s, winner {env.winner}')
