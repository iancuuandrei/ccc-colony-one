"""Check net.Features win flag (R1 or R2) against engine brute force on self-play positions.

python test_win.py [n_positions]
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'env')); sys.path.insert(0, str(ROOT / 'verify_win'))
from colony_one import constants
from colony_one.board import Board
from verify_win import brute_force
from net import Features

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
obs = np.concatenate([np.load(f)['obs'] for f in sorted((ROOT / 'data' / 'gen00').glob('chunk*.npz'))])
rng = np.random.default_rng(1)
idx = rng.choice(len(obs), min(n, len(obs)), replace=False)
x = torch.tensor(obs[idx].astype(np.float32))
with torch.no_grad():
    _, win, _ = Features()(x)
win = win.reshape(-1).numpy() > 0.5
viol = miss = kills = 0; examples = []
for j, i in enumerate(idx):
    o = obs[i].astype(np.float32)
    al = int(round(float(o[13, 0, 0]) * 2))
    kill = brute_force(Board.from_observation(o, constants.CYAN), al)[0]
    kills += kill
    if win[j] and not kill:
        viol += 1; examples.append(int(i))
    if kill and not win[j]:
        miss += 1
print(json.dumps({'checked': len(idx), 'kills': int(kills), 'flagged': int(win.sum()),
                  'violations': viol, 'misses': miss, 'violation_examples': examples[:10]}))
