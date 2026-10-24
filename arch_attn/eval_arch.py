"""Evaluate checkpoints (attn or cnn) on chunk folders with train.py's holdout metrics.

wsl -e bash .../wpy.sh arch_attn/eval_arch.py --ckpt arch_attn/ckpt_attn.pt arch_attn/ckpt_cnn.pt --data data/sp_...
Policy KL to the search policy over legal moves (rows with a policy target), value MSE to the
train.py target (1 - 0.2*q_valid) * z + 0.2*q_valid * q, both without augmentation.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F
from train import chunk_files, load      # noqa: E402
from net_attn import load_ckpt           # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--ckpt', nargs='+', required=True)
p.add_argument('--data', nargs='+', required=True)
p.add_argument('--qmix', type=float, default=0.2)
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
files, _ = chunk_files(a.data, False)
H = {k: torch.tensor(v).to(dev) for k, v in load(files).items()}
res = {'rows': int(len(H['z'])), 'policy_rows': int(H['has_pol'].sum()), 'files': len(files)}
for c in a.ckpt:
    net, ck = load_ckpt(c)
    net.to(dev).eval()
    kl_sum = 0.0; mse_sum = 0.0; pw = 0.0; vw = 0.0; top1 = 0.0
    with torch.no_grad():
        for s in range(0, len(H['z']), 4096):
            sl = slice(s, s + 4096)
            logits, v, _ = net.forward_train(H['obs'][sl].float())
            pi = H['pidx'][sl].long(); pv = H['pval'][sl].float(); valid = pi >= 0
            lg = torch.gather(logits, 1, pi.clamp(min=0)).masked_fill(~valid, -1e9)
            kl = -(pv * F.log_softmax(lg, 1)).sum(1) + (pv * torch.log(pv.clamp(min=1e-8))).sum(1)
            w = H['has_pol'][sl].float(); hv = H['has_value'][sl].float()
            agree = (lg.argmax(1) == pv.argmax(1)).float()
            mix = a.qmix * H['q_valid'][sl].float()
            tgt = (1 - mix) * H['z'][sl] + mix * H['q'][sl]
            kl_sum += float((kl * w).sum()); top1 += float((agree * w).sum()); pw += float(w.sum())
            mse_sum += float((((v - tgt) ** 2) * hv).sum()); vw += float(hv.sum())
    res[c] = {'arch': ck.get('arch', 'cnn'), 'policy_kl': kl_sum / pw, 'top1_vs_search': top1 / pw,
                         'value_mse': mse_sum / vw}
print(json.dumps(res))
