"""Train the policy/value net on self-play chunks with 8-fold board symmetry.

python train.py --data data/gen00 data/gen01 --init ckpt/g00.pt --out ckpt/g01.pt --steps 6000
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from net import Net, export

def sq_maps():
    maps = []
    for t in range(8):
        m = np.zeros(64, np.int64)
        for s in range(64):
            r, c = divmod(s, 8)
            for _ in range(t % 4):
                r, c = c, 7 - r          # rotate 90
            if t >= 4:
                c = 7 - c                # mirror
            m[s] = r * 8 + c
        maps.append(m)
    return np.stack(maps)

def chunk_files(dirs, holdout_last):
    files = [f for d in dirs for f in sorted(Path(d).glob('chunk*.npz'))]
    if holdout_last:
        last = sorted(Path(dirs[-1]).glob('chunk*.npz'))[-1]
        return [f for f in files if f != last], [last]
    return files, []

def load(files):
    parts = []
    for f in files:
        if True:
            z = np.load(f)
            d = {k: z[k] for k in z.files}
            if 'q_valid' not in d:  # legacy chunks: Q may come from a pool opponent's search
                d['q_valid'] = np.zeros(len(d['z']), bool)
            parts.append(d)
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return cat

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', nargs='+', required=True)
    p.add_argument('--holdout-last', action='store_true', help='hold out the last chunk of the last data dir')
    p.add_argument('--init')
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--wd', type=float, default=3e-5)
    p.add_argument('--qmix', type=float, default=0.2)
    p.add_argument('--channels', type=int, default=48)
    p.add_argument('--blocks', type=int, default=5)
    p.add_argument('--emb', type=int, default=32)
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_files, hold_files = chunk_files(a.data, a.holdout_last)
    D = load(train_files)
    n = len(D['z'])
    T = lambda k, dt: torch.tensor(D[k]).to(dev, dt)
    obs = T('obs', torch.float16); pidx = T('pidx', torch.long); pval = T('pval', torch.float32)
    z = T('z', torch.float32); q = T('q', torch.float32); qv = T('q_valid', torch.float32); has = T('has_pol', torch.bool); left = T('left', torch.float32)
    H = None
    if hold_files:
        Dh = load(hold_files)
        H = {k: torch.tensor(Dh[k]).to(dev) for k in Dh}
    maps = torch.tensor(sq_maps(), device=dev)
    if a.init:
        ck = torch.load(a.init, map_location='cpu')
        net = Net(**ck['cfg']); net.load_state_dict(ck['state'])
    else:
        net = Net(a.channels, a.blocks, a.emb)
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)

    def augment(o, pi, t):
        b = o.shape[0]
        m = maps[t]                                    # (B,64)
        flat = o.reshape(b, 16, 64).float()
        out = torch.empty_like(flat)
        out.scatter_(2, m[:, None, :].expand(b, 16, 64), flat)
        valid = pi >= 0
        fr, to = pi.clamp(min=0) // 64, pi.clamp(min=0) % 64
        npi = torch.gather(m, 1, fr) * 64 + torch.gather(m, 1, to)
        return out.reshape(b, 16, 8, 8), torch.where(valid, npi, torch.full_like(npi, -1))

    def losses(o, pi, pv, zz, qq, hh, ll, qvv, train=True):
        logits, v, aux = net.forward_train(o)
        valid = pi >= 0
        lg = torch.gather(logits, 1, pi.clamp(min=0)).masked_fill(~valid, -1e9)
        logp = F.log_softmax(lg, 1)
        ce = -(pv * logp).sum(1)
        ent = -(pv * torch.log(pv.clamp(min=1e-8))).sum(1)
        pl = ((ce - ent) * hh).sum() / hh.sum().clamp(min=1)       # KL to search policy
        mix = a.qmix * qvv
        tgt = (1 - mix) * zz + mix * qq
        vl = F.mse_loss(v, tgt)
        al = ((torch.tanh(aux[:, 0]) - qq) ** 2 * qvv).sum() / qvv.sum().clamp(min=1)             + 0.05 * F.mse_loss(aux[:, 1], torch.log(ll))
        return pl, vl, al

    t0 = time.time(); log = []
    for step in range(a.steps):
        net.train()
        i = torch.randint(0, n, (a.batch,), device=dev)
        t = torch.randint(0, 8, (a.batch,), device=dev)
        o, pi = augment(obs[i], pidx[i], t)
        pl, vl, al = losses(o, pi, pval[i], z[i], q[i], has[i].float(), left[i], qv[i])
        loss = pl + 1.5 * vl + 0.25 * al
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()
        if (step + 1) % 500 == 0 or step + 1 == a.steps:
            rec = {'step': step + 1, 'policy_kl': float(pl), 'value_mse': float(vl), 'aux': float(al),
                   'sec': round(time.time() - t0, 1)}
            if H is not None:
                net.eval()
                with torch.no_grad():
                    hl = [0.0, 0.0]; m = len(H['z']); cnt = 0; pcnt = 0
                    for s in range(0, m, 4096):
                        sl = slice(s, s + 4096)
                        o = H['obs'][sl].float(); w = H['has_pol'][sl].float()
                        pl2, vl2, _ = losses(o, H['pidx'][sl].long(), H['pval'][sl].float(), H['z'][sl],
                                             H['q'][sl], w, H['left'][sl], H['q_valid'][sl].float(), train=False)
                        k = len(H['z'][sl]); kp = float(w.sum())
                        hl[0] += float(pl2) * kp; hl[1] += float(vl2) * k; cnt += k; pcnt += kp
                rec.update(hold_policy_kl=hl[0] / max(pcnt, 1), hold_value_mse=hl[1] / cnt)
            log.append(rec); print(json.dumps(rec), flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    export(net, out.with_suffix('.onnx'))          # ONNX first: a .pt means a complete export
    torch.save({'cfg': net.cfg, 'state': state, 'log': log, 'args': vars(a), 'positions': n,
                'train_files': [str(f) for f in train_files], 'holdout_files': [str(f) for f in hold_files]}, out)
    print(json.dumps({'saved': str(out), 'onnx': str(out.with_suffix('.onnx')), 'positions': n}))

if __name__ == '__main__':
    main()
