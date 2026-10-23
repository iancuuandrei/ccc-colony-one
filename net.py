"""Policy/value network for Colony: One with exact in-graph attack features.

Everything exported uses allow-listed ONNX ops (MatMul, Conv, Clip, ReduceMean, ...).
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

COORDS = np.array([(i // 8, i % 8) for i in range(64)])
ORTH = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL8 = ORTH + DIAG
KNIGHT = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

def shift(dr, dc):
    m = np.zeros((64, 64), np.float32)
    for p, (r, c) in enumerate(COORDS):
        rr, cc = r + dr, c + dc
        if 0 <= rr < 8 and 0 <= cc < 8:
            m[p, rr * 8 + cc] = 1
    return m

N_FEAT = 16 + 12 + 5  # raw planes, attack maps, empty + attack totals + kernel-zone + win flag

class Features(nn.Module):
    """Attack maps per piece type and team, following rules.py (rays stop at pieces)."""

    def __init__(self):
        super().__init__()
        self.register_buffer('S', torch.tensor(np.stack([shift(*d) for d in ALL8])))
        self.register_buffer('ADJ', torch.tensor(sum(shift(*d) for d in ALL8)))
        self.register_buffer('KN', torch.tensor(sum(shift(*d) for d in KNIGHT)))
        self.register_buffer('ONES', torch.ones(1, 64))

    def rays(self, src, dirs, lo, hi, empty):
        out = 0
        for d in dirs:
            cur = src @ self.S[ALL8.index(d)]
            for k in range(1, hi + 1):
                if k >= lo:
                    out = out + cur
                if k < hi:
                    cur = (cur * empty) @ self.S[ALL8.index(d)]
        return out

    def forward(self, x):
        f = x.reshape(-1, 16, 64)
        occ = f[:, 0:12].sum(1)
        locked = f[:, 12]
        empty = torch.clamp(1.0 - occ - locked, 0.0, 1.0)
        maps = []
        totals = []
        raw = []
        for base in (0, 6):
            ping, dae, exp, tra, ice, ker = [f[:, base + i] for i in range(6)]
            team = [
                self.rays(ping, DIAG, 1, 2, empty),
                dae @ self.ADJ + self.rays(dae, ORTH, 2, 3, empty),
                exp @ self.KN,
                self.rays(tra, DIAG, 2, 3, empty),
                self.rays(ice, ALL8, 1, 2, empty),
                ker @ self.ADJ,
            ]
            maps += [torch.clamp(t, 0.0, 1.0) for t in team]
            totals.append(sum(team) * 0.25)
            raw.append(sum(team))
        kzone = (f[:, 5] @ self.ADJ) + (f[:, 11] @ self.ADJ)
        # Exact immediate win: an attack reaches the enemy Kernel and it has fewer
        # guards than we have actions left (each redirect removes one guard and
        # leaves the attacker in place, so the attack repeats).
        ek = f[:, 11]
        hits = torch.clamp((raw[0] * ek).sum(1, keepdim=True), 0.0, 1.0)
        guards = ((ek @ self.ADJ) * f[:, 6:11].sum(1)).sum(1, keepdim=True)
        acts = f[:, 13].mean(1, keepdim=True) * 2.0
        win = hits * torch.clamp(acts - guards, 0.0, 1.0)            # (B,1)
        extra = torch.stack(maps + [empty, totals[0], totals[1], kzone, win * self.ONES], 1)
        return torch.cat([f, extra], 1).reshape(-1, N_FEAT, 8, 8), win, ek


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Linear(c, c // r)
        self.fc2 = nn.Linear(c // r, 2 * c)

    def forward(self, x):
        s = x.mean((2, 3))
        g, bias = self.fc2(F.relu(self.fc1(s))).chunk(2, 1)
        return x * torch.sigmoid(g)[:, :, None, None] + bias[:, :, None, None]


class Block(nn.Module):
    def __init__(self, c, se):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(c)
        self.se = SE(c) if se else None

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        if self.se is not None:
            y = self.se(y)
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self, channels=48, blocks=5, emb=32):
        super().__init__()
        self.cfg = dict(channels=channels, blocks=blocks, emb=emb)
        self.feat = Features()
        self.stem = nn.Sequential(nn.Conv2d(N_FEAT, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(channels, se=(i % 2 == 1)) for i in range(blocks)])
        self.emb = emb
        self.p_conv = nn.Conv2d(channels, 2 * emb + 2, 1)
        self.v_conv = nn.Conv2d(channels, 8, 1)
        self.v_fc1 = nn.Linear(8 * 64 + channels, 128)
        self.v_fc2 = nn.Linear(128, 1)
        self.aux = nn.Linear(128, 2)  # root-Q and log remaining actions (training only)

    def trunk(self, x):
        feats, win, ek = self.feat(x)
        return self.blocks(self.stem(feats)), win, ek

    def heads(self, h):
        p = self.p_conv(h).reshape(-1, 2 * self.emb + 2, 64)
        fr, to = p[:, :self.emb], p[:, self.emb:2 * self.emb]
        logits = torch.matmul(fr.transpose(1, 2), to) * (self.emb ** -0.5)
        logits = logits + p[:, 2 * self.emb][:, :, None] + p[:, 2 * self.emb + 1][:, None, :]
        v = F.relu(self.v_conv(h)).reshape(-1, 8 * 64)
        v = F.relu(self.v_fc1(torch.cat([v, h.mean((2, 3))], 1)))
        return logits.reshape(-1, 4096), v

    def outputs(self, x):
        h, win, ek = self.trunk(x)
        logits, v = self.heads(h)
        value = torch.tanh(self.v_fc2(v)).reshape(-1)
        w = win.reshape(-1)
        value = value + w * (0.99 - value)
        # Every legal action aimed at the enemy Kernel square is an attack on it.
        logits = (logits.reshape(-1, 64, 64) + (win * 30.0 * ek).reshape(-1, 1, 64)).reshape(-1, 4096)
        return logits, value, v

    def forward(self, x):
        logits, value, _ = self.outputs(x)
        return logits, value

    def forward_train(self, x):
        logits, value, v = self.outputs(x)
        return logits, value, self.aux(v)


def export(net: Net, path):
    net = net.eval().cpu()
    x = torch.zeros(2, 16, 8, 8)
    torch.onnx.export(net, (x,), str(path), input_names=['observation'], output_names=['policy', 'value'],
                      dynamic_axes={'observation': {0: 'batch'}, 'policy': {0: 'batch'}, 'value': {0: 'batch'}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    import onnx
    m = onnx.load(str(path))
    onnx.save(m, str(path), save_as_external_data=False)
