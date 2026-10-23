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
        win1 = hits * torch.clamp(acts - guards, 0.0, 1.0)           # rule R1, (B,1)
        # Rule R2 (two actions left): a piece relocates (empty square or capturing a
        # non-Kernel enemy) to a square from which it attacks an unguarded enemy Kernel;
        # with exactly one guard, the relocation may capture that guard.
        own = [f[:, i] for i in range(6)]
        enk = f[:, 6:11].sum(1)
        land = empty + enk
        dest = [
            self.rays(own[0], ORTH, 1, 2, empty) * empty + self.rays(own[0], DIAG, 1, 2, empty) * enk,
            self.rays(own[1], ORTH, 1, 3, empty) * empty + self.rays(own[1], ORTH, 2, 3, empty) * enk,
            (own[2] @ self.KN) * land,
            self.rays(own[3], ORTH, 1, 1, empty) * empty,
            self.rays(own[4], ALL8, 1, 2, empty) * land,
            (own[5] @ self.ADJ) * land,
        ]
        guard_sq = (ek @ self.ADJ) * enk
        g0 = torch.clamp(1.0 - guards, 0.0, 1.0)
        g1 = torch.clamp(1.0 - torch.abs(guards - 1.0), 0.0, 1.0)
        ok = g0 + g1 * guard_sq                                        # (B,64)
        r2 = 0
        for t in range(6):
            e2 = torch.clamp(empty + own[t], 0.0, 1.0)                 # origin vacated
            rev = [
                lambda: self.rays(ek, DIAG, 1, 2, e2),
                lambda: ek @ self.ADJ + self.rays(ek, ORTH, 2, 3, e2),
                lambda: ek @ self.KN,
                lambda: self.rays(ek, DIAG, 2, 3, e2),
                lambda: self.rays(ek, ALL8, 1, 2, e2),
                lambda: ek @ self.ADJ,
            ][t]()
            r2 = r2 + (dest[t] * rev * ok).sum(1, keepdim=True)
        two = torch.clamp(acts - 1.0, 0.0, 1.0)
        win2 = torch.clamp(r2, 0.0, 1.0) * two
        win = torch.clamp(win1 + win2, 0.0, 1.0)
        extra = torch.stack(maps + [empty, totals[0], totals[1], kzone, win * self.ONES], 1)
        return torch.cat([f, extra], 1).reshape(-1, N_FEAT, 8, 8), win, ek * win1


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Linear(c, c // r)
        self.fc2 = nn.Linear(c // r, 2 * c)
        self.c = c

    def forward(self, x):
        s = x.mean((2, 3))
        y = self.fc2(F.relu(self.fc1(s)))
        g, bias = y[:, :self.c], y[:, self.c:]
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
        # Export-time calibration for the fixed referee search: policy softmax
        # temperature and value scale (value scale s acts like c_puct / s).
        self.register_buffer('ptemp', torch.ones(1), persistent=False)
        self.register_buffer('vscale', torch.ones(1), persistent=False)

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
        value = torch.tanh(self.v_fc2(v)).reshape(-1) * self.vscale
        logits = logits / self.ptemp
        w = win.reshape(-1)
        value = value + w * (0.99 - value)
        # Every legal action aimed at the enemy Kernel square is an attack on it.
        # ek here is already gated by R1: every legal action onto the Kernel square is an attack.
        logits = (logits.reshape(-1, 64, 64) + (30.0 * ek).reshape(-1, 1, 64)).reshape(-1, 4096)
        return logits, value, v

    def forward(self, x):
        logits, value, _ = self.outputs(x)
        return logits, value

    def forward_train(self, x):
        logits, value, v = self.outputs(x)
        return logits, value, self.aux(v)


def export(net: Net, path, ptemp=1.0, vscale=1.0):
    net = net.eval().cpu()
    net.ptemp.fill_(ptemp); net.vscale.fill_(vscale)
    x = torch.zeros(2, 16, 8, 8)
    torch.onnx.export(net, (x,), str(path), input_names=['observation'], output_names=['policy', 'value'],
                      dynamic_axes={'observation': {0: 'batch'}, 'policy': {0: 'batch'}, 'value': {0: 'batch'}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(path))
    # The server allow-list has no Constant op: turn every Constant node into an initializer.
    keep = []
    for n in m.graph.node:
        if n.op_type == 'Constant':
            t = onnx.helper.get_attribute_value(n.attribute[0])
            arr = numpy_helper.to_array(t) if isinstance(t, onnx.TensorProto) else np.asarray(t)
            m.graph.initializer.append(numpy_helper.from_array(arr, n.output[0]))
        else:
            keep.append(n)
    del m.graph.node[:]
    m.graph.node.extend(keep)
    # Declare the exact contract signature (the exporter loses static dims through Reshape(-1)).
    from onnx import TensorProto, helper
    sig_in = helper.make_tensor_value_info('observation', TensorProto.FLOAT, ['batch', 16, 8, 8])
    sig_out = [helper.make_tensor_value_info('policy', TensorProto.FLOAT, ['batch', 4096]),
               helper.make_tensor_value_info('value', TensorProto.FLOAT, ['batch'])]
    del m.graph.input[:]; m.graph.input.extend([sig_in])
    del m.graph.output[:]; m.graph.output.extend(sig_out)
    init_names = {t.name for t in m.graph.initializer}
    kept_vi = [v for v in m.graph.value_info if v.name not in ('policy', 'value', 'observation')]
    del m.graph.value_info[:]; m.graph.value_info.extend(kept_vi)
    onnx.checker.check_model(m)
    onnx.save(m, str(path), save_as_external_data=False)
    net.ptemp.fill_(1.0); net.vscale.fill_(1.0)
