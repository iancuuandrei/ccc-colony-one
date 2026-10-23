"""Hand-built policy/value ONNX covering all six Level 4 pieces.

Attack maps follow env/colony_one/rules.py, including blocking along rays.
python build_heuristic.py out.onnx
"""
import sys, json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / 'model-tactical-v2.onnx')
W = dict(  # tunable weights
    mat=(1.0, 3.0, 2.2, 2.2, 3.2), guard=0.55, threat_own=0.55, threat_enemy=0.30,
    kernel_hit=1.6, ring=0.9, doomed=0.8, mobility=0.03, scale=0.30,
)
for arg in sys.argv[2:]:
    k, v = arg.split('=')
    W[k] = eval(v)

nodes, init = [], []

def const(x, dtype=np.float32):
    name = f'c{len(init)}'
    init.append(nh.from_array(np.asarray(x, dtype=dtype), name))
    return name

def op(kind, *args, **attrs):
    name = f'n{len(nodes)}'
    nodes.append(h.make_node(kind, list(args), [name], name=name, **attrs))
    return name

def C(x): return x if isinstance(x, str) else const(x)
def add(*a):
    x = C(a[0])
    for y in a[1:]: x = op('Add', x, C(y))
    return x
def mul(a, b): return op('Mul', C(a), C(b))
def sub(a, b): return op('Sub', C(a), C(b))
def mm(a, m): return op('MatMul', a, C(m))
def rs(a, shape): return op('Reshape', a, const(shape, np.int64))
def sum1(a): return op('ReduceSum', a, const([1], np.int64), keepdims=1)
def max1(a): return op('ReduceMax', a, axes=[1], keepdims=1)
def flt(a): return op('Cast', a, to=T.FLOAT)
def outer(a, b): return op('MatMul', rs(a, [-1, 64, 1]), rs(b, [-1, 1, 64]))
def clip(a, lo=0., hi=1.): return op('Clip', a, const(lo), const(hi))

coords = np.array([(i // 8, i % 8) for i in range(64)])
def shift(dr, dc):
    m = np.zeros((64, 64), np.float32)
    for p, (r, c) in enumerate(coords):
        rr, cc = r + dr, c + dc
        if 0 <= rr < 8 and 0 <= cc < 8: m[p, rr * 8 + cc] = 1
    return m
ORTH = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL8 = ORTH + DIAG
S = {d: const(shift(*d)) for d in ALL8}
adj8 = sum(shift(*d) for d in ALL8)
knight = sum(shift(*d) for d in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)])
ADJ, KN = const(adj8), const(knight)
ring = np.minimum.reduce([coords[:, 0], coords[:, 1], 7 - coords[:, 0], 7 - coords[:, 1]]).astype(np.float32)

x = rs('observation', [-1, 16, 64])
P = [op('Gather', x, const(i, np.int64), axis=1) for i in range(16)]   # each (B,64)
locked = P[12]
actions_left = op('ReduceMax', P[13], axes=[1], keepdims=1)  # 1.0 or 0.5
urgency = op('ReduceMax', P[14], axes=[1], keepdims=1)
occ = add(*P[:12])
empty = clip(sub(sub(1., occ), locked))
open_sq = sub(1., locked)

def rays(src, dirs, lo, hi):
    """Squares at distance lo..hi along dirs, path through empty squares."""
    out = []
    for d in dirs:
        cur = mm(src, S[d])
        for k in range(1, hi + 1):
            if k >= lo: out.append(cur)
            if k < hi: cur = mm(mul(cur, empty), S[d])
    return add(*out)

def team(base):
    ping, dae, exp, tra, ice, ker = P[base:base + 6]
    att = {
        'ping': rays(ping, DIAG, 1, 2),
        'dae': add(mm(dae, ADJ), rays(dae, ORTH, 2, 3)),
        'exp': mm(exp, KN),
        'tra': rays(tra, DIAG, 2, 3),
        'ice': rays(ice, ALL8, 1, 2),
        'ker': mm(ker, ADJ),
    }
    attacks = clip(add(*att.values()), 0., 4.)
    vals = W['mat']
    pieces_v = add(mul(ping, vals[0]), mul(dae, vals[1]), mul(exp, vals[2]), mul(tra, vals[3]), mul(ice, vals[4]))
    mobility = add(mm(ping, adj8), mm(dae, adj8), mm(ice, adj8), mm(tra, adj8))
    others = add(ping, dae, exp, tra, ice)
    guards = clip(sum1(mul(mm(ker, ADJ), others)), 0., 3.)
    # Actual sacrifice order is Ping, Exploit, Trace, Daemon, Ice.
    adj_kernel = mm(ker, ADJ)
    available = const(1.)
    sacrificed = const(0.)
    for piece, val in [(ping, vals[0]), (exp, vals[2]), (tra, vals[3]), (dae, vals[1]), (ice, vals[4])]:
        present = clip(sum1(mul(adj_kernel, piece)))
        sacrificed = add(sacrificed, mul(mul(available, present), val))
        available = mul(available, sub(1., present))
    ring_k = sum1(mul(ker, const(ring)))
    return dict(ker=ker, attacks=attacks, pv=pieces_v, others=others, guards=guards,
                guard_value=sacrificed, material=sum1(pieces_v), mobility=sum1(mul(mobility, empty)), ring=ring_k)

own, en = team(0), team(6)
att_own = clip(own['attacks'])
att_en = clip(en['attacks'])

# Capturable value: enemy piece values on squares we attack; a guarded kernel costs a guard.
own_threat = max1(mul(en['pv'], att_own))
en_threat = max1(mul(own['pv'], att_en))
own_hits_kernel = sum1(mul(en['ker'], att_own))
en_hits_kernel = sum1(mul(own['ker'], att_en))
unguarded_en = op('Relu', sub(1., en['guards']))
unguarded_own = op('Relu', sub(1., own['guards']))
# If currently legal attack reaches Kernel and the remaining actions exceed
# guard count, repeating that same attack wins before opponent can react.
# Guard redirection leaves attacker put and removes no square on its clear ray.
forced_win = mul(own_hits_kernel, flt(op('Greater', mul(actions_left, 2.), en['guards'])))
# Conservative sufficient certificate for quiet approach followed by Kernel
# capture. Current source occupancy is retained in reverse rays. Removing that
# source after the quiet move can only clear a line, never invalidate it.
# This intentionally omits tactics requiring source vacancy to open the ray.
approach_enabled = mul(flt(op('Greater', actions_left, const(0.75))), unguarded_en)
reverse_attacks = [rays(en['ker'], DIAG, 1, 2),
                   add(mm(en['ker'], ADJ), rays(en['ker'], ORTH, 2, 3)),
                   mm(en['ker'], KN), rays(en['ker'], DIAG, 2, 3),
                   rays(en['ker'], ALL8, 1, 2), mm(en['ker'], ADJ)]
quiet_moves = [rays(P[0], ORTH, 1, 2), rays(P[1], ORTH, 1, 3),
               mm(P[2], KN), add(*(mm(P[3], S[d]) for d in ORTH)),
               rays(P[4], ALL8, 1, 2), mm(P[5], ADJ)]
approach_policy = []
for piece, quiet, attack_from in zip(P[:6], quiet_moves, reverse_attacks):
    good_dest = mul(mul(mul(quiet, empty), clip(attack_from)), approach_enabled)
    approach_policy.append(outer(piece, good_dest))
approach_pairs = clip(add(*approach_policy))
approach_exists = rs(max1(rs(approach_pairs, [-1, 4096])), [-1, 1])
certified_win = clip(add(forced_win, approach_exists))
own_threat = op('Max', own_threat, mul(own_hits_kernel, en['guard_value']))
en_threat = op('Max', en_threat, mul(en_hits_kernel, own['guard_value']))
own_vulnerable = flt(op('Less', own['guards'], const(2.)))

# Firewall: the ring that locks next is the lowest open ring; urgency near 1 means imminent.
next_ring = op('ReduceMin', add(const(ring), mul(locked, 9.)), axes=[1], keepdims=1)
ring_vec = const(ring)
doomed_sq = mul(op('Relu', sub(1., op('Abs', sub(ring_vec, next_ring)))), clip(sub(mul(urgency, 4.), 3.)))
own_k_doomed = sum1(mul(own['ker'], doomed_sq))
en_k_doomed = sum1(mul(en['ker'], doomed_sq))

v = add(
    sub(own['material'], en['material']),
    mul(sub(clip(own['guards'], 0, 2), clip(en['guards'], 0, 2)), W['guard']),
    mul(own_threat, mul(actions_left, W['threat_own'])),
    mul(en_threat, -W['threat_enemy']),
    mul(mul(own_hits_kernel, unguarded_en), W['kernel_hit'] * 2),
    mul(mul(en_hits_kernel, unguarded_own), -W['kernel_hit']),
    mul(mul(en_hits_kernel, sub(own_vulnerable, unguarded_own)), -W['kernel_hit']),
    mul(sub(own['ring'], en['ring']), mul(urgency, W['ring'])),
    mul(sub(en_k_doomed, own_k_doomed), W['doomed'] * 4),
    mul(sub(own['mobility'], en['mobility']), W['mobility']),
)
value = rs(op('Where', op('Greater', certified_win, const(0.5)), const(1.), mul(op('Tanh', mul(v, W['scale'])), 0.97)), [-1])

# Policy: from/to scores plus moving-piece value times destination danger.
kval = 6.0
kernel_capture = add(mul(en['guard_value'], 1.6), mul(unguarded_en, kval))
cap_to = add(mul(en['pv'], 1.6), mul(en['ker'], kernel_capture))
own_val = add(own['pv'], mul(own['ker'], kval))
escape_from = mul(own_val, mul(att_en, 0.8))
to_score = add(cap_to, mul(ring_vec, mul(urgency, 0.4)), mul(mm(own['ker'], ADJ), 0.3))
pair = outer(own_val, mul(att_en, -0.7))
policy = add(pair, rs(escape_from, [-1, 64, 1]), rs(to_score, [-1, 1, 64]))
# A shot or redirected attack leaves attacker at source, so destination
# danger, source escape, centrality and own-Kernel proximity do not apply.
daemon_shot = mul(outer(P[1], add(*P[6:12])), const(adj8))
trace_shot = outer(P[3], add(*P[6:12]))
guard_redirect = outer(add(*P[:6]), mul(en['ker'], sub(1., unguarded_en)))
stationary = clip(add(daemon_shot, trace_shot, guard_redirect))
movement_only = add(pair, rs(escape_from, [-1, 64, 1]), rs(sub(to_score, cap_to), [-1, 1, 64]))
policy = sub(policy, mul(stationary, movement_only))
# Strong exact winning prior compensates for the referee's deliberate lack
# of cached expansion after a bodyguard redirect. Illegal pairs are masked.
policy = add(policy, mul(outer(add(*P[:6]), en['ker']), rs(mul(forced_win, 10.), [-1, 1, 1])))
policy = add(policy, mul(approach_pairs, 10.))
policy = rs(policy, [-1, 4096])

nodes.append(h.make_node('Identity', [policy], ['policy']))
nodes.append(h.make_node('Identity', [value], ['value']))
graph = h.make_graph(nodes, 'ColonyOneL4Heuristic',
                     [h.make_tensor_value_info('observation', T.FLOAT, ['batch', 16, 8, 8])],
                     [h.make_tensor_value_info('policy', T.FLOAT, ['batch', 4096]),
                      h.make_tensor_value_info('value', T.FLOAT, ['batch'])], init)
model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], producer_name='CCC L4 heuristic')
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, OUT)
OUT.with_suffix('.nodes.json').write_text(json.dumps({'approach_pairs':approach_pairs,'certified_win':certified_win}), encoding='utf-8')
print({'file': str(OUT), 'bytes': OUT.stat().st_size, 'nodes': len(nodes)})
