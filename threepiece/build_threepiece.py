"""Wrap a base ONNX with Codex's exact three-piece capture-to-bare win certificate.

Port of codex/endgame/three_piece_certificate.py::certificate (helper SHA256
c97f7dc7ff2823d9a731a363fb1d8ac7907c379e5ba9ae523e165d8841d4f9f2) as conventional
graph nodes around an unchanged base, in the style of codex/endgame/build_endgame.py.

Family (canonical observation of the actual side to move):
  * own pieces: exactly one Kernel s (plane 5), no other own piece (planes 0-4);
  * enemy pieces: exactly one Kernel e (plane 11) and exactly one non-Kernel t
    (planes 6-10, any type);
  * s, e, t on distinct, unlocked squares (plane 12);
  * actions_left plane 13 uniform 0.5 or 1.0; side plane 15 uniform 0 or 1;
  * Chebyshev(t, e) > 2 and inward ring R(t) > R(e), or R(t) == R(e) with actual
    mover Magenta (plane 15 == 1), R(p) = min(r, c, 7-r, 7-c);
  * one action left: Chebyshev(s, t) == 1, certified first action s->t;
    two actions left: Chebyshev(s, t) == 2, certified first actions s->m for every
    unlocked EMPTY m with Chebyshev(s, m) == Chebyshev(m, t) == 1; at least one m.
Rotation by 180 degrees preserves Chebyshev distance, ring index, adjacency and the
lock plane, so the rule is evaluated directly in canonical coordinates and the
bonus lands on canonical action indices from*64+to.

Outputs. In the family: value := +0.99 and policy := base + 30 on the certified
first actions only. Policy is formed as base - Bn with Bn = -30 on certified
actions and exactly +0.0 everywhere else (IEEE 754: x - (+0) == x bitwise for every
x, including -0.0; x - (-30) == x + 30). Bn = s (x) (dest * flag * -30) + zero,
where the outer product yields -30 or a signed zero and adding a runtime +0.0
(Cast(flag)*0, not an initializer, so no no-op elimination can drop it) turns every
zero into +0.0 (-0 + +0 = +0 under round-to-nearest).
Value is passed through by Where. Outside the family both outputs are bitwise
identical to the base (UNKNOWN). All shape/axes/split inputs are initializers.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T

PREFIX = 'tp_'
BONUS = 30.0
WIN_VALUE = 0.99


def constants_np():
    coords = np.array([(q // 8, q % 8) for q in range(64)])
    dist = np.abs(coords[:, None] - coords[None, :]).max(2)
    ring = np.minimum.reduce([coords[:, 0], coords[:, 1], 7 - coords[:, 0], 7 - coords[:, 1]])
    far = dist > 2
    g_cy = (far & (ring[:, None] > ring[None, :])).astype(np.float32)   # [t, e]
    g_eq = (far & (ring[:, None] == ring[None, :])).astype(np.float32)  # [t, e]
    adj = (dist == 1).astype(np.float32)
    # Selection rows over the 16 planes:
    # 0 s=own Kernel, 1 e=enemy Kernel, 2 t=enemy non-Kernels,
    # 3 z=all pieces + locked (max 1 <=> distinct pieces, none locked),
    # 4 actions_left, 5 side, 6 own non-Kernels.
    sel = np.zeros((7, 16), np.float32)
    sel[0, 5] = 1; sel[1, 11] = 1; sel[2, 6:11] = 1; sel[3, 0:13] = 1
    sel[4, 13] = 1; sel[5, 15] = 1; sel[6, 0:5] = 1
    col = lambda v: np.array(v, np.float32).reshape(1, 7, 1)
    # Row checks: count == TC + 64*max*U (weights WC) and (max - C1)(max - C2) == 0.
    checks = dict(tc=col([1, 1, 1, 0, 0, 0, 0]), u64=col([0, 0, 0, 0, 64, 64, 0]),
                  wc=col([1, 1, 1, 0, 1, 1, 1]),
                  c1=col([1, 1, 1, 1, .5, 0, 0]), c2=col([1, 1, 1, 1, 1, 1, 0]))
    return dict(adj=adj, g_cy=g_cy, g_eq=g_eq, sel=sel, **checks)


def build(base, output):
    base = Path(base); output = Path(output)
    model = onnx.load(str(base))
    out_names = [x.name for x in model.graph.output]
    assert out_names == ['policy', 'value'], out_names
    existing = {n.name for n in model.graph.node} | {i.name for i in model.graph.initializer}
    existing |= {o for n in model.graph.node for o in n.output}
    assert not any(x.startswith(PREFIX) for x in existing), 'prefix collision'
    renamed = {'policy': PREFIX + 'base_policy', 'value': PREFIX + 'base_value'}
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in renamed: node.input[i] = renamed[name]
        for i, name in enumerate(node.output):
            if name in renamed: node.output[i] = renamed[name]

    K = constants_np()
    nodes, init, cache = [], [], {}

    def c(value, dtype=np.float32):
        arr = np.asarray(value, dtype=dtype)
        key = (arr.dtype.str, arr.shape, arr.tobytes())
        if key not in cache:
            name = PREFIX + 'c' + str(len(init))
            init.append(nh.from_array(arr, name)); cache[key] = name
        return cache[key]

    def op(kind, *args, n_out=1, **attrs):
        base_name = PREFIX + 'n' + str(len(nodes))
        outs = [base_name] if n_out == 1 else [f'{base_name}_{k}' for k in range(n_out)]
        nodes.append(h.make_node(kind, list(args), outs, name=base_name, **attrs))
        return outs[0] if n_out == 1 else outs

    def i64(v): return c(v, np.int64)

    zero, half, one = c(0.0), c(0.5), c(1.0)

    x = op('Reshape', 'observation', i64([-1, 16, 64]))            # [B,16,64]
    y = op('MatMul', c(K['sel']), x)                                # [B,7,64]
    s, e, t, z, _, _, _ = op('Split', y, i64([1] * 7), axis=1, n_out=7)  # [B,1,64]
    cnt = op('ReduceSum', y, i64([2]), keepdims=1)                  # [B,7,1]
    mx = op('ReduceMax', y, axes=[2], keepdims=1)                   # [B,7,1]
    # Exact piece counts (s,e,t = 1; own non-Kernels = 0); actions_left/side uniform
    # (sum == 64*max) with max in {0.5,1} / {0,1}; max(z) == 1 (distinct, unlocked).
    dev_cnt = op('Mul', op('Abs', op('Sub', cnt, op('Add', op('Mul', mx, c(K['u64'])), c(K['tc'])))), c(K['wc']))
    dev_max = op('Abs', op('Mul', op('Sub', mx, c(K['c1'])), op('Sub', mx, c(K['c2']))))
    domain_ok = op('Equal', op('ReduceSum', op('Add', dev_cnt, dev_max), i64([1]), keepdims=1), zero)  # [B,1,1]
    mxs = op('Split', mx, i64([1] * 7), axis=1, n_out=7)
    is_two = op('Equal', mxs[4], one)                               # actions_left == 2
    m = mxs[5]                                                      # 1.0 iff actual mover Magenta
    free = op('Sub', one, z)                                        # empty and unlocked
    adj = c(K['adj'])
    adj_s = op('MatMul', s, adj)
    adj_t = op('MatMul', t, adj)
    d1 = op('Mul', adj_s, t)                                        # s->t iff Cheb(s,t)==1
    not_adj = op('Sub', one, op('ReduceSum', d1, i64([2]), keepdims=1))
    # common empty unlocked neighbour of s and t, s and t not adjacent => Cheb(s,t)==2
    d2 = op('Mul', op('Mul', op('Mul', adj_s, adj_t), free), not_adj)
    dest = op('Where', is_two, d2, d1)                              # [B,1,64]
    any_path = op('Greater', op('ReduceSum', dest, i64([2]), keepdims=1), half)
    # Final bare-Kernel adjudication after capturing t: Cheb(t,e)>2 and ring rule.
    fin_row = op('Add', op('MatMul', t, c(K['g_cy'])), op('Mul', op('MatMul', t, c(K['g_eq'])), m))
    fin_ok = op('Greater', op('ReduceSum', op('Mul', fin_row, e), i64([2]), keepdims=1), half)
    enabled = op('And', op('And', domain_ok, fin_ok), any_path)     # [B,1,1]
    en_f = op('Cast', enabled, to=T.FLOAT)
    dest_neg = op('Mul', dest, op('Mul', en_f, c(-BONUS)))          # -30 or -0.0
    # K=1 MatMul outer product: -30 exactly on (s, certified m), a signed zero elsewhere
    # (zero sign is implementation-defined); adding the runtime +0.0 makes every zero +0.0.
    outer = op('MatMul', op('Reshape', s, i64([-1, 64, 1])), dest_neg)  # [B,64,64]
    bn = op('Reshape', op('Add', outer, op('Mul', en_f, zero)), i64([-1, 4096]))  # -30 or +0.0
    flag = op('Reshape', enabled, i64([-1]))                        # [B] bool
    proof_nodes = list(nodes)
    nodes.append(h.make_node('Sub', [PREFIX + 'base_policy', bn], ['policy'], name=PREFIX + 'policy'))
    nodes.append(h.make_node('Where', [flag, c(WIN_VALUE), PREFIX + 'base_value'], ['value'], name=PREFIX + 'value'))
    # Proof-only diagnostic: certified-action mask from Bn (not part of the upload graph).
    proof_nodes.append(h.make_node('Less', [bn, c(-BONUS / 2)], [PREFIX + 'mask'], name=PREFIX + 'mask'))

    opset = [o for o in model.opset_import if o.domain in ('', 'ai.onnx')][0].version
    used = {i for n in proof_nodes for i in n.input}
    proof = h.make_model(h.make_graph(proof_nodes, 'ThreePieceProof', list(model.graph.input), [
        h.make_tensor_value_info(flag, T.BOOL, ['batch']),
        h.make_tensor_value_info(PREFIX + 'mask', T.BOOL, ['batch', 4096]),
        h.make_tensor_value_info(bn, T.FLOAT, ['batch', 4096]),
    ], [x for x in init if x.name in used]), opset_imports=[h.make_opsetid('', opset)])
    proof.ir_version = model.ir_version
    onnx.checker.check_model(proof, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    proof_path = output.with_suffix('.proof.onnx')
    onnx.save(proof, str(proof_path))

    main_used = {i for n in nodes for i in n.input}
    model.graph.node.extend(nodes); model.graph.initializer.extend([x for x in init if x.name in main_used])
    model.graph.name += '_ExactThreePiece'
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    added_init = [x for x in init if x.name in main_used]
    manifest = dict(base=str(base), base_sha256=sha(base), output=str(output), sha256=sha(output),
                    bytes=output.stat().st_size, proof=str(proof_path), proof_sha256=sha(proof_path),
                    base_nodes=len(model.graph.node) - len(nodes), added_nodes=len(nodes),
                    total_nodes=len(model.graph.node), added_initializers=len(added_init),
                    total_initializers=len(model.graph.initializer),
                    added_weight_bytes=sum(len(x.raw_data) for x in added_init), opset=opset,
                    bonus=BONUS, win_value=WIN_VALUE, builder_sha256=sha(__file__),
                    helper='codex/endgame/three_piece_certificate.py',
                    helper_sha256='c97f7dc7ff2823d9a731a363fb1d8ac7907c379e5ba9ae523e165d8841d4f9f2')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); mf = build(a.base, a.output)
    Path(a.output).with_suffix('.build.json').write_text(json.dumps(mf, indent=2))
    print(json.dumps(mf, indent=2))
