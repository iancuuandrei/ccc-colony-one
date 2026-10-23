"""Append an exact bare-two-kernel own-turn solver to an existing ONNX.

Only this directory is owned by this worker. Never mutates the base model.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T


def build(base, output):
    model = onnx.load(str(base))
    original_outputs = [x.name for x in model.graph.output]
    renamed = {original_outputs[0]: 'eg_base_policy', original_outputs[1]: 'eg_base_value'}
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in renamed: node.input[i] = renamed[name]
        for i, name in enumerate(node.output):
            if name in renamed: node.output[i] = renamed[name]
    nodes, init = [], []
    def c(value, dtype=np.float32):
        name = 'eg_c' + str(len(init))
        init.append(nh.from_array(np.asarray(value, dtype=dtype), name))
        return name
    def op(kind, *args, **attrs):
        name = 'eg_n' + str(len(nodes))
        nodes.append(h.make_node(kind, list(args), [name], name=name, **attrs))
        return name
    def rs(x, shape): return op('Reshape', x, c(shape, np.int64))
    def red(x): return op('ReduceSum', x, c([1], np.int64), keepdims=1)
    def mul(x,y): return op('Mul',x,y)
    def add(x,y): return op('Add',x,y)
    def gt(x,y): return op('Greater',x,y)
    def eq(x,y): return op('Equal',x,y)
    def AND(x,y): return op('And',x,y)
    def OR(x,y): return op('Or',x,y)
    def flt(x): return op('Cast',x,to=T.FLOAT)
    coords = np.array([(s//8,s%8) for s in range(64)])
    distances = np.abs(coords[:,None] - coords[None,:]).max(2)
    adj = c((distances == 1).astype(np.float32))
    far = c((distances > 2).astype(np.float32))
    ring = c(np.minimum.reduce([coords[:,0],coords[:,1],7-coords[:,0],7-coords[:,1]]))
    zero, one, half = c(0), c(1), c(.5)
    flat = rs('observation',[-1,16,64])
    planes = [op('Gather',flat,c(i,np.int64),axis=1) for i in range(16)]
    own, enemy, locked = planes[5],planes[11],planes[12]
    open_sq = op('Sub',one,locked)
    side = op('ReduceMax',planes[15],axes=[1],keepdims=1)
    acts = op('ReduceMax',planes[13],axes=[1],keepdims=1)
    enemy_ring = red(mul(enemy,ring))
    better_ring = OR(gt(ring,enemy_ring),AND(eq(ring,enemy_ring),gt(side,half)))
    # Bare adjudication is terminal after the current turn, before firewall.
    final_good = mul(op('MatMul',enemy,far),flt(better_ring))
    final_good = op('Max',enemy,final_good)  # capture wins before adjudication
    final_good = mul(final_good,open_sq)
    # A two-action first move wins iff it captures immediately OR admits a
    # winning second move. Only the own kernel moves and vacates its source.
    next_good = gt(op('MatMul',final_good,adj),zero)
    two_good = OR(gt(enemy,half),next_good)
    action_good = op('Where',gt(acts,c(.75)),flt(two_good),final_good)
    first_legal = mul(op('MatMul',own,adj),open_sq)
    any_win = gt(red(mul(first_legal,action_good)),zero)
    exact_value = rs(op('Where',any_win,one,c(-1)),[-1])
    exact_logits = rs(op('MatMul',c(np.ones((64,1),np.float32)),rs(op('Sub',mul(action_good,c(40)),c(20)),[-1,1,64])),[-1,4096])
    others = planes[0]
    for i in [1,2,3,4,6,7,8,9,10]: others = add(others,planes[i])
    enabled = AND(eq(red(others),zero),AND(eq(red(own),one),eq(red(enemy),one)))
    enabled = AND(enabled,eq(red(mul(own,enemy)),zero))
    enabled = AND(enabled,eq(red(mul(add(own,enemy),locked)),zero))
    enabled = AND(enabled,OR(eq(acts,one),eq(acts,half)))
    enabled = AND(enabled,OR(eq(side,zero),eq(side,one)))
    value_enabled = rs(enabled,[-1])
    nodes.append(h.make_node('Where',[enabled,exact_logits,'eg_base_policy'],['policy'],name='eg_policy'))
    nodes.append(h.make_node('Where',[value_enabled,exact_value,'eg_base_value'],['value'],name='eg_value'))
    # expose diagnostics only in an independent proof graph
    proof_nodes = list(nodes[:-2])
    proof = h.make_model(h.make_graph(proof_nodes,'BareKernelProof',list(model.graph.input),[
        h.make_tensor_value_info(action_good,T.FLOAT,['batch',64]),
        h.make_tensor_value_info(exact_value,T.FLOAT,['batch']),
        h.make_tensor_value_info(enabled,T.BOOL,['batch',1]),
    ],init),opset_imports=[h.make_opsetid('',17)])
    proof.ir_version = 10
    # proof graph must not depend on renamed base outputs
    onnx.checker.check_model(proof)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(proof,str(output.with_suffix('.proof.onnx')))
    model.graph.node.extend(nodes); model.graph.initializer.extend(init)
    model.graph.name += '_ExactBareKernels'
    onnx.checker.check_model(model)
    onnx.save(model,str(output))
    manifest=dict(base=str(base),base_sha256=hashlib.sha256(Path(base).read_bytes()).hexdigest(),
                  output=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  bytes=output.stat().st_size,added_nodes=len(nodes),added_weight_bytes=sum(len(x.raw_data) for x in init))
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(Path(a.base),Path(a.output))
