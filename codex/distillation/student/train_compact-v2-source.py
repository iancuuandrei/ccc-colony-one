"""Learn a compact Astra adapter reusing frozen v2+endgame graph features.

Adds at most 94 graph nodes. Exact +/-1 values AND their complete policy
vectors bypass all learned corrections. A source-regenerated v2 prefix is
compared byte-for-byte with the frozen composed trunk before binding tensors.
"""
from __future__ import annotations

import argparse
import copy
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, TensorProto as T

import train_student as lib

HERE, ROOT = lib.HERE, lib.ROOT


def binding(base_path):
    source = ROOT / "codex/tactical/build_tactical_v2.py"
    symbolic = HERE / "symbolic-v2.onnx"
    old = sys.argv
    try:
        sys.argv = [str(source), str(symbolic)]
        d = runpy.run_path(str(source))
    finally:
        sys.argv = old
    rebuilt = onnx.load(symbolic)
    original = onnx.load(base_path)
    # Output Identity nodes are renamed by the endgame wrapper. Every internal
    # node/tensor actually referenced by the student must retain exact bytes.
    named = {n.output[0]: n for n in original.graph.node}
    for n in rebuilt.graph.node:
        if n.output[0] in ("policy", "value"):
            continue
        if n.output[0] not in named or n.SerializeToString() != named[n.output[0]].SerializeToString():
            raise ValueError("Frozen v2 internal tensor drift: " + n.output[0])
    original_init = {n.name: n for n in original.graph.initializer}
    for n in rebuilt.graph.initializer:
        if n.name not in original_init or n.SerializeToString() != original_init[n.name].SerializeToString():
            raise ValueError("Frozen v2 initializer drift: " + n.name)
    def find(kind, inputs):
        return next(n.output[0] for n in rebuilt.graph.node if n.op_type == kind and list(n.input) == inputs)
    fields = {k: d[k] for k in ("P", "own", "en", "actions_left", "urgency", "own_threat", "en_threat", "own_vulnerable", "stationary", "pair", "movement_only", "cap_to", "escape_from", "certified_win", "ADJ")}
    fields["own_adj"] = find("MatMul", [d["own"]["ker"], d["ADJ"]])
    fields["enemy_adj"] = find("MatMul", [d["en"]["ker"], d["ADJ"]])
    proof = dict(source=str(source), source_sha256=lib.sha(source),
                 regenerated_v2_sha256=lib.sha(symbolic), base_sha256=lib.sha(base_path),
                 status="PASS", internal_nodes_verified=len(rebuilt.graph.node)-2,
                 initializers_verified=len(rebuilt.graph.initializer), fields=fields)
    return original, fields, proof


def compact_features(b, d, base_policy, base_value):
    """Only new guard interactions need graph nodes; other fields reuse trunk."""
    c, op = b.constant, b.op
    rs = lambda a,shape: op("Reshape", a, c(shape))
    mul = lambda a,z: op("Mul",a,z)
    sub = lambda a,z: op("Sub",a,z)
    fr = lambda a: rs(a,[-1,64,1])
    to = lambda a: rs(a,[-1,1,64])
    move = sub(c(1.),d["stationary"])
    kernel_from = fr(d["P"][5])
    guarddelta = mul(mul(sub(to(d["own_adj"]),fr(d["own_adj"])),sub(c(1.),kernel_from)),move)
    last = rs(sub(c(2.),mul(d["actions_left"],c(2.))),[-1,1,1])
    guardlast = mul(guarddelta,last)
    guardrisk = mul(guardlast,rs(d["own_vulnerable"],[-1,1,1]))
    neighbours = op("MatMul",d["own"]["others"],d["ADJ"])
    kguard = mul(mul(sub(to(neighbours),fr(neighbours)),kernel_from),move)
    enemy_guard_capture = to(mul(d["enemy_adj"],d["en"]["others"]))
    ringdelta = mul(move,c((lib.RING[None,:]-lib.RING[:,None])))
    pf=[to(d["cap_to"]),fr(d["escape_from"]),d["pair"],d["movement_only"],d["stationary"],
        guarddelta,guardlast,guardrisk,kguard,enemy_guard_capture,ringdelta,rs(base_policy,[-1,64,64])]
    pn=["capture_value","escape_source","destination_danger","movement_prior","stationary",
        "own_guard_delta","last_action_guard_delta","last_action_vulnerable_guard_delta",
        "kernel_guard_delta","capture_enemy_guard","ring_delta","baseline_policy_calibration"]
    pscale=np.array([6.,6.,6.,6.,1.,1.,1.,1.,5.,1.,1.,10.],np.float32)
    vf=[rs(base_value,[-1,1]),sub(d["own"]["material"],d["en"]["material"]),
        d["own"]["guards"],d["en"]["guards"],d["own_threat"],d["en_threat"],
        sub(d["own"]["ring"],d["en"]["ring"]),d["actions_left"],d["urgency"],
        op("ReduceMean",d["P"][15],axes=[1],keepdims=1)]
    vn=["baseline_value","material_difference","own_guards","enemy_guards","own_threat",
        "enemy_threat","kernel_ring_difference","actions_left","urgency","magenta"]
    vscale=np.array([1.,10.,5.,5.,6.,6.,3.,1.,1.,1.],np.float32)
    return pf,pn,pscale,vf,vn,vscale


def renamed(model):
    m=copy.deepcopy(model)
    for n in m.graph.node:
        for i,name in enumerate(n.output):
            if name in ("policy","value"):n.output[i]="distill_base_"+name
        for i,name in enumerate(n.input):
            if name in ("policy","value"):n.input[i]="distill_base_"+name
    return m


def extract(base_model,d,obs):
    m=renamed(base_model)
    b=lib.GraphOps(m)
    pf,pn,ps,vf,vn,vs=compact_features(b,d,"distill_base_policy","distill_base_value")
    del m.graph.output[:]
    names=["distill_base_policy","distill_base_value"]+pf+vf
    for i,name in enumerate(names):
        if i==0:shape=["batch",4096]
        elif i==1:shape=["batch"]
        elif i<2+len(pf):shape=None
        else:shape=["batch",1]
        # ORT permits a diagnostic intermediate output with omitted shape;
        # this graph is never a submission candidate.
        m.graph.output.append(h.make_tensor_value_info(name,T.FLOAT,shape))
    diagnostic=HERE/"compact-feature-diagnostic.onnx"
    onnx.save(m,diagnostic)
    outputs=lib.session(diagnostic).run(None,{"observation":obs})
    bp,bv=outputs[:2]
    policy=np.stack([np.broadcast_to(a,(len(obs),64,64)) / s for a,s in zip(outputs[2:2+len(pf)],ps)],axis=-1)
    value=np.concatenate(outputs[2+len(pf):],axis=1)/vs
    # Exact solved states are excluded from correction learning as well as
    # bypassed by the deployed graph. They remain visible in teacher metrics.
    solved=np.abs(bv)>.98
    policy[solved]=0;value[solved]=0
    return bp,bv,policy,value,pn,vn


def export(base_model,d,wp,wv,out_path):
    m=renamed(base_model);base_nodes=len(m.graph.node)
    b=lib.GraphOps(m);c,op=b.constant,b.op
    pf,pn,ps,vf,vn,vs=compact_features(b,d,"distill_base_policy","distill_base_value")
    # Fold feature normalization into learned constants. No feature-sized
    # constant allocation or additional Div nodes are needed at inference.
    terms=[op("Mul",a,c(float(w/s))) for a,w,s in zip(pf,wp,ps)]
    pc=__import__("functools").reduce(lambda a,z:op("Add",a,z),terms)
    pc=op("Reshape",op("Clip",pc,c(-lib.POLICY_CAP),c(lib.POLICY_CAP)),c([-1,4096]))
    newp=op("Add","distill_base_policy",pc)
    keep=op("Greater",op("Abs","distill_base_value"),c(.98))
    newp=op("Where",op("Reshape",keep,c([-1,1])),"distill_base_policy",newp)
    vcat=op("Concat",*vf,axis=1)
    vc=op("MatMul",vcat,c((wv/vs)[:,None]))
    vc=op("Reshape",op("Clip",vc,c(-lib.VALUE_CAP),c(lib.VALUE_CAP)),c([-1]))
    newv=op("Clip",op("Add","distill_base_value",vc),c(-1.),c(1.))
    newv=op("Where",keep,"distill_base_value",newv)
    m.graph.node.append(h.make_node("Identity",[newp],["policy"]))
    m.graph.node.append(h.make_node("Identity",[newv],["value"]))
    m.producer_name="CCC Astra compact shared-feature residual distillation v2"
    m.doc_string="Trained Astra labels; frozen proved tactical and endgame trunk; exact solved policy/value bypass. No playing-strength qualification."
    onnx.checker.check_model(m)
    if len(m.graph.node)>1024 or len(m.graph.initializer)>512:
        raise ValueError(f"Compact graph budget exceeded: {len(m.graph.node)} nodes, {len(m.graph.initializer)} initializers")
    onnx.save(m,out_path)
    return dict(nodes=len(m.graph.node),added_nodes=len(m.graph.node)-base_nodes,
                initializers=len(m.graph.initializer),bytes=out_path.stat().st_size)


def parity(base_model,d,base_path,candidate,obs,wp,wv):
    checks=[];student=lib.session(candidate)
    for n in (1,8,16):
        x=np.stack([obs[i%len(obs)] for i in range(n)])
        bp,bv,pf,vf,*_=extract(base_model,d,x)
        ep=bp+np.clip(pf.reshape(n,4096,-1)@wp,-lib.POLICY_CAP,lib.POLICY_CAP)
        ev=np.clip(bv+np.clip(vf@wv,-lib.VALUE_CAP,lib.VALUE_CAP),-1,1)
        solved=np.abs(bv)>.98
        ep[solved]=bp[solved];ev[solved]=bv[solved]
        ap,av=student.run(None,{"observation":x})
        pe,ve=float(np.max(np.abs(ep-ap))),float(np.max(np.abs(ev-av)))
        if pe>2e-5 or ve>2e-5 or not np.isfinite(ap).all() or not np.isfinite(av).all():raise AssertionError((n,pe,ve))
        if not np.array_equal(ap[solved],bp[solved]) or not np.array_equal(av[solved],bv[solved]):raise AssertionError("Exact solved output changed")
        checks.append(dict(batch=n,policy_max_error=pe,value_max_error=ve,exact_solved_rows=int(solved.sum()),exact_solved_outputs_equal=True))
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument("--packet",type=Path,action="append",required=True)
    p.add_argument("--labels",type=Path,action="append",required=True)
    p.add_argument("--baseline",type=Path,default=ROOT/"codex/tactical/model-tactical-v2-endgame.onnx")
    p.add_argument("--output",type=Path,default=HERE/"model-distill-v2.onnx")
    a=p.parse_args();started=time.time()
    if len(a.packet)!=len(a.labels):raise ValueError("Packet/label count mismatch")
    packets=[];labels=[];records=[]
    for packet,label in zip(a.packet,a.labels):
        pack,lab,rows=lib.load_data(packet,label);packets.append(pack);labels.append(lab);records.extend(rows)
    if len({r["position_id"] for r in records})!=len(records):raise ValueError("Duplicate positions across packets")
    lib.assign_splits(records)
    base,d,bound=binding(a.baseline)
    obs=np.stack([r["obs"] for r in records]).astype(np.float32)
    bp,bv,pf,vf,pn,vn=extract(base,d,obs)
    for i,r in enumerate(records):
        r["features"]=pf[i].reshape(4096,-1)[r["legal"]]
        r["value_features"]=vf[i];r["base_policy"]=bp[i,r["legal"]].astype(np.float64);r["base_value"]=float(bv[i])
    wp,history=lib.fit_policy(records);wv=lib.fit_value(records)
    graph=export(base,d,wp,wv,a.output)
    checks=parity(base,d,a.baseline,a.output,obs,wp,wv)
    np.savez(a.output.with_suffix(".npz"),policy_weights=wp,value_weights=wv,policy_names=pn,value_names=vn)
    report=lib.metrics(records,wp,wv)
    manifest=dict(version=2,packet_ids=[p["packet_id"] for p in packets],teacher=[l["teacher"] for l in labels],
                  inputs=dict(trainer_sha256=lib.sha(__file__),shared_trainer_sha256=lib.sha(lib.__file__),
                              packets=[dict(path=str(p.resolve()),sha256=lib.sha(p)) for p in a.packet],
                              labels=[dict(path=str(p.resolve()),sha256=lib.sha(p)) for p in a.labels],
                              baseline=dict(path=str(a.baseline.resolve()),sha256=lib.sha(a.baseline))),
                  binding=bound,output=dict(path=str(a.output.resolve()),sha256=lib.sha(a.output),**graph),
                  policy_weights=dict(zip(pn,map(float,wp))),value_weights=dict(zip(vn,map(float,wv))),
                  training=dict(policy_l2=.1,value_l2=2.,steps=1500,policy_cap=lib.POLICY_CAP,value_cap=lib.VALUE_CAP,
                                learned_parameters=len(wp)+len(wv),completed_outcomes_used=False,
                                history=history,groups=[dict(position_id=r["position_id"],source_game=r["group"],split=r["split"],claim=r["claim"]) for r in records]),
                  metrics=report,parity=dict(status="PASS",checks=checks),
                  qualification=dict(arena="NOT RUN",deployment="NOT QUALIFIED",promotion="NOT PERFORMED"),elapsed_seconds=time.time()-started)
    a.output.with_suffix(".json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(dict(output=manifest["output"],metrics={k:v["mean"] for k,v in report.items()},parity=checks),indent=2))


if __name__=="__main__":main()
