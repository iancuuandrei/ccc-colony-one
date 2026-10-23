"""Small, reproducible Astra preference distillation, with a frozen tactical trunk.

Only teacher-reviewed JSON labels are learning targets. Historical outcomes are
never passed to this optimizer. Shared board/action features generalize across
positions; there is no board-key lookup, transposition table, or memorized patch.
Run with --packet and --labels after the independent label validator succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as h, numpy_helper as nh

HERE = Path(__file__).resolve().parent
DISTILL = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(DISTILL))
sys.path.insert(0, str(ROOT / "env"))
from queue import observation  # noqa: E402 -- local packet encoder

COORD = np.array([(i // 8, i % 8) for i in range(64)])
DIST = np.max(np.abs(COORD[:, None, :] - COORD[None, :, :]), axis=2).astype(np.float32)
ADJ = (DIST == 1).astype(np.float32)
RING = np.min(np.c_[COORD, 7 - COORD], axis=1).astype(np.float32) / 3
POLICY_CAP = 1.0
VALUE_CAP = 0.12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def session(path):
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])


class NumpyOps:
    def constant(self, x): return np.asarray(x)
    def op(self, kind, *a, **kw):
        if kind == "Reshape": return np.reshape(a[0], a[1])
        if kind == "Gather": return np.take(a[0], a[1], axis=kw["axis"])
        if kind == "Add": return a[0] + a[1]
        if kind == "Sub": return a[0] - a[1]
        if kind == "Mul": return a[0] * a[1]
        if kind == "Div": return a[0] / a[1]
        if kind == "MatMul": return a[0] @ a[1]
        if kind == "ReduceSum": return np.sum(a[0], axis=tuple(a[1]), keepdims=bool(kw.get("keepdims", 1)))
        if kind == "ReduceMean": return np.mean(a[0], axis=tuple(kw["axes"]), keepdims=bool(kw.get("keepdims", 1)))
        if kind == "Clip": return np.clip(a[0], a[1], a[2])
        if kind == "Abs": return np.abs(a[0])
        if kind == "Less": return a[0] < a[1]
        if kind == "Greater": return a[0] > a[1]
        if kind == "Cast": return a[0].astype(np.float32)
        if kind == "Where": return np.where(*a)
        raise ValueError(kind)


class GraphOps:
    def __init__(self, model):
        self.model = model
        self.cache = {}
        self.counter = 0

    def constant(self, x):
        x = np.asarray(x)
        if x.dtype.kind in "iu": x = x.astype(np.int64)
        elif x.dtype.kind != "b": x = x.astype(np.float32)
        key = (x.dtype.str, x.shape, x.tobytes())
        if key not in self.cache:
            name = f"distill_c{len(self.cache)}"
            self.cache[key] = name
            self.model.graph.initializer.append(nh.from_array(x, name))
        return self.cache[key]

    def op(self, kind, *a, **kw):
        name = f"distill_n{self.counter}"
        self.counter += 1
        self.model.graph.node.append(h.make_node(kind, list(a), [name], name=name, **kw))
        return name


def features(backend, obs, base_policy, base_value):
    """Same arithmetic constructs numpy reference features and an ONNX graph."""
    c, op = backend.constant, backend.op
    rs = lambda a, shape: op("Reshape", a, c(shape))
    add = lambda a, b: op("Add", a, b)
    sub = lambda a, b: op("Sub", a, b)
    mul = lambda a, b: op("Mul", a, b)
    div = lambda a, b: op("Div", a, b)
    mm = lambda a, b: op("MatMul", a, b)
    sm = lambda a: op("ReduceSum", a, c([1]), keepdims=1)
    clip = lambda a, lo=0., hi=1.: op("Clip", a, c(lo), c(hi))
    x = rs(obs, [-1, 16, 64])
    p = [op("Gather", x, c(i), axis=1) for i in range(16)]
    sum_arrays = lambda a: __import__("functools").reduce(add, a)
    own_nonker = sum_arrays(p[:5]); enemy_nonker = sum_arrays(p[6:11])
    enemy = add(enemy_nonker, p[11])
    own_adj = mm(p[5], c(ADJ)); enemy_adj = mm(p[11], c(ADJ))
    own_guards = sm(mul(own_adj, own_nonker))
    enemy_guards = sm(mul(enemy_adj, enemy_nonker))
    own_neighbours = mm(own_nonker, c(ADJ))
    owndist = mm(p[5], c(DIST / 7)); endist = mm(p[11], c(DIST / 7))
    # Helper projections create full (B,64,64) pair fields by broadcasting.
    zero_pair = c(np.zeros((1, 64, 64), np.float32))
    fr = lambda a: rs(a, [-1, 64, 1])
    to = lambda a: rs(a, [-1, 1, 64])
    full = lambda a: add(a, zero_pair)
    outer = lambda a, b: mm(fr(a), to(b))
    # Stationary legal actions: Daemon adjacent shots, Trace shots, guarded
    # Kernel hits. Movement-only guard/ring features must vanish on these.
    shot = add(mul(outer(p[1], enemy), c(ADJ)), outer(p[3], enemy))
    redirect = outer(sum_arrays(p[:6]), mul(p[11], clip(enemy_guards)))
    moving = sub(c(1.), clip(add(shot, redirect)))
    capture = full(to(enemy))
    iskernel = full(fr(p[5]))
    nonkernel = sub(c(1.), iskernel)
    guard_delta = mul(moving, mul(nonkernel, sub(to(own_adj), fr(own_adj))))
    kguard_delta = mul(moving, mul(iskernel, sub(to(own_neighbours), fr(own_neighbours))))
    urgency = op("ReduceMean", p[14], axes=[1], keepdims=1)
    actions = op("ReduceMean", p[13], axes=[1], keepdims=1)
    al = rs(actions, [-1, 1, 1]); ug = rs(urgency, [-1, 1, 1])
    ring_delta = mul(moving, c(RING[None, :] - RING[:, None]))
    policy_features, policy_names = [], []

    def pf(name, a):
        policy_names.append(name); policy_features.append(full(a))

    for i, name in enumerate(("ping", "daemon", "exploit", "trace", "ice", "kernel")):
        pf(f"{name}_quiet", mul(fr(p[i]), sub(c(1.), capture)))
        pf(f"{name}_capture", mul(fr(p[i]), capture))
    for i, name in enumerate(("ping", "daemon", "exploit", "trace", "ice", "kernel")):
        pf(f"target_{name}", to(p[6+i]))
    pf("own_guard_delta", guard_delta)
    pf("own_guard_delta_low_count", div(guard_delta, rs(add(c(1.), own_guards), [-1, 1, 1])))
    pf("kernel_guard_delta", div(kguard_delta, c(5.)))
    pf("capture_enemy_guard", to(mul(enemy_adj, enemy_nonker)))
    pf("stationary", sub(c(1.), moving))
    pf("move_distance", mul(moving, c(DIST / 7)))
    pf("toward_own_kernel", mul(moving, sub(fr(owndist), to(owndist))))
    pf("toward_enemy_kernel", mul(moving, sub(fr(endist), to(endist))))
    pf("ring_delta", ring_delta)
    pf("ring_delta_urgency", mul(ring_delta, ug))
    pf("kernel_ring_delta_urgency", mul(mul(ring_delta, ug), iskernel))
    pf("kernel_toward_enemy", mul(mul(moving, iskernel), sub(fr(endist), to(endist))))
    pf("guard_delta_last_action", mul(guard_delta, sub(c(1.), al)))
    pf("capture_last_action", mul(capture, sub(c(1.), al)))
    pf("baseline_logit_calibration", rs(div(clip(base_policy, -10., 10.), c(10.)), [-1, 64, 64]))

    value_features, value_names = [], []

    def vf(name, a):
        value_names.append(name); value_features.append(a)

    vf("bias", add(mul(actions, c(0.)), c(1.)))
    vf("baseline_value", rs(base_value, [-1, 1]))
    for i, name in enumerate(("ping", "daemon", "exploit", "trace", "ice")):
        vf(f"material_diff_{name}", div(sm(sub(p[i], p[6+i])), c(4.)))
    vf("own_guards", div(own_guards, c(5.)))
    vf("enemy_guards", div(enemy_guards, c(5.)))
    vf("kernel_ring_diff", sm(mul(sub(p[5], p[11]), c(RING))))
    vf("urgency", urgency)
    vf("actions_left", actions)
    vf("magenta", op("ReduceMean", p[15], axes=[1], keepdims=1))
    vf("unguarded_own", sub(c(1.), clip(own_guards)))
    vf("unguarded_enemy", sub(c(1.), clip(enemy_guards)))
    vf("kernel_distance", sm(mul(p[5], endist)))
    return policy_features, policy_names, value_features, value_names


def softmax(a):
    e = np.exp(a - np.max(a))
    return e / e.sum()


def game_group(position):
    src = position["source"]
    if "match" in src:
        return ("logical-site-match", str(src["match"]))
    key = next((src[k] for k in ("match", "seed", "game_index") if k in src), None)
    if key is None:
        raise ValueError("Missing source game identity: " + str(src))
    return (src["path"], str(key))


def load_data(packet_path, labels_path):
    from validation import validate_file
    result = validate_file(packet_path, labels_path)
    if result.get("errors"):
        raise ValueError(result["errors"])
    packet = json.loads(Path(packet_path).read_text(encoding="utf-8"))
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    lookup = {p["position_id"]: p for p in packet["positions"]}
    records = []
    for label in labels["labels"]:
        pos = lookup[label["position_id"]]
        legal = np.array([a["action_index"] for a in pos["legal_actions"]], np.int64)
        ranked = {r["action_index"]: r for r in label["policy_rankings"]}
        # Unranked legal actions retain a conservative broad background tier;
        # omitted actions are never asserted illegal or given zero probability.
        background_weight = min(r["weight"] for r in ranked.values())
        weights = np.array([ranked.get(int(a), {}).get("weight", background_weight) for a in legal], np.float64)
        teacher = weights
        teacher /= teacher.sum()
        records.append(dict(position_id=pos["position_id"], group=game_group(pos),
                            obs=observation(pos["state"]), legal=legal, target=teacher,
                            confidence=float(label["confidence"]),
                            value_confidence=float(label.get("value_confidence", min(.25, label["confidence"]))),
                            value=float(label["value"]), claim=label["claim"]))
    assign_splits(records)
    return packet, labels, records


def assign_splits(records):
    groups = sorted({r["group"] for r in records})
    # Stable whole-source-game holdout independent of position contents/labels.
    ordered = sorted(groups, key=lambda g: hashlib.sha256(repr(g).encode()).hexdigest())
    holdout = set(ordered[:max(1, int(np.ceil(len(groups) * .25)))])
    for r in records: r["split"] = "holdout" if r["group"] in holdout else "train"
    if not any(r["split"] == "train" for r in records): raise ValueError("No training groups")


def fit_policy(records, l2=.1, steps=1500):
    train = [r for r in records if r["split"] == "train"]
    k = train[0]["features"].shape[1]
    w = np.zeros(k, np.float64)
    denom = sum(r["confidence"] for r in train)
    if denom <= 0: raise ValueError("No positive-confidence training labels")
    history = []
    for step in range(steps):
        grad = l2 * w; loss = .5*l2*float(w@w)
        for r in train:
            raw = r["features"] @ w
            pred = softmax(r["base_policy"] + np.clip(raw, -POLICY_CAP, POLICY_CAP))
            alpha = r["confidence"] / denom
            loss -= alpha * float(r["target"] @ np.log(np.maximum(pred, 1e-12)))
            grad += alpha * (r["features"].T @ ((pred-r["target"]) * (np.abs(raw) < POLICY_CAP)))
        w -= (.08 / (1+step/1000)) * grad
        if step % 100 == 0: history.append(dict(step=step, objective=loss, grad_norm=float(np.linalg.norm(grad))))
    return w.astype(np.float32), history


def fit_value(records, l2=2.):
    train = [r for r in records if r["split"] == "train"]
    X = np.stack([r["value_features"] for r in train]).astype(np.float64)
    y = np.array([r["value"]-r["base_value"] for r in train])
    alpha = np.array([r["value_confidence"] for r in train])
    # A proved baseline terminal value is never changed by teacher prediction.
    alpha *= np.array([abs(r["base_value"]) < .98 for r in train])
    return np.linalg.solve(X.T@(alpha[:, None]*X) + l2*np.eye(X.shape[1]), X.T@(alpha*y)).astype(np.float32)


def metrics(records, wp, wv):
    out = {}
    for split in ("train", "holdout"):
        rows = [r for r in records if r["split"] == split]
        items = []
        for r in rows:
            bp = r["base_policy"]; pred = bp + np.clip(r["features"]@wp, -POLICY_CAP, POLICY_CAP)
            target_best = set(np.flatnonzero(r["target"] == r["target"].max()).tolist())
            bv = r["base_value"]
            value = bv if abs(bv)>=.98 else np.clip(bv+np.clip(r["value_features"]@wv,-VALUE_CAP,VALUE_CAP),-1,1)
            items.append(dict(position_id=r["position_id"], baseline_ce=float(-r["target"]@np.log(np.maximum(softmax(bp),1e-12))),
                              student_ce=float(-r["target"]@np.log(np.maximum(softmax(pred),1e-12))),
                              baseline_top_tier=int(int(np.argmax(bp)) in target_best),
                              student_top_tier=int(int(np.argmax(pred)) in target_best),
                              baseline_value_abs_error=abs(bv-r["value"]),
                              student_value_abs_error=float(abs(value-r["value"])),
                              policy_confidence=r["confidence"],value_confidence=r["value_confidence"]))
        out[split] = dict(positions=len(rows), source_games=len({r["group"] for r in rows}),
                          mean={k:float(np.mean([r[k] for r in items])) for k in ("baseline_ce","student_ce","baseline_top_tier","student_top_tier","baseline_value_abs_error","student_value_abs_error")} if items else {},
                          records=items)
    return out


def export(base_path, out_path, wp, wv):
    model = onnx.load(base_path)
    for node in model.graph.node:
        for i, name in enumerate(node.output):
            if name in ("policy", "value"): node.output[i] = "distill_base_" + name
        for i, name in enumerate(node.input):
            if name in ("policy", "value"): node.input[i] = "distill_base_" + name
    b = GraphOps(model); c, op = b.constant, b.op
    pf, pn, vf, vn = features(b, "observation", "distill_base_policy", "distill_base_value")
    def combine(fs, ws):
        fields = [op("Mul", f, c(float(w))) for f,w in zip(fs,ws)]
        return __import__("functools").reduce(lambda a,z:op("Add",a,z),fields)
    pc = op("Clip", combine(pf, wp), c(-POLICY_CAP), c(POLICY_CAP))
    newp = op("Add", "distill_base_policy", op("Reshape", pc, c([-1,4096])))
    vc = op("Reshape", op("Clip", combine(vf, wv), c(-VALUE_CAP), c(VALUE_CAP)), c([-1]))
    value = op("Clip", op("Add", "distill_base_value", vc), c(-1.), c(1.))
    keep = op("Greater", op("Abs", "distill_base_value"), c(.98))
    newv = op("Where", keep, "distill_base_value", value)
    model.graph.node.append(h.make_node("Identity", [newp], ["policy"]))
    model.graph.node.append(h.make_node("Identity", [newv], ["value"]))
    model.producer_name = "CCC Astra shared-feature residual distillation v1"
    model.doc_string = "Astra teacher preference labels; frozen tactical baseline; capped learned residual. Not arena-qualified."
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    return dict(nodes=len(model.graph.node),initializers=len(model.graph.initializer),bytes=Path(out_path).stat().st_size)


def parity(base_path, student_path, observations, wp, wv):
    base, student = session(base_path), session(student_path)
    checks=[]
    for n in (1,8,16):
        x = np.stack([observations[i%len(observations)] for i in range(n)]).astype(np.float32)
        bp,bv=base.run(None,{"observation":x})
        pf,_,vf,_=features(NumpyOps(),x,bp,bv)
        pc=sum(f*w for f,w in zip(pf,wp)).reshape(n,4096)
        vc=sum(f*w for f,w in zip(vf,wv)).reshape(n)
        ep=bp+np.clip(pc,-POLICY_CAP,POLICY_CAP)
        ev=np.where(np.abs(bv)>.98,bv,np.clip(bv+np.clip(vc,-VALUE_CAP,VALUE_CAP),-1,1))
        ap,av=student.run(None,{"observation":x})
        pe,ve=float(np.max(np.abs(ep-ap))),float(np.max(np.abs(ev-av)))
        if pe>2e-5 or ve>2e-5:raise AssertionError((n,pe,ve))
        if ap.shape!=(n,4096) or av.shape!=(n,):raise AssertionError((ap.shape,av.shape))
        checks.append(dict(batch=n,policy_max_error=pe,value_max_error=ve,finite=bool(np.isfinite(ap).all() and np.isfinite(av).all())))
    return checks


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--packet",type=Path,required=True,action="append");p.add_argument("--labels",type=Path,required=True,action="append")
    p.add_argument("--baseline",type=Path,default=ROOT/"codex/tactical/model-tactical.onnx")
    p.add_argument("--output",type=Path,default=HERE/"model-distill-v1.onnx")
    args=p.parse_args();started=time.time()
    if len(args.packet)!=len(args.labels):raise ValueError("Pass one --labels for each --packet")
    packets=[];all_labels=[];records=[]
    for packet_path,label_path in zip(args.packet,args.labels):
        packet,labels,rows=load_data(packet_path,label_path)
        packets.append(packet);all_labels.append(labels);records.extend(rows)
    if len({r["position_id"] for r in records})!=len(records):raise ValueError("Duplicate position IDs across packets")
    assign_splits(records)
    obs=np.stack([r["obs"] for r in records]);base=session(args.baseline)
    bp,bv=base.run(None,{"observation":obs})
    pf,pn,vf,vn=features(NumpyOps(),obs,bp,bv)
    for i,r in enumerate(records):
        r["features"]=np.stack([f[i].reshape(-1)[r["legal"]] for f in pf],axis=1)
        r["value_features"]=np.array([f[i,0] for f in vf],np.float32)
        r["base_policy"]=bp[i,r["legal"]].astype(np.float64);r["base_value"]=float(bv[i])
    wp,history=fit_policy(records);wv=fit_value(records)
    report=metrics(records,wp,wv)
    graph=export(args.baseline,args.output,wp,wv)
    checks=parity(args.baseline,args.output,[r["obs"] for r in records],wp,wv)
    ckpt=args.output.with_suffix(".npz")
    np.savez(ckpt,policy_weights=wp,value_weights=wv,policy_names=pn,value_names=vn)
    manifest=dict(version=1,packet_ids=[p["packet_id"] for p in packets],teacher=[l.get("teacher") for l in all_labels],
                  inputs={"trainer":{"path":str(Path(__file__).resolve()),"sha256":sha(__file__)},
                          "packets":[{"path":str(p.resolve()),"sha256":sha(p)} for p in args.packet],
                          "labels":[{"path":str(p.resolve()),"sha256":sha(p)} for p in args.labels],
                          "baseline":{"path":str(args.baseline.resolve()),"sha256":sha(args.baseline)}},
                  output={"path":str(args.output.resolve()),"sha256":sha(args.output),**graph},
                  checkpoint_sha256=sha(ckpt),policy_weights=dict(zip(pn,map(float,wp))),value_weights=dict(zip(vn,map(float,wv))),
                  training=dict(algorithm="Confidence-weighted legal-action soft-target cross-entropy, capped shared linear residual, L2=.1; confidence-weighted value residual ridge L2=2",
                                policy_target="Teacher-provided weights normalized over legal actions; unranked legal actions get minimum explicit weight",
                                policy_cap=POLICY_CAP,value_cap=VALUE_CAP,steps=1500,
                                groups=[dict(position_id=r["position_id"],source_game=r["group"],split=r["split"],claim=r["claim"]) for r in records],
                                history=history,completed_outcomes_used=False,game_group_overlap=False),
                  metrics=report,parity=dict(status="PASS",checks=checks),
                  qualification=dict(teacher_agreement="MEASURED",arena="NOT RUN",deployment="NOT QUALIFIED",promotion="NOT PERFORMED"),
                  elapsed_seconds=time.time()-started)
    args.output.with_suffix(".json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"model":str(args.output),"sha256":sha(args.output),"graph":graph,"metrics":{k:v["mean"] for k,v in report.items()},"parity":checks,"qualification":manifest["qualification"]},indent=2))


if __name__=="__main__":main()
