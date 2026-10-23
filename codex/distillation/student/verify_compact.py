"""Operational evidence: exact solved bypass, dynamic batches and finite bounds."""
import argparse
import json
from pathlib import Path

import numpy as np
import train_student as lib


def main():
    p=argparse.ArgumentParser();p.add_argument("--model",type=Path,default=lib.HERE/"model-distill-v2.onnx")
    p.add_argument("--baseline",type=Path,default=lib.ROOT/"codex/tactical/model-tactical-v2.onnx")
    a=p.parse_args();obs=[]
    files=sorted((lib.ROOT/"data/gen00").glob("chunk*.npz"))
    for file in files[::max(1,len(files)//8)][:8]:
        with np.load(file) as data:
            obs.extend(data["obs"][np.linspace(0,len(data["obs"])-1,min(256,len(data["obs"])),dtype=int)].astype(np.float32))
    for file in sorted((lib.DISTILL/"packets").glob("packet*.json")):
        packet=json.loads(file.read_text(encoding="utf-8"));obs.extend(lib.observation(r["state"]) for r in packet["positions"])
    base,student=lib.session(a.baseline),lib.session(a.model)
    count=0;changed=0;vmax=0.;pmax=0.;exact=[]
    for i in range(0,len(obs),16):
        x=np.stack(obs[i:i+16]).astype(np.float32)
        bp,bv=base.run(None,{"observation":x});sp,sv=student.run(None,{"observation":x})
        if not np.isfinite(sp).all() or not np.isfinite(sv).all() or np.max(np.abs(sv))>1:raise AssertionError("Nonfinite/unbounded outputs")
        keep=np.abs(bv)>.98
        if not np.array_equal(bp[keep],sp[keep]) or not np.array_equal(bv[keep],sv[keep]):raise AssertionError("Exact solved policy/value changed")
        count+=int(keep.sum());changed+=int(np.any(bp!=sp,axis=1).sum())
        if keep.any():exact.extend(x[keep])
        pmax=max(pmax,float(np.max(np.abs(sp-bp))));vmax=max(vmax,float(np.max(np.abs(sv-bv))))
    if pmax>lib.POLICY_CAP+1e-5 or vmax>lib.VALUE_CAP+1e-5:raise AssertionError("Residual cap violated")
    result=dict(status="PASS",model_sha256=lib.sha(a.model),baseline_sha256=lib.sha(a.baseline),
                positions=len(obs),exact_solved_positions=count,exact_policy_bitwise_equal=True,exact_value_bitwise_equal=True,
                positions_with_changed_policy=changed,maximum_policy_residual=pmax,maximum_value_residual=vmax,
                finite_bounded_outputs=True,completed_outcome_labels_used=False)
    a.model.with_suffix(".operational.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
