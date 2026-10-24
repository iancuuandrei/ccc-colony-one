"""How often certified firewall positions occur in self-play (raw chunks, duplicates kept).
Writes freq.json."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final  # noqa: E402

ROOT = os.path.dirname(HERE)


def main():
    folders = sorted(glob.glob(os.path.join(ROOT, "data", "sp_*"))) + sorted(glob.glob(os.path.join(ROOT, "data", "vps_*")))
    xs, zs = [], []
    total = 0
    for f in folders:
        for ch in sorted(glob.glob(os.path.join(f, "chunk*.npz"))):
            d = np.load(ch)
            o = d["obs"]
            total += len(o)
            m = o[:, 14, 0, 0].astype(np.float32) > 0.999
            xs.append(o[m]); zs.append(d["z"][m])
    x = np.concatenate(xs).astype(np.float32)
    z = np.concatenate(zs)
    res = {k: [] for k in ("WIN", "LOSS", "WIN_M", "WIN_C", "LOSS_M", "LOSS_C")}
    t1 = {k: [] for k in ("WIN", "LOSS")}
    for i in range(0, len(x), 8192):
        d = final.certify(x[i:i + 8192])
        for k in res:
            res[k].append(d[k])
        d1 = final.certify(x[i:i + 8192], tier1=True)
        for k in t1:
            t1[k].append(d1[k])
    res = {k: np.concatenate(v) for k, v in res.items()}
    t1 = {k: np.concatenate(v) for k, v in t1.items()}
    mag = x[:, 15, 0, 0] > 0.5
    nl = x[:, 12].sum((1, 2))
    c = (nl > 1).astype(int) + (nl > 30) + (nl > 50)
    out = dict(positions_total=int(total), shrink_round_positions=int(len(x)),
               shrink_round_share=round(len(x) / total, 4))
    for k, v in res.items():
        out[k] = int(v.sum())
    out["certified_share_of_shrink_round"] = round(float((res["WIN"] | res["LOSS"]).mean()), 4)
    out["certified_share_of_all_positions"] = round(float((res["WIN"] | res["LOSS"]).sum() / total), 5)
    out["tier1_certified"] = int((t1["WIN"] | t1["LOSS"]).sum())
    out["by_shrink"] = {int(s): dict(positions=int((c == s).sum()), WIN=int((res["WIN"] & (c == s)).sum()),
                                     LOSS=int((res["LOSS"] & (c == s)).sum())) for s in range(4)}
    out["by_mover"] = {"CYAN": dict(positions=int((~mag).sum()), WIN=int((res["WIN"] & ~mag).sum()), LOSS=int((res["LOSS"] & ~mag).sum())),
                       "MAGENTA": dict(positions=int(mag.sum()), WIN=int((res["WIN"] & mag).sum()), LOSS=int((res["LOSS"] & mag).sum()))}
    # agreement with the recorded game outcome z (mover's perspective): a sanity check only
    zw = z[res["WIN"]]
    zl = z[res["LOSS"]]
    out["z_agree_WIN"] = round(float((zw > 0).mean()), 4) if len(zw) else None
    out["z_agree_LOSS"] = round(float((zl < 0).mean()), 4) if len(zl) else None
    json.dump(out, open(os.path.join(HERE, "freq.json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
