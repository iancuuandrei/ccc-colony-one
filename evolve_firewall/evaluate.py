"""Score every certificate generation against the brute-force ground truth; log to generations.json.

usage: python evaluate.py [ground_*.npz ...]   (default: all ground_*.npz here)
Metrics per certificate and set:
  checked      positions with a solved ground truth (status ok) in the set
  flagged      positions the certificate fires on
  violations   flagged positions whose brute-force value differs from the claim (must be 0)
  certain      positions whose brute-force value equals the claim (+1 / -1)
  fw_certain   ... of those, decided by the firewall (value with firewall off differs)
  coverage / fw_coverage = correct flags / certain, / fw_certain
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certs as C  # noqa: E402
from candidates import GENERATIONS  # noqa: E402


def run_all(x, gens, bs=8192):
    """Chunks outer, generations inner (so candidates.base() memoises the shared primitives)."""
    out = {g: {} for g in gens}
    for i in range(0, len(x), bs):
        ch = x[i:i + bs]
        for gname, gen in gens.items():
            for k, (claim, f) in gen(ch).items():
                out[gname].setdefault(k, (claim, []))[1].append(f)
    return {g: {k: (cl, np.concatenate(fs)) for k, (cl, fs) in d.items()} for g, d in out.items()}


def lemma_report(x, g):
    d = C.certificates(x)
    ok = g["value"] != 9
    own, en, L, a, u, mag, c, R, I = C.split(x[:1])  # noqa
    Kon = d["_Kon"]
    m = ok & Kon
    esc_f, esc_b = d["_esc"][m], g["esc"][m].astype(bool)
    nk = d["_nokill"]
    kill = g["kill"].astype(bool)
    o_, e_, L_, a_, *_ = C.split(x)
    ks = C.kill_sound(o_, e_, L_, a_)
    return dict(
        esc_checked=int(m.sum()),
        esc_formula1_brute0=int((esc_f & ~esc_b).sum()),
        esc_formula0_brute1=int((~esc_f & esc_b).sum()),  # must be 0 (soundness of 'no escape')
        nokill_checked=int(ok.sum()),
        nokill_flagged=int((nk & ok).sum()),
        nokill_violations=int((nk & ok & kill).sum()),  # must be 0
        nokill_recall=round(float((nk & ok).sum() / max(1, (ok & ~kill).sum())), 4),
        quiet_mismatch=int((d["_quiet"][ok] != g["quiet"][ok].astype(bool)).sum()),
        killsound_flagged=int((ks & ok).sum()),
        killsound_violations=int((ks & ok & ~kill).sum()),  # must be 0 (R1/R2 sufficiency)
    )


def score(g, d):
    v = g["value"]
    ok = v != 9
    fw = ok & (g["value_nofw"] != v) if "value_nofw" in g else ok
    mag = g["obs"][:, 15, 0, 0] > 0.5
    rep = {}
    for name, (claim, flags) in d.items():
        f = flags & ok
        # domain: certificates named *_C* apply to Cyan to move, *_M* to Magenta to move
        tag = name.split("_")[1][:1] if "_" in name else ""
        dom = ~mag if tag == "C" else (mag if tag == "M" else np.ones_like(mag))
        cert = ok & (v == claim) & dom
        fwc = cert & fw
        rep[name] = dict(checked=int(ok.sum()), flagged=int(f.sum()),
                         violations=int((f & (v != claim)).sum()),
                         certain=int(cert.sum()), fw_certain=int(fwc.sum()),
                         coverage=round(float((f & cert).sum() / max(1, cert.sum())), 4),
                         fw_coverage=round(float((f & fwc).sum() / max(1, fwc.sum())), 4))
    return rep


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "ground_*.npz")))
    log_path = os.path.join(HERE, "generations.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else []
    sets = {}
    for f in files:
        z = np.load(f)
        sets[os.path.basename(f)[7:-4]] = ({k: z[k] for k in z.files})
    lemmas = {}
    for sname, g in sets.items():
        x = g["obs"].astype(np.float32)
        lemmas[sname] = lemma_report(x, g)
        print(sname, "lemmas", lemmas[sname])
    sel = os.environ.get("GENS")
    gens = {k: v for k, v in GENERATIONS.items() if not sel or k in sel.split(",")}
    flags = {sname: run_all(g["obs"].astype(np.float32), gens) for sname, g in sets.items()}
    for gname, gen in gens.items():
        entry = dict(generation=gname, doc=" ".join((gen.__doc__ or "").split()),
                     time=time.strftime("%Y-%m-%d %H:%M:%S"), sets={})
        for sname, g in sets.items():
            entry["sets"][sname] = score(g, flags[sname][gname])
        tot = {}
        for sname, rep in entry["sets"].items():
            for cname, m in rep.items():
                t = tot.setdefault(cname, dict(flagged=0, violations=0, certain=0, fw_certain=0, correct=0, fw_correct=0))
                t["flagged"] += m["flagged"]
                t["violations"] += m["violations"]
        entry["totals"] = tot
        entry["zero_violations"] = all(t["violations"] == 0 for t in tot.values())
        print(gname, json.dumps(entry["sets"], indent=None))
        log = [e for e in log if e.get("generation") != gname]
        log.append(entry)
    np.savez_compressed(os.path.join(HERE, f"flags_{sel or 'all'}.npz"),
                        **{f"{s}|{g}|{k}": v[1] for s, d in flags.items() for g, dd in d.items() for k, v in dd.items()})
    log = [e for e in log if e.get("generation") != "lemmas"]
    log.append(dict(generation="lemmas", time=time.strftime("%Y-%m-%d %H:%M:%S"), sets=lemmas))
    json.dump(log, open(log_path, "w"), indent=1)


if __name__ == "__main__":
    main()
