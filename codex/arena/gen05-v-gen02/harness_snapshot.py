"""Durable exact-referee comparison, paired by opening RNG seed and both colors.

Each seed gives two games with the same initial RNG state. Different legal
branches consume different randomness, so this is variance reduction rather
than an assertion that the openings are identical. Fixed seeds are for screening;
reserve a disjoint seed range for confirmation after selecting a candidate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "env"))
_evaluators = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluator(path):
    if path not in _evaluators:
        from colony_one.onnx_runner import OnnxEvaluator
        _evaluators[path] = OnnxEvaluator(path, num_threads=1)
    return _evaluators[path]


def play(job):
    a, b, seed, a_cyan, sims = job
    from colony_one import constants, mcts
    from colony_one.env import ColonyOneEnv
    ea, eb = evaluator(a), evaluator(b)
    cfg = mcts.MCTSConfig(simulations=sims)
    rng = np.random.default_rng(seed)
    env = ColonyOneEnv(seed=seed)
    history = []
    start = time.monotonic()
    while not env.terminated:
        a_turn = (env.current_team == constants.CYAN) == a_cyan
        action = mcts.search_action(env.board, env.current_team, env.actions_left,
                                   env.round_number, ea if a_turn else eb, cfg, rng=rng)
        history.append({"team": env.current_team, "round": env.round_number,
                        "actions_left": env.actions_left,
                        "from": list(action.from_pos), "to": list(action.to_pos)})
        env.step(action)
    a_team = constants.CYAN if a_cyan else constants.MAGENTA
    return {"seed": seed, "a_cyan": a_cyan, "a_won": env.winner == a_team,
            "winner": env.winner, "actions": len(history), "round": env.round_number,
            "seconds": time.monotonic() - start, "history": history}


def wilson(wins, n, z=1.959963984540054):
    if not n:
        return None
    p = wins / n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]


def summarize(rows, expected):
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["a_cyan"]] = row
    pairs = [v for v in by_seed.values() if len(v) == 2]
    scores = np.array([sum(r["a_won"] for r in v.values()) / 2 for v in pairs])
    n = len(scores)
    out = {"games": len(rows), "expected_games": expected,
           "complete": len(rows) == expected, "a_wins": sum(r["a_won"] for r in rows),
           "a_rate": sum(r["a_won"] for r in rows)/len(rows) if rows else None,
           "complete_pairs": n, "per_color": {},
           "uncertainty_note": "Wilson intervals are per-color; paired bootstrap resamples complete seeds, preserving within-pair dependence. Fixed-size exploratory evidence, not tournament acceptance or multiple-selection corrected proof."}
    for color, c in (("CYAN", True), ("MAGENTA", False)):
        rr = [r for r in rows if r["a_cyan"] == c]
        w = sum(r["a_won"] for r in rr)
        out["per_color"][color] = {"wins": w, "games": len(rr),
                                   "rate": w/len(rr) if rr else None,
                                   "wilson95": wilson(w, len(rr))}
    if n:
        boot = np.random.default_rng(928572).choice(scores, size=(20000,n)).mean(axis=1)
        lower, upper = np.quantile(boot, [0.025, 0.975]).tolist()
        pair_wins, splits, losses = (int(np.sum(scores == x)) for x in (1, .5, 0))
        unequal = pair_wins + losses
        p_sign = min(1., 2*sum(math.comb(unequal,k) for k in range(min(pair_wins, losses)+1))/2**unequal) if unequal else 1.
        out.update({"paired_score_mean": float(scores.mean()),
                    "paired_bootstrap95": [lower,upper],
                    "paired_hoeffding95": [max(0,float(scores.mean())-math.sqrt(math.log(40)/(2*n))),
                                            min(1,float(scores.mean())+math.sqrt(math.log(40)/(2*n)))],
                    "pair_outcomes": {"a_sweeps":pair_wins,"splits":splits,"b_sweeps":losses},
                    "sweep_sign_test_two_sided_p":p_sign,
                    "evidence": "A_FAVORED" if n>=64 and lower>.5 else "B_FAVORED" if n>=64 and upper<.5 else "INCONCLUSIVE"})
    return out


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a"); p.add_argument("b")
    p.add_argument("--pairs", type=int, default=64)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    o = p.parse_args()
    if o.pairs<1 or o.workers<1 or o.workers>8:
        p.error("pairs >=1, 1<=workers<=8 required")
    from colony_one import mcts
    import onnxruntime as ort
    o.out.mkdir(parents=True, exist_ok=True)
    a,b = str(Path(o.a).resolve()), str(Path(o.b).resolve())
    engine_files = sorted((ROOT/"env"/"colony_one").glob("*.py")) + [ROOT/"env"/"colony_one"/"scene.json"]
    file_hashes = {str(x.relative_to(ROOT)):digest(x) for x in engine_files}
    protocol = {"a":a,"b":b,"a_sha256":digest(a),"b_sha256":digest(b),
                "start":o.start,"pairs":o.pairs,"rng":"numpy.default_rng(seed), same seed across swapped colors",
                "config":asdict(mcts.MCTSConfig(simulations=o.sims)),
                "engine_files":file_hashes,
                "engine_sha256":hashlib.sha256(json.dumps(file_hashes,sort_keys=True).encode()).hexdigest(),
                "python":platform.python_version(),"numpy":np.__version__,"onnxruntime":ort.__version__,
                "harness_sha256":digest(__file__)}
    protocol = json.loads(json.dumps(protocol))  # normalize tuple config fields for durable resume
    manifest = o.out/"manifest.json"
    if manifest.exists():
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        if prior["protocol"] != protocol:
            raise RuntimeError("Cannot resume: model, engine, harness, runtime or protocol changed")
    else:
        write_json(manifest,{"protocol":protocol,"created_utc":datetime.now(timezone.utc).isoformat()})
    records = o.out/"games.jsonl"
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()] if records.exists() else []
    seen = {(r["seed"],r["a_cyan"]) for r in rows}
    if len(seen)!=len(rows):
        raise RuntimeError("Duplicate completed-game records")
    jobs = [(a,b,s,c,o.sims) for s in range(o.start,o.start+o.pairs) for c in (True,False) if (s,c) not in seen]
    expected = o.pairs*2
    print(json.dumps({"event":"starting","pending":len(jobs),"workers":o.workers,"a_sha256":protocol["a_sha256"],"b_sha256":protocol["b_sha256"]}),flush=True)
    with records.open("a",encoding="utf-8") as out, ProcessPoolExecutor(o.workers) as ex:
        futures = {ex.submit(play,j):j for j in jobs}
        for f in as_completed(futures):
            row = f.result()
            out.write(json.dumps(row,separators=(",",":"))+"\n")
            out.flush(); os.fsync(out.fileno())
            rows.append(row)
            report = summarize(rows,expected)
            write_json(o.out/"summary.json",report)
            if len(rows)%16==0 or len(rows)==expected:
                print(json.dumps({"event":"progress","games":len(rows),"a_wins":report["a_wins"],
                                  "complete_pairs":report["complete_pairs"],"a_rate":report["a_rate"]}),flush=True)
    if {str(x.relative_to(ROOT)):digest(x) for x in engine_files} != file_hashes or digest(a)!=protocol["a_sha256"] or digest(b)!=protocol["b_sha256"]:
        raise RuntimeError("Model or engine changed during execution: evidence invalid")
    report = summarize(rows,expected)
    report["end_provenance_verified"] = True
    write_json(o.out/"summary.json",report)
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
