"""Operational audit of durable arena histories and cross-harness agreement."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/"env"))
from colony_one.env import ColonyOneEnv


def main():
    report = {"history_replay": [], "original_arena_agreement": []}
    for folder in sorted(HERE.iterdir()):
        if not folder.is_dir() or not (folder/"games.jsonl").exists():
            continue
        summary = json.loads((folder/"summary.json").read_text())
        if not summary["complete"]:
            continue
        rows = [json.loads(x) for x in (folder/"games.jsonl").read_text().splitlines()]
        for row in rows:
            env = ColonyOneEnv(seed=row["seed"])
            for h in row["history"]:
                assert (env.current_team,env.round_number,env.actions_left)==(h["team"],h["round"],h["actions_left"])
                legal = [a for a in env.legal_actions() if list(a.from_pos)==h["from"] and list(a.to_pos)==h["to"]]
                assert len(legal)==1
                env.step(legal[0])
            assert env.terminated and env.winner==row["winner"] and env.round_number==row["round"]
            assert len(row["history"])==row["actions"]
        report["history_replay"].append({"folder":folder.name,"games":len(rows),"status":"PASS"})
    spec = importlib.util.spec_from_file_location("original_arena",ROOT/"arena.py")
    arena = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arena)
    folder = HERE/"h1-v-l3-holdout10000"
    manifest = json.loads((folder/"manifest.json").read_text())["protocol"]
    rows = [json.loads(x) for x in (folder/"games.jsonl").read_text().splitlines()]
    for target_seed,cyan in ((10000,False),(10001,True)):
        prior = next(r for r in rows if r["seed"]==target_seed and r["a_cyan"]==cyan)
        rerun = arena.play((manifest["a"],manifest["b"],5000,cyan,100))
        expected = [[h["team"][0],h["from"],h["to"]] for h in prior["history"]]
        assert rerun["history"]==expected and rerun["winner"]==prior["winner"]
        report["original_arena_agreement"].append({"rng_seed":target_seed,"a_cyan":cyan,"status":"PASS"})
    (HERE/"evidence-audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=="__main__":
    main()
