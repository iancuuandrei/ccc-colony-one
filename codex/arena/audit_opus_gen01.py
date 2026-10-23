"""Replay Opus's durable gen01 arena history without modifying its artifacts."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/"env"))
from colony_one.env import ColonyOneEnv

source = ROOT/"logs"/"gen01-vs-tactical-v2.json"
data = json.loads(source.read_text())
rows = data["games"]
assert len(rows)==400 and sum(r["a_won"] for r in rows)==388
seen = set()
for row in rows:
    key = row["seed"],row["a_cyan"]
    assert key not in seen
    seen.add(key)
    env = ColonyOneEnv()
    for team,from_pos,to_pos in row["history"]:
        assert env.current_team.startswith(team)
        legal = [a for a in env.legal_actions() if list(a.from_pos)==from_pos and list(a.to_pos)==to_pos]
        assert len(legal)==1
        env.step(legal[0])
    assert env.terminated and env.winner==row["winner"] and env.round_number==row["round"]
    assert env.action==row["actions"]==len(row["history"])+1
    assert row["a_won"]==(env.winner==("CYAN" if row["a_cyan"] else "MAGENTA"))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
released = ROOT/"release"/"gen01"/"model.onnx"
checkpoint = ROOT/"ckpt"/"gen01.onnx"
assert sha(released)==sha(checkpoint)=="ed45eae71a9332d941f5f02d9e5a469f8372af56e8d31ccf0d4b5cdbe87f6c7a"
report = {"status":"PASS","history_games":len(rows),"a_wins":388,
          "seed_min":min(r["seed"] for r in rows),"seed_max":max(r["seed"] for r in rows),
          "source_sha256":sha(source),"released_model_sha256":sha(released),
          "checkpoint_matches_release":True,
          "scope":"All saved histories are legal under the current supplied engine and reach recorded winners. Original run had no stored engine/model hash manifest; current checkpoint and release hashes agree. Fresh independent arena binds full provenance separately."}
(HERE/"opus-gen01-history-audit.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report))
