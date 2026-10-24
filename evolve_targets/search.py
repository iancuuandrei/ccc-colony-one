"""Collect raw search statistics for the policy-target experiment.

python evolve_targets/search.py sample --n 620 --seed 7          -> positions.npz
python evolve_targets/search.py run [--limit K] [--workers 4]   -> raw.jsonl (resumable)

Per position (no root noise anywhere):
  * root prior/value of gen04-endgame
  * referee-like search (100 sims), self-play-like search (250 sims), deep search (DEEP sims)
  * independent per-action values: a CHILD-sim search from every legal child position
  * 100-sim searches with the root prior replaced by each candidate target ("root injection")
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "env"))
sys.path.insert(0, str(HERE))

MODEL = str(ROOT / "ckpt" / "gen04-endgame.onnx")
DEEP, MID, REF, CHILD = 3200, 250, 100, 128
_ev = {}


def ev():
    if "e" not in _ev:
        from colony_one.onnx_runner import OnnxEvaluator
        _ev["e"] = OnnxEvaluator(MODEL)
    return _ev["e"]


def round_of(ob):
    locked = int(round(float(ob[12].sum())))
    done = {0: 0, 28: 1, 48: 2, 60: 3}.get(locked, 0)
    nxt = (8, 16, 24, 26)[min(done, 3)]
    return max(1, int(round(nxt - 8 * (1 - float(ob[14, 0, 0])))))


# ------------------------------------------------------------------ sampling
def cmd_sample(a):
    dirs = sorted(d for d in (ROOT / "data").glob("sp_gen04-endgame_*") if (d / "summary.json").exists())
    cands = []
    for d in dirs:
        for f in sorted(d.glob("chunk*.npz")):
            z = np.load(f)
            rows = np.nonzero(z["left"] > 3)[0]
            cands += [(str(f.relative_to(ROOT)), int(r)) for r in rows]
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(cands), size=a.n, replace=False)
    pick = sorted(pick)
    obs, src = [], []
    cache = {}
    for i in pick:
        f, r = cands[i]
        if f not in cache:
            cache[f] = np.load(ROOT / f)["obs"]
        obs.append(cache[f][r].astype(np.float32))
        src.append(f"{f}:{r}")
    np.savez_compressed(HERE / "positions.npz", obs=np.stack(obs), src=np.array(src))
    print(json.dumps({"dirs": [d.name for d in dirs], "candidates": len(cands), "sampled": len(obs)}))


# ------------------------------------------------------------------ search helpers
def root_stats(root, legal_idx):
    N = np.array([root.N.get(i, 0) for i in legal_idx], np.float64)
    W = np.array([root.W.get(i, 0.0) for i in legal_idx], np.float64)
    return N, W


def search_value(board, team, al, rnd, sims, rng):
    """MCTS-consistent value of a position for `team`: (v_net + sum W) / (1 + sum N)."""
    from colony_one import encoding, mcts
    planes = encoding.encode_board(board, team, al, rnd)
    _, v = ev()(planes[None])
    v0 = float(v[0])
    root = mcts.run(board, team, al, rnd, ev(), mcts.MCTSConfig(simulations=sims, add_root_noise=False), rng)
    n = sum(root.N.values())
    w = sum(root.W.values())
    return (v0 + w) / (1 + n)


def child_value(board, team, al, rnd, a_idx, sims, rng):
    """Value (root team's view) after root action a_idx, from an independent child search."""
    from colony_one import constants, encoding
    from colony_one.pieces import PieceType
    opp = constants.MAGENTA if team == constants.CYAN else constants.CYAN
    action = encoding.decode_action(a_idx, team, board)
    rec = board.apply_action(action)
    try:
        if rec.captured_piece is not None and rec.captured_piece.piece_type is PieceType.KERNEL:
            return 1.0, True
        if al - 1 > 0:
            return search_value(board, team, al - 1, rnd, sims, rng), False
        if board.only_kernels_remain():
            w = board.kernel_endgame_winner(acting_team=team)
            return (1.0 if w == team else -1.0), True
        fw = None
        if team == constants.MAGENTA and rnd in constants.SHRINK_ROUNDS:
            fw = board.apply_firewall_shrink()
        try:
            if fw is not None:
                kk = [p for _, _, p in fw.killed if p.piece_type is PieceType.KERNEL]
                if kk:
                    w = constants.MAGENTA if len(kk) == 2 else (opp if kk[0].team == team else team)
                    return (1.0 if w == team else -1.0), True
            nxt = rnd + 1 if team == constants.MAGENTA else rnd
            return -search_value(board, opp, constants.MAX_ACTION_PER_TURN, nxt, sims, rng), False
        finally:
            if fw is not None:
                board.undo_firewall_shrink(fw)
    finally:
        board.undo_action(rec)


class RootOverride:
    """Evaluator wrapper: the first call (the root expansion) returns the given logits."""

    def __init__(self, base, logits):
        self.base, self.logits, self.first = base, logits, True

    def __call__(self, planes):
        lg, v = self.base(planes)
        if self.first:
            self.first = False
            lg = np.array(lg, copy=True)
            lg[0] = self.logits
        return lg, v


# ------------------------------------------------------------------ worker
def work(job):
    pid, ob = job
    from colony_one import constants, encoding, mcts
    from colony_one.board import Board
    from colony_one.rules import legal_actions_for_team
    import targets as T
    rng = np.random.default_rng(1000 + pid)
    team = constants.MAGENTA if ob[15, 0, 0] > 0.5 else constants.CYAN
    al = int(round(2 * float(ob[13, 0, 0])))
    rnd = round_of(ob)
    board = Board.from_observation(ob, team)
    planes = encoding.encode_board(board, team, al, rnd)
    enc_diff = float(np.abs(planes - ob).max())
    legal = legal_actions_for_team(board, team)
    legal_idx = [encoding.action_index(x, team) for x in legal]
    out = dict(id=pid, team=team, actions_left=al, round=rnd, n_legal=len(legal_idx), enc_diff=enc_diff)
    if len(legal_idx) < 2:
        out["skip"] = "fewer than 2 legal actions"
        return out
    lg, v = ev()(planes[None])
    prior = T.softmax(np.asarray(lg[0], np.float64)[legal_idx])
    out.update(legal=legal_idx, prior=prior.tolist(), v_net=float(v[0]))
    tm = {}
    for name, sims in (("ref", REF), ("mid", MID), ("deep", DEEP)):
        t0 = time.perf_counter()
        root = mcts.run(board, team, al, rnd, ev(), mcts.MCTSConfig(simulations=sims, add_root_noise=False), rng)
        tm[name] = time.perf_counter() - t0
        N, W = root_stats(root, legal_idx)
        out[name] = dict(N=N.tolist(), W=W.tolist())
    # independent per-action values
    t0 = time.perf_counter()
    qc, term = [], []
    for i in legal_idx:
        q, t = child_value(board, team, al, rnd, i, CHILD, rng)
        qc.append(q); term.append(t)
    tm["child"] = time.perf_counter() - t0
    out["child"] = dict(Q=qc, terminal=term)
    # root injection: candidate targets from the 250-sim search (and the deep visits as oracle)
    Nm = np.array(out["mid"]["N"]); Wm = np.array(out["mid"]["W"])
    Qm = np.where(Nm > 0, Wm / np.maximum(Nm, 1), np.nan)
    tg = T.all_targets(prior, Nm, Qm, out["v_net"])
    tg["deep_visits_oracle"] = T.visits(out["deep"]["N"])
    inj = {}
    t0 = time.perf_counter()
    for name, pi in tg.items():
        logits = np.full(encoding.NUM_ACTIONS, -60.0, np.float32)
        logits[legal_idx] = np.log(np.maximum(pi, 1e-6)).astype(np.float32)
        root = mcts.run(board, team, al, rnd, RootOverride(ev(), logits),
                        mcts.MCTSConfig(simulations=REF, add_root_noise=False), rng)
        N, _ = root_stats(root, legal_idx)
        inj[name] = N.tolist()
    tm["inject_each"] = (time.perf_counter() - t0) / len(tg)
    out["inject"] = inj
    out["time"] = tm
    return out


def cmd_run(a):
    pos = np.load(HERE / "positions.npz")
    obs = pos["obs"]
    raw = HERE / (a.out or "raw.jsonl")
    done = set()
    if raw.exists():
        for line in raw.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    ids = [i for i in range(len(obs)) if i not in done]
    if a.limit:
        ids = ids[: a.limit]
    t = time.time()
    with ProcessPoolExecutor(a.workers) as ex, raw.open("a") as fh:
        for k, r in enumerate(ex.map(work, [(i, obs[i]) for i in ids], chunksize=1)):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if (k + 1) % 20 == 0 or k + 1 == len(ids):
                print(json.dumps({"done": k + 1, "of": len(ids), "seconds": round(time.time() - t, 1)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sample"); s.add_argument("--n", type=int, default=620); s.add_argument("--seed", type=int, default=7)
    r = sp.add_parser("run"); r.add_argument("--limit", type=int, default=0); r.add_argument("--workers", type=int, default=4)
    r.add_argument("--out", default="")
    a = p.parse_args()
    {"sample": cmd_sample, "run": cmd_run}[a.cmd](a)
