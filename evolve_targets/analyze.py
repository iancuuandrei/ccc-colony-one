"""Score candidate policy targets against deep-search ground truth -> results.json.

python evolve_targets/analyze.py [--raw raw.jsonl]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import targets as T
import search as S

EPS = 1e-3
B = 2000


def first_argmax(x, legal):
    """argmax with ties broken by smallest action index (mcts.select_action_index order)."""
    x = np.asarray(x, np.float64)
    order = np.argsort(legal, kind="stable")
    return int(order[int(np.argmax(x[order]))])


def qarr(d):
    N = np.asarray(d["N"], np.float64); W = np.asarray(d["W"], np.float64)
    return N, np.where(N > 0, W / np.maximum(N, 1), np.nan)


def boot(x, rng):
    x = np.asarray(x, np.float64)
    m = x.mean()
    idx = rng.integers(0, len(x), size=(B, len(x)))
    bs = x[idx].mean(1)
    return [round(float(m), 5), round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--raw", default="raw.jsonl"); a = p.parse_args()
    recs = [json.loads(l) for l in (HERE / a.raw).read_text().splitlines() if l.strip()]
    skipped = [r for r in recs if "skip" in r]
    recs = sorted([r for r in recs if "skip" not in r], key=lambda r: r["id"])
    rng = np.random.default_rng(0)

    per = {}  # metric key -> list over positions
    def add(k, v): per.setdefault(k, []).append(float(v))
    tgt_time = []
    meta = dict(actions_left=[], round=[], n_legal=[], deep_top_share=[], gt_agree=[], mid_argmax_is_best=[],
                visits_ne_gumbel_argmax=[])

    for r in recs:
        legal = np.asarray(r["legal"]); L = len(legal)
        prior = np.asarray(r["prior"]); v = r["v_net"]
        Nd, Qd = qarr(r["deep"])
        Qc = np.asarray(r["child"]["Q"], np.float64)
        best = first_argmax(Nd, legal)
        bestc = first_argmax(Qc, legal)
        Qfill = np.where(Nd > 0, Qd, Qc)
        pdeep = Nd / Nd.sum()
        Hdeep = float(-(pdeep[pdeep > 0] * np.log(pdeep[pdeep > 0])).sum())
        meta["actions_left"].append(r["actions_left"]); meta["round"].append(r["round"]); meta["n_legal"].append(L)
        meta["deep_top_share"].append(float(Nd.max() / Nd.sum())); meta["gt_agree"].append(float(best == bestc))
        add("gt/deep_entropy", Hdeep)

        for src in ("mid", "ref"):
            N, Q = qarr(r[src])
            t0 = time.perf_counter()
            tg = T.all_targets(prior, N, Q, v)
            if src == "mid":
                tgt_time.append(time.perf_counter() - t0)
                meta["mid_argmax_is_best"].append(float(first_argmax(N, legal) == best))
                meta["visits_ne_gumbel_argmax"].append(float(first_argmax(tg["visits"], legal) != first_argmax(tg["gumbel"], legal)))
            for name, pi in tg.items():
                k = f"target_{src}/{name}"
                am = first_argmax(pi, legal)
                pis = (1 - EPS) * pi + EPS / L
                add(k + "/top1_deep", am == best)
                add(k + "/p_best_deep", pi[best])
                add(k + "/regret_deep", Qd[best] - float((pi * Qfill).sum()))
                add(k + "/mass_unvisited_by_deep", float(pi[Nd == 0].sum()))
                add(k + "/ce_deep_visits", -float((pdeep * np.log(pis)).sum()))
                add(k + "/top1_child", am == bestc)
                add(k + "/regret_child", Qc[bestc] - float((pi * Qc).sum()))

        # referee-like 100-sim decision, plain and with root-injected targets
        dec = {"referee_plain": np.asarray(r["ref"]["N"])}
        dec.update({f"inject/{k}": np.asarray(n) for k, n in r["inject"].items()})
        dec["selfplay_mid_250"] = np.asarray(r["mid"]["N"])
        for name, N in dec.items():
            c = first_argmax(N, legal)
            k = f"decision_100/{name}"
            add(k + "/top1_deep", c == best)
            add(k + "/regret_deep", Qd[best] - Qfill[c])
            add(k + "/top1_child", c == bestc)
            add(k + "/regret_child", Qc[bestc] - Qc[c])

    n = len(recs)
    arr = {k: np.asarray(v) for k, v in per.items()}

    def summarise(prefix, ref_name, names, ref2=None):
        out = {}
        for nm in names:
            base = f"{prefix}/{nm}"
            ms = sorted({k[len(base) + 1:] for k in arr if k.startswith(base + "/")})
            out[nm] = {}
            for m in ms:
                x = arr[f"{base}/{m}"]
                d = {"mean_ci95": boot(x, rng)}
                for rn in (ref_name, ref2):
                    if rn and nm != rn:
                        y = arr[f"{prefix}/{rn}/{m}"]
                        d[f"diff_vs_{rn.replace('/', '_')}_ci95"] = boot(x - y, rng)
                out[nm][m] = d
        return out

    tnames = list(T.all_targets(np.ones(2) / 2, np.ones(2), np.zeros(2), 0.0).keys())
    dnames = ["referee_plain"] + [f"inject/{k}" for k in tnames + ["deep_visits_oracle"]] + ["selfplay_mid_250"]
    tm = {k: [] for k in recs[0]["time"]}
    for r in recs:
        for k, x in r["time"].items():
            tm[k].append(x)
    al = np.asarray(meta["actions_left"])
    by_al = {}
    for v_ in (1, 2):
        mask = al == v_
        by_al[f"actions_left={v_}"] = {
            "n": int(mask.sum()),
            **{f"{t}/{m}": round(float(arr[f"target_mid/{t}/{m}"][mask].mean()), 5)
               for t in tnames for m in ("top1_deep", "regret_deep", "regret_child")},
        }
    def sub(mask, ms):
        if not mask.any():
            return {"n": 0}
        return {"n": int(mask.sum()),
                **{f"{t}/{m}": round(float(arr[f"target_mid/{t}/{m}"][mask].mean()), 5) for t in tnames for m in ms}}
    contested = np.asarray(meta["mid_argmax_is_best"]) < 0.5
    cont = sub(contested, ("top1_deep", "regret_deep", "p_best_deep", "top1_child"))
    disagree = np.asarray(meta["visits_ne_gumbel_argmax"]) > 0.5
    disg = sub(disagree, ("top1_deep", "regret_deep", "top1_child", "regret_child"))

    res = {
        "setup": {
            "model": S.MODEL, "positions_sampled": n + len(skipped), "positions_used": n,
            "skipped": len(skipped), "source": "data/sp_gen04-endgame_* (completed dirs), key left > 3, seed 7",
            "sims": {"deep_ground_truth": S.DEEP, "selfplay_target_search": S.MID, "referee": S.REF,
                     "independent_child_search_per_action": S.CHILD},
            "search": "mcts.run, MCTSConfig(simulations=..., add_root_noise=False), c_puct 1.5, batch 8",
            "gumbel": "pi'=softmax(log prior + (50+maxN)*c_scale*q01), completedQ: Q(a) visited, v_mix (mctx, prior-weighted) unvisited, "
                      "q01 = min-max over legal; c_scale 0.1 ('gumbel'), 1.0 ('gumbel_c1'); 'gumbel_vw' uses v_mix=(v_net+sumW)/(1+sumN)",
            "softq": "softmax(Q/tau) over visited actions, raw Q in [-1,1]",
            "ground_truth_deep": "best = argmax N of the deep search; Q_deep = W/N; actions the deep search never visited are filled "
                                 "with their independent child-search value for regret",
            "ground_truth_child": "Q_child(a) = (v_net + sumW)/(1+sumN) of a 128-sim search from the child position, sign-corrected; "
                                  "terminal transitions exact; best = argmax Q_child",
            "ce": f"-sum p_deep log((1-{EPS})pi + {EPS}/L)",
            "decision_100": "argmax visits of a 100-sim no-noise search; inject/X = root prior replaced by target X "
                            "(interior nodes keep the network prior): perfect-fit-at-root proxy, not a trained network",
            "ci": f"bootstrap over positions, {B} resamples; diff_* are paired",
        },
        "ground_truth_stats": {
            "deep_top_action_visit_share_mean": round(float(np.mean(meta["deep_top_share"])), 4),
            "deep_best_equals_child_best": round(float(np.mean(meta["gt_agree"])), 4),
            "mid250_argmax_equals_deep_best": round(float(np.mean(meta["mid_argmax_is_best"])), 4),
            "deep_visit_entropy_mean": round(float(arr["gt/deep_entropy"].mean()), 4),
            "n_legal_mean": round(float(np.mean(meta["n_legal"])), 2),
            "actions_left_counts": {str(k): int((al == k).sum()) for k in (1, 2)},
        },
        "targets_from_250": summarise("target_mid", "visits", tnames),
        "targets_from_100": summarise("target_ref", "visits", tnames),
        "decision_100sim": summarise("decision_100", "referee_plain", dnames, ref2="inject/visits"),
        "breakdown_targets_from_250": {"by_actions_left": by_al, "contested_positions_mid_argmax_ne_deep_best": cont,
                                       "positions_where_visits_and_gumbel_argmax_differ": disg},
        "cost": {
            "wall_seconds_mean_per_position": {k: round(float(np.mean(v)), 4) for k, v in tm.items()},
            "wall_seconds_median_per_position": {k: round(float(np.median(v)), 4) for k, v in tm.items()},
            "all_targets_compute_seconds_mean": float(np.mean(tgt_time)),
            "note": "4 processes on a machine at ~100% CPU from other jobs; times are relative, not absolute",
        },
    }
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"n": n, "skipped": len(skipped)}))


if __name__ == "__main__":
    main()
