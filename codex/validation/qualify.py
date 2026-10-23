"""Extra exact-model evidence. Official validators are never modified.

Run with --mode graph, hard, or level4. Heavy modes stay single-process.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("model", type=Path)
p.add_argument("--mode", choices=("graph", "hard", "level4"), required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--output", type=Path)
args = p.parse_args()
root = Path(__file__).resolve().parents[2]
envpath = root / "env"
sys.path.insert(0, str(envpath))
model = args.model.resolve()
result = {"model": str(model), "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
          "bytes": model.stat().st_size, "mode": args.mode, "python": sys.executable}
result["source_sha256"] = {
    name: hashlib.sha256((envpath / name).read_bytes()).hexdigest()
    for name in ("colony_one/scene.json", "colony_one/submission_policy.py",
                 "colony_one/submission_check.py", "colony_one/mcts.py",
                 "tools/validate_submission.py")
}

if args.mode == "graph":
    import numpy as np
    import onnx
    import onnxruntime as ort
    from colony_one import submission_policy as policy
    result["versions"] = {"numpy": np.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__}
    proto = onnx.load(str(model), load_external_data=False)
    onnx.checker.check_model(model.read_bytes())
    policy.validate_submission_contract(proto)
    result["structural_contract"] = "PASS"
    result["opsets"] = [(x.domain, x.version) for x in proto.opset_import]
    result["operators"] = dict(collections.Counter(n.op_type for n in proto.graph.node))
    result["nodes"] = len(proto.graph.node)
    result["initializers"] = len(proto.graph.initializer)
    result["constant_indices"] = "PASS (authoritative check_graph_structure)"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    t = time.perf_counter()
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    result["session_wall_seconds"] = time.perf_counter() - t
    cases = policy._validation_corpus()
    result["corpus_cases"] = len(cases)
    result["batches"] = []
    for batchsize in (1, 8, 16):
        # Every fixed legal/numeric-extreme case occurs, including batch1.
        batches = [np.stack([cases[(offset + j) % len(cases)] for j in range(batchsize)])
                   for offset in range(0, len(cases), batchsize)]
        t = time.perf_counter()
        vmin, vmax = 1., -1.
        for x in batches:
            pol, val = session.run(None, {session.get_inputs()[0].name: x})
            policy._check_batch_outputs(pol, val, batchsize)
            vmin, vmax = min(vmin, float(val.min())), max(vmax, float(val.max()))
        result["batches"].append({"batch": batchsize, "calls": len(batches), "result": "PASS",
                                  "wall_seconds": time.perf_counter() - t,
                                  "value_min": vmin, "value_max": vmax})

elif args.mode == "hard":
    if sys.platform == "win32":
        raise SystemExit("hard mode requires POSIX resource limits")
    import resource
    from colony_one import submission_policy as policy
    # Apply the supplied numeric caps in an isolated interpreter before any
    # NumPy/ONNX import. The authoritative Linux wrapper is not supplied.
    child = envpath / "colony_one" / "submission_check.py"
    result["checks"] = []
    for mode, address_mib, cpu_limit, wall_limit in (
        ("--child", policy.CHECK_ADDRESS_SPACE_MIB, policy.CHECK_CPU_SECONDS, policy.VALIDATION_TIMEOUT_SECONDS),
        ("--match-child", policy.MATCH_ADDRESS_SPACE_MIB, policy.MATCH_CPU_SECONDS, policy.MATCH_TIMEOUT_SECONDS),
    ):
        def limit_child():
            resource.setrlimit(resource.RLIMIT_AS, (address_mib * 1024**2,) * 2)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit,) * 2)
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        item = {"mode": mode, "caps": {"address_space_MiB": address_mib,
                "cpu_seconds": cpu_limit, "wall_seconds": wall_limit, "open_files": 32}}
        t = time.perf_counter()
        try:
            completed = subprocess.run([sys.executable, "-I", str(child), mode, str(model)],
                                       env=policy.child_environment(), stdin=subprocess.DEVNULL,
                                       capture_output=True, timeout=wall_limit, preexec_fn=limit_child)
            item.update(returncode=completed.returncode, stdout=completed.stdout.decode("utf-8", "replace"),
                        stderr=completed.stderr.decode("utf-8", "replace"),
                        result="PASS" if completed.returncode == 0 else "FAIL")
        except subprocess.TimeoutExpired:
            item.update(result="FAIL", reason=f"published {wall_limit}s wall timeout")
        item["wall_seconds"] = time.perf_counter() - t
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        item["child_usage"] = {"cpu_seconds": usage.ru_utime + usage.ru_stime - before.ru_utime - before.ru_stime,
                               "max_rss_KiB_cumulative": usage.ru_maxrss}
        result["checks"].append(item)
    result["result"] = "PASS" if all(x["result"] == "PASS" for x in result["checks"]) else "FAIL"

elif args.mode == "level4":
    import numpy as np
    import psutil
    from colony_one import constants, mcts
    from colony_one.env import ColonyOneEnv
    from colony_one.onnx_runner import OnnxEvaluator
    t = time.process_time()
    evaluator = OnnxEvaluator(model, num_threads=1)
    result["initialization_cpu_seconds"] = time.process_time() - t
    game = ColonyOneEnv()
    rng = np.random.default_rng(args.seed)
    cfg = mcts.MCTSConfig(simulations=100)
    side = {str(constants.CYAN): {"actions": 0, "cpu_seconds": 0., "wall_seconds": 0.},
            str(constants.MAGENTA): {"actions": 0, "cpu_seconds": 0., "wall_seconds": 0.}}
    peak = psutil.Process().memory_info().rss
    start = time.perf_counter()
    while not game.terminated:
        team = str(game.current_team)
        t, w = time.process_time(), time.perf_counter()
        action = mcts.search_action(game.board, game.current_team, game.actions_left,
                                    game.round_number, evaluator, cfg, rng=rng)
        side[team]["cpu_seconds"] += time.process_time() - t
        side[team]["wall_seconds"] += time.perf_counter() - w
        if action is None:
            break
        side[team]["actions"] += 1
        game.step(action)
        peak = max(peak, psutil.Process().memory_info().rss)
    result.update(result="PASS" if game.terminated else "INCOMPLETE", seed=args.seed,
                  simulations=100, batch_size=cfg.batch_size, sides=side, winner=game.winner,
                  total_wall_seconds=time.perf_counter() - start, peak_rss_bytes=peak)

output = json.dumps(result, indent=2)
print(output, flush=True)
if args.output:
    args.output.write_text(output + "\n", encoding="utf-8")
