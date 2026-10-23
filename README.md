# Colony: One — private Level 4 collaboration

This private repository snapshots the current ONNX submission, baselines, training scripts, supplied referee environment, task PDF/scene and bounded evidence. It is independent of CCC-OMNI infrastructure.

## Artifacts and evidence

- `model.onnx` is the **current GEN01 final**, identical to `release/gen01/model.onnx` and `model-gen01.onnx`: 1,439,594 bytes; SHA-256 `ed45eae71a9332d941f5f02d9e5a469f8372af56e8d31ccf0d4b5cdbe87f6c7a`. The importer explicitly reads the GEN01 release, because the older workspace `model.onnx` still contains h1.
- `checkpoints/gen01.pt` preserves the small retraining checkpoint (SHA-256 `710a69d2297cf4422b257af313fdfdf3fa12404d8bea82bf46fa560f39bbb239`). Its configuration/arguments are extracted in `provenance/gen01-checkpoint.json`. This enables a model warm start; full optimizer/scheduler/RNG continuation is not established.
- `model-h1.onnx` remains the 311,345-byte historical heuristic with SHA-256 `0940ddada8c2b761126ffbc5532d3103c3b8cb4eb20d89f7923f70d03f2e8b65`; `build_heuristic.py` is its generator. Named earlier models are preserved.
- `model-l3.onnx` is the earlier Level 3 baseline; `random.onnx` is the supplied random baseline.
- `evidence/comparison-h1.json` records a historical local 84/128 (65.6%) result against Level 3. The underlying game histories were unavailable. Its validator ACCEPTED claim conflicts with the later Windows timing rejection in `evidence/h1-validation-windows.txt`; both are preserved.
- `evidence/gen00-summary.json` records 6,000 generated games and 203,193 positions. Bulk NPZ files are excluded. This does not establish a trained checkpoint.

Andrei uploaded GEN01 himself; the upload owner observed the site displaying `model.onnx` at 1.37 MiB with “Model validated and deployed to the tournament”. No duplicate upload was performed. `codex/coordination.md` records this observation. Earlier h1 acceptance was observed at approximately 16:46:40 Europe/Bucharest on 2026-10-23. No sustained tournament score improvement is claimed. Browser screenshots and account state are excluded.

GEN01 won **63/64** in the fresh independent paired arena against tactical v2, using the exact Level 4 scene and 100 simulations; all 64 histories replayed legally. A separate audit replayed the 400 saved Opus histories and verified the reported 388/400 wins against v2. That historical run did not store its original full engine/model manifest; the independent arena has separate provenance. Native Linux graph, hard resource caps and validator evidence are in `codex/validation/gen01-*`; earlier mounted-path timeouts are retained. These checks qualify their measured scopes only.

Tactical v1/v2 and other earlier named models remain available with immutable hash pins. The current final is GEN01. Never infer deployment from an arena win rate or a local PASS.

`codex/arena/` and tactical arena directories contain bounded manifests, summaries and available harness snapshots. Some runs are partial: inspect `complete` and `expected_games` before interpreting rates. Raw game histories remain in the source workspace; replay claims in imported audit files describe that source evidence. `codex/validation/` preserves exact-model reports including Linux PASS and Windows timing failures. `codex/engine_audit/` and `codex/matches/` contain source and bounded diagnostics. Raw site records, account/browser data and bulk self-play files are excluded.

`codex/distillation/` now includes **32 genuine Astra labels** (7 exact wins, 2 exact losses, 23 predictions), packets, schema, independent engine validation, target arrays, student training source and frozen v1/v2 candidates with learning evidence. Compact v2 is a 22-parameter adapter; teacher-agreement gains did not establish stronger play (121/256 against tactical v2), so it was **not promoted**. Student recovery manifests bind executed source snapshots and artifacts. Operational fixture/diagnostic models are excluded.

The historical teacher-ready reports name validator hash `c74610db…`, while the synchronized current validator has subsequently changed. Its original source snapshot is unavailable. Packet, label and target hashes still match those reports; `provenance/collaboration-integrity.json` records this exact limit. No historical validator-source reproduction is claimed.

`codex/synthetic/` contains completed LLM-authored constructed proposals and bounded certified corpora, separate from real games. The current curriculum manifest reports 224 constructed states across 5 motif groups; full-game values remain unknown where explicitly masked. Opening reachability is unproved; symmetry/family siblings belong in the same split. Per-action certification does not prove eventual victory. Reports retain their precise source hashes; duplicate raw validation snapshots are omitted. These synthetic states and Astra labels are **not claimed to have trained GEN01**.

Larger networks under the 5 MiB budget, incremental comparisons, and analysis of real wins/losses remain ongoing. Losses/hard decisions have priority, while wins also supply evidence; the repository does not claim these ongoing requests are complete.

`release/gen01/source-current/` freezes the neural scripts captured during this sync, each hash-bound in the manifest. Their exact identity at the historical GEN01 generation build is **UNKNOWN**; checkpoint configuration is stronger evidence than reconstructing unrecorded code history. Root scripts track current collaboration code. Do not confuse the frozen current snapshot with a proven historical build source.

## Setup and evaluation

Python 3.12 on Linux/WSL2 is preferred for heavy computation. The supplied package declares Python 3.11–3.14. Run from the repository root:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/verify_snapshot.py
python scripts/verify_collaboration.py
python env/tools/validate_submission.py model.onnx
mkdir -p logs
python arena.py model-h1.onnx model-l3.onnx --games 64 --sims 100 --workers 8 --out logs/h1-vs-l3.json
```

On PowerShell create logs with `New-Item -ItemType Directory -Force logs` instead. Comparable evaluation uses the supplied Level 4 scene and **100 referee simulations**. `--games 64` means 64 seeds, both colours, **128 actual games**. Other simulation counts are diagnostics. The validator applies CPU/wall/memory limits; timing varies by host. Root requirements install declared dependency ranges. The supplied exact pins remain in `env/requirements.txt` (run those from `env/` if required).

## Training

Select the appropriate PyTorch CPU/CUDA installation; the optional requirements do not pin a CUDA build.

```bash
python -m pip install -r requirements-train.txt
python selfplay.py --model model.onnx --pool model-h1.onnx codex/tactical/model-tactical-v2.onnx --games 2000 --workers 8 --out data/local-gen02
python train.py --data data/local-gen02 --init checkpoints/gen01.pt --out ckpt/local-gen02.pt --steps 6000
python env/tools/validate_submission.py ckpt/local-gen02.onnx
python arena.py ckpt/local-gen02.onnx model.onnx --games 64 --sims 100 --workers 8 --out logs/local-gen02-gate.json
```

These commands start a new experiment. Omitted datasets/checkpoints are not available for continuation. `loop.py` requires an existing `loop_state.json` containing `best`, `pool`, `latest_ckpt` and `history`; initialize it only after a real checkpoint exists and use relative paths. Data, logs and checkpoints remain ignored until a particular artifact is reviewed explicitly. Promote only using comparable match evidence.

## Incremental synchronization

`sync_allowlist.json` lists exact source/destination files. Import is one-way from CCC-OMNI and never changes it, commits, pushes or reads credentials. Unchanged files are skipped. Conflicting local edits are protected and must be reconciled before import.

```powershell
cd D:\dev\CCC-Colony-One
git pull --ff-only
python scripts/sync_from_ccc.py --source D:\dev\CCC-OMNI --dry-run
python scripts/sync_from_ccc.py --source D:\dev\CCC-OMNI
python scripts/verify_snapshot.py
git diff --stat
git add --all
git commit -m "Refresh reviewed Colony One model, code and evidence"
git push
```

On WSL use `--source /mnt/d/dev/CCC-OMNI`. Coordinate file ownership: remote code edits are not copied back to the canonical workspace. Add new candidate/checkpoint/evidence files through exact allowlist entries after review. An entry with `sha256` pins an immutable candidate: preserve an existing artifact and add a newly versioned entry when its bytes change. `snapshot_once` preserves an already imported frozen source snapshot even as its working source evolves, while checking its stored hash. Files over 20 MiB or paths escaping the roots are rejected. Old artifacts are preserved. Bulk self-play data, arbitrary logs, credentials, `.env`, caches, runtimes and virtual environments are never automatically imported. The small checkpoint/target/corpus files tracked here are exact reviewed exceptions listed in `.gitignore`.

`provenance/snapshot.json` records every imported file's source-relative path, size and SHA-256, plus source infrastructure HEAD and snapshot time. Generated workspace files were ignored in CCC-OMNI, so source HEAD alone cannot reconstruct them. This repository commit and manifest bind the snapshot.

`env/provenance.json` describes the original supplied archive. Its original scene was replaced with the exact Level 4 scene. The current `env/colony_one/scene.json` matches `task/level-4_scene.json` byte-for-byte: SHA-256 `5489fce67399e60fa5d291d249cb4e524dd90705da5221cb9e2f1a690338270f`. The current manifest describes the modified package.

The supplied copyright/license is preserved in `env/LICENSE`, including CCC 2026 permission and restrictions on commercial/event use. No broader license is asserted for the repository or task material.
