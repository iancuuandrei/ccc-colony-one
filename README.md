# Colony: One — private Level 4 collaboration

This private repository snapshots the current ONNX submission, baselines, training scripts, supplied referee environment, task PDF/scene and bounded evidence. It is independent of CCC-OMNI infrastructure.

## Artifacts and evidence

- `model.onnx` is currently identical to `model-h1.onnx`: 311,345 bytes; SHA-256 `0940ddada8c2b761126ffbc5532d3103c3b8cb4eb20d89f7923f70d03f2e8b65`. `build_heuristic.py` generates this hand-built heuristic.
- `model-l3.onnx` is the earlier Level 3 baseline; `random.onnx` is the supplied random baseline.
- `evidence/comparison-h1.json` records a historical local 84/128 (65.6%) result against Level 3. The underlying game histories were unavailable. Its validator ACCEPTED claim conflicts with the later Windows timing rejection in `evidence/h1-validation-windows.txt`; both are preserved.
- `evidence/gen00-summary.json` records 6,000 generated games and 203,193 positions. Bulk NPZ files are excluded. This does not establish a trained checkpoint.

The upload owner recorded that the site accepted and deployed the exact h1 model at approximately **16:46:40 Europe/Bucharest on 2026-10-23**, replying “Model validated and deployed to the tournament”; see `codex/coordination.md`. This is an operator observation, distinct from local validation. No tournament score improvement is claimed. The misleading screenshot named `h1-deployed.jpg` is excluded.

Tactical v1 and v2 are included as **candidates**, with pinned immutable SHA-256 values in `sync_allowlist.json` and exact hashes in `provenance/snapshot.json`. V2 has local Linux resource/graph evidence, but broader paired evaluation and site deployment remain pending at this snapshot. Never infer deployment from an arena win rate or local PASS.

`codex/arena/` and tactical arena directories contain bounded manifests, summaries and available harness snapshots. Some runs are partial: inspect `complete` and `expected_games` before interpreting rates. Raw game histories remain in the source workspace; replay claims in imported audit files describe that source evidence. `codex/validation/` preserves exact-model reports including Linux PASS and Windows timing failures. `codex/engine_audit/` and `codex/matches/` contain source and bounded diagnostics. Raw site records, account/browser data and bulk self-play files are excluded.

Astra distillation is active in the source workspace. No teacher labels, student model or operational fixture are published until the completed set has explicit model/label/schema provenance and validation. Main neural training is also active; no current trained checkpoint is included. A complete training/loop run was not qualified by repository publication.

## Setup and evaluation

Python 3.12 on Linux/WSL2 is preferred for heavy computation. The supplied package declares Python 3.11–3.14. Run from the repository root:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/verify_snapshot.py
python env/tools/validate_submission.py model.onnx
mkdir -p logs
python arena.py model-h1.onnx model-l3.onnx --games 64 --sims 100 --workers 8 --out logs/h1-vs-l3.json
```

On PowerShell create logs with `New-Item -ItemType Directory -Force logs` instead. Comparable evaluation uses the supplied Level 4 scene and **100 referee simulations**. `--games 64` means 64 seeds, both colours, **128 actual games**. Other simulation counts are diagnostics. The validator applies CPU/wall/memory limits; timing varies by host. Root requirements install declared dependency ranges. The supplied exact pins remain in `env/requirements.txt` (run those from `env/` if required).

## Training

Select the appropriate PyTorch CPU/CUDA installation; the optional requirements do not pin a CUDA build.

```bash
python -m pip install -r requirements-train.txt
python selfplay.py --model model-h1.onnx --pool model-l3.onnx random.onnx --games 2000 --workers 8 --out data/gen01
python train.py --data data/gen01 --out ckpt/gen01.pt --steps 6000
python env/tools/validate_submission.py ckpt/gen01.onnx
python arena.py ckpt/gen01.onnx model-h1.onnx --games 64 --sims 100 --workers 8 --out logs/gen01-gate.json
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

On WSL use `--source /mnt/d/dev/CCC-OMNI`. Coordinate file ownership: remote code edits are not copied back to the canonical workspace. Add new candidate/checkpoint/evidence files through exact allowlist entries after review. An entry with `sha256` pins an immutable candidate: preserve an existing artifact and add a newly versioned entry when its bytes change. Files over 20 MiB or paths escaping the roots are rejected. Old artifacts are preserved. Bulk self-play data, arbitrary logs, credentials, `.env`, caches, runtimes and virtual environments are never automatically imported.

`provenance/snapshot.json` records every imported file's source-relative path, size and SHA-256, plus source infrastructure HEAD and snapshot time. Generated workspace files were ignored in CCC-OMNI, so source HEAD alone cannot reconstruct them. This repository commit and manifest bind the snapshot.

`env/provenance.json` describes the original supplied archive. Its original scene was replaced with the exact Level 4 scene. The current `env/colony_one/scene.json` matches `task/level-4_scene.json` byte-for-byte: SHA-256 `5489fce67399e60fa5d291d249cb4e524dd90705da5221cb9e2f1a690338270f`. The current manifest describes the modified package.

The supplied copyright/license is preserved in `env/LICENSE`, including CCC 2026 permission and restrictions on commercial/event use. No broader license is asserted for the repository or task material.
