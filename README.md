# Colony: One — private Level 4 collaboration

This private repository snapshots the current ONNX submission, baselines, training scripts, supplied referee environment, task PDF/scene and bounded evidence. It is independent of CCC-OMNI infrastructure.

GEN06 is the **current deployed release**. The upload owner immediately verified SHA-256 `123467a930e0598c464ae7694800ad565aeac71680b88dafc5f014bfe7cf50ac`, selected `release/gen06/model.onnx`, clicked DEPLOY and observed explicit platform acceptance at **18:38 Europe/Bucharest on 2026-10-23**. Its size is 1,479,383 bytes with 696 nodes. Native Linux graph, hard caps, Level 4 check and official validator reports PASS. Independent final scores are 335/400 against GEN03 and 186/200 against fixed GEN02; all 600 histories legally replay PASS. Saved Opus gate replay is 297/400 against GEN05. Final GEN05 comparison artifacts (286/400 against GEN03 and 172/200 against GEN02, all 600 replay PASS) remain archived separately. No sustained live rating improvement is claimed.

GEN06 frozen 48-state decisions retain zero certified failures; its 22-site diagnostics reduce loss failures 7→6 and missed wins 7→1, with seven winning follow-throughs checked. This is bounded diagnostic evidence, not complete tactical remediation. GEN06 source-current snapshots were captured later and are not asserted to be the exact historical build source. The override-integration audit names earlier train/learner hashes that differ from this capture; actor matches. The integrity report exposes those differences.

`data/ext_site_deep10910_v1/` releases two independently verified rows: the actual pre40 state has certified value −1 and no policy label; its legal counterfactual has a bounded-horizon safety policy and an UNKNOWN full-game value. Both disable Q supervision. Source-game aliases stay together, frozen holdout collisions are zero, and seven unclassified earlier actions remain UNKNOWN. Proof, packet and loader-audit hashes are preserved. This does not assert the data trained any released generation.

Export (5) is now finalized. Only its stable bounded manifest, summary and provenance/identity receipts are shared here: 111 local complete episodes/3,777 rows plus seven external episode aliases/328 rows, totaling **118 unique episodes/4,105 rows**. Manifest SHA-256 is `50ce9af40cd67e8e7c27addd300dbf3dab00625daff47f37ed42b02169624b39`. Existing external numerical rows were preserved; fresh export identities supersede older inferred user sides. Raw bulk exports and full episode chunks are excluded. Played moves and moves in won games are not automatically policy labels; actual protocol winners determine mover-perspective outcomes. Unqualified synthetic-28 work and unfinished 14958 work are not published as training-ready.

## Artifacts and evidence

- `model.onnx` is the **qualified and deployed GEN06 release**, identical to `release/gen06/model.onnx` and `model-gen06.onnx`: 1,479,383 bytes; SHA-256 `123467a930e0598c464ae7694800ad565aeac71680b88dafc5f014bfe7cf50ac`. The importer explicitly reads this named release. Later internal work does not advance this alias. GEN03 is preserved as the previous deployed incumbent, with SHA-256 `0cf937446ba89dc9b5b3652edbcb4025030bc785e924551782f4b2d298ebe96f`.
- **GEN02 remains the fixed comparison baseline** requested by the user: `release/gen02/model.onnx`, SHA-256 `579d3ec01da7328dc706003b8dcc5eba357bb74902252187e45633ca0595a9a0`. Its checkpoint and exact endgame wrapper provenance are preserved. GEN01 and every earlier named model remain historical artifacts.
- GEN03 model checkpoint and additional recovery artifact are in `checkpoints/gen03.pt` and `checkpoints/gen03.resume.pt`; extracted configuration is in `provenance/gen03-checkpoint.json`. Recovery itself was not exercised by this sync. Current source may have changed after generation; exact build source is UNKNOWN.
- `checkpoints/gen01.pt` preserves the small retraining checkpoint (SHA-256 `710a69d2297cf4422b257af313fdfdf3fa12404d8bea82bf46fa560f39bbb239`). Its configuration/arguments are extracted in `provenance/gen01-checkpoint.json`. This enables a model warm start; full optimizer/scheduler/RNG continuation is not established.
- `model-h1.onnx` remains the 311,345-byte historical heuristic with SHA-256 `0940ddada8c2b761126ffbc5532d3103c3b8cb4eb20d89f7923f70d03f2e8b65`; `build_heuristic.py` is its generator. Named earlier models are preserved.
- `model-l3.onnx` is the earlier Level 3 baseline; `random.onnx` is the supplied random baseline.
- `evidence/comparison-h1.json` records a historical local 84/128 (65.6%) result against Level 3. The underlying game histories were unavailable. Its validator ACCEPTED claim conflicts with the later Windows timing rejection in `evidence/h1-validation-windows.txt`; both are preserved.
- `evidence/gen00-summary.json` records 6,000 generated games and 203,193 positions. Bulk NPZ files are excluded. This does not establish a trained checkpoint.

Historically, Andrei explicitly confirmed his GEN03 upload and the upload owner observed site acceptance at 1.41 MiB. GEN06 acceptance is the newer direct upload-owner observation described above and in `provenance/current-release.json`. The UI exposes no stored SHA; identity relies on immediate file hash verification and the selected upload file. No duplicate upload was performed by this sync. Earlier GEN01 and h1 acceptance remain historical observations. Browser screenshots and account state are excluded.

GEN03 native Linux graph, hard resource limits, Level 4 check and official validator reports PASS in `codex/validation/gen03-*`. The research review legally replayed its saved **500 games**: 296/400 versus fixed GEN02 and 98/100 versus tactical v2. These replay checks do not regenerate historical searches. Frozen 48-state diagnostics found zero certified move-selection failures for either GEN02 or GEN03; the separate 22 actual-site decisions showed three GEN03 improvements and unresolved failures. The 48 constructed variants are correlated and remain **TEST_ONLY**, never training data. Full bounded results and provenance are in `codex/arena/gen03-tactical-audit/` and `codex/research/evaluation-review.md`.

GEN01 won **63/64** in the fresh independent paired arena against tactical v2, using the exact Level 4 scene and 100 simulations; all 64 histories replayed legally. A separate audit replayed the 400 saved Opus histories and verified the reported 388/400 wins against v2. That historical run did not store its original full engine/model manifest; the independent arena has separate provenance. Native Linux graph, hard resource caps and validator evidence are in `codex/validation/gen01-*`; earlier mounted-path timeouts are retained. These checks qualify their measured scopes only.

Tactical v1/v2 and other earlier named models remain available with immutable hash pins. The current release alias is GEN06; comparisons continue to use fixed GEN02, and GEN03 remains archived as the prior incumbent. Never infer deployment from an arena win rate or a local PASS.

`codex/arena/` and tactical arena directories contain bounded manifests, summaries and available harness snapshots. Some runs are partial: inspect `complete` and `expected_games` before interpreting rates. Raw game histories remain in the source workspace; replay claims in imported audit files describe that source evidence. `codex/validation/` preserves exact-model reports including Linux PASS and Windows timing failures. `codex/engine_audit/` and `codex/matches/` contain source and bounded diagnostics. Raw site records, account/browser data and bulk self-play files are excluded.

`codex/distillation/` now includes **32 genuine Astra labels** (7 exact wins, 2 exact losses, 23 predictions), packets, schema, independent engine validation, target arrays, student training source and frozen v1/v2 candidates with learning evidence. Compact v2 is a 22-parameter adapter; teacher-agreement gains did not establish stronger play (121/256 against tactical v2), so it was **not promoted**. Student recovery manifests bind executed source snapshots and artifacts. Operational fixture/diagnostic models are excluded.

The historical teacher-ready reports name validator hash `c74610db…`, while the synchronized current validator has subsequently changed. Its original source snapshot is unavailable. Packet, label and target hashes still match those reports; `provenance/collaboration-integrity.json` records this exact limit. No historical validator-source reproduction is claimed.

`codex/synthetic/` contains completed LLM-authored constructed proposals and bounded certified corpora, separate from real games. The current curriculum manifest reports 224 constructed states across 5 motif groups; full-game values remain unknown where explicitly masked. Opening reachability is unproved; symmetry/family siblings belong in the same split. Per-action certification does not prove eventual victory. Reports retain their precise source hashes; duplicate raw validation snapshots are omitted. These synthetic states and Astra labels are **not claimed to have trained GEN01**.

Larger networks under the 5 MiB budget, incremental comparisons, and analysis of real wins/losses remain ongoing. Losses/hard decisions have priority, while wins also supply evidence; the repository does not claim these ongoing requests are complete.

`release/gen01/source-current/` freezes the neural scripts captured during this sync, each hash-bound in the manifest. Their exact identity at the historical GEN01 generation build is **UNKNOWN**; checkpoint configuration is stronger evidence than reconstructing unrecorded code history. Root scripts track current collaboration code. Do not confuse the frozen current snapshot with a proven historical build source.

`release/gen03/source-current/` similarly freezes the working scripts captured for this later sync. They may already contain subsequent p2-head changes; they are not asserted to have produced GEN03. New reports and helpers in `codex/research/` are a bounded capture while other workers continued. `research_inputs/` preserves the user-shared dossier and CSV index as untrusted source material, not repository instructions or verified bibliography.

Reviewed small external training snapshots are under `data/ext_astra_policy_v1/` (144 policy-only rows), `data/ext_astra_site_gen02_v1/` (55 rows: 15 actual and 40 constructed; 12 exact values, 43 masked unknown values), and `data/ext_site_tactics_v1/` (certificate-guided policy repairs with proof and override bindings). Full-game unknown values are never interpreted as losses/draws. The first 144 rows have no value/Q supervision. Teacher-authored scenarios, engine certificates and observed replay data keep their distinct authority. Source-game/family grouping and frozen-holdout quarantine still apply. Presence here does not claim these new inputs trained GEN03.

The finalized site-tactics snapshot has **19 rows**: six byte-preserved projections of existing teacher policies and 13 **PROOF_ONLY_UNIFORM** targets over certified admissible actions. No new 800-simulation teacher policy is asserted. All 19 disable value/Q/length supervision; zero rows await teacher data and three no-admissible-action states remain excluded. Manifest SHA-256 is `05455b009a522c31d4d194f5d00072edc569fc9886cbe85953f18d6ebc7c6013`; override SHA-256 is `39f2fd439f8bada43812cf19b99737e547697a4f0c8f191e6f5105cbda766276`. Validation covers bindings and dry/in-memory routing; actual trainer integration was not performed by the qualification worker. The coherent earlier six-row dataset is preserved under `training_snapshots/ext_site_tactics_v1-six/`, with its own archive hash receipt.

`codex/endgame/three_piece_certificate.py` and its reports preserve a pure checked helper. Sharing it creates no ONNX model change or deployment claim.

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
