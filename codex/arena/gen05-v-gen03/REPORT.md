# GEN05 shared gate against deployed GEN03

Candidate: `release/gen05/model.onnx`, SHA256 `af45a5f85894e694f7ddab47811c4fb45b9f9536345d78b0f2fa3ef06c99c91e`.

**GEN05 wins286/400 (71.5%) against deployed GEN03, with exactly143/200 wins on each color.** Its separate fixed-GEN02 anchor is172/200 (86%). All600 saved game histories independently replayed legally to their recorded winners. No automatic promotion or upload was performed.

| Gate | Seed pairs | Games /GEN05 wins | Cyan | Magenta | Paired bootstrap95% |
|---|---|---|---|---|---|
|Direct GEN05 vs deployed GEN03|250000–250199,200 pairs|400 /286 (71.5%)|143/200|143/200|67.25–75.75%|
|GEN05 vs fixed GEN02 anchor|251000–251099,100 pairs|200 /172 (86%)|87/100|85/100|81.0–90.5%|

The direct gate has99 GEN05 sweeps,88 split pairs and13 GEN03 sweeps; conservative paired Hoeffding95% interval61.90–81.10%. The anchor has74 GEN05 sweeps,24 splits and2 GEN02 sweeps; conservative interval72.42–99.58%. These are local opponent results, not a guarantee about unknown tournament opponents or proof of universal strength.

GEN03 SHA256: `0cf937446ba89dc9b5b3652edbcb4025030bc785e924551782f4b2d298ebe96f`.
GEN02 SHA256: `579d3ec01da7328dc706003b8dcc5eba357bb74902252187e45633ca0595a9a0`.

## Protocol and provenance

The supplied exact MCTS used100 fixed simulations and all other referee defaults, including the published root noise and first-eight-action temperature schedule. Each seed was played with colors swapped, starting the same NumPy RNG state in each game; the seed pair is the uncertainty unit. Divergent branches may consume different randomness, so openings are not asserted identical. Each ORT evaluator used one CPU inference thread; at most8 search workers ran. The two gates were sequential.

Engine/scene bundle SHA256: `2e6a926a41230225c7f7c58855b704d95c3c4fee1025211f860a34c766230ecd`; Level4 scene SHA256: `5489fce67399e60fa5d291d249cb4e524dd90705da5221cb9e2f1a690338270f`. Each run's manifest pins every engine member, model/runtime/harness identities and settings. Model/engine hashes were checked again at completion and during the independent history audit. Both exact harness sources are preserved as `harness_snapshot.py` beside the manifests.

Each directory contains its complete flushed `games.jsonl`, `manifest.json`, final `summary.json` and `engine-history-audit.json`. Direct games JSONL SHA256 is `aa659fb5c86046056a0419627323bdd2990dedc0bb2dbba3c38291900b4d943a`; anchor games SHA256 is `ea7e2968fd5dc95620456b8280bc2f04650f1e2d14bffb749b84c658c5eeba92`. `replay.py` validates the600 existing histories without creating new games.

## Tactical tradeoff that must accompany the strength result

The independent audit in `../gen05-tactical-audit/REPORT.md` compared both GEN03 and GEN05 on the unchanged frozen48 and all22 current actual-site hard cases. Frozen holdout failures remain0 missed own-turn wins /0 avoidable next-turn losses for both models; exact-value MAE improves0.382681→0.337635. All48 earlier GEN03 choices reproduced exactly, and frozen input hashes were preserved. This cohort remains quarantined test-only.

On site cases GEN05 fixes three avoidable loss decisions:8079/30,9366/52 and9928/28. However, **8623/action33 regresses**: GEN03 selects `E6->E5` and can complete the terminal win with `E5->G7`; GEN05 selects `E6->F5` and discards the available own-turn win. This is not a claim of eventual full-game loss. Site avoidable-loss flags decrease10→7, while missed-win flags increase6→7. The local tactical result is a tradeoff, not complete remediation.

## Runtime and CPU coordination

No tactical or strength evaluation began before `/root/onnx_validation` confirmed native qualification PASS for this exact candidate hash. That owner reported unchanged Linux validator ACCEPTED, hard check CPU0.8024s/wall0.8104s, hard Level3 CPU1.2254s/wall1.2271s, raw batches1/8/16 across24 corpus inputs PASS, and full100-simulation48-action selfmatch CPU1.986/2.014s per side plus0.395s initialization, peak RSS85,532,672 bytes (81.57MiB). The qualification artifacts remain owned by that validator; parallel arena timings are not used as resource qualification.

Arena compute and history replay have ended; no owned GEN05 processes remain. Root was notified that all8 arena slots are free and owns actor restoration and the publication decision. No variants, training, model creation, old-GEN01 comparisons or additional strength gates were started.

