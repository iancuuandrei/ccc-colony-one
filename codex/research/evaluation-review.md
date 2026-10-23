# Evaluation review: fixed GEN02 and a frozen tactical holdout

Reviewed 2026-10-23. This work replayed existing histories and read artifacts and primary publisher sources. It created no models, training jobs, new arena games, or publication. Opus remains the model/training/export owner. Future candidate comparisons use GEN02 as requested by the user; GEN03 is audited below without silently changing that baseline.

## Existing evidence verified

Every saved action was resolved against the current engine's legal-action list and replayed. Recorded team, final winner, round, action count, candidate-win flag, duplicate game identity, and summary win counts were checked.

| Existing artifact | Replayed | Candidate wins | Cyan | Magenta | Exact seed range, both colors |
|---|---:|---:|---:|---:|---|
| `logs/gen02-gate.json` | PASS 400/400 | 310/400 (77.5%) vs GEN01 | 155/200 | 155/200 | 902000–902199 |
| `logs/gen02-tactical.json` | PASS 200/200 | 200/200 (100%) vs tactical v2 | 100/100 | 100/100 | 802000–802099 |
| `logs/gen03-gate.json` | PASS 400/400 | 296/400 (74%) vs GEN02 | 145/200 | 151/200 | 903000–903199 |
| `logs/gen03-tactical.json` | PASS 100/100 | 98/100 (98%) vs tactical v2 | 48/50 | 50/50 | 803000–803049 |

All 1,100 existing histories passed. These are legal-history and artifact-identity checks, not independently regenerated MCTS games or proof of historical machine limits.

Current artifact identities:

| Artifact | SHA256 |
|---|---|
| `release/gen02/model.onnx` and `ckpt/gen02-endgame.onnx` | `579d3ec01da7328dc706003b8dcc5eba357bb74902252187e45633ca0595a9a0` |
| `release/gen03/model.onnx` and `ckpt/gen03-endgame.onnx` | `0cf937446ba89dc9b5b3652edbcb4025030bc785e924551782f4b2d298ebe96f` |
| Frozen tactical v2, `codex/tactical/model-tactical-v2.onnx` | `0a95cc4b1b17c5300667f6630cc9f21c391d0bc0a2eb02547916b47a3caefd76` |
| `ckpt/gen01.onnx` (historical opponent only) | `ed45eae71a9332d941f5f02d9e5a469f8372af56e8d31ccf0d4b5cdbe87f6c7a` |
| Supplied `env/colony_one/scene.json` | `5489fce67399e60fa5d291d249cb4e524dd90705da5221cb9e2f1a690338270f` |
| Current engine bundle | `2e6a926a41230225c7f7c58855b704d95c3c4fee1025211f860a34c766230ecd` |

The engine bundle is SHA256 of UTF-8 `json.dumps(file_hashes, sort_keys=True)`, with Windows-relative paths for every top-level `env/colony_one/*.py` plus `scene.json`; this matches the earlier arena manifest method. Important members: MCTS `1374b55952f08fcbe411307fd96ec62832d2712362b2b649a2307948ffbf355e`, environment `0bef52825302c38d8c84e6ed02b0502ef1295ab50c1543d0bdda8936fc0044f5`, board `a1b23455c22047db17f835c2fad2004754ecf9e5479b17afb1a9bce3281c8f5d`.

Log file SHA256 values:

- GEN02 gate: `80379f89388b6b8c72bf2b425cb2d7c9ea837499f05aa5fa7299be44ac1e4438`.
- GEN02 anchor: `9e4854aad32ef5decacf5364769bc461411039ae53d028de9d58fff7f2224ced`.
- GEN03 gate: `439913e7d85fc5f79de7edcf357a08dbbe7e1b308a2cef563dbb0d7eb4b1d85d`.
- GEN03 anchor: `026e7e10be856b2643a926d2c241f1d7265b6f58c7dcc8c542440bb5ab35dafc`.

Historical logs do not contain a run-time engine hash manifest. Current identity plus legal replay cannot establish that historical hashes were identical. GEN03's text log does bind its wrapper build to the present model hash and records arena commands. The current `arena.py` defaults to 100 simulations; those commands omit `--sims`. Its RNG is `seed*2+int(a_is_cyan)`, so these are color-balanced games with different RNG streams across colors, not the common-RNG protocol of `paired_arena.py`.

GEN03 operational timing evidence: `logs/gen03.txt` records `check_onnx.py` dry run 1.0963 s, batch-8 inference 4.394 ms, and 682 nodes for 1,477,761 bytes. Gate/anchor wall times were 82.38/21.73 s with 24 workers. **CPU hard-budget qualification: NOT ESTABLISHED by these artifacts.** Parallel arena wall time and a dry-run timer are not isolated CPU-time/RSS evidence; the root qualifier owns that check.

## Verified literature and transfer limits

The supplied bibliography and CSV were read at `D:/Downloads/colony_one_research_for_astra_opus.md` and `D:/Downloads/colony_one_research_index.csv` (the actual CSV filename). Three requested topics were verified against publisher records; the metamorphic topic also required its later replication caveat.

1. Glickman and Jones, **Models and Rating Systems for Head-to-Head Competition** (2025): [publisher and DOI](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-040722-061813). The publisher abstract discusses probabilistic strength models, ties, and side/home advantage. It is a statistical review, not a Colony trial. It supports explicitly modeling side effects; it does **not** prescribe our shared seeds, paired bootstrap, or sample size. Those are design choices below. Elo is a model-dependent secondary summary, not evidence that one opponent's win rate transfers to every opponent.

2. Tseng et al., **Can Go AIs Be Adversarially Robust?** (AAAI 2025): [primary proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/34980), [DOI](https://doi.org/10.1609/aaai.v39i26.34980). Its publisher abstract reports that three defenses improved resistance to known attacks but failed against adaptive attacks. Transfer to Colony is a research inference: preserve targeted tactical checks alongside average head-to-head strength. This source does not prove any specific Colony exploit or justify expensive adversary training now.

3. Méndez et al., **Metamorphic Testing of Chess Engines** (2023): [primary publisher](https://www.sciencedirect.com/science/article/pii/S0950584923001179), [DOI](https://doi.org/10.1016/j.infsof.2023.107263). Its indexed publisher abstract describes related-position transformations to expose evaluation inconsistencies when a perfect oracle is unavailable. **Necessary caveat:** Martin et al.'s [2025 replication](https://www.sciencedirect.com/science/article/pii/S0950584925000187), DOI [10.1016/j.infsof.2025.107679](https://doi.org/10.1016/j.infsof.2025.107679), found sensitivity to positions/depth and implementation explanations for discrepancies. Thus a transformed-position score mismatch is a diagnostic, not automatically an engine bug. Publisher-indexed abstracts were available; direct ScienceDirect full-page fetches returned 403, so no uninspected full-text detail is claimed.

## Practical shared gate changes

These are recommendations, not executed new experiments or an acceptance-state workflow.

1. **Freeze the comparison before the next candidate.** Record candidate hash, fixed GEN02 hash above, frozen tactical-v2 hash above, Level 4 scene/engine hashes, exact 100 simulations and referee defaults, runtime version, and explicit seed/RNG scheme. Opus runs the agreed gate; Codex validates artifacts/proofs and replays its durable histories. Root assigns any genuinely useful independent holdout. Do not start duplicate old-generation comparisons.

2. **Use the seed pair as the uncertainty unit.** Play both colors. Common RNG starts across swapped colors preserve each game's marginal distribution but do not guarantee identical openings once branches differ. Report Cyan and Magenta separately, complete-pair score, sweep/split/loss counts, and a seed-resampled interval. Do not analyze `2*n` correlated games as independent Bernoulli trials. Keep descriptive per-color Wilson intervals separate. Existing Opus logs must retain their recorded RNG interpretation; do not relabel them as common-RNG runs.

3. **Confirm a positive candidate, not the least-bad grid member.** Exploratory seeds may rank hypotheses, but winning an eight-variant search does not establish improvement. Freeze one candidate before genuinely fresh confirmation, including fresh seeds and excluded training/relabel positions. Predeclare sample size and a useful effect threshold; do not repeatedly inspect an ordinary fixed-sample CI and stop when it first passes. A sequential rule needs a valid predeclared sequential method. Strong margins may justify small bounded gates; remove the misleading implication that a hardcoded minimum pair count is itself the scientific decision.

4. **Require GEN02 gain plus tactical retention.** Direct GEN02 head-to-head is primary; the frozen tactical opponent and position holdout check regression. Compare candidate and GEN02 against the same anchor seed/position sets and preserve those pairs when estimating their difference. Predeclare a tolerated regression; a nonsignificant difference is not proof of equivalence. An anchor near 100% saturates, so retain per-case legal-oracle tactical results and failures rather than trusting its aggregate rate alone. No claim of universally strongest play follows from this small opponent panel.

5. **Keep a small frozen adversarial holdout.** Reuse distinct legal cases for guarded-kernel redirection, two-action same-turn tactics, firewall timing and simultaneous kernel burns, and acting-team-sensitive bare-kernel outcomes. Record scene, round, actions remaining, legal action mapping, source, and exact expected fact. Cases used as training labels become regression cases; a separate excluded set is needed for unseen generalization. Check each tactical stratum and worst cases alongside overall wins. No new adversary/model pipeline is proposed.

6. **Apply only sound metamorphic relations.** Engine apply/undo identity, legal replay, action encode/decode consistency, no search-state mutation, deterministic same-input rerun, and numerically tolerant batch partition/reorder consistency are useful bounded checks when assigned. Prove any geometric/color transform preserves the full state and rules before requiring equivalent actions or values. Colony's Magenta-only shrink timing and Magenta tiebreak make naive color-swap symmetry unsafe. A neural score asymmetry alone is not a correctness failure unless an exact invariant or output contract requires equality.

Runtime/ONNX qualification stays separate from strength evidence. The root qualifier verifies the exact candidate under the published native Linux resource limits; only then can root combine strength, robustness, and runtime evidence for its publication decision.
