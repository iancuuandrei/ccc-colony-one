# Quarantined deep-loss packet: site 10910

Status: **PENDING INDEPENDENT VERIFICATION; NO ext_* RELEASE BY THIS OWNER.**
Files `packet.json`, `training_packet.npz`, and `certificates.json` are a single
hash-bound packet. The source is the actual match-10910 replay, plus the
explicit one-step counterfactual Kernel F6→F7. `build_packet.py` constructed the
packet using engine replay and bounded exact oracles only. No model, MCTS,
training, or export was run.

| Row | Position | Policy evidence | Value |
|---|---|---|---|
| A | Actual pre-action39, MAGENTA Kernel F6 | `PROVED_SAFE_SUBSET`: first2738 (F6→F7) has demonstrated continuation3164; actual2729 (F6→G6) is proved unsafe; other7 first actions UNKNOWN | UNKNOWN |
| B | Actual pre-action40 after Kernel F6→G6 | All6 legal completions admit forced terminal defeat by second following CYAN turn; no preferred policy action | Exactly -1 |
| C | Counterfactual pre-action40 after Kernel F6→F7 | Complete6-action classification: sole horizon escape3164 (Trace G7→D4 SHOOT), other5 proved unsafe | UNKNOWN |

The scope is `TERMINAL_LOSS_THROUGH_SECOND_FOLLOWING_OPPONENT_TURN`.
Horizon escape is not an eventual win. The packet does not reuse a
one-opponent-turn certificate label for this longer statement.

`demonstrated_safe_mask`, `proved_unsafe_mask`, and `unknown_action_mask`
partition the complete legal set. `classification_complete=False` on A is
mandatory. There is deliberately no generic `safe_mask` that could conceal
the partial classification.

A's ordinary `has_pol=False` and zero `pval` prevent the present full-legal KL
loader from treating all UNKNOWN actions as false targets. The packet stores
only the explicit preference pair `2738 > 2729` using separate pairwise fields.
A requires intentional partial-supervision support, or remains metadata-only.
It must not be silently converted into a one-hot/full-safe policy target.

B has `has_value=True, z=-1, has_pol=False`. Its positive `left=5` is the
actual remaining recorded episode length, not a fabricated minimax duration.
C has `has_pol=True`, a normalized one-hot over the complete legal `pidx`
support, and `has_value=False`. A and C have finite zero value placeholders;
all rows have `q_valid=False`. C's `left=1` is a finite placeholder and its
length auxiliary is masked together with its unknown value. No counterfactual
observed outcome is invented: C has `has_observed_outcome=False`.

All rows share `source_group=site:10910`. B overlaps an actual episode already
eligible for outcome data; this is additional certificate information, not a
new independent match. Keep all three rows and any symmetry/derived copies in
one training/holdout group, and make any duplicate weighting intentional.

The packet records 890,424 exact endpoint-oracle nodes and 441 cached endpoint
positions across B/C. The successful C action covers all 239 CYAN first-turn
completions without UNKNOWN endpoints. Detailed branches and endpoint oracle
maps are preserved in `certificates.json` for independent replay.

Local operational validation passed hashes, finite numeric arrays, complete
legal `pidx`, disjoint/exhaustive class masks, policy mass `[0,0,1]`, explicit
partial-policy disabling, masked unknown values, and common source grouping.
These checks do not replace the independent Astra/training-audit proof review.

Packet hashes:

- `training_packet.npz`: `f7e8d935be6f1e4ee5adfb242ccc7e36957b046530f7b8ab6a9b9112a4cc466d`.
- `certificates.json`: `d57155ba7fec0e4c60af4dd1331e0f6428844502ac681a602a9e53479fe8ae95`.

The original full analysis and reproduction helper are in the parent research
directory: `loss-10910-review.md` and `loss_10910_review.py`. Do not rerun the
builder over these frozen packet artifacts while independent review is active.
