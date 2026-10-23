# Three-row certified proposal: STAGED, NOT RELEASED

The reviewable chunk is `ext_deep19490_proposal/chunk0000.npz`, SHA256 **42e9d162b2e4105d1f8e5bec262ce62e7813d1d0a57cd506666191ff6c349bc0**. `packet.json` binds the full independent proof, source-equivalence receipt, exact observations, legal actions, and the current loader revision. No live `data/ext_*` folder, policy override, model, or training process was modified.

| Position | Full legal count | Policy / pval_g | Value | Q | Length |
|---|---:|---|---|---|---|
| Before 68 | 5 | One-hot Kernel E6→F5, index 2786; `has_pol=True` | UNKNOWN, `z=0`, `has_value=False` | `q=0`, `q_valid=False` | UNKNOWN, finite `left=1`, `has_length=False` |
| Before 71 | 7 | Zero targets, `has_pol=False` | Exact −1, `has_value=True` | `q=0`, `q_valid=False` | Observed 4 actions, `has_length=True` |
| Before 72 | 5 | Zero targets, `has_pol=False` | Exact −1, `has_value=True` | `q=0`, `q_valid=False` | Observed 3 actions, `has_length=True` |

All legal actions are retained in padded `pidx`, including zero-target alternatives. `pval_g` equals `pval`, so policy mixing cannot restore probability to the certified losing choices. Pre68's legal set is fully classified through two future CYAN turns; 2786 is its unique bounded-safe action, surviving all 59 complete next-CYAN turns. This does **not** prove eventual victory or a game length. At 71/72 every legal continuation loses by the next CYAN turn, so the proposal supplies value without inventing a policy preference. Lengths 4/3 are `74-action+1` from the two identical recorded endings, rather than optimal mate-distance claims.

`has_length` is explicit. Executing the current pure `train.load` definitions from its parsed source preserved every training key exactly, classified the staged `ext_` subfolder as external, and produced no next-policy auxiliary labels. Policy mixtures 0, 0.5, and 1 retained the same pre68 target. This loader check does not train or import the staged data into a live dataset.

## Duplicate ending and existing outcomes

The independent equivalence receipt proves full deterministic state, bit-exact observation, and legal-index equality for 19101/19490 at actions 63, 65, 67, 68, 71, and 72. There are **two source games but one repeated ending**. The proposal contains three unique states, each with both source aliases and `source_group=site-ending-19101-19490`. Link `site-game-19101`, `site-game-19490`, and this derived group for splitting and sampling. Do not put aliases in separate train/holdout sets or count the proof twice.

19101 already has complete 74-row outcome coverage in `data/ext_site_v1/chunk-site-19101.npz`. The proposal observations exactly match its zero-based rows 67, 70, and 71. These rows are intentional certified emphasis on already covered observations, not new independent outcome data. No 19490 outcome episode was imported. A bounded-safe move also does not invalidate the historical −1 outcome after the actual losing move.

The staged audit scanned the current live external NPZ files for these three exact float16 observations. It found only those three raw 19101 rows; **no matching approximate teacher row was found in this snapshot**. The scan results and source hashes are recorded in `packet.json`. Future regenerated teacher data can introduce overlap, so this check must be repeated against the actual next training input list.

## Minimal integration recommendation to Opus

1. Keep this packet staged until the owner resolves observation-level target precedence across the complete training input list. At pre68, an approximate teacher policy must not retain mass on the four proven losing actions, including Trace F6→F5, which survives only the shorter one-enemy-turn horizon. Replace that row's policy and `pval_g` with the deep certificate or suppress its policy supervision; do not average contradictory labels. Its policy-only replacement must also mask approximate value, Q, and unknown length supervision. Exact-value-only rows 71/72 need `has_pol=False`, `has_value=True`, `z=-1`, and no arbitrary repaired policy.
2. Coordinate the representation with `onnx_validation`, which owns the live tactics dataset and override manifest. Current `train.apply_overrides` unconditionally sets `has_pol=True`, `has_value=False`, and `q_valid=False`; it cannot express these value-only rows, and it does not set `has_length`. Repeated observation hashes from different override sources also overwrite by iteration order rather than certificate horizon. A single resolved manifest should carry explicit per-head masks/targets, proof scope, precedence, and linked-source identity, or the owner should exclude conflicting teacher rows and load this separate certified chunk. **Do not silently merge this two-turn packet into the existing one-turn policy-only manifest.**

The chunk's explicit masks work with the current loader as a separate external source; the remaining integration concern is how all sources sharing the same observation are reconciled. Opus retains sole training/model/export ownership. No release decision is implied by this proposal.

Reproduce the staged artifact with `python codex/synthetic/validation/deep19490-stage/build_stage.py` from the level directory. The operation reads frozen proofs and existing data, then writes only inside this staging directory.
