# Distillation review for the single Opus pipeline

Reviewed 2026-10-23. Scope: five primary papers, the supplied dossier's distillation entries, and read-only inspection of `train.py` / `research_ext.py`. No model, search experiment, training, export, or Opus source edit was performed. Model ownership remains Opus; comparisons use the authoritative GEN02-endgame baseline, SHA256 `579d3ec01da7328dc706003b8dcc5eba357bb74902252187e45633ca0595a9a0`.

**Recommendation:** retain the existing legal-action policy KL and outcome/value masks. Integrate independently proved tactical action sets as corrections to existing teacher targets. A separate student architecture or generic high-temperature KD does not address the identified target-quality issue. An optional set-mass objective can avoid imposing arbitrary preferences among equally certified actions.

## Five verified primary sources

| Source | Verified claim and boundary | Applicable decision |
|---|---|---|
| [Rusu et al., Policy Distillation, 2015/2016, §2 and §4.2](https://arxiv.org/html/1511.06295v2) | Compares chosen-action NLL, Q-vector MSE, and KL from temperature-transformed teacher Q values. Its Atari experiment favors **sharpening**, with teacher temperature 0.01; the student softmax in its displayed KL is untempered. This is not a universal recommendation to soften targets or copy that temperature to visit counts. | Existing visit-distribution KL already transfers more information than an argmax action. Do not apply the paper's Q temperature directly to `pval`, which is normalized visit counts rather than raw Q. |
| [Ruoss et al., ChessBench, 2024, §2.1–2.3 and Appendix B.5](https://arxiv.org/html/2402.04494v2) | Uses Stockfish 16 annotations for state values, **all legal action values**, and best-action cloning. Appendix B.5 explicitly controls training-data count: the action-value advantage over state-value prediction largely disappears, while both outperform best-action cloning in their comparison. It does not establish that adding an action-value head is automatically better under equal compute. | Preserve full legal action annotations and retain uncertainty/provenance. Extract the data-target lesson, without introducing a large transformer, new action-value inference API, or a claimed architecture win. |
| [Anthony et al., Expert Iteration, 2017, §3.3 and §4.2](https://arxiv.org/html/1705.08439v4) | Defines tree-policy targets as normalized root visit counts, versus chosen-action one-hot targets; discusses cost sensitivity, dataset aggregation, and collecting states closer to the apprentice's visited distribution. Its experiments are Hex, not Colony. | `research_ext.py` plus the current policy KL already implements the central tree-policy-target idea. Re-search actual GEN02 site failures and retain distributions; this is incremental improvement to the existing pipeline. |
| [Chang et al., Learning to Search Better Than Your Teacher, ICML 2015](https://proceedings.mlr.press/v37/changb15.html), [paper](https://proceedings.mlr.press/v37/changb15.pdf) | LOLS offers local-deviation guarantees in its structured-prediction setting using learned-policy roll-ins and mixed rollouts. It is not a guarantee that an imitator of a fixed poor teacher, or an 800-simulation MCTS, improves game performance. | The transferable idea is to evaluate alternatives on states reached by the learner. Exact alternative-action certificates can legitimately contradict the current teacher; simple imitation of its failure cannot fix the failure. |
| [Zhao et al., Decoupled Knowledge Distillation, CVPR 2022, §3](https://arxiv.org/html/2203.08679v2), [CVPR entry](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html) | Decomposes single-target-class KL into target-versus-rest KL plus conditional non-target KL, weighted in ordinary KD by `1 - teacher_target_probability`; DKD separates those weights. Evaluations concern classification/detection, not strategic games with several equally good actions. | A certified action **set** is a better analogue here than an arbitrary single target. A set-mass/conditional-ranking split below is our adaptation, not a result demonstrated by the DKD paper. |

These five references resolve to the advertised works; Expert Iteration is in the dossier's foundations section. The CVPR HTML page initially returned HTTP 403 to direct opening; its primary search entry and the authors' arXiv full text supplied the metadata and equations. The remaining distillation entries in the supplied dossier were inspected for scope but were not independently verified in this bounded review.

## What is already implemented

At inspected `train.py:103–128`, policy training gathers the represented `pidx` entries, masks padding, and minimizes

\[
L_\pi=\operatorname{mean}_{\mathrm{has\_pol}}
  D_{KL}(\pi_{\mathrm{target}}\Vert\pi_\theta),\qquad
L=L_\pi+1.5L_v+0.25L_{aux}.
\]

Subtracting teacher entropy in `ce - ent` changes the reported KL but not the student gradient. This is already distributional policy distillation. The softmax support is the supplied `pidx` set, so data validation must ensure it includes all legal actions, including zero-target actions; padding is not a substitute for a complete legal mask.

`research_ext.py:31–42` uses no root noise, an 800-simulation default, normalized root visits, and root-average `W/N` as Q. `visit_distribution` returns root actions sorted by index; `idx[:K]` truncates to 128 **by action index**, not by teacher quality. Assert that no intended legal support is lost and recheck probability normalization after any truncation. The present small branching factor makes this a guardrail, not an observed defect in this review.

`train.py:113–116` uses `(1-qmix*q_valid)*z + qmix*q_valid*q` for value targets, gated by `has_value`; the length auxiliary is also gated by `has_value`. **The Q auxiliary is independently gated only by `q_valid`.** `research_ext.py:65–66` emits `has_value=False` but `q_valid=True`: these purported policy-only rows therefore still train the shared trunk through the auxiliary Q regression. This does not duplicate terminal outcome labels, but it does distill an uncertified search estimate. Truly policy-only repaired/UNKNOWN rows need both flags false, unless Opus deliberately retains that Q signal as a separately identified experiment.

The existing fixed `--ext-frac` gives all external rows a shared batch fraction; it does not balance each external source, game, seed, or motif family. `--holdout-last` excludes the last chunk of the last directory, not every derived copy of its games. A site's outcome rows and re-search rows must share one split identity; synthetic D4/paired siblings must remain within their motif/seed group. These are data/sampling choices, not new networks.

## Why exact labels can beat the current teacher

Finite search is an estimator with learned priors and leaf values. Eight hundred simulations is a budget, not a correctness certificate. More confident visits can reinforce a systematically missed guard/shot/two-action tactic. The comment in `mcts.py:684–688` describing visits as a strictly better policy is an algorithmic aspiration, not a theorem applicable to each finite search result.

Current local evidence already certifies avoidable GEN02 100-simulation failures at site games 7525, 8079, 8185 and 8443; `codex/matches/site-gen02-v1-report.md` records their witnesses and alternatives. **This review did not measure whether the corresponding 800-simulation target still chooses those failures.** That is being checked by the independent policy-repair owner from existing targets, without duplicate search.

The superiority claim is limited and precise: an engine-certified current-turn winning action is better than an action certified to permit a forced terminal loss. When two actions merely survive the next opponent turn, their eventual ranking remains unknown. A state with at least one such survival is not automatically `z=+1`; an actual recorded loss is an outcome under played policies, not proof of a forced loss. Certificates must include actions remaining, side, firewall clock, and the actual turn/terminal ordering. Never negate a backed-up value between two actions of the same team.

## Two minimal integration proposals

### 1. Repair existing teacher targets before the existing KL — first priority

For each complete legal action set `A`, obtain independently verified winning-first actions `W`, safe-continuation actions `S`, and proven unsafe actions `U`, with UNKNOWN kept distinct. Choose `C=W` when `W` is nonempty; otherwise use the certified safety set only when the safety classification is complete enough to justify that support. Do not silently label `A\U` as certified safe when some actions remain UNKNOWN.

For an existing teacher distribution `q` with positive mass on `C`, use

\[
p^*(a)=\frac{q(a)\mathbf1[a\in C]}{\sum_{b\in C}q(b)}.
\]

This preserves the teacher's relative preferences inside `C` and is the minimizer of `KL(p||q)` among distributions supported on `C`. It is a mathematical projection of the target; the unchanged trainer still minimizes `KL(p*||pi_theta)`, which is a different KL direction. If `q(C)=0`, record an explicit uniform-on-`C` or epsilon-floor fallback: there is no finite `KL(p||q)` solution supported there. When every action is proved unsafe, there is no safe action target to fabricate.

Keep all legal `pidx`, setting excluded target probabilities to zero. Use `has_value=False, q_valid=False` for UNKNOWN values; exact ±1 certificates may set `has_value=True, z=±1, q_valid=False`. Preserve real-game outcome rows separately. Publish source/model/engine/certificate hashes and source-game/motif identities. This requires no inference graph change and is already assigned to `onnx_validation`'s `certified_policy_repair.py` / `ext_site_tactics_v1`; this review does not duplicate that implementation.

### 2. Optional set-mass objective if uniform targets suppress useful preferences

For a nonempty certified acceptable set `C`, define `m_theta=sum_C pi_theta(a)` over the **full legal softmax**. A minimal alternative is

\[
L_C=-\log m_\theta,
\qquad
L_{rank}=D_{KL}\left(q(\cdot\mid C)\Vert
                     \pi_\theta(\cdot\mid C)\right).
\]

`L_C` rewards allocating probability to any certified acceptable action; it does not force uniformity among them. Add a small, separately weighted `L_rank` only when the teacher's within-set distinctions are worth retaining. This is a set-based adaptation inspired by the DKD decomposition, not literal DKD and not a proven Colony improvement. The algebraic conditional-KL decomposition is valid; the useful coefficients require an ablation in Opus's single training pipeline. Use proposal 1 first because it works with the current loss and schema.

Measure exact unsafe policy mass, certified-winning-set recall, value error only on appropriate outcome/proof labels, and independently held-out game/motif performance. Teacher KL alone cannot establish improved strength. Final candidate comparisons remain GEN02 at the actual 100-simulation referee setting; no comparison or training was executed for this review.

## Inspected-input bindings

- Supplied dossier `D:\Downloads\colony_one_research_for_astra_opus.md`: SHA256 `21f71ba01290fe5caec56ec519dca92a4af2af4f6a5555676f7c5fc1610f2413`.
- Supplied index `D:\Downloads\colony_one_research_index.csv`: SHA256 `ee4b342ef3df0d882f5d619e84c90158e57ace1bb18d5a7c8f5336076e98e965`.
- Inspected `train.py`: SHA256 `81ea60002338705f743dc82de4f18c320d9d36bc5d9bb06c4be7ff949b87bd66`.
- Inspected `research_ext.py`: SHA256 `27439648ec2dfb85433e43f8312454b011e28a3d1cc372bc212b6e427b53686a`.

Line references bind to those inspected bytes. Opus may subsequently change its owned files; these findings are a dated review, not a claim about later revisions.
