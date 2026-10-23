# Research integration driven by observed losses

2026-10-23. One learner and model/export owner: Opus. Codex supplies independently checked data, mathematical certificates, research and evaluation. GEN02 remains the fixed comparison baseline. GEN03 is deployed, confirmed by Andrei and the site's acceptance message.

## What the research changes now

The immediate bottleneck is incorrect policy supervision on observed failures. Five existing GEN02 800-simulation teacher distributions assign 56–96% probability to actions with exact next-turn losing witnesses. A sixth assigns 85.76% outside an exact current-turn winning set. More simulations did not repair these examples. This is measured in `data/ext_site_tactics_v1`, not an inference from a paper.

1. **Repair teacher targets using exact game certificates.** The current policy-distillation / Expert Iteration machinery already exists. We add verified action sets and replace contradictory approximate targets on matching observations. `policy-overrides.json` binds original source rows, observation hashes, proof records and corrected distributions. Six corrections are released; further proved positions await the existing teacher's incremental output. The loader integration is Opus-owned and must be verified separately from the successful dry application.
2. **Use certified tactical curricula with correct masks.** Released corpora include 80 exact winning states, 144 bounded-safety policy-only states, and 55 states derived from observed mistakes and adversarial perturbations. In the last corpus, 43 full-game values remain unknown and are masked; 12 exact terminal outcomes are separately proved. All legal actions stay in the policy denominator, even when their target probability is zero. Counterfactual safety is not labeled as a full-game win.
3. **Give informative cases bounded sampling priority.** Retain broad uniform coverage and source-game groups. A small, capped increase for certificate-conflicting positions within the existing external-data share is preferable to repeatedly copying the same six failures. The precise multiplier is an engineering choice to evaluate, not a theorem from the literature.
4. **Add a narrow exact three-piece certificate through Opus.** Some guard redirects prevent the fixed search from caching/deepening a child. A bare Kernel that can capture the final opposing non-Kernel and win the subsequent bare-Kernel adjudication needs an exact leaf value. A sufficient-rule helper and engine proof are being built; no model change is qualified merely from the proposed formula.

## Mathematical target

For legal action set L, let A be the independently certified winning-first-action set, or otherwise the set of actions admitting a continuation that avoids an opponent terminal win on the next turn. Given teacher probabilities q, use

`p*(a) = q(a) / sum_{b in A} q(b)` for a in A, and zero otherwise.

When the denominator is positive this is the minimum-change projection minimizing `KL(p || q)` subject to support restricted to A. A zero denominator requires an explicitly recorded fallback; no fallback was required in the first six cases. Empty A is not an arbitrary training target: skip the forced-losing policy row and inspect earlier recoverable positions. This proof is about the projection, not eventual game strength.

For two consecutive own actions, a safe first move requires an **existing safe own completion**; declaring it unsafe requires **every own completion** to admit an opponent terminal win. Value perspective changes only when the actual team changes. The original real-game result remains a valid result of the observed trajectory, but is not evidence that a counterfactual continuation wins.

The policy-only rows use finite `z=q=0,left=1`, `has_value=False,q_valid=False`. Both flags matter: the Q auxiliary is independent of the value mask. The actual loader and CPU loss-gradient checks passed before release.

## Source grounding and transfer limits

- [Expert Iteration](https://arxiv.org/abs/1705.08439) supplies the search-to-policy iteration already implemented. [Policy Distillation](https://arxiv.org/abs/1511.06295) supports distributional targets, but its teacher-temperature choices do not transfer mechanically to normalized visit counts. The new contribution here is exact correction of a failing teacher on our game states.
- [MuZero Reanalyse](https://arxiv.org/abs/2104.06294) motivates refreshing targets on stored observations. Our environment is already exact, so a learned transition model is unnecessary. Refreshed targets remain estimates until checked.
- [KataGo methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#policy-surprise-weighting) and [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) motivate informative sampling with retained coverage. Their objectives and importance-weight conventions differ; we explicitly intend a bounded tactical emphasis rather than claim an unbiased uniform objective.
- [AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision](https://arxiv.org/html/2607.08984v1), a 2026 preprint, adds cached, masked oracle policy supervision while retaining the learning/search loop. Its exact optimal-move oracle is stronger than our bounded safety oracle. Our policy-support projection is an adaptation, not the paper's identical loss or a guaranteed Colony improvement.
- [Go-Exploit](https://arxiv.org/abs/2302.12359) motivates restarting from archived states. Opus has implemented hard-state restarts; independent reconstruction, terminal-state and validation-group checks are required. New continuation outcomes are distinct from simply repeating an old result.
- [Engineering Efficient Self-Play Chess](https://arxiv.org/html/2609.37447v1), a 2026 preprint, reports an integrated recipe and explicitly limits attribution to individual mechanisms. Its capped policy-surprise mixture and target-quality failures are useful cautions. Its Elo, hardware and parameter counts are not evidence for this contest.

The supplied bibliography contains 106 entries. We inspected the dossier and index and independently verified a focused subset through primary sources; we did not claim to have fully reviewed 106 papers. Detailed reviews are `search-replay-review.md`, `distillation-review.md`, and `evaluation-review.md`.

## Evidence and remaining scope

The fourth export brings the shared corpus to 89 completely replayed games / 2,947 positions, including 30 losses. The latest three losses include two immediately avoidable mistakes and a longer-horizon failure whose last turn was already lost. Independent reviewers inspect earlier transitions rather than inventing a winning label for an impossible final defense.

GEN03 passed native Linux contract/resource qualification and all 500 existing gate/anchor replays. It scored 296/400 against GEN02. On the frozen 48-state test cohort it introduces no certified action regression; on 22 actual hard cases it fixes three decisions but still has ten avoidable losses and six missed immediate wins. These figures describe this finite cohort, not overall correctness.

Every change still needs the shared next-generation model, exact artifact validation, fixed-referee matches against GEN02/current incumbent, and tactical regression checks. Data release, trainer integration, candidate strength, and website deployment are distinct observable results.
