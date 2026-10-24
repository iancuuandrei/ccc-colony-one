# Research verification: search control, sample weighting, value targets, from-to heads

Date: 2026-10-23. Web-only verification. Evidence labels used below:

- **[paper]**: stated in the source (numbers quoted as given).
- **[code]**: read in KataGo source or config files.
- **[inference]**: my extrapolation to our setup. Not claimed by any source.

How I read the sources: Go-Exploit (2302.12359) and Uncertainty-Guided Exploration (NeurIPS 2025) were read as raw PDF text (pdftotext), so confidence is high. RGSC (2602.20809), AZAL (2607.08984), the chess-engineering paper (2609.37447), Chessformer (2605.19091), Monroe & Chalmers (2409.12272) and the KataGo doc and code were read through WebFetch, which summarises pages with a small model. Their numbers are probably right, but check them against the PDF before relying on any single one. Nothing was entirely unopenable. The specific gaps are listed per source.

---

## 0. Our setup in numbers [inference, from the brief]

- New data per cycle: 160 pos/s x 360 s = **~57.6k positions**. Window 600k, so turnover takes **~10.4 cycles (~62 min)**. About 20 cycles remain, which adds ~1.15M positions in total.
- **Consequence:** any change on the self-play side (restarts, branching) touches only ~10% of the training window after one cycle. **It cannot be judged by one gate.** Only training-side changes, which act on the whole 600k window at once, can be tested in a single cycle.
- **Gate power:** 400 games, SE about 0.025 (binomial upper bound; pairing lowers it somewhat).
  - True +20 Elo (p=0.529): passes the 55% gate about **20%** of the time.
  - +35 Elo (p=0.55): about 50%.
  - +60 Elo (p=0.585): about 92%.
  - A null change (p=0.50): about 2.3%.
  - So the gate mostly rejects real but small gains. The chess paper's final gate is a non-regression gate: the candidate must score **>=0.48 in two consecutive paired matches** [paper]. That paper's earlier loss-based gate promoted a net "approximately 270 Elo weaker" [paper].
- **Playout-cap work split** under KataGo semantics, if p=0.3 is the full-search (250) probability: fast moves use 0.7x100 / (0.3x250 + 0.7x100) = **48% of search work** and produce no policy target. If p=0.3 is instead the fast probability, the share is 15%. The chess paper dropped its fast/full scheme (150 visits on most moves, 600 on a quarter) because "cheap moves consumed 42.9% of nominal search work" and batch slots went unused [paper]. It reports no Elo comparison for that decision.

---

## 1. RGSC: Regret-Guided Search Control (arXiv 2602.20809)

Status: arXiv preprint (Feb 24, 2026). The arXiv comments field says "Accepted by ICLR 2026"; I did not check this on OpenReview.

- **Regret** [paper, Eq. 2]: R(s_t) = 1/(T-t) * sum_{i=t..T} (V_sel(s_i) - z)^2. V_sel is "the MCTS value of the selected action" at s_i, and z is the game outcome. R is computed only after the game ends. The sign convention for z is unspecified. [inference] Use z from the side-to-move perspective at each s_i.
- **Regret network** [paper]: two extra heads on the AlphaZero backbone.
  - A ranking head gives a score gamma; restart probabilities are softmax(gamma); loss L_rank = -log sum rho(s|S) exp(R(s)).
  - A value head estimates regret for MCTS-tree nodes that are off the trajectory. Its training target is not stated.
- **Candidates** [paper]: all trajectory states and all MCTS tree nodes of each game are scored, and the single top-ranked state becomes that game's candidate.
- **Prioritised Regret Buffer** [paper]:
  - Size kappa = **100**; 500 in the well-trained setting. Sizes 100, 500 and 1000 showed "no significant difference".
  - Each worker keeps its own buffer.
  - A candidate enters a full buffer only if its regret exceeds the lowest stored regret.
  - Sampling P(s) proportional to R(s)^(1/tau) with **tau = 0.1**. Tested 0.1, 0.5 and 1; 0.1 was best.
  - When a stored state is replayed, its regret is updated by EMA with **alpha = 0.5**. Tested 0.1, 0.5 and 1.
- **Restart fraction** [paper]: **lambda = 0.5**. Each self-play game starts from a buffer state with probability 0.5, otherwise from the initial state. Tested 0.2, 0.5 and 0.9; 0.9 was unstable on Hex.
- **Settings** [paper]: 200 simulations (400 for the well-trained run); 3 blocks x 256 filters from scratch; 160k states per iteration on Go (120k otherwise); 300 iterations; batch 1024; SGD at lr 0.02.
- **Results** [paper], Elo vs AlphaZero / vs Go-Exploit at the final checkpoint:
  - 9x9 Go +76 / +96; 10x10 Othello +70 / +50; 11x11 Hex +84 / +122. Averages: "77" and "89".
  - Well-trained 9x9 Go, 40 iterations: +42 vs AlphaZero. Win rate vs KataGo went 69.3% -> 78.2%; AlphaZero reached 70.2% and Go-Exploit 69.2%.
  - Compute: about 150 A6000 GPU-hours per from-scratch run.
  - Overhead: 1.25x per iteration with the 3-block net, about 1.03x with the 15-block net.
- **No speedup figure** (iterations needed to reach baseline strength) is reported, and the ablations are plotted only, with no Elo numbers.
- [inference] Implied by the numbers above: in RGSC's own setup, Go-Exploit came out about 20 Elo below AlphaZero on Go, +20 on Othello and -38 on Hex. Go-Exploit's gains did not replicate there.
- [inference] A "lite" version for us skips the regret network: trajectory candidates only, using computed R. The paper mentions such a variant ("w/o regret ranking network") only in an appendix on opening length, with no Elo, so its effectiveness is **unverified**.

## 2. Go-Exploit (arXiv 2302.12359)

Status: arXiv preprint; the comments field says accepted to AAMAS 2023. Read in full from the PDF.

- **Archive contents** [paper]:
  - Visited States (GEV*): the non-terminal states visited in self-play, added after each game.
  - Search States (GES*): all non-terminal search-tree states from separate archive actors that always play full games from s0.
- **Archive structures** [paper]:
  - Expanding: every state is kept.
  - Circular: the most recent N states.
  - Reservoir sampling.
- **Selection** [paper]: always **uniform** over the archive. Duplicates make it frequency-weighted.
- **Restart probability** [paper, Tables 6 and 8]: lambda here is the probability of starting from the *initial* state, swept over {0, 0.01, 0.1, 0.25}.
  - Best on 9x9 Go: **lambda = 0.1 for all variants**, so **90% of games restart from the archive**.
  - Best on Connect Four: 0.1, 0.1, 0.0 and 0.01 for GEVE, GEVC, GESR and GESC respectively.
- **Archive size** [paper]:
  - Go: GEVC 10^6, GESR and GESC 2x10^6.
  - Connect Four: GEVC and GESR 10^6, GESC 10^5.
- **Exploration** [paper]: the number of temperature-sampled opening moves m drops; on Go, AlphaZero used 2 and the Go-Exploit variants mostly used 1. Dirichlet noise was otherwise similar.
- **Settings** [paper]: 100 simulations (Connect Four) and 400 (Go); 10 blocks x 256 filters; 600 and 900 learning steps; 700 training actors and 50 archive actors.
- **Results** [paper]:
  - 9x9 Go head-to-head vs AlphaZero: **0.753-0.795 at checkpoint 300** and **0.641-0.670 at checkpoint 900**.
  - Connect Four: 0.593-0.643 at checkpoint 600.
  - Trajectories per learning step: Connect Four 147 -> 323+; Go 74.8 -> 147+.
  - Value loss on visited states, Go: 0.293 -> 0.214-0.227.
  - Beat KataGo's own search control (branching and trajectory initialisation) on Connect Four. Adding playout-cap randomisation and forced playouts on top of Go-Exploit improved it further.

## 3. Uncertainty-Guided Exploration (NeurIPS 2025, Cheng, Tsai, Hong, Kandemir)

Status: peer-reviewed proceedings. Read in full from the PDF.

- **LCR** [paper]: LCR(s) = Pr(Z_i != Z_j) = 2w(1-w), estimated from the search value v in [0,1] as 2v(1-v). An optional Beta posterior is (alpha + s) / (alpha + beta + m) from paired outcomes. The worked example uses (alpha, beta) = (4.8, 5.2), so the prior strength is 10.
- **Branching** [paper]: per game, positions with LCR >= **LCR0 = 0.48** get weight u = t (the move number), else 0; the branch point is sampled proportional to u. LCR0 = 0.48 corresponds to v = 0.6, the empty-board value, so the paper branches only from positions at least as uncertain as the opening. The branch move is **resampled from the MCTS visit policy pi**, backtracking if that move was already explored. This repeats up to **V = 10 variations per game**.
- **Value label** [paper]: each position's value label is the **mean outcome over all variations passing through it**.
- **Settings** [paper]: 400 simulations, c_puct 1.25, Dirichlet 0.25; 9x9 net 6 blocks x 96 channels; 0.3M states per iteration.
- **Results** [paper]:
  - 9x9 Go: reaches the baseline's strength with **59.6% of the samples** on average, and leads by +99.3 Elo at equal samples.
  - 19x19 Go, warm-started: 66.4% of samples, +75 Elo.
  - V=10 beat V=1 at a 69% win rate after 12M positions.
  - **Uniform random branching was worse than no branching** (35% win rate vs baseline). Choosing *where* to branch matters.
- [inference] For us: v' = (v+1)/2; set LCR0 = 2 v0 (1-v0) from the opening value. This is a self-play-side change, so it cannot be judged in one cycle.

## 4. AZAL: AlphaZero in Sparsely Rewarded Games (arXiv 2607.08984)

Status: arXiv preprint, no venue; 3 seeds; toy games.

- **Auxiliary loss** [paper]: L = L_pi + L_v + lambda_aux * L_aux, with **lambda_aux = 1.0**.
  - L_aux is cross-entropy to q(a|s) proportional to (1[a in B(s)] + eps) * 1[a legal], where **eps = 1e-3** and B(s) is the oracle-optimal move set.
  - It is averaged over labelled states only: the Pons solver for Connect Four, Grundy numbers for Chomp.
  - Value targets are unchanged. **Oracle value supervision was not tested.**
- **Results** [paper]:
  - Connect Four oracle-match 0.785 -> 0.849; first-failure ply 0.867 -> 5.283; sampled states 0.589 -> 0.768.
  - Chomp 10x11: perfect traces 0/60 -> 60/60. Chomp 9x10: 0/60 -> 34/60.
  - Multi-frame inputs did not help.
  - Strength vs opponents was not measured. Settings: 800-1000 simulations, 9 blocks x 128 channels, 700 games per iteration.
- [inference] For us this helps only on states where an offline oracle knows more than the deployed net plus in-graph rules. Where the in-graph rules already override the outputs, it adds nothing at play time.

## 5. Engineering Efficient Self-Play Chess (arXiv 2609.37447, B. Braun)

Status: arXiv preprint, single author; 8 GPUs for 2.5 days; 3,251 benchmark Elo.

- **Replay** [paper]:
  - Window grows from **0.6M to 20M** rows.
  - **4 training presentations per new row**. Ratios 4, 6.25 and 8 gave "comparable strength" in short controls; 4 was kept for freshness.
- **Restarts** [paper]:
  - 50% of games start after 0-8 uniform random plies; 50% start from an archive.
  - Archive eligibility: >=15 plies remaining, |root value| <= 0.8, the top 2-3 actions cover 85% of visits.
  - The game is replayed down an unplayed alternative, marked used.
  - Selection is 30% uniform, 70% proportional to sqrt(0.5 * |searched root value - raw net value|).
  - Archive size is not stated.
- **Replay sampling** [paper]: 70% by policy surprise with the weight **capped at 2.0**, 30% uniform. The exact formula is not given.
- **Value target** [paper]:
  - WDL head; outcome discounted 0.998 per ply.
  - Root-value blend ramps 0 -> **0.10**.
  - Auxiliary losses: next-policy 0.15, remaining-length 0.1.
- **Search** [paper]: visits 300 -> 800, the same on every move; c_puct 1.5; 60% subtree reuse. A frozen-net sweep scored 0.318 at 200 visits vs 0.748 at 1600.
- **From-to head** [paper]: query-key over 64 squares, 1,880 actions, 50k parameters vs 484k for a dense head. Policy gap improved by **0.0298 nats** at about a 1.9% serving cost.
- **Negative or neutral results** [paper]:
  - Learned visit allocator: about -60 to -100 Elo.
  - Search stopping: unresolved, +1.7 ±9.9 Elo.
  - Graph search: about 8% slower.
  - Inference cache: 0.97% hit rate.
  - Parallel leaves: -6.4 Elo per doubling.
  - Move-plane head: weaker.
  - Larger WDL head: rejected.
  - INT8 post-training quantisation: rejected.
  - Distillation: -176 to -291 Elo.
  - Reanalysis: not retained, unmeasured.
  - Publishing every 100 steps: too costly.
  - Model growth: no clear gain.
  - Discounting: benefit "not established".
- **Credited components** [paper]: compact structured policy head, shared conv trunk, progressive sizing, **targeted restart states, prioritised replay**. **No per-component Elo ablation is given.** The author calls the result "the integrated system".

## 6. KataGo Methods doc and code

Status: documentation and source code, not peer-reviewed.

- **Policy surprise weighting** [doc]: about half the frequency weight is uniform (baseline 0.5 per full-search row); the other half is proportional to KL(policy target || prior). Rows are repeated floor(w) times, plus one more with probability equal to the fractional remainder. No importance scaling is applied.
- **Formula in play.cpp** [code], per game:
  - w_i = (1 - 0.5 - vs) * t_i + 0.5 * P_i * sum(t) / sum(P) + vs * V_i * sum(t) / sum(V).
  - P_i = t_i * KL_i + (1 - t_i) * max(0, KL_i - 1.5 * mean_t(KL)). Fast rows (t=0) earn weight only from surprise above 1.5x the game's weighted mean.
  - V_i = t_i * min(1, KL(smoothed outcome || raw NN WDL)), with smoothing nowFactor = 1/(1 + 0.016 * area).
  - vs is scaled down when the game's mean value surprise is below 0.010.
  - Rounding: floor plus Bernoulli on the remainder.
- **Config** [code, selfplay1.cfg]:
  - `policySurpriseDataWeight = 0.5`, `valueSurpriseDataWeight = 0.1`, so the current split is 40% uniform / 50% policy-surprise / 10% value-surprise.
  - `cheapSearchProb = 0.75`, `cheapSearchVisits = 100`, `maxVisits = 600`, `cheapSearchTargetWeight = 0.0`.
  - `reducedVisitsWeight = 0.1` at a 0.9 win rate.
  - `sidePositionProb = 0.020`, `earlyForkGameProb = 0.04`.
  - The doc calls value surprise "much more experimental and unproven".
- **Short-term value targets** [code, trainingwrite.cpp]:
  - nowFactor = 1/(1 + area*0.176), 1/(1 + area*0.056), 1/(1 + area*0.016).
  - target_t = sum_k nf (1-nf)^k Q_{t+k}, with the leftover mass on the outcome.
  - The doc gives horizons of about 50, 16 and 6 turns on 19x19.
  - Loss weights per the doc caption (final, 50, 16, 6): legacy [1.0, 0.458, 0.458, 0.125], newer [0.6, 0.6, 0.6, 0.6].
  - Benefit [doc]: nets "train slightly faster"; **no Elo** is reported.
  - [inference] For an 8x8 board (area 64): nowFactor **0.0815 / 0.218 / 0.494**, giving horizons of about 12 / 4.6 / 2 positions.

## 7. Chessformer (arXiv 2605.19091) and Monroe & Chalmers (arXiv 2409.12272)

Status: 2605.19091 has an arXiv comment naming ICLR 2026, not verified on OpenReview. 2409.12272 is a preprint (v2, Oct 2024).

- **Policy head** [paper, both]:
  - A dense layer over each square's token, then linear projections to a source-square query and a destination-square key.
  - Logits are the scaled dot product, a 64x64 matrix with illegal moves masked.
  - Promotions use an additive per-piece bias projected from the last-rank keys.
- **Position bias in the head** [paper, 2409.12272]: none. The authors "did not see performance gain" from any position representation in the policy head.
- **GAB** [paper, 2605.19091]: Geometric Attention Bias is a *body* attention bias generated from the board, h x 64 x 64. Leela ablation: +1.9 policy accuracy, +83 Elo. In the body of 2409.12272, Shaw relative encoding beat absolute encoding by 1.83% policy accuracy.
- [inference] Our bilinear from x to head is this same family (q^T W k), so I recommend no change. A zero-initialised 64x64 static pair bias would preserve the current function and be cheap, but the evidence predicts no gain.

---

## Top 5 one-cycle-testable changes (training side only; ranked)

Caveat: no source reports a per-cycle effect for any of these. A single 55%/400 gate reliably detects only gains of about +50 Elo or more.

1. **KataGo surprise-weighted sampling over the 600k window.**
   - Parameters [code]: 0.4 uniform / 0.5 policy-surprise / 0.1 value-surprise, using the formula in section 6. Normalise per game; fast rows earn weight only from excess surprise above 1.5x the game mean.
   - Sample with replacement proportional to w; do not reweight the loss.
   - Simpler alternative [paper, chess]: 30% uniform + 70% proportional to min(KL_i / mean KL, 2.0).
   - Needs the stored root prior (or a recompute). Expected gain: unquantified in both sources. Risk: low.
2. **Drop or downweight (0.1) rows whose outputs the in-graph exact rules or solver override.** [inference] This frees capacity and stops exact ±1 states from dominating the value head. Do it only if those rows are more than about 10% of the window. Trivial to implement.
3. **Short-term value auxiliary heads** [code].
   - Three tanh heads with nowFactor 0.494 / 0.218 / 0.0815.
   - Loss weights 0.125 / 0.458 / 0.458 relative to the main value loss (KataGo legacy).
   - Drop them at ONNX export, so inference cost is zero.
   - Gain [doc]: "slightly faster" training. Warm-start risk: the new heads perturb the trunk.
4. **Set the replay ratio to about 4 presentations per new row** [paper, chess]: about 230k samples per cycle at 57.6k new rows. If you currently train a full pass over 600k (about 10.4x), cutting it shortens the cycle and allows more gates. Strength was comparable at 4-8.
5. **AZAL oracle policy cross-entropy** [paper]: lambda_aux = 1.0, eps = 1e-3, averaged over labelled states, applied only where an offline solver labels states the in-graph rules do not cover. Otherwise skip it.

**Not testable in one cycle, but the best-evidenced:** RGSC-lite restarts.
- lambda = 0.5; per-worker buffer of 100; trajectory regret R as in section 1.
- Insert only if R exceeds the buffer minimum; P proportional to R^10; EMA alpha = 0.5 on replay.
- Reported +70 to +84 Elo vs AlphaZero after 300 iterations. Judge it only after 5 or more cycles.

**Process note** [inference]: the 55%/400 gate likely discards most real gains of 10-30 Elo. The chess paper's ">=0.48 twice consecutively" non-regression rule is the documented alternative.
