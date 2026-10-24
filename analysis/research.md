# Research: squeezing a fixed 100-sim PUCT referee from the network side

Date: 2026-10-23. Scope: web/arXiv only. Labels used throughout:
**[E]** = published evidence (paper ablation or engine write-up), **[D]** = derivation from the search formula (mathematically certain if the referee is standard PUCT, but not empirically measured here), **[S]** = speculation / engineering judgement.

Setting recap: fixed referee, PUCT c=1.5, 100 sims, batch 8 with virtual loss 1.0, root Dirichlet alpha 0.3 / eps 0.15, unvisited child Q = 0 (FPU 0), temperature sampling for the first 8 actions then greedy. We control only logits and value. Budget about 3 h, 32 cores (about 14 games/s), one RTX 5050.

---

## Key facts gathered

### Search-side levers the network controls
- **Lc0 default policy softmax temperature is 1.359** (range 0.1–10; "higher values make priors of move candidates closer to each other, widening the search"). Lc0 also uses FPU strategy "reduction" with value 0.330 and cPUCT 1.745. Lc0 tunes these by match play. [E] Source: lc0 `src/neural/shared_params.cc`, `src/search/classic/params.cc` (https://github.com/LeelaChessZero/lc0).
- **KataGo** uses a root policy softmax temperature of 1.25 early, decaying to 1.1 (paper: 1.03), an FPU reduction of 0.2 (0 at the root when noise is on), and cPUCT 1.1. "Tuning a constant cPUCT on the latest nets gains about 25 Elo"; dynamic variance-scaled cPUCT plus uncertainty weighting gives about 50–75 Elo. [E] https://arxiv.org/abs/1902.10565, https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
- **Value scale is the same as cPUCT.** [D] The PUCT score is Q + c·P·sqrt(N)/(1+n). If the network outputs v' = s·v, every network-derived Q scales by s, and the argmax is unchanged when c is replaced by c/s. FPU = 0 is invariant under scaling (0·s = 0). So exporting `v' = s·v` sets the effective cPUCT to 1.5/s. Exception: terminal values come from the rules (±1) and are **not** scaled. With s < 1, proven wins and losses weigh relatively more, which is usually desirable. Use an odd, monotone map such as `tanh(s·atanh(v))` or plain `s·v` with s ≤ 1, so negamax symmetry is kept.
- **Logit temperature is the policy-temperature knob.** [D] Exporting `logits / T` reproduces Lc0's `policy-softmax-temp` exactly.
- **FPU = 0 asymmetry.** [D] With Q_unvisited = 0 from the mover's perspective:
  - In **winning** positions (visited Q > 0), unvisited moves look worse than visited ones, so the search narrows onto the top-prior moves. Policy errors are then hard to correct.
  - In **losing** positions (visited Q < 0), unvisited moves look better, so the search tries almost every child once. 100 sims spread across all legal actions and the search becomes shallow.
  - The network can soften both effects only through value magnitude (s) and prior sharpness (T). Verify in the referee code that Q is from the mover's view and how the sign flips across the two actions of one turn: same player twice means no flip between them.
- **Virtual loss 1.0 with batch 8** pushes the 8 leaves of a batch apart. With about 12 batches per action, the search is wider than a sequential 100-sim search. This argues for slightly sharper priors (T < 1) than a sequential search would want, but only measurement decides. [D/S]
- **Value offset (optimism or pessimism) is not recommended.** [D/S] A constant b added to v becomes +b for same-player children and −b for opponent children under negamax. It changes behaviour at alternate depths in a way that depends on the two-action turn structure, not a clean contempt. No source supports it for this setting.

### Policy targets
- **AlphaZero can fail to improve its policy when not all root actions are visited.** Gumbel MuZero learns reliably on 9x9 Go with 2 simulations, while MuZero fails at 16 sims or fewer. [E] Improved target: π' = softmax(logits + σ(completedQ)), with σ(q) = (c_visit + max_b N(b))·c_scale·q, c_visit = 50, c_scale = 1.0 (Go/chess) or 0.1 with Q rescaled to [0,1] (Atari; mctx default value_scale 0.1, maxvisit_init 50). Unvisited actions get the mixed value v_mix = (v_net + ΣN·Σ_visited P·Q / Σ_visited P)/(1 + ΣN). Sources: Danihelka et al., ICLR 2022 (https://openreview.net/forum?id=bERaNdoegnO), mctx `qtransforms.py` (https://github.com/google-deepmind/mctx).
- **Grill et al. (ICML 2020):** visit counts only approximate a regularized policy optimum. Using the exact solution π̄ = λ_N·π_θ/(α − q), with λ_N = c·sqrt(ΣN)/(|A| + ΣN) and α found by bisection, gives the largest gains at low simulation budgets. [E] https://arxiv.org/abs/2007.12509
- **KataGo forced playouts and policy target pruning:** n_forced = sqrt(k·P(c)·ΣN) with k = 2. Removing it cost a 1.25x training-efficiency factor. [E]
- **KataGo playout cap randomization:** p = 0.25 full searches (N = 600, later 1000) and n = 100 (later 200) fast searches. **Only full-search turns are recorded as policy targets**, and fast searches disable noise. Removing it cost 1.37x. [E]
- **KataGo auxiliary heads (training-only):** opponent-reply policy at weight 0.15 (removal cost 1.30x) [E]. Soft policy target π^(1/4) renormalised at weight 8, described as "greatly improving policy learning speed" (no number) [E-weak]. Policy surprise weighting (half uniform, half proportional to KL(target‖prior)) is "one of the larger improvements" (no number) [E-weak].

### Value targets
- **Willemsen, Baier, Kaisers (2021):** a family of value targets. soft-Z, A0C and A0GB (greedy backups: do not let exploratory moves contaminate z) train faster and perform better than plain z on Connect-Four and Breakthrough. [E] https://doi.org/10.1007/s00521-021-05928-5
- **Boige, Boumaza, Scherrer, arXiv 2609.09094 (Sep 2026):** approximate value iteration with **one-step negamax backups** from a frozen network (target = max_a[r − V_θ̄(s')], MSE, ε-greedy ε = 0.15–0.3, Adam 3e-4, batch 256) learns **more accurate values than AlphaZero's z-target**, even against AlphaZero at 512 sims, on Connect Four and Hex 7x7 with exact oracles, with no divergence over 20 seeds. Caveat: it is greedy play without search, and weaker than MiniZero search on Othello and 9x9 Go. [E] https://arxiv.org/abs/2609.09094
- **KataGo short-term value targets:** exponentially averaged future MCTS values with horizons of about 6/16/50 turns on 19x19 (shorter on small boards). They gave slightly faster training and slightly better final value loss, but are only partly retested. KataGo keeps z as the main target, weighted 1.5. [E-weak]

### Robustness
- **Wang et al., ICML 2023:** adversarial policies beat KataGo (above 97% at superhuman settings) with under 5% of its training compute. The flaws persist after adversarial training. Self-play nets have non-transitive blind spots. [E] https://arxiv.org/abs/2211.00241, https://goattack.far.ai
- **AlphaStar league / prioritized fictitious self-play (PFSP):** sample opponents in proportion to how hard they are for the current agent. [E, different domain] Vinyals et al., Nature 2019.
- **Policy-value misalignment in AlphaZero:** for many states the policy and value disagree. VISA-VIS reduces misalignment by up to 76% and value error by up to 55%. [E] https://arxiv.org/abs/2301.11857

### Architecture
- **KataGo global pooling** (per channel: mean, size-scaled mean, max, then an FC layer that biases the channels) in 2–3 blocks plus the heads. Removal cost 1.60x, the largest general technique. Game-specific input features: removal cost 1.55x. The smallest net, b6c96, already reached strong amateur level in Go. [E] Global pooling is the mechanism KataGo uses for board-size awareness, which applies directly to a shrinking board. [S]
- **Lc0** uses SE (squeeze-excitation) residual blocks, which are a close relative of global pooling. [E: engine practice, no ablation found]
- **KataGo nested bottleneck blocks** suit large nets. Simple single-bottleneck blocks were "substantially worse" than normal residual blocks. [E] For a tiny net, use plain residual blocks.

---

## Ranked recommendations (top 8)

### 1. Calibrate T (logit temperature) and s (value scale) against an exact local replica of the referee, then bake them into the ONNX
- **What:** export `logits/T` and `s·v` (s ≤ 1, or `tanh(s·atanh v)`). Coarse grid: T ∈ {0.7, 0.85, 1.0, 1.2, 1.4} × s ∈ {0.5, 0.7, 1.0}, 400 games per cell against the untuned net plus 2 pool members. Refine around the best cell.
- **Why:** the referee's cPUCT, FPU and policy temperature were not tuned for our net. Lc0 and KataGo both find non-unit temperatures and tuned cPUCT worth about 25 Elo [E]. s is the same as cPUCT [D].
- **Cost:** about 15 cells × 400 games at about 14 games/s, roughly 7–10 min of CPU. No retraining.
- **Expected gain:** +20–60 Elo [S, anchored on KataGo's 25 Elo for cPUCT alone]. Highest benefit per minute of compute.

### 2. Train the prior toward a search-improved policy, not raw visit counts
- **What:** compute π' = softmax(logits + σ(completedQ)) from each full search (c_visit = 50, c_scale = 0.1 with Q min-max rescaled to [0,1], unvisited filled with v_mix). Use it as the policy target, or blend it 50/50 with visit counts. Alternative: KataGo forced playouts (k = 2) plus policy target pruning in self-play.
- **Why:** at 100 sims the referee mostly ratifies the prior. Gumbel and Grill both show their largest gains at low budgets [E].
- **Caveat:** applying the Gumbel target to PUCT-collected statistics is heuristic [S]. Switching self-play to Gumbel Sequential Halving (m = 16, n = 400) makes it principled but changes the data distribution.
- **Cost:** small code change, no inference cost.

### 3. Better value target: short-horizon bootstrapping and no exploration contamination
- **What:** value target = 0.5·z + 0.5·TD(λ) over successive root Qs, with λ ≈ 0.8 per action (mean horizon about 5 actions; KataGo uses roughly 2–20% of game length).
- **A0GB rule:** for positions before an exploratory action (temperature-sampled first 8 actions, or a Dirichlet-driven deviation from argmax visits), replace z with the Q at that action.
- **Why:** Willemsen et al. and arXiv 2609.09094 both show bootstrapped targets beat pure z on small games [E]. The current 0.2·rootQ only partly does this.
- **Cost:** a few lines in the target builder.

### 4. Make self-play, PCR and gating match the referee
- **What:**
  - Record policy targets only from the 400-sim searches, with noise only there; use the 100-sim positions for value only [E: KataGo].
  - Gate and evaluate at exactly the referee settings: 100 sims, batch 8, VL 1.0, FPU 0, eps 0.15, temperature over the first 8 actions.
  - Use SPRT (Elo0 = 0, Elo1 = 35) instead of a fixed 400 games to save time [S].
- **Why:** a net that is stronger at 400 sims is not guaranteed to be stronger at 100.

### 5. Training-only auxiliary heads, stripped from the ONNX
- **What:**
  - Opponent-next-action policy, weight 0.15 [E: 1.30x].
  - Soft policy π^(1/4), weight 8 [E-weak].
  - 1–2 short-term value heads [E-weak].
  - Policy surprise weighting [E-weak].
- **Cost:** zero inference cost.

### 6. Architecture: small residual net with global pooling and game features
- **What:**
  - Plain residual tower of about 6 blocks × 48–64 channels on 8x8, with KataGo global-pooling bias in 2 blocks and in the policy and value heads [E: 1.60x].
  - Input planes for action-in-turn (1st/2nd), shrink countdown or ring mask, and move number [E: features 1.55x].
  - No attention. Fold BatchNorm into the convs before export.
- **Latency estimate:** 6×64 is about 28M MAC per position, so a batch of 8 is about 0.45 GFLOP, roughly 5–10 ms on one ORT thread [S]. Benchmark with the referee's ORT version. Drop to 48 channels (about 16M MAC per position) if over budget.

### 7. Robustness for a Glicko-2 field
- **What:**
  - PFSP opponent sampling, weight ∝ (1 − winrate)², about 25–30% of self-play games against the pool.
  - Add fixed weak but different baselines to the pool and to gating: random, 1-ply greedy on value, a heuristic bot.
  - Promote a candidate only if it is not worse than about −2σ against any pool member [S].
- **Why:** self-play nets have non-transitive blind spots [E: Wang et al.]. Glicko-2 rewards consistency across the field.

### 8. In-graph symmetry averaging, if the latency budget allows
- **What:** average policy (after un-transforming) and value over 2 or 4 dihedral transforms inside the ONNX graph, at a cost of 2–4x inference.
- **Why:** reduces prior and value noise, and helps with the policy-value inconsistency documented in 2301.11857.
- **Evidence:** speculative for this setting. AlphaGo (2016) averaged 8 symmetries; later engines picked one at random for speed. Measure it against a 2x wider net at equal latency.

**Not recommended:**
- Value offset for optimism or pessimism [D/S, see above].
- Big architecture changes (attention, bottlenecks).
- Lowering gating noise by raising sims.
