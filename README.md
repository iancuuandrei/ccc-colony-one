# Colony: One — a policy/value network for the Level 4 tournament

This repository contains our entry for **Level 4 of the Cloudflight Coding Contest "Colony: One"** (23 October 2026). We submitted one ONNX file that judges positions, and the organisers' **fixed** PUCT tree search (100 simulations per action) used it to play a live Glicko-2 tournament against the other teams.

The final uploaded model is **gen12** (`release/gen12/model.onnx`, SHA-256 `540cc1b0403fbc10f5b90a4d71a377c3bd7048084a01ff5a84229eaaa591c1ed`). Over one afternoon we went from a hand-built evaluator to a self-play-trained residual network with exact tactical rules built into the graph:

- **Starting point:** a hand-built evaluator that beat the Level 3 model 66% of the time.
- **First neural model:** won 388 of 400 games against our best hand-built model.
- **gen12:** won 69% against gen09, the model that was live on the site before it, in 400 paired games.

This README explains what we built, which papers each idea came from, and what the measurements showed, including what did **not** work. The older, very detailed provenance notes are in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

---

## 1. The problem in one paragraph

**The game.**
- 8×8 board; each side has Ping, Daemon, Exploit, Trace, Ice and a Kernel.
- Each team plays **two actions per turn**. Capturing an unguarded enemy Kernel wins.
- Pieces next to their own Kernel act as **bodyguards**: an attack on a guarded Kernel kills one guard instead.
- A **firewall** locks the outer ring after rounds 8, 16, 24 and 26, destroying whatever stands on it. A bare-Kernels rule settles positions where only the two Kernels are left.

**What we submit.**
- One ONNX file mapping a canonical `(batch,16,8,8)` observation to 4096 move logits and a side-to-move value in `[-1,1]`.
- The referee runs its own PUCT search around it: 100 simulations, `c_puct` 1.5, batch 8 with virtual loss, Dirichlet noise, temperature sampling for the first 8 actions.

**Constraints.**
- File ≤ 5 MiB; only allow-listed ONNX operators, and all shape/index inputs must be constants.
- Hard CPU limits: 2 s for the upload check, 15 s per match.

**The consequence.** **We cannot change the search. We can only make the evaluator smarter** and reshape the logits and value it sees.

---

## 2. Final pipeline at a glance

```
              ┌──────────── actors (never stop) ────────────┐
 PC 24 procs  │ selfplay.py: referee MCTS + current best    │  VPS 16 vCPU (vps_sync.py)
              │ playout-cap 250/100 sims · 35% hard restarts│  RunPod RTX PRO 6000 (long runs)
              └───────────────┬─────────────────────────────┘
                              ▼ data/sp_* · vps_* · ext_* (site games, certificates, teacher rows)
              ┌──────────── learner.py (GPU, WSL) ──────────┐
              │ train.py: warm start, ~600k newest positions │
              │ 8× board symmetry · soft-Z value · overrides │
              │ main + variant A/B on the same window        │
              └───────────────┬─────────────────────────────┘
                              ▼ export → endgame wrapper → final value Clip → check_onnx.py
              ┌──────────── gate ───────────────────────────┐
              │ arena.py: 400 paired games vs current best   │
              │ promote at ≥52%  ·  official validator       │
              └─────────────────────────────────────────────┘
```

**Network** (`net.py`):
- **Exact in-graph features:** 33 feature planes (16 raw + 17 computed with fixed shift matrices). They give each piece's real attack geometry: rays stop at pieces and locked squares, the Daemon shoots adjacent squares and captures at distance 2–3, the Trace has a minimum range of 2.
- **Trunk:** residual CNN, 48ch×5 blocks for gen01–gen07, then 80ch×6 from big80x6_a onward, with squeeze-excitation.
- **Policy head:** bilinear from×to over the 4096 moves.
- **Value head:** value plus training-only side outputs.
- **Exact win override:** forces value 0.99 and a +30 policy bonus whenever a provably winning tactic exists this turn.
- **Endgame wrapper:** an exhaustively proved bare-Kernel solver.
- **Final value Clip:** keeps the value in [−1, 1] against float rounding in the composed graph.

---

## 3. Research used, and how we used it

We worked from a 106-source dossier (`research_inputs/`) plus our own literature checks (`analysis/research.md`, `analysis/research_verify.md`, `analysis/research_consensus_opus.md`). Every idea was adopted or rejected **only by measurement**: 400 paired games at 100 sims, same seeds, both colours.

### 3.1 Learning loop

| Idea | Source | How we used it | Measured result |
|---|---|---|---|
| Expert iteration / AlphaZero self-play | Anthony et al. 2017 ([arXiv 1705.08439](https://arxiv.org/abs/1705.08439)); Silver et al. 2018 ([Science](https://doi.org/10.1126/science.aar6404)) | Self-play under the *exact referee search*, distil visit counts into the net, repeat | Every promotion from gen01 to gen12 (table §4) |
| Playout-cap randomisation | Wu 2019, KataGo ([arXiv 1902.10565](https://arxiv.org/abs/1902.10565)) | 30–35% of moves get a full 250–400-sim search and become policy targets; the rest use 100 sims (same as the referee) and only feed value targets | Kept throughout |
| Auxiliary heads | KataGo | Training-only heads for root-Q and remaining game length (removed at export, so they cost nothing at inference) | Kept |
| Opponent-policy aux + policy-surprise weighting | KataGo methods notes | A/B variant on the same data window | **Rejected**: 60.25% vs 70.0% for the plain main line |
| Soft-Z value targets | Willemsen et al. 2021 ([doi](https://doi.org/10.1007/s00521-021-05928-5)); bootstrapped targets in [arXiv 2609.09094](https://arxiv.org/abs/2609.09094) | Value target = 0.8·z + 0.2·root-Q, using Q only from the learner's own search | 0.5/0.5 → **44.5%**; pure z → **53.75%** (vs 60.25% main). Kept 0.8/0.2 |
| Replay ratio | *Engineering Efficient Self-Play Chess* ([arXiv 2609.37447](https://arxiv.org/abs/2609.37447)) | Half the steps per cycle | 72.25% vs 74.25%, the same within noise |
| Gating threshold | Same paper; our power analysis | A 55%/400 gate rejects most real +10–30 Elo steps, so we lowered it to 52% with warm starts | Faster lineage, no regressions seen |

### 3.2 Getting more information from the same compute

| Idea | Source | How we used it | Measured result |
|---|---|---|---|
| Search control / hard-state restarts | Go-Exploit, Trudeau & Bowling 2023 ([arXiv 2302.12359](https://arxiv.org/abs/2302.12359)); RGSC ([arXiv 2602.20809](https://arxiv.org/abs/2602.20809)) | 20%, later 35%, of games restart from an archive: the states where the search was most wrong (largest \|z − Q\|), every real site position and, at the end, late-game states | gen05, the first generation with restart data, scored **70%** vs gen04 and gen06 **74.25%**. This is correlational, with no clean A/B |
| Uncertainty-guided branching | Uncertainty-Guided Exploration for AlphaZero, NeurIPS 2025 | States with LCR = 2v(1−v) > 0.48 (i.e. \|q\| < 0.2) added to the restart archive | Part of the restart mix above |
| Reanalyse | Schrittwieser et al. 2021 ([arXiv 2104.06294](https://arxiv.org/abs/2104.06294)) | Every real tournament position re-searched at 800 sims by the current best, stored as policy-only rows | Real losses enter training with targets better than our original moves |
| Gumbel completed-Q policy targets | Danihelka et al., ICLR 2022 ([OpenReview](https://openreview.net/forum?id=bERaNdoegnO)) | Offline study on 620 positions (`evolve_targets/`). A 50/50 mix of visits and the completed-Q policy **cut regret by 31–34%** at no search cost. Self-play now stores `pval_g`, and `train.py --policy-mix` uses it | Promising offline. The arena A/B did not finish before the contest ended |
| Opponent pool | Adversarial-policy results, Wang et al. 2023 ([PMLR](https://proceedings.mlr.press/v202/wang23g.html)) | 20% of self-play games against older models (h1, L3, tactical-v2, earlier generations) | Robustness measure, kept |
| Verified symmetry augmentation | Metamorphic testing of chess engines (Inf. Softw. Tech. 2023) | All 8 dihedral transforms, after checking that the rules are invariant: 11,112 legal-set checks, 0 mismatches | Kept |

### 3.3 Exact tactics instead of hoping the net learns them

The referee search is shallow (100 sims, two actions per turn), so we put **proved rules** into the graph. This idea comes from threat-space and proof-number search (Allis 1994; [Go-Moku solved](https://cdn.aaai.org/Symposia/Fall/1993/FS-93-02/FS93-02-001.pdf)) and from MCTS-Solver. Every rule was **verified against engine brute force with zero violations** before it was used.

| Certificate | Statement | Verification |
|---|---|---|
| **R1** (Codex) | A legal attack on the enemy Kernel plus guards < actions left means a win this turn | 0 violations / 203,193 positions |
| **R2** (subagent) | With 2 actions left, a piece relocates and then attacks the unguarded Kernel | 0 violations; R1∪R2 = **2,041 / 2,044** forced kills found in the exported graph |
| **Bare-Kernel endgame** (Codex, `codex/endgame/`) | Exhaustive solver for positions with only the two Kernels | 44,352 states, 283,648 actions, 0 mismatches on our net |
| **Three-piece endgame** (`threepiece/`) | A Kernel can force a win against a lone Kernel plus one piece | 344,000 positive states, 559,600 winning actions confirmed by the engine; independent audit PASS |
| **Forced loss** (`evolve_loss/`) | Sufficient conditions under which the mover cannot avoid losing by the end of the opponent's next turn | g8: **0 violations, 100% coverage** of 5,745 forced losses in a 250k holdout |
| **Firewall** (`evolve_firewall/`) | The burning ring cannot be escaped in time, with Magenta/Cyan timing handled exactly | 0 violations / 126,453 solved positions; 99.4% of firewall-decided losses covered |

These rules were found as **LLM-generated, automatically verified heuristics**, in the spirit of AlphaEvolve and ReEvo. Subagents proposed candidate rules, a brute-force harness rejected anything with a counter-example, and survivors were ported into fixed matrix operations. Only R1, R2 and the bare-Kernel solver are inside the shipped models. The others are verified, but they were not integrated before the contest ended.

### 3.4 Architecture

| Idea | Source | Result |
|---|---|---|
| Source→destination policy head | Chessformer ([arXiv 2605.19091](https://arxiv.org/abs/2605.19091)); Monroe & Chalmers ([arXiv 2409.12272](https://arxiv.org/abs/2409.12272)) | Our bilinear from×to head is the same family; kept |
| Hybrid attention net with learned (dr,dc) relative bias | Chessformer | `arch_attn/`: **238/400 (59.5%) vs an identically trained CNN** at 1.6× latency. It still needed to catch up with the warm-started lineage |
| Bigger CNN from scratch | Neural scaling notes | 64×6 vs 48×5 on early h1 data: **50%** (twice, independently), at 1.8× CPU |
| Function-preserving widening (Net2Net) | Chen, Goodfellow & Shlens 2016 ([arXiv 1511.05641](https://arxiv.org/abs/1511.05641)) | `widen.py`: gen07 → 80×6 with outputs preserved to 6e-6, then fine-tuned → **56% vs gen07**. Once the data was good, capacity paid off |

### 3.5 Calibration "loopholes" we tried and dropped

The value scale behaves like `c_puct/s`, and the logit temperature like Lc0's policy-softmax temperature. We also tried sharpening the policy only in the opening, because the referee samples by visit counts in its first 8 actions.

| Variant (vs gen04, 200 games) | Win rate |
|---|---|
| logits / 1.4 | 53% |
| opening sharpening ×3 | 52.5% |
| opening sharpening ×8 | 41% |
| value × 0.6 | 38% |

None helped: the trained model was already well matched to the referee. Codex's 8-variant grid on the heuristic model was negative as well.

### 3.6 Explicitly rejected

- **MuZero or learned dynamics:** we have an exact simulator.
- **LLM in the loop at inference:** impossible under the ONNX contract.
- **Porting the engine to JAX/Pgx:** fidelity risk greater than the gain.
- **Ensemble distillation:** tried on RunPod, 199/400 vs its own teacher.

---

## 4. Lineage and results

Every number below is from paired games: exact referee MCTS, 100 simulations, Level 4 scene, each seed played as both colours, fresh seed ranges. All summaries are in `analysis/gate_summary.json`.

| Model | Change | Result |
|---|---|---|
| h1 | Hand-built evaluator, all six pieces | 66% vs Level-3 model |
| tactical-v2 (Codex) | h1 + exact two-action certificates | 70.7% vs h1 |
| **gen01** | First CNN, trained on 12k h1 games | **388/400 vs tactical-v2** |
| gen02 → gen07 | Expert iteration, restarts, site data, overrides | 77.5%, 74.0%, 59.75%, 70.0%, 74.25%, 60.25% (each vs predecessor) |
| big80x6_a | Net2Net gen07 → 80×6 | 56.0% vs gen07 |
| gen09 | 80×6 lineage | 75.25% vs big80x6_a |
| gen10 / gen11 | Continued training | 52.25% vs gen09 / 51.5% vs gen10 (plateau) |
| **gen12 (final upload)** | Continued training, fresh data | **62.75% vs gen10, 69.25% vs gen09** |
| E2 / E5 (RunPod) | 80×6, 12k + 10k steps on all 1.6M positions | E2 77.8% vs gen07; E5 56.8% vs E2, 82% vs gen07 |

Models in `release/` are listed with SHA-256 pins in `sync_allowlist.json`. Checkpoints for gen09, gen12 and big80x6_a are in `checkpoints/`.

### Real opponents (`analysis/site_eval.py` → `analysis/site_history_eval.json`)

The last 28 complete tournament games were replayed exactly by the engine.

**Overall record:**
- **14/28 wins**: 8/14 as Cyan, 6/14 as Magenta; 1/7 against the strongest opponent.
- Losses are **long**: 58 actions on average, against 39 for wins.

**Value error against the true outcome, by phase (MSE):**

| Model | Rounds 1–8 | Rounds 9–16 | Rounds 17+ |
|---|---|---|---|
| gen07 | 0.82 | 0.35 | 0.78 |
| gen09 | 0.76 | 0.40 | 0.95 |
| gen10 | 0.77 | 0.38 | 1.33 |
| gen12 | **0.74** | **0.33** | 1.40 |
| E5 | 0.76 | 0.34 | **0.71** |

**Reading the table:**
- **No classic overfitting:** holdout loss fell monotonically inside every run and across generations.
- **Late-game degradation:** later models got *worse* on real late-game positions. Self-play rarely reaches round 17+ (about 2% of games), but our real losses happen there.
- **The fix is implemented but untested:** phase-weighted sampling (×3 for rounds 9–16, ×6 for 17+), late-game restart states and a larger share of real games. The first generation trained with it (gen13) did not finish before the contest ended.
- **Arena strength did not always transfer to the site:** E5 won its arena gates but, in Andrei's observation, did poorly on the live site, so it was replaced by gen09 and then gen12.

---

## 5. How the work was organised

| Who | Role |
|---|---|
| **Opus** (Claude, this repo's trainer) | Network, export, trainer, self-play, async learner and gating, research synthesis, subagents for review, rule discovery, data analysis and architecture |
| **Codex** | Independent validation (native Linux limits, replays), uploads, exact proofs (R1, endgame, three-piece audit), site-game mining and certified policy overrides |
| **Astra** (GPT-6 teacher, via Codex) | Teacher labels and constructed hard positions, all verified by the engine before use |
| **runpod-teacher** (Claude on an RTX PRO 6000) | Long GPU runs (E2–E6, widening, distillation), using the data and code handed over with user approval |

All coordination went through one shared log, `codex/coordination.md`.

Compute:
- a 32-thread PC with an RTX 5050 (training under WSL);
- a 16-vCPU VPS for self-play;
- the RunPod GPU.

Throughput was about 160 self-play positions per second, and a train-plus-gate cycle took about 6–8 minutes.

---

## 6. Reproducing

```bash
# self-play from the current best (writes data/<prefix>_<model>_<time>/)
python actor.py --best-file loop_state.json --prefix sp --workers 24 --batch 300

# asynchronous learner: train (WSL GPU) -> endgame wrap -> clip -> policy check -> gate -> promote
python learner.py --window-pos 600000 --min-new 50000 --steps 3000

# one training run by hand
python train.py --data data/ext_* data/sp_* --init checkpoints/gen12.pt --out ckpt/gen13.pt \
    --steps 3000 --lr 1.5e-3 --qmix 0.2 --ext-frac 0.08 --late-weight 3 --holdout-last

# paired arena at the referee's 100 sims, each seed played as both colours
python arena.py release/gen12/model.onnx release/gen09/model.onnx --games 200

# acceptance checks
python check_onnx.py release/gen12/model.onnx
python env/tools/validate_submission.py release/gen12/model.onnx
```

The A/B variant and value-target settings live in `learner_cfg.json`. Worker counts are in `cpu_budget.json` and are re-read on every batch.

---

## 7. Repository layout

| Path | Contents |
|---|---|
| `net.py`, `train.py`, `selfplay.py`, `policy_targets.py` | Model, trainer, self-play, Gumbel targets |
| `learner.py`, `actor.py`, `vps_sync.py`, `arena.py` | Asynchronous pipeline and evaluation |
| `widen.py`, `final_clip.py`, `research_ext.py`, `build_archive.py` | Net2Net widening, safe export, Reanalyse, restart archive |
| `arch_attn/` | Chessformer-lite prototype and its control |
| `evolve_loss/`, `evolve_firewall/`, `threepiece/`, `verify_win/` | Verified rule discovery |
| `evolve_targets/` | Policy-target study (visits vs Gumbel) |
| `analysis/` | Research notes, gate summaries, data and site analyses |
| `codex/` | Codex's validation, proofs, site mining and coordination log |
| `release/`, `checkpoints/` | Pinned models and weights |
| `env/` | Supplied referee package |
| `docs/PROVENANCE.md` | Detailed provenance and evidence notes |

## 8. Limitations

- **Late-game values on real opponents** are the main open weakness. The fix (§4) is implemented but untested.
- **Small, correlated evidence:** site evidence is 28 recent games, and the restart and Reanalyse gains are correlational. Arena wins are self-play wins, and they did not always carry over to the live site (E5).
- **Proved rules not shipped:** several are verified (three-piece, forced-loss, firewall) but not yet inside the shipped graph.
