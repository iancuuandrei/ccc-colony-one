# Opus position on Andrei's research dossier (106 sources) — 18:15

Grounded in our measured evidence first, literature second. "DONE" = already in the live pipeline with evidence.

## Already in the pipeline (dossier tier-A items)
| Dossier item | Status | Evidence |
|---|---|---|
| Expert iteration / AlphaZero loss (Anthony 2017, Silver 2018) | DONE | gen01→gen03 each promoted: 97% vs tactical-v2, 77.5% vs gen01, 74% vs gen02 (400 paired games each) |
| KataGo playout-cap randomisation, aux heads | DONE | full 250 sims p=0.3 / fast 100 (= referee); aux Q + length heads |
| Verified symmetry augmentation (dossier §4 warns) | DONE | 8 D4 maps, 11,112 legal-set checks + Codex 960 checks, 0 mismatches |
| Exact tactics / threat-space / proof labels (Allis; PNS; MCTS-Solver idea) | DONE in-graph | R1 (Codex proof) ∪ R2: 2,041/2,044 forced kills, 0 false flags; value 0.99 + policy prior |
| Endgame tablebase (Checkers/Lc0-vs-tablebase papers) | DONE in-graph | Codex bare-Kernel solver, 44,352 states / 283,648 actions, 0 mismatches on our net |
| Soft-Z value target (Willemsen 2021) | DONE | 0.8 z + 0.2 rootQ, Q only from learner's own search |
| Reanalyse (Schrittwieser 2021), partial | DONE for site games | 1,965 real-site positions re-searched at 800 sims (policy-only rows) |
| Bilinear from–to policy head (Chessformer-style QK) | DONE | 32-dim from/to embeddings + from/to biases |
| Paired-seed arena, both colours (A7) | DONE | every gate 200 seeds × 2 colours, fresh seed ranges |
| Capacity ablation (A5) | DONE, negative | 64×6 vs 48×5: 50.0% (mine, 300 g) and 50.78% (Codex, 256 g), 1.8× CPU → keep 48×5 |
| Calibration T/s (A8) on tactical-v2 | DONE by Codex, negative | 8 variants × 512 games, none beat baseline |

## Ranked next steps (all cheap, each judged by the 400-game gate)
1. **Search control / hard-state restarts (Go-Exploit 2302.12359, RGSC 2602.20809)** — LIVE since 18:12 on the local actor: 20% of games restart from an archive of 30k states with the largest |z − rootQ| (search most wrong) + all real-site positions; archive refreshed every 10 min. VPS stays uniform as a control stream. Decision rule: keep if the next two gates promote at ≥ the recent rate (74–77%); drop if a gen trained with restarts fails a gate that uniform data passed.
2. **Value-target ablation A3** (z vs 0.8z+0.2Q vs 0.5z+0.5Q) — run as a side-by-side training on the same window while the main learner runs; pick by direct paired match between the two candidates (400 games). Neural self-play Q is now far more reliable than the h1-era Q that motivated 0.2.
3. **Policy-surprise sample weighting (KataGo methods)** — weight positions by KL(search policy || prior) at training time; cheap to add, test via A/B like (2).
4. **Reanalyse at scale** — periodically re-search the hard archive with the newest net at 800 sims and add as policy-only rows (same contract as ext_site_v1_rs).
5. **Opponent-pool robustness (Adversarial policies 2023)** — keep 20% pool games (h1, L3, tactical-v2, older gens) and Codex's site-loss mining; never drop real-site data.

## Explicitly rejected / deprioritised (with reason)
- MuZero/learned dynamics, giant transformers, LLM-as-evaluator: exact simulator exists; ONNX ≤5 MiB and CPU budget; dossier §4 agrees.
- Bigger nets: our A5 ablation was negative twice.
- Pgx/JAX port or bitboard engine: self-play throughput is ~160 positions/s and cycles are already ~6 min; porting the rules with exact fidelity in <2 h is a correctness risk larger than the gain.
- Gumbel offline teacher: the referee is PUCT; Gumbel targets on PUCT statistics are heuristic (survey). Could revisit only if (2)/(3) stall.
- More T/s calibration: Codex's negative result on tactical-v2; could re-check once on the final neural candidate if CPU allows.

## Questions for Codex / Astra (to converge)
- Do you agree with ranks 1–3? Any source in the dossier you rank above them for *this* game?
- Astra held-out set: please report gen03 vs gen02 on your TEST_ONLY states (tactical choices + value calibration) — a second, non-arena signal for promotions.
- Site mining: which failure classes of gen02 on the live site are not covered by R1/R2/endgame? Those become archive seeds.
