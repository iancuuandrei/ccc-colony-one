# Exact-referee local candidate comparisons

`paired_arena.py` uses the unmodified supplied engine and `OnnxEvaluator` with one
CPU inference thread. MCTS uses 100 fixed simulations and every other published
default, including noise and the first-eight-action temperature schedule. The
Level 4 supplied `scene.json` is included in the engine manifest hashes.

Each seed is played twice with models swapping colors. Both games start
`numpy.default_rng(seed)` at the same state. This common-random-number design
reduces opening noise and isolates color effects better than the existing arena's
different seed per color. It preserves the correct marginal randomness of each
game. Different branches may consume different random numbers; it does not force
identical openings. Each color is reported separately.

The independent unit for combined uncertainty is the seed, not an individual
game. Reports include deterministic 20,000-resample paired bootstrap 95% intervals
over two-game seed scores, conservative bounded-score Hoeffding 95% intervals,
per-color Wilson 95% intervals, and a two-sided exact sign test on swept pairs.
Bootstrap evidence is approximate. The Hoeffding interval requires only independent
sampled seeds and remains valid with within-pair dependence. These finite local
seed samples do not qualify the unknown tournament opponent distribution.

Seed ranges were separated before the tactical comparisons: 10000-10063 baseline,
20000-20127 screening, 30000-30127 confirmation, and 40000-40127 secondary-opponent
robustness. The tactical model remained fixed across screening and confirmation.
Confirmation is the relevant held-out check; pooling screening and confirmation
is supplementary exploratory precision and is not selection-corrected evidence.
The same 40000 range was also used for h1-v-l3 to avoid mistaking the large seed
variation seen in l3 matchups for candidate regression.

Each run writes immutable protocol/model/engine/runtime/harness SHA256 metadata,
completed games with full histories in flushed JSONL, and an atomic summary.
Resumption checks that the entire protocol still matches before reusing completed
games. End hashes are checked against initial hashes. The first baseline's exact
harness source is saved beside its manifest; subsequent runs use the resume-fixed
current harness. A completed confirmation resume reported zero pending games.

`audit_evidence.py` replayed all 1152 completed histories legally to their recorded
winners. Two baseline games were independently reproduced through the original
`arena.py` using its seed transformation, with identical complete histories and
winners. Results are saved in `evidence-audit.json`.

Local tactical result: screen144/256, confirmation141/256; pooled285/512 (55.66%),
paired bootstrap95 52.54%-58.79%. Held-out confirmation alone is55.08%, interval
50.39%-59.77%, exact sweep sign p=.0470. Cyan confirmation58/128 and Magenta83/128
show a side gap, while pooled Cyan130/256 and Magenta155/256 show no observed
pooled Cyan loss. This supports a modest overall tactical preference over h1,
with limited statistical margin on the standalone confirmation.

Against l3 on the shared40000 range tactical wins172/256 and h1 wins171/256.
The paired seed bootstrap95 interval for tactical-minus-h1 is[-4.69,+5.86]
percentage points. This shows similar observed robustness; it is not an equivalence
or noninferiority guarantee. Format/resource qualification remains a separate
validator responsibility, and local wins do not prove grader acceptance.
