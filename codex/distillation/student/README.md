# Astra distillation student

The current candidate is `model-distill-v2.onnx`, a 22-parameter compact
adapter trained from 32 independently validated Astra labels over frozen
`model-tactical-v2.onnx`. It reuses the trunk's existing feature tensors and
adds 69 nodes (911 total), with 246 initializers and a 365,027-byte artifact.
It holds out 12 positions by complete source game and trains on 20 positions.
Held-out teacher cross entropy improves from 3.6349 to 3.4632; value mean
absolute error improves from 0.2202 to 0.2151; top preference remains 7/12.
These are teacher-agreement measurements; independent arenas must determine
playing strength. No automatic promotion occurs.

The entire policy vector and value bypass learning when the frozen trunk
returns exact ±1. An operational run over 2,080 positions found 164 such
positions with bitwise-equal policy/value outputs. New guard features include
guard changes conditional on the last action and on low guard count. Prefix
binding regenerates and compares all 840 v2 internal nodes and 220 initializers
before reusing them. Raw ONNX/numpy parity passes batches 1/8/16. Standalone
graph/dry-run validation passes at 0.7963 seconds and approximately 1.331 ms
per batch of 8 on the observed Windows run. Full independent qualification
and arena results are owned by the lead agent.

```powershell
.venv/Scripts/python.exe workspace/colony-one-level4/codex/distillation/student/train_compact.py --packet workspace/colony-one-level4/codex/distillation/packets/packet0001.json --labels workspace/colony-one-level4/codex/distillation/teacher_labels/astra_packet0001.json --packet workspace/colony-one-level4/codex/distillation/packets/packet0002.json --labels workspace/colony-one-level4/codex/distillation/teacher_labels/astra_packet0002.json --baseline workspace/colony-one-level4/codex/tactical/model-tactical-v2.onnx --output workspace/colony-one-level4/codex/distillation/student/model-distill-v2-reproduced.onnx
```

`v2-recovery-hashes.json` binds the checkpoint, output, exact executed source
snapshots, metrics manifest, and operational reports. Source snapshots preserve
the code used to generate v2 even if the main scripts are subsequently extended.

`train_student.py` learns 33 shared action-feature coefficients and 16 value
coefficients from independently validated GPT-6-Astra labels. The baseline
policy/value graph is frozen. Learned policy corrections are bounded to ±1
logit; value corrections are bounded to ±0.12, with baseline exact ±1 values
preserved. There is no board-key lookup and no MCTS/outcome label substitution.

The policy objective is confidence-weighted cross entropy against normalized
teacher-provided legal-action weights, with L2 regularization. The value head
uses separately confidence-weighted ridge regression against teacher values.
Features include moving/target piece identity, guard changes, proximity,
distance, firewall urgency, capture timing, and baseline calibration. All
features are shared across positions and have no absolute-square parameters.

Splits hold out complete source games, using source path plus match ID, seed,
or game index. Source-game identities are hashed for a fixed split independent
of label contents. Outputs include a weights checkpoint, input hashes, split
membership, teacher-agreement metrics, and numpy/ONNX parity on batch 1/8/16.
These metrics measure agreement with the teacher, not playing strength.

Reproduction:

```powershell
.venv/Scripts/python.exe workspace/colony-one-level4/codex/distillation/student/train_student.py --packet workspace/colony-one-level4/codex/distillation/packets/packet0001.json --labels workspace/colony-one-level4/codex/distillation/teacher_labels/astra_packet0001.json --output workspace/colony-one-level4/codex/distillation/student/model-reproduced.onnx
```

Multiple packets can be supplied by repeating both `--packet` and `--labels`
in corresponding order. Existing candidates should receive new output names.

The first real candidate is `model-distill-v1.onnx`, trained from 20 genuine
Astra position labels. Nine positions train, 11 are held out by source game.
Held-out teacher cross entropy improves from 3.7981 to 3.5016, and value mean
absolute error improves from 0.2598 to 0.2196. Top teacher preference remains
6/11. Training top preference worsens from 1/9 to 0/9 despite cross-entropy
improvement. These limited labels do not establish a playing-strength gain.

`operational-fixture.onnx` uses random nonzero coefficients and exists solely
to check the exporter; it is not a trained candidate. The actual v1 candidate
passes raw ONNX/numpy parity and standalone graph/dry-run checks. A full Windows
validator invocation under active load rejected stage 1 for check time limits;
the candidate has not been claimed accepted. Arena and deployment qualification
remain pending independent checks. No candidate has been promoted by this code.
