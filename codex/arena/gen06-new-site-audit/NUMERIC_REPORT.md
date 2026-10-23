# GEN06 raw numeric observed-corpus scan

PASS on all 4,105 observed legal decision states from 118 complete website games, evaluated at batch sizes 1, 8 and 16. Each batch run reached minimum -1.0 and maximum 1.0, with zero out-of-range values, nonfinite values or nonfinite policy entries.

The immutable deployed model SHA256 is `123467a930e0598c464ae7694800ad565aeac71680b88dafc5f014bfe7cf50ac`. Model and corpus hashes matched before and after scanning. Exact float32 observations were reconstructed by complete authoritative engine replays and verified against the 111 local stored game chunks plus seven external episode aliases. Observation collection SHA256 is `e11060ea414b9fbe5fd6a27452522629df260354fb1ca91d943b0b91675bf6d8`.

The scan calls raw ONNX Runtime `session.run` directly, bypassing the normal evaluator's output sanitization and clipping. It uses Windows ONNX Runtime 1.31.0 CPU, one process and one thread. Last partial batches were padded by repeating their last observation to exercise exactly the requested batch size; statistics count each original state once per batch size. All 12,315 state evaluations passed. Total time including replay and identity checks was 20.66 seconds.

There is no observed-corpus evidence requiring an immediate clipping backport to GEN06. This is bounded evidence; it does not prove numeric safety for every legal or search state, or for Linux runtime kernels. The final graph node being `Where` establishes no failure on its own. Historical source model identities remain UNKNOWN.

See `numeric-summary.json` for source hashes and batch statistics, `numeric-observation-index.jsonl` for every observation's game/action/hash, and `numeric-violations.json` (empty) for failures. No model, training data or exporter was changed. No strength games were run.
