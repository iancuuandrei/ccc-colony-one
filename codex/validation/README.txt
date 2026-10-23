h1 exact artifact SHA256:
0940ddada8c2b761126ffbc5532d3103c3b8cb4eb20d89f7923f70d03f2e8b65

Unmodified supplied tools/validate_submission.py accepted h1 twice after
parallel Windows selfplay ended. Saved second transcript. This is local
acceptance-policy evidence, not remote grader acceptance.

Windows caveat: the supplied psutil monitor watches the Windows venv launcher,
which reports approximately 4 MiB RSS and 0.02s CPU despite actual Python
runtime/session work. Those monitor figures cannot qualify hard Linux limits.
qualify.py --mode level4 measures the actual running Python process instead.

Supplied policy constants are authoritative for our local hard-limit probe:
CHECK_ADDRESS_SPACE_MIB=384, CHECK_CPU_SECONDS=2, timeout=3;
MATCH_ADDRESS_SPACE_MIB=512, MATCH_CPU_SECONDS=15, timeout=20.
The docs also contain an outdated 192 MiB address-space sentence. No supplied
server submission_validation.py is available, so a custom POSIX cap probe
can reproduce numerical caps but does not certify unavailable server code.

qualify.py graph uses onnx.checker plus unchanged validate_submission_contract,
then runs raw ORT outputs on all 24 fixed legal/numeric-extreme corpus cases
in batch1,8,16. No evaluator nan_to_num/clipping hides invalid output.
Level4 uses unchanged MCTSConfig(simulations=100), batch8, seeded RNG, both
sides running the same exact model, and reports measured CPU per side.

Environment provenance: 24 executable/docs files match extracted archive
provenance. The sole difference, colony_one/scene.json, is the intentional
Level 4 task-supplied scene. Its bytes exactly match the preserved raw
challenge/current/task/level-4_scene.json (SHA2565489fce6...338270f).
See level4-scene-lineage.json and env-provenance-check.json. Neither raw
task data nor extracted environment provenance was altered by validation.

Linux qualification completed with prepared isolated Python3.12.13 runtime:
/home/andrei/.cache/colony-one-validator/bin/python
numpy2.5.3 / onnx1.23.0 / onnxruntime1.30.0 (task pinned versions).

h1 SHA0940ddad...f2e8b65: hard check and hard full L3 match PASS.
tactical v1 SHA8fef2d4a...7487b5b: unchanged validator ACCEPTED twice;
hard check and hard full L3 match PASS; full raw batch1/8/16 corpus PASS;
full100sim selfmatch24actions, sideCPU0.1295/0.1105s, initCPU0.0881s,
peakRSS82,481,152bytes.
tactical v2 SHA0a95cc4b...caefd76: unchanged validator ACCEPTED;
hard check and hard full L3 match PASS; full raw batch1/8/16 corpus PASS;
full100sim selfmatch34actions, sideCPU0.2093/0.1871s, initCPU0.1261s,
peakRSS84,623,360bytes.

Hard checks used RLIMIT_AS384MiB/RLIMIT_CPU2s/wall3s/NOFILE32 for model
check and RLIMIT_AS512MiB/RLIMIT_CPU15s/wall20s/NOFILE32 for L3 match,
all before isolated child exec. Full numerical results in *-hard-linux.json.
These preserve the supplied limits; Linux server wrapper itself was not
supplied. Local checks do not establish remote grader acceptance.
