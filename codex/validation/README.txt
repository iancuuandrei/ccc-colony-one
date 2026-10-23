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

Additional candidates qualified in the same pinned Linux runtime:
- tactical-v2-endgame SHA31361de0...a191599,381756bytes,930nodes259init:
  unchanged validator ACCEPTED; hard check0.710CPU/1.872wall;
  hard fullL3match0.874CPU/2.044wall/30actions; raw batch1/8/16 PASS;
  full100sim34actions, sideCPU0.2315/0.2083s, init0.1511s, RSS84.24MB.
- distill-v1 SHA9fd3c095...c0f44a5,422140bytes,884nodes267init:
  unchanged validator ACCEPTED; hard check0.259CPU/0.558wall;
  hard fullL3match0.346CPU/0.658wall/18actions; raw batch1/8/16 PASS;
  full100sim32actions, sideCPU0.3010/0.2758s, init0.1620s, RSS84.43MB.
These are format/resource results only, with no candidate strength claim.

Compact distill-v2 SHA9ad48af6...6f94163c,365027bytes,911nodes246init:
unchanged Linux validator ACCEPTED; hardcheck0.264CPU/0.573wall;
hardL3match0.358CPU/0.664wall/26actions; raw batch1/8/16 PASS;
full100sim32actions sideCPU1.424/1.232s+init0.604s, RSS83.95MB.

Neural gen01 SHAed45eae7...e87f6c7a,1439594bytes,594nodes196init:
unchanged Linux validator ACCEPTED on mounted workspace and native snapshot.
The first mounted hardcheck failed3s wall while using only0.337CPU, with
peakRSS31MiB, before full runtime startup. This failure remains preserved.
Mounted hardretry PASS: check0.986CPU/2.093wall, L3match2.043CPU/2.824wall.
Native snapshot copies19files byte-for-byte (gen01-native-copy-manifest.json)
to /home/andrei/.cache/colony-one-validation-runs/gen01, excluding /mnt/d
filesystem latency. Nativehardcheck PASS0.905CPU/0.912wall; nativehardL3
PASS2.198CPU/2.200wall/34actions, maxRSS92132KiB. Raw batch1/8/16 PASS
on both paths. Fullnative100sim44actions: sideCPU6.365/6.261s plus
init0.731s, RSS85.75MB. The mounted100sim run used4.657/4.600s per side;
these are different contemporaneous hostload observations, not evidence of
a performance regression caused by native copying. Same model/source hashes.
