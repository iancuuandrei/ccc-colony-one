# Actual overrides integration audit

PASS for the current single-manifest pipeline. Executed `overrides-integration-audit.py` through the prepared WSL interpreter, importing the actual `train.load` and executing the current nested loss functions on CPU leaf tensors. No Net instance, training, model inference, actor launch, or live-job interruption occurred.

Audited `train.py` SHA256 `b0249ff4b5a6195b7ec3e5ec156568ef5c662b12b6672b69f453990f9f60e255`; actor SHA256 `2f3dd0b1cfa11ba6e81aba596730ba9116a9e0dc0f355181245806958ea2d5ca`. Both stayed unchanged during execution. Full numerical evidence: `overrides-integration-audit.json`.

## Executed evidence

- The actual 1,984-row load (19 certified targets plus 1,965 approximate teacher rows) applied exactly six replacements: source rows 1680, 1737, 1808, 1872, 1909, 1962. All other teacher fields/rows remained unchanged. This is execution of Opus's current implementation, beyond the earlier separate dry replacement.
- Current override SHA256 `39f2fd439f8bada43812cf19b99737e547697a4f0c8f191e6f5105cbda766276` contains 19 unique contiguous-float16 observation keys. All 19 replacement chunk hashes, observations, complete legal sets, policy supports and proof-file/record hashes match. All six existing teacher-row file hashes and observation identities match. Oracle trees were not re-solved in this integration audit; their previously independently checked certificates are bound by hash.
- All 19 keys apply in the designated approximate source folder, including 13 positions without existing teacher rows. The same observations in the raw outcome folder receive zero replacements. Perturbations that preserve float16 bytes still match; altered float16 bytes do not.
- Actual mixed load of 5,132 rows includes new55, policy144 and deep10910 two rows plus tactical, approximate and true-outcome inputs. Separate true-outcome fields remain identical. All external rows have next-policy auxiliary disabled.
- New inputs retain 13 exact-value rows and 197 policy rows across 201 positions: 55 has 12 values/52 policies; 144 has 0 values/144 policies; deep2 has 1 value/1 policy. All Q masks are false; finite positive length placeholders remain masked when value is unknown.
- Executed current losses with Q mixing .2, opponent auxiliary .15 and surprise weighting .5. UNKNOWN value/length gradients are exactly zero; invalid Q gradients are zero; policy-only rows have nonzero policy gradients; rows without policy have zero policy gradient; external next-policy gradients are zero.
- Executed current actor main control flow using terminating sleep/subprocess stubs: worker counts 0 and -1 sleep20 before reading model metadata or launching work; count8 constructs a launch command. No actor process was started.

## Limits and follow-up

GEN07's command log contains all three new folders and completed training metrics. It does not record imported source hashes, override hash, or actual replacement count. Therefore the code/data integration is ACTUAL PASS now; attributing these exact code bytes to an already-running historical GEN07 process remains NOT VERIFIED. Log these three items at loader startup for future auditability.

The following source-inspected weaknesses do not invalidate the current single-manifest PASS:

- `load` discovers every override manifest but forwards only `next(iter(ovf))` to `apply_overrides`. With multiple distinct override folders, only one arbitrary manifest is loaded unless another happens to reside in a target source folder. Forward all manifests explicitly when expanding this mechanism.
- Runtime override application trusts manifest contents and does not check proof/chunk/source hashes. Current bindings passed this independent audit. Fail closed on malformed or changed bindings before future use.
- Actor reads `workers()` both for the pause check and command construction. A change to zero between those reads can still construct a zero-worker command. Capture one value per iteration for a complete race fix. The ordinary zero-before-iteration pause case passed.
- Group IDs live in provenance, not the tensor loader. Current learner holds out a self-play chunk from its last window directory; it does not claim a new site-game or motif-level holdout. This audit does not establish held-out generalization.

Only this audit script/report/evidence were written; Opus source files and datasets were preserved.
