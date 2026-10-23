# Independent deep 10910 qualification

**PASS.** Released only two qualified rows to `data/ext_site_deep10910_v1`. The research packet and earlier frozen curricula were preserved.

The verifier uses its own `env.clone/step` enumeration of complete turns, independent of the research minimax helpers. At each live defender endpoint it asks whether **some** complete defender turn avoids **every** terminal enemy response on the next turn. Leaf terminal wins come from the separately qualified own-turn oracle, and every positive leaf witness is replayed through `env.step`.

- **A, before action 39:** Kernel F6-F7 has a demonstrated bounded-safe completion; played F6-G6 is proved losing. Seven other first actions remain unclassified. Full-game value is UNKNOWN. No standard policy/value row is released; only the partial preference remains in research metadata.
- **B, after played F6-G6:** Every one of six legal second actions permits an enemy first turn after which every defender response admits an enemy terminal win. Thus value -1 is exact; no policy preference is invented.
- **C, after alternative F6-F7:** All six second actions are classified. Trace G7-D4 is uniquely bounded-safe: every one of 239 enemy first-turn completions permits a defender response avoiding the second enemy turn's terminal win. This is a complete policy target for the stated horizon; full-game value remains UNKNOWN and masked.

The scope is terminal defeat through the **second following enemy turn**, not merely the immediate enemy turn. For B/C, this can include seven individual actions from the decision point. The C certificate is not an eventual winning strategy.

`proof.json` records 705 distinct endpoint states, 3,930 complete defender turns and 3,844 leaf-oracle checks. All 44 source actions were independently replayed. `mechanics.json` separately checks every typed capture and the stationary Trace shot/bodyguard redirect. `loader_audit.json` executes the current `train.load` function and records the exact policy/value/Q/length masking expressions. B has `has_value=true, has_pol=false`; C has `has_value=false, has_pol=true`; both have `q_valid=false`.

Source group `site:10910`, alias `site-game-10910`, must remain in one split across replay and counterfactual corpora. There are zero D4 orbit intersections with the 48 frozen held-out states, and all holdout hashes remain unchanged. No model, GPU, training, ONNX export or upload was used.
