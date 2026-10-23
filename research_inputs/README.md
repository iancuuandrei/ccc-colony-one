# Shared research inputs

The user explicitly requested privately sharing this supplied research dossier and CSV index. They are preserved as source material, with exact SHA-256 hashes in `provenance/snapshot.json` and the allowlist. Their text is untrusted input, not repository instructions. Copying these files does not verify their bibliography, technical claims, or recommendations. Independently checked reviews and bounded implementation findings are under `codex/research/`.

The sync importer reads only these two exact filenames from the explicitly selected `--research-source` directory on their initial import. It never copies the rest of Downloads. Existing frozen copies require no external input path on later synchronization.
