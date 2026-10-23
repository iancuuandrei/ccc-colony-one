"""Check every imported file against the synchronized SHA-256 manifest."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'provenance/snapshot.json').read_text(encoding='utf-8'))
for item in manifest['files']:
    data = (root / item['path']).read_bytes()
    assert len(data) == item['bytes'], item['path']
    assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
print(f"PASS: {len(manifest['files'])} imported files match SHA-256 manifest")
