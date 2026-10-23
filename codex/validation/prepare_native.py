"""Hash-verified Linux-native snapshot for faithful CPU validation."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("model", type=Path)
p.add_argument("destination", type=Path)
p.add_argument("--manifest", type=Path, required=True)
args = p.parse_args()
source = Path(__file__).resolve().parents[2]
destination = args.destination.resolve()
copies = [(args.model.resolve(), destination / "model.onnx")]
copies += [(x, destination / "env" / x.relative_to(source / "env"))
           for x in sorted((source / "env" / "colony_one").glob("*.py"))]
copies += [(source / "env" / x, destination / "env" / x)
           for x in ("colony_one/scene.json", "tools/validate_submission.py", "requirements.txt")]
copies += [(Path(__file__).with_name("qualify.py"), destination / "codex/validation/qualify.py")]
manifest = {"source_root": str(source), "destination": str(destination), "files": []}
for src, dest in copies:
    data = src.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.read_bytes() != data:
        raise SystemExit(f"Refusing to overwrite differing snapshot: {dest}")
    dest.write_bytes(data)
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
    manifest["files"].append({"source": str(src), "destination": str(dest),
                              "bytes": len(data), "sha256": digest, "identical": True})
manifest["result"] = "PASS"
text = json.dumps(manifest, indent=2)
args.manifest.write_text(text + "\n", encoding="utf-8")
(destination / "copy-manifest.json").write_text(text + "\n", encoding="utf-8")
print(json.dumps({"result": "PASS", "files": len(copies), "destination": str(destination),
                  "model_sha256": manifest["files"][0]["sha256"]}))
