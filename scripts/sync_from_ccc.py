"""Incremental allowlisted import from CCC-OMNI; never commits or pushes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path outside root: {relative}')
    return path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO.parent / 'CCC-OMNI')
    parser.add_argument('--research-source', type=Path, help='Directory holding explicitly shared research inputs; needed only on their first import')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--only', action='append', default=[], help='Refresh only this exact destination path; preserve other manifest entries')
    args = parser.parse_args()
    source = args.source.resolve()
    entries = json.loads((REPO / 'sync_allowlist.json').read_text(encoding='utf-8'))['files']
    old_path = REPO / 'provenance' / 'snapshot.json'
    old = json.loads(old_path.read_text(encoding='utf-8')) if old_path.exists() else {'files': []}
    previous = {item['path']: item for item in old['files']}
    records, pending, missing = [], [], []
    for entry in entries:
        if args.only and entry['path'] not in args.only:
            if entry['path'] not in previous:
                raise ValueError(f'Unimported entry outside --only selection: {entry["path"]}')
            records.append(previous[entry['path']])
            continue
        dst = inside(REPO, entry['path'])
        if entry.get('snapshot_once') and dst.is_file():
            current = digest(dst.read_bytes())
            if current != entry['sha256'] or entry['path'] not in previous:
                raise RuntimeError(f'Frozen snapshot has changed or lost provenance: {dst}')
            records.append(previous[entry['path']])
            continue
        if entry.get('source_group') == 'research_inputs':
            if args.research_source is None:
                raise ValueError('First import of shared research inputs requires --research-source')
            src = inside(args.research_source.resolve(), entry['source'])
        else:
            src = inside(source, entry['source'])
        if not src.is_file():
            if entry.get('optional'):
                missing.append(entry['source'])
                if dst.exists() and entry['path'] in previous:
                    records.append(previous[entry['path']])
                continue
            raise FileNotFoundError(src)
        before = src.stat()
        data = src.read_bytes()
        after = src.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Source changed while reading: {src}; retry after write completes')
        if len(data) > 20 * 1024 * 1024:
            raise ValueError(f'Allowlisted file exceeds 20 MiB review limit: {src}')
        sha = digest(data)
        expected = entry.get('sha256')
        if expected and sha != expected:
            raise RuntimeError(f'Pinned candidate changed: {src}; preserve it and add a new versioned entry')
        current = digest(dst.read_bytes()) if dst.exists() else None
        prior = previous.get(entry['path'], {}).get('sha256')
        if current and current != sha and current != prior:
            raise RuntimeError(f'Local edits would be overwritten: {dst}; reconcile them first')
        record = {'path': entry['path'], 'source': entry['source'], 'sha256': sha, 'bytes': len(data)}
        if entry.get('source_group'):
            record['source_group'] = entry['source_group']
        records.append(record)
        if current != sha:
            pending.append((dst, data))
    print(json.dumps({'changed': [str(p.relative_to(REPO)) for p, _ in pending], 'missing_optional': missing}))
    if args.dry_run:
        return
    for dst, data in pending:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + '.sync-tmp')
        tmp.write_bytes(data)
        os.replace(tmp, dst)
    try:
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    except subprocess.CalledProcessError:
        revision = None
    records.sort(key=lambda rec: rec['path'])
    content = {'schema': 1, 'source_repo': 'iancuuandrei/CCC-OMNI', 'source_head': revision,
               'scope': 'Allowlisted working files; source_head does not assert these generated files were tracked.',
               'files': records, 'missing_optional': missing}
    comparable = {k: v for k, v in old.items() if k != 'snapshot_utc'}
    if content != comparable:
        content['snapshot_utc'] = datetime.now(timezone.utc).isoformat()
        old_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.write_text(json.dumps(content, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
