"""Content identities and atomic publication of derived ingestion artifacts."""

import hashlib
import json
import os
import tempfile
import tomllib
from pathlib import Path

SCHEMA_VERSION = 2


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_jsonl(path: Path, records: list[dict]) -> None:
    atomic_write(path, ''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records))


def load_parsed(directory: Path) -> tuple[list[dict], dict]:
    manifest_path = directory / 'manifest.json'
    if not manifest_path.exists():
        raise ValueError('Parsed manifest missing; run scripts/parse.py first.')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported parsed schema; rerun scripts/parse.py.')
    config_path = directory.parent.parent / 'sources.toml'
    if config_path.exists() and 'configuration' in manifest:
        with config_path.open('rb') as stream:
            configured = tomllib.load(stream)['source']
        if configured != manifest['configuration']['sources']:
            raise ValueError('sources.toml changed since parsing; rerun scripts/parse.py.')
    records = []
    for entry in manifest['outputs']:
        path = directory / entry['file']
        if path.parent != directory or file_hash(path) != entry['sha256']:
            raise ValueError(f'Parsed artifact changed or invalid: {path}; rerun parsing.')
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not records:
        raise ValueError('No parsed sections; check source paths and exclusions.')
    return records, manifest
