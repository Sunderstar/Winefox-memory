"""Scan PUBLIC source files or a release ZIP for obvious private artifacts.

Tracked git files are authoritative when in a git repository. Untracked local
runtime data are deliberately skipped; this is NOT permission to zip the entire
working directory. Before publishing any ZIP run `--zip <zip-file>`.
This detects common problems only: human review is still required.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
BAD_PARTS={'data','__pycache__','.venv','venv','backups','.pytest_cache','.git'}
BAD_SUFFIXES={'.pyc','.pyo','.db','.sqlite','.sqlite3','.maid','.nbt','.log','.jsonl'}
TEXT_SUFFIXES={'.py','.md','.txt','.json','.yml','.yaml','.toml','.html','.bat','.sh','.ini','.example'}
SECRET_PATTERNS=[
    re.compile(r'(?i)\bsk-[a-z0-9_-]{16,}\b'),
    re.compile(r'(?i)\b(?:ghp|gho|ghu|ghs)_[a-z0-9]{20,}\b'),
    re.compile(r'(?i)\b(?:api_key|secret_key|access_token)\s*[:=]\s*["\']?[a-z0-9_-]{20,}'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
]


def _files():
    try:
        output=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT,stderr=subprocess.DEVNULL,timeout=5)
        if output:
            return [ROOT / name.decode('utf-8') for name in output.split(b'\0') if name]
    except (subprocess.SubprocessError,OSError):
        pass
    # With no git repository, inspect source-like paths only. Runtime private
    # files have not been added to the release. ZIP builds get audited below.
    return [x for x in ROOT.rglob('*') if x.is_file() and not any(p in BAD_PARTS for p in x.relative_to(ROOT).parts)]


def _check_name(name: str) -> str | None:
    parts=Path(name).parts
    if any(p in BAD_PARTS for p in parts):return 'private/runtime directory'
    p=Path(name)
    if p.name=='.env' or (p.name.startswith('.env.') and p.name!='.env.example'):
        return 'private API config'
    if p.suffix.lower() in BAD_SUFFIXES:return 'private data/cache extension'
    return None


def _audit(entries):
    problems=[];count=0
    for name,content in entries:
        count+=1
        reason=_check_name(name)
        if reason:
            problems.append((name,reason));continue
        if len(content)>4_000_000:
            problems.append((name,'unexpected large file'));continue
        try:
            text=content.decode('utf-8')
        except UnicodeDecodeError:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                problems.append((name,'possible secret-like text'));break
    return count,problems


def check(zip_path: Path | None = None):
    if zip_path:
        with ZipFile(zip_path) as z:
            files=((n,z.read(n)) for n in z.namelist() if not n.endswith('/'))
            count,problems=_audit(files)
    else:
        files=((str(p.relative_to(ROOT)),p.read_bytes()) for p in _files() if p.is_file())
        count,problems=_audit(files)
    print(f'Checked {count} public-source files; flagged {len(problems)}.')
    for name,reason in problems:print(f'CHECK: {name}: {reason}')
    if not zip_path:print('Warning: local untracked data are not scanned. Use --zip for the actual release.')
    return 1 if problems else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--zip',type=Path,default=None,help='Scan a release ZIP instead of source files')
    args=parser.parse_args()
    sys.exit(check(args.zip))
