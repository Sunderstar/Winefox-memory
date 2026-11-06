"""Explicitly shareable, strictly allowlisted offline diagnostics for WineFox.

Do NOT add raw config keys, exception messages, log lines, SQL text fields,
filesystem paths, message IDs, prompts, or environment variables here.
No API/network calls, imports of embedding models, or mutations are performed.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPORT_FORMAT = "winefox-diagnostic-v1"
PUBLIC_RELEASE = "0.3b.4-community.1"

# All permitted fields are enumerated. Unknown keys/status values are discarded.
RETRIEVAL_FIELDS = {
    'enabled': bool, 'semantic_backend': ('lexical', 'auto', 'sentence_transformers'),
    'max_input_tokens': int, 'memory_budget_tokens': int, 'core_budget_tokens': int,
    'recent_chat_messages': int, 'recent_chat_budget_tokens': int,
    'episode_cards_target': int, 'episode_cards_max': int, 'episode_cards_min': int,
    'cand_limit': int, 'cand_min': int, 'episode_detail_max': int,
    'episode_tool_enabled': bool, 'episode_tool_max_items': int,
    'episode_tool_extra_cards': int, 'episode_tool_budget_tokens': int,
    'context_query_messages': int, 'context_query_max_chars': int,
    'spontaneous_recall_probability': (int, float),
}
AUTO_FIELDS = {
    'enabled': bool, 'trigger_messages': int, 'keep_recent_messages': int,
    'max_messages': int, 'max_chars': int,
}
STATUS_VALUES = {
    'candidates': {'pending', 'approved', 'rejected', 'auto_active', 'duplicate', 'superseded'},
    'extracted_episodes': {'pending', 'approved', 'rejected', 'auto_active'},
    'memories': {'active', 'superseded', 'deleted', 'rejected', 'inactive'},
    'extraction_runs': {'succeeded', 'failed', 'running', 'pending', 'partial', 'skipped'},
    'auto_archive_jobs': {'running', 'completed', 'failed', 'queued', 'succeeded', 'skipped'},
}
# Database queries only use these fixed literal identifiers. Never use an ID from user data.
TABLE_COUNTS = (
    'candidates', 'extracted_episodes', 'memories', 'extraction_runs',
    'processed_records', 'embedding_vectors', 'core_auto_entries',
    'core_auto_runs', 'auto_archive_jobs', 'retrieval_metadata',
)


def _small_json_object(file: Path) -> dict:
    if not file.is_file() or file.stat().st_size > 200_000:
        return {}
    try:
        value = json.loads(file.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _safe_settings(value: dict, allowed: dict) -> dict:
    answer = {}
    for k, spec in allowed.items():
        v = value.get(k)
        if isinstance(spec, tuple) and all(isinstance(x, str) for x in spec):
            if type(v) is str and v in spec:
                answer[k] = v
        elif spec is bool:
            if type(v) is bool:
                answer[k] = v
        elif spec is int:
            if type(v) is int and 0 <= v <= 500000:
                answer[k] = v
        elif isinstance(spec, tuple) and type(v) in spec and 0 <= v <= 1:
            answer[k] = v
    return answer


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open('rb') as stream:
        line_breaks = 0
        last_byte = b''
        for block in iter(lambda: stream.read(1024*1024), b''):
            line_breaks += block.count(b'\n')
            last_byte = block[-1:]
        return line_breaks + int(bool(last_byte) and last_byte != b'\n')


def _safe_package_version(name: str) -> str | None:
    try:
        version = importlib.metadata.version(name)
        # Permit only conventional numeric version parts; never package metadata.
        match = re.match(r'^(\d+(?:\.\d+){0,3})', version)
        return match.group(1) if match else 'installed'
    except importlib.metadata.PackageNotFoundError:
        return None


def _db_counts(path: Path) -> dict:
    if not path.is_file():
        return {'present': False, 'tables': {}}
    report = {'present': True, 'tables': {}}
    try:
        # Open in *read-only* mode; no database creation/migrations.
        con = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=1)
        try:
            existing = {row[0] for row in con.execute('SELECT name FROM sqlite_master WHERE type="table"')}
            for table in TABLE_COUNTS:
                if table not in existing:
                    continue
                n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                entry = {'count': int(n)}
                if table in STATUS_VALUES:
                    col = 'review_status' if table == 'extracted_episodes' else 'status'
                    cols = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
                    if col in cols:
                        rows = con.execute(f'SELECT "{col}",COUNT(*) FROM "{table}" GROUP BY "{col}"')
                        allowed = STATUS_VALUES[table]
                        groups = Counter()
                        for status, count in rows:
                            groups[status if status in allowed else 'other'] += int(count)
                        entry['states'] = {k: groups[k] for k in sorted(groups)}
                report['tables'][table] = entry
        finally:
            con.close()
    except (sqlite3.Error, OSError, ValueError):
        # Report an enum only. SQLite errors can include private paths or data.
        report['status'] = 'read_error'
    return report


def _recall_statistics(path: Path) -> dict:
    """Read a bounded tail and export ONLY finite numeric/boolean aggregates."""
    result = {'log_present': path.is_file(), 'sample_count': 0}
    if not path.is_file():
        return result
    from collections import deque
    rows = deque(maxlen=100)
    try:
        with path.open(encoding='utf-8', errors='replace') as stream:
            for line in stream:
                if len(line) < 400000:
                    rows.append(line)
        core_truncated = tool_calls = 0
        input_tokens = []
        for line in rows:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            result['sample_count'] += 1
            core_truncated += row.get('core_truncated') is True
            expansion = row.get('tool_expansions')
            tool_calls += int(isinstance(expansion, list) and len(expansion) > 0)
            value = row.get('estimated_total_input_tokens')
            if type(value) is int and 0 <= value <= 1_000_000:
                input_tokens.append(value)
        result['core_truncated_turns'] = core_truncated
        result['episode_expansion_turns'] = tool_calls
        if input_tokens:
            result['mean_estimated_input_tokens'] = round(sum(input_tokens) / len(input_tokens))
    except (OSError, UnicodeError):
        result['readable'] = False
    return result


def build_diagnostic_report(data_dir: str | Path, project_dir: str | Path) -> dict[str, Any]:
    """Build an offline report with a fixed schema and no raw user information."""
    data = Path(data_dir)
    project = Path(project_dir)
    system = platform.system()
    result = {
        'report_format': REPORT_FORMAT,
        'version': PUBLIC_RELEASE,
        'privacy': {
            'no_raw_chat': True, 'no_persona': True, 'no_core_text': True,
            'no_api_keys': True, 'no_paths_or_ids': True,
            'no_exception_messages': True, 'no_network_request': True,
            'review_before_sharing': True,
        },
        'runtime': {
            'os_family': system if system in ('Windows', 'Linux', 'Darwin') else 'Other',
            'python': '.'.join(map(str, sys.version_info[:3])),
            'sqlite': sqlite3.sqlite_version,
            'python_310_plus': sys.version_info >= (3, 10),
        },
        'optional_packages': {
            'numpy': _safe_package_version('numpy'),
            'sentence_transformers': _safe_package_version('sentence-transformers'),
        },
        'storage': {
            'chat_message_lines': _count_lines(data / 'chat_history.jsonl'),
            'self_talk_lines': _count_lines(data / 'autonomous_history.jsonl'),
            'persona_file_present': (data / 'persona.md').is_file(),
            'core_file_present': (data / 'core_memory.md').is_file(),
            'model_env_file_present': (project / '.env').is_file(),
        },
        'retrieval_settings': _safe_settings(_small_json_object(data / 'memory_retrieval.json'), RETRIEVAL_FIELDS),
        'auto_archive_settings': _safe_settings(_small_json_object(data / 'auto_archive_settings.json'), AUTO_FIELDS),
        'database': _db_counts(data / 'memory.db'),
        'recent_recall_metrics': _recall_statistics(data / 'logs' / 'memory_recall.jsonl'),
        'share_note': 'Only counts, allowlisted settings, package versions and coarse system info are included. Review this JSON before posting publicly.',
    }
    return result


def write_diagnostic_report(data_dir: str | Path, project_dir: str | Path, output: Path | None = None) -> Path:
    """Write diagnostic JSON atomically; never overwrite an existing file."""
    from datetime import datetime, timezone
    import secrets
    data = Path(data_dir)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        output = data / 'diagnostics' / f'winefox-diagnostic-{stamp}-{secrets.token_hex(3)}.json'
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('诊断文件已存在，不会覆盖。')
    report = build_diagnostic_report(data, project_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return output
