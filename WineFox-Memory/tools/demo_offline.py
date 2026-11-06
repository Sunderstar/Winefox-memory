"""Disposable synthetic two-pass memory demo. Zero external API requests.

Usage: python tools/demo_offline.py
Creates a temporary data directory and removes it automatically.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maidcore import MaidCore  # noqa: E402
from maidcore.auto_archive import AutoArchiveManager  # noqa: E402
from maidcore.improved_extraction import build_batches, run_extraction  # noqa: E402
from maidcore.memory_verification import VerifiedMemoryDatabase  # noqa: E402
from maidcore.retrieval import MemoryRetriever  # noqa: E402


class SyntheticModel:
    """Deterministic in-process stand-in; never calls an LLM provider."""
    config = SimpleNamespace(model='synthetic-offline', reasoning_effort='none', max_tokens=8192, timeout=10)
    last_usage = None

    def complete(self, messages):
        payload = json.loads(messages[-1]['content'].split('\n', 1)[1])
        ids = [row['id'] for row in payload['messages']]
        if 'episodes' not in payload:
            return json.dumps({'episodes': [{
                'id': 'e1', 'title': '参观虚构博物馆',
                'detail': '用户与助手讨论参观了一座虚构的博物馆和看到的恐龙化石。',
                'summary': '参观博物馆并讨论恐龙化石',
                'event_status': 'occurred', 'plans': [], 'evidence_ids': ids[:2],
            }]}, ensure_ascii=False)
        return json.dumps({'memories': [{
            'content': '用户曾参观虚构的蓝石博物馆，并喜欢地球科学展区。',
            'category': 'event', 'subject': 'user', 'fact_key': '',
            'importance': 0.7, 'confidence': 0.95,
            'evidence_ids': [ids[0], ids[4]], 'episode_ids': ['e1'],
        }]}, ensure_ascii=False)


def run_demo():
    fixture = json.loads((ROOT / 'examples' / 'fictional_conversation.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='winefox-offline-demo-') as temp:
        core = MaidCore(Path(temp) / 'data')
        for item in fixture:
            core.append_chat(item['role'], item['content'])
        db = VerifiedMemoryDatabase(core.root / 'memory.db')
        batch = build_batches(core, 'chat', max_messages=40, max_chars=8000)[0]
        run = run_extraction(db, batch, SyntheticModel(), mode='two-pass')
        published = AutoArchiveManager(core, db=db).publish_pending(dry_run=False)
        retriever = MemoryRetriever(core)
        retriever.update_settings({'semantic_backend': 'lexical', 'spontaneous_recall_probability': 0})
        recalled = retriever.prepare('我们以前参观过博物馆吗？')
        assert run.get('run_id')
        assert published['auto_episodes'] >= 1
        assert len(recalled['cards']) >= 1
        assert len(core.chat_history()) == len(fixture)
        # Only aggregate data, never a real data root or personal chats.
        return {'offline': True, 'chat_messages': len(fixture),
                'published_episode_count': published['auto_episodes'],
                'published_candidate_count': published['auto_candidates'],
                'retrieved_episode_count': len(recalled['cards']),
                'retrieved_candidate_count': len(recalled['cand']),
                'real_api_requests': 0, 'temporary_data_removed': True}


if __name__ == '__main__':
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2))
