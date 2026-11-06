"""v0.3b lightweight, budgeted recall over approved memories and episodes.

This module never deletes or edits the source archive. Optional local embeddings
are opt-in and never transmit texts to a third-party embedding endpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import random
import struct
import re
import threading
from pathlib import Path

from .long_memory import now
from .episode_memory import local_iso, utc_iso


@dataclass
class RetrievalSettings:
    enabled: bool = True
    memory_budget_tokens: int = 6000
    max_input_tokens: int = 24000
    core_budget_tokens: int = 600
    directory_budget_tokens: int = 3000
    detail_budget_tokens: int = 2200
    episode_cards_target: int = 24
    episode_cards_max: int = 64
    episode_detail_max: int = 3
    cand_limit: int = 8
    # 0 minima: never pad ordinary conversation with irrelevant history.
    episode_cards_min: int = 0
    cand_min: int = 0
    episode_relevance_threshold: float = 0.0
    cand_relevance_threshold: float = 0.055
    detail_relevance_threshold: float = 0.015
    spontaneous_recall_count: int = 1
    recent_chat_messages: int = 200
    recent_chat_budget_tokens: int = 12000
    semantic_weight: float = 0.65
    recency_weight: float = 0.20
    importance_weight: float = 0.15
    mention_weight: float = 0.0
    spontaneous_recall_probability: float = 0.08
    recall_cooldown_turns: int = 6
    session_activation_turns: int = 4
    semantic_backend: str = "auto"  # auto=local cached model or lexical fallback
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    raw_excerpt_limit: int = 3
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    # v0.3b.4: retrieve from the current message AND a bounded recent topic.
    context_query_messages: int = 6
    context_query_max_chars: int = 380
    context_weight_normal: float = 0.25
    context_weight_followup: float = 0.70
    # Optional model-chosen local detail read. No extra model call unless used.
    episode_tool_enabled: bool = True
    episode_tool_max_items: int = 4
    episode_tool_extra_cards: int = 3  # low-relevance association seeds, only titles
    episode_tool_budget_tokens: int = 3000
    core_auto_enabled: bool = False  # explicit opt-in: triggers paid API calls on approvals
    core_auto_min_new: int = 3
    core_auto_batch_limit: int = 24
    core_auto_max_items: int = 8
    core_auto_text_budget_tokens: int = 500

    @classmethod
    def load(cls, data_dir: str | Path):
        path = Path(data_dir) / 'memory_retrieval.json'
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('记忆召回配置必须为 JSON 对象')
        return cls.validate(data)

    @classmethod
    def validate(cls, overrides):
        if not isinstance(overrides, dict):
            raise ValueError('配置必须为 JSON 对象')
        allowed = {f.name for f in fields(cls)}
        extra = set(overrides) - allowed
        if extra:
            raise ValueError('未知记忆配置项：' + ', '.join(sorted(extra)))
        default = asdict(cls())
        value = {}
        for name, old in default.items():
            new = overrides.get(name, old)
            if type(old) is bool:
                if type(new) is not bool: raise ValueError(name+' 必须为布尔值')
            elif type(old) is int:
                if type(new) is not int: raise ValueError(name+' 必须为整数')
            elif type(old) is float:
                if type(new) not in (int, float) or not math.isfinite(new): raise ValueError(name+' 必须为有限数字')
                new = float(new)
            elif type(old) is str:
                if not isinstance(new, str): raise ValueError(name+' 必须为字符串')
            value[name] = new
        c = cls(**value)
        if c.semantic_backend not in ('auto', 'lexical', 'sentence_transformers'):
            raise ValueError('semantic_backend 应为 auto、lexical、sentence_transformers')
        if not re.fullmatch(r'[\w./:\\-]{1,200}', c.embedding_model):
            raise ValueError('embedding_model 不能包含空白或控制字符')
        if len(c.query_instruction)>200: raise ValueError('query_instruction 最多 200 字符')
        for k, low, high in [
            ('memory_budget_tokens', 100, 200000), ('max_input_tokens', 1000, 1000000), ('core_budget_tokens', 0, 50000),
            ('directory_budget_tokens', 100, 100000), ('detail_budget_tokens', 0, 100000),
            ('episode_cards_target', 1, 512), ('episode_cards_max', 1, 512),
            ('episode_detail_max', 0, 30), ('cand_limit', 0, 100),
            ('episode_cards_min',0,512),('cand_min',0,100),('spontaneous_recall_count',0,30),
            ('recent_chat_messages',0,5000),('recent_chat_budget_tokens',0,200000),
            ('recall_cooldown_turns', 0, 1000), ('session_activation_turns', 0, 100),
            ('raw_excerpt_limit', 0, 15),
            ('core_auto_min_new',1,200),('core_auto_batch_limit',1,100),
            ('core_auto_max_items',1,30),('core_auto_text_budget_tokens',100,20000),
            ('context_query_messages',0,20),('context_query_max_chars',0,800),
            ('episode_tool_max_items',0,4),('episode_tool_extra_cards',0,20),
            ('episode_tool_budget_tokens',0,10000)]:
            if not low <= getattr(c,k) <= high: raise ValueError(f'{k} 允许范围 {low}～{high}')
        if c.memory_budget_tokens > c.max_input_tokens: raise ValueError('长期记忆预算不能超过总输入预算')
        if c.episode_cards_target > c.episode_cards_max: raise ValueError('target 不可大于 max')
        if c.episode_cards_min > c.episode_cards_target: raise ValueError('episode_cards_min 不可大于目标卡数')
        if c.cand_min > c.cand_limit: raise ValueError('cand_min 不可大于 cand_limit')
        for name in ('semantic_weight','recency_weight','importance_weight','mention_weight','spontaneous_recall_probability', 'episode_relevance_threshold',
                     'cand_relevance_threshold', 'detail_relevance_threshold',
                     'context_weight_normal', 'context_weight_followup'):
            if not 0 <= getattr(c,name) <= 1: raise ValueError(name+' 应在 0～1 之间')
        if not c.semantic_weight+c.recency_weight+c.importance_weight+c.mention_weight>0:
            raise ValueError('排序权重不能全部为零')
        return c

    def save(self, data_dir):
        target = Path(data_dir) / 'memory_retrieval.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        tmp = target.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(target)
        return target


def estimated_tokens(text):
    """Conservative character-based *estimate*, not provider billing tokens."""
    return math.ceil(len(str(text)) * 1.1)


def _slice_budget(text, available):
    if available <= 0: return ''
    return text[: max(0, int(available / 1.1))]


def _features(text):
    t = str(text).casefold()
    out = set()
    for w in re.findall(r'[\u4e00-\u9fff]+|[a-z0-9_]+', t):
        if re.fullmatch(r'[\u4e00-\u9fff]+',w):
            out.update(w[i:i+n] for n in (2,3) for i in range(len(w)-n+1))
            if len(w) <= 4: out.add(w)
        elif len(w)>2:
            out.add(w)
    return out



def _explicit_topic_mention(user_text, memory_text):
    """Conservative heuristic: user explicitly repeats a distinctive phrase.

    Memory retrieval, context text and model replies do NOT count as mentions.
    A false negative is preferable to a self-reinforcing hot-memory loop.
    """
    u=str(user_text).casefold();m=str(memory_text).casefold()
    if not u or not m:return False
    stop={'主人','酒狐','主人说','酒狐说','你们','我们','之前','以前','现在','可以','已经','这个','那个',
          '知道','觉得','事情','为什么','什么','这样','时候','一次','记忆','回忆','对话',
          '的事情','的时候','我们之','们之前','前是不','是不是','有过类','过类似','类似的','这件事'}
    # Exact four-character+ Chinese sequences are strong signals. Short unique
    # Three-character topic anchors are accepted only when the entire
    # matched token is three characters; this is a generic rule.
    for token in re.findall(r'[\u4e00-\u9fff]{3,}|[a-z0-9_]{4,}',u):
        if token in stop:continue
        if token in m and len(token)>=3:return True
        if re.fullmatch(r'[\u4e00-\u9fff]+',token):
            for n in (4,3):
                for i in range(len(token)-n+1):
                    piece=token[i:i+n]
                    if piece not in stop and piece in m and (n==4 or (n==3 and token==piece)):return True
    return False

def _cos(a,b):
    return sum(x*y for x,y in zip(a,b)) / max(1e-9, math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b)))


def _trusted_time(value):
    """Parse an offset-aware timestamp only; never guess a timezone."""
    return utc_iso(value)


def _time_span(start, end):
    """Source timestamp range rendered in Beijing time, or None if unknown.

    This refers to the recorded conversation, NOT necessarily when a fact
    became true or when an event occurred in the fictional/game world.
    """
    times = sorted({t for t in (_trusted_time(start), _trusted_time(end)) if t})
    if not times:
        return None
    labels = [local_iso(t) for t in (times[0], times[-1])]
    return labels[0] if labels[0] == labels[1] else labels[0] + ' ～ ' + labels[1]


def _memory_time_label(entry):
    """Use evidence time first. Archive creation is explicitly not event time."""
    span = _time_span(entry.get('time_start'), entry.get('time_end'))
    if span:
        return '来源对话时间（北京时间）：' + span
    recorded = local_iso(entry.get('recorded_at'))
    if recorded:
        return '仅知收录时间（非事件发生时间，北京时间）：' + recorded
    return '时间未知'


def _source_time_ranges(con):
    """Recover timestamps from original cited messages, not archive creation.

    Multiple Cand can refer to the same extraction batch. Decode its saved
    input only once; retrieval must remain affordable with a large archive.
    """
    rows=con.execute('''SELECT m.id, m.evidence_ids, m.event_time, a.run_id
        FROM memories m
        LEFT JOIN candidate_audit a ON a.candidate_id=m.origin_candidate_id
        WHERE m.status='active' AND a.run_id IS NOT NULL''').fetchall()
    eligible=[r for r in rows if not _trusted_time(r['event_time'])]
    run_ids=sorted({r['run_id'] for r in eligible})
    index={}
    for pos in range(0,len(run_ids),300):
        group=run_ids[pos:pos+300]
        if not group:
            continue
        placeholders=','.join('?' for _ in group)
        for run in con.execute(f'SELECT id,input_payload FROM extraction_runs WHERE id IN ({placeholders})',group):
            try:
                records=json.loads(run['input_payload']).get('messages',[])
                index[run['id']]={item.get('id'):_trusted_time(item.get('event_time') or item.get('created_at'))
                                  for item in records if isinstance(item,dict)}
            except (TypeError, ValueError, AttributeError):
                continue
    result={}
    for r in eligible:
        try:
            ids=json.loads(r['evidence_ids'])
            recorded=index.get(r['run_id'],{})
            times=sorted({recorded[i] for i in ids if recorded.get(i)})
        except (TypeError, ValueError, KeyError):
            continue
        if times:
            result[r['id']]=(times[0],times[-1])
    return result


def _elapsed_days(iso):
    if not iso: return None
    try:
        t = datetime.fromisoformat(str(iso).replace('Z','+00:00'))
        if t.tzinfo is None: return None
        return max(0., (datetime.now(timezone.utc)-t.astimezone(timezone.utc)).total_seconds()/86400)
    except (ValueError, TypeError):
        return None


class MemoryRetriever:
    """Session-scoped recall, persistent metadata and per-turn introspection.

    Previews do not mutate counters. Only successful model replies call commit().
    """
    def __init__(self, core):
        self.core = core
        self.data_dir = core.root
        self._lock = threading.RLock()
        self._embedder = None
        self._embed_cache = {}
        self._failed_model = ''
        self._curator_instance = None
        self._session_scores = {}
        self._turn = 0
        self._last_spontaneous = -10000
        self._session_seed = random.randrange(1<<40)
        from .memory_verification import VerifiedMemoryDatabase
        # Construct once per session instead of migrating/inspecting SQLite
        # on every retrieval, important when the archive becomes large.
        self._db_instance = VerifiedMemoryDatabase(self.data_dir / 'memory.db')
        self._ensure_schema()

    def _db(self):
        return self._db_instance

    def _ensure_schema(self):
        db=self._db()
        with db.connect() as con:
            con.executescript('''CREATE TABLE IF NOT EXISTS retrieval_metadata (
                kind TEXT NOT NULL, item_id TEXT NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                importance_override REAL,
                mention_count INTEGER NOT NULL DEFAULT 0,
                recall_count INTEGER NOT NULL DEFAULT 0,
                last_mentioned_at TEXT,
                last_recalled_at TEXT,
                PRIMARY KEY(kind,item_id)
            );
            CREATE TABLE IF NOT EXISTS embedding_vectors (
                model TEXT NOT NULL, kind TEXT NOT NULL, item_id TEXT NOT NULL,
                content_sha TEXT NOT NULL, dims INTEGER NOT NULL,
                vec BLOB NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(model,kind,item_id,content_sha)
            );
            CREATE INDEX IF NOT EXISTS idx_embed_model ON embedding_vectors(model);
            ''')

    def get_settings(self): return RetrievalSettings.load(self.data_dir)

    def update_settings(self, changes):
        previous=asdict(self.get_settings())
        c=previous.copy(); c.update(changes)
        valid=RetrievalSettings.validate(c)
        valid.save(self.data_dir)
        history=self.data_dir/'logs'/'retrieval_config_history.jsonl'
        history.parent.mkdir(parents=True,exist_ok=True)
        with self._lock, history.open('a',encoding='utf-8') as fp:
            fp.write(json.dumps({'at':now(),'previous':previous,'current':asdict(valid)},ensure_ascii=False)+'\n')
        return asdict(valid)

    def manual_core_text(self):
        p=self.data_dir/'core_memory.md'
        if p.exists():return p.read_text(encoding='utf-8'), 'core_memory.md'
        return self.core.read_memory_document(), 'memory.md (legacy fallback)'

    def auto_core_entries(self):
        # Import lazily to avoid a module cycle; core curator is independent of
        # core_text retrieval, and does not call itself recursively.
        if self._curator_instance is None:
            from .core_curator import CoreCurator
            self._curator_instance=CoreCurator(self)
        entries=self._curator_instance.entries()
        if not entries:return entries
        # A human rejection of an auto-published source immediately hides any
        # AI Core item derived from it; its historical version is still archived.
        with self._db().connect() as con:
            valid_c={'cand:'+x[0] for x in con.execute("SELECT id FROM memories WHERE status='active'")}
            valid_e={'episode:'+x[0] for x in con.execute("SELECT id FROM extracted_episodes WHERE review_status IN ('approved','auto_active')")}
        allowed=valid_c|valid_e
        return [e for e in entries if e['source_ids'] and all(i in allowed for i in e['source_ids'])]

    def core_text(self):
        manual,source=self.manual_core_text()
        auto=self.auto_core_entries()
        if not auto:return manual,source
        section='重要共同经历与长期认识（AI 整理，非人格命令）：\n'+\
                '\n'.join('- '+x['text'] for x in auto)
        combined=(manual.rstrip()+'\n\n' if manual.strip() else '')+section
        return combined,source+' + AI Core'

    def save_core_text(self, value):
        if not isinstance(value,str) or len(value)>25000: raise ValueError('Core Memory 最多 25000 字符')
        p=self.data_dir/'core_memory.md'
        with self._lock:
            existing,origin=self.manual_core_text()
            if value==existing and p.exists():return {'file':str(p),'characters':len(value),'changed':False}
            # Snapshot the previous content even if it was legacy memory.md.
            history=self.data_dir/'core_memory_history'
            history.mkdir(parents=True,exist_ok=True)
            version=now().replace(':','-').replace('+','_')+'_'+sha256(existing.encode()).hexdigest()[:12]+'.md'
            (history/version).write_text(existing,encoding='utf-8')
            tmp=p.with_suffix('.md.tmp');tmp.write_text(value,encoding='utf-8');tmp.replace(p)
        return {'file':str(p),'characters':len(value),'changed':True,'previous_version':version}

    def core_versions(self):
        folder=self.data_dir/'core_memory_history'
        if not folder.exists():return []
        return [p.name for p in sorted(folder.glob('*.md'),reverse=True)[:100]]

    def restore_core(self,version):
        if not isinstance(version,str) or not re.fullmatch(r'[0-9A-Za-z_.-]+\.md',version):
            raise ValueError('非法历史版本')
        p=self.data_dir/'core_memory_history'/version
        if not p.is_file():raise ValueError('找不到此历史版本')
        return self.save_core_text(p.read_text(encoding='utf-8'))

    def metadata(self, kind, item_id):
        if kind not in ('cand','episode','legacy'):raise ValueError('无效记忆类型')
        db=self._db()
        with db.connect() as con:
            r=con.execute('SELECT * FROM retrieval_metadata WHERE kind=? AND item_id=?',(kind,item_id)).fetchone()
        return dict(r) if r else {'kind':kind,'item_id':item_id,'pinned':0,'importance_override':None,
                                   'mention_count':0,'recall_count':0,'last_mentioned_at':None,'last_recalled_at':None}

    def set_metadata(self, kind, item_id, *, pinned, importance_override=None):
        if kind not in ('cand','episode','legacy'):raise ValueError('无效记忆类型')
        if type(pinned) is not bool:raise ValueError('pinned 必须为 true 或 false')
        if importance_override is not None:
            if type(importance_override) not in (int,float) or not 0<=importance_override<=1:
                raise ValueError('重要性覆盖值应在 0～1 内')
        db=self._db()
        with db.connect() as con:
            if kind=='cand': exists=con.execute("SELECT 1 FROM memories WHERE id=? AND status='active'",(item_id,)).fetchone()
            elif kind=='episode': exists=con.execute("SELECT 1 FROM extracted_episodes WHERE id=? AND review_status IN ('approved','auto_active')",(item_id,)).fetchone()
            else: exists=any(x['id']==item_id for x in self.core.list_memories())
            if not exists:raise ValueError('只能调整已批准/已保存的记忆')
            con.execute('''INSERT INTO retrieval_metadata(kind,item_id,pinned,importance_override)
             VALUES(?,?,?,?) ON CONFLICT(kind,item_id) DO UPDATE SET pinned=excluded.pinned,
             importance_override=excluded.importance_override''',(kind,item_id,int(pinned),importance_override))
        return self.metadata(kind,item_id)

    def _entries(self):
        db=self._db(); meta={}
        with db.connect() as con:
            meta={(x['kind'],x['item_id']):dict(x) for x in con.execute('SELECT * FROM retrieval_metadata')}
            eps=[dict(e) for e in con.execute("""SELECT id,title,summary,detail,event_status,
                time_start_utc,time_end_utc FROM extracted_episodes
                WHERE review_status IN ('approved','auto_active')""")]
            cand_spans=_source_time_ranges(con)
        out=[]
        for m in db.all_memories():
            start,end=cand_spans.get(m['id'], (m.get('event_time'),m.get('event_time')))
            if _trusted_time(m.get('event_time')):
                start=end=m['event_time']
            entry={'kind':'cand','id':m['id'],'title':m['category'],
             'content':m['content'],'search':m['content']+' '+m.get('fact_key',''),
             'importance':float(m.get('importance') or .5),'event_time':m.get('event_time'),
             'time_start':start,'time_end':end,'recorded_at':m.get('created_at'),
             'metadata':meta.get(('cand',m['id']),{}),'detail':'', 'category':m['category']}
            entry['time_label']=_memory_time_label(entry)
            out.append(entry)
        for e in eps:
            entry={'kind':'episode','id':e['id'],'title':e['title'],
             'content':e['summary'],'search':' '.join((e['title'],e['summary'])),
             'importance':.55,'event_time':e['time_end_utc'],'metadata':meta.get(('episode',e['id']),{}),
             'time_start':e['time_start_utc'],'time_end':e['time_end_utc'], 'recorded_at':None,
             'detail':e['detail'],'category':e['event_status']}
            entry['time_label']=_memory_time_label(entry)
            out.append(entry)
        for m in self.core.list_memories():
            entry={'kind':'legacy','id':m['id'],'title':m.get('category','fact'),
             'content':m['content'],'search':m['content']+' '+' '.join(m.get('tags',[])),
             'importance':.55,'event_time':m.get('created_at'),'metadata':meta.get(('legacy',m['id']),{}),
             'time_start':None,'time_end':None,'recorded_at':m.get('created_at'),
             'detail':'','category':m.get('category','fact')}
            entry['time_label']=_memory_time_label(entry)
            out.append(entry)
        return out

    def embedding_index_stats(self):
        with self._db().connect() as con:
            count=con.execute('SELECT COUNT(*) FROM embedding_vectors').fetchone()[0]
            models=[dict(r) for r in con.execute(
                'SELECT model,COUNT(*) AS records FROM embedding_vectors GROUP BY model')]
        settings=self.get_settings()
        active=self._entries()
        active_keys={(e['kind'],e['id'],sha256(e['search'].encode()).hexdigest()) for e in active}
        with self._db().connect() as con:
            indexed={(r['kind'],r['item_id'],r['content_sha']) for r in con.execute(
                'SELECT kind,item_id,content_sha FROM embedding_vectors WHERE model=?',
                (settings.embedding_model,))}
        valid=len(active_keys & indexed)
        return {'stored_vectors':count,'models':models,
                'eligible_active_records':len(active_keys),
                'valid_for_configured_model':valid,
                'missing_or_outdated':len(active_keys)-valid,
                'configured_model':settings.embedding_model,
                'loaded_model':self._failed_model if self._embedder is not None else None,
                'cache_note':'索引按需计算：未检索过的条目可以没有向量；持久 SQLite BLOB 按模型、条目 ID 和正文 SHA 失效。'}

    def _embeddings(self, queries, entries, settings):
        backend=settings.semantic_backend
        if backend=='lexical':return None,'lexical only'
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            if self._embedder is None or self._failed_model!=settings.embedding_model:
                # Never silently download model weights on normal chat startup.
                self._embedder=SentenceTransformer(settings.embedding_model,local_files_only=True)
                self._embed_cache={};self._failed_model=settings.embedding_model
            model=settings.embedding_model
            keys=[(r['kind'],r['id'],sha256(r['search'].encode()).hexdigest()) for r in entries]
            missing=[(r,key) for r,key in zip(entries,keys) if key not in self._embed_cache]
            if missing:
                with self._db().connect() as con:
                    for r,key in missing:
                        hit=con.execute('''SELECT dims,vec FROM embedding_vectors
                            WHERE model=? AND kind=? AND item_id=? AND content_sha=?''',
                            (model,*key)).fetchone()
                        if hit and len(hit['vec'])==int(hit['dims'])*4:
                            self._embed_cache[key]=np.frombuffer(hit['vec'],dtype='<f4').copy()
                uncached=[(r,key) for r,key in missing if key not in self._embed_cache]
                # BGE small has a short input limit; for episode indexes use
                # title + summary only instead of truncating mid-detail.
                for start in range(0,len(uncached),32):
                    batch=uncached[start:start+32]
                    vecs=self._embedder.encode([r['search'][:1200] for r,key in batch],
                                               normalize_embeddings=True)
                    with self._db().connect() as con:
                        for (r,key),v in zip(batch,vecs):
                            arr=np.asarray(v,dtype='<f4').reshape(-1)
                            self._embed_cache[key]=arr
                            con.execute('''INSERT OR REPLACE INTO embedding_vectors
                                (model,kind,item_id,content_sha,dims,vec,updated_at)
                                VALUES(?,?,?,?,?,?,?)''',
                                (model,*key,int(arr.size),arr.tobytes(),now()))
                            # Trim outdated versions for the same memory and model.
                            con.execute('''DELETE FROM embedding_vectors WHERE model=? AND kind=?
                                AND item_id=? AND content_sha<>?''',(model,*key))
            # Encode two short queries together; a long concatenated query would
            # be truncated by the embedding model and lose the current topic.
            query_vecs=self._embedder.encode(
                [settings.query_instruction+q for q in queries],normalize_embeddings=True)
            result=[]
            for qv in query_vecs:
                vec=np.asarray(qv)
                result.append({(r['kind'],r['id']):float(np.dot(vec,self._embed_cache[key]))
                               for r,key in zip(entries,keys)})
            return result, 'local sentence-transformers (persistent SQLite)'
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return None, 'lexical fallback: '+type(exc).__name__

    def prepare(self, question, *, context_query="", settings=None, preview=False, fixed_seed=None):
        settings=settings or self.get_settings()
        if not settings.enabled:return {'memory_text':'','core_text':'','cards':[], 'cand':[], 'details':[], 'selected':[],
                                         'estimated_tokens':0,'backend':'off','settings':asdict(settings),'question':question}
        entries=self._entries()
        with self._lock:
            turn=self._turn
            seed=fixed_seed if fixed_seed is not None else self._session_seed+turn
        context_query=str(context_query or '')[:settings.context_query_max_chars].strip()
        # Elliptical follow-ups rely more on the preceding exchange. A concrete
        # current query remains the primary signal otherwise.
        followup=bool(re.search(r'类似|刚才|之前.*(这种|这样|类似|说过)|这件事|那件事|这个|那个|还记得|是不是.*(过|之前)',question))
        context_weight=(settings.context_weight_followup if followup else
                        settings.context_weight_normal) if context_query else 0.
        queries=[question]+([context_query] if context_query else [])
        embedding_maps,backend=self._embeddings(queries,entries,settings)
        # Precompute lexical IDF separately for each query, preserving short
        # exact names while retaining topic continuity across turns.
        query_features=[_features(q) for q in queries]
        entry_features={(e['kind'],e['id']):_features(e['search']) for e in entries}
        lexical_maps=[]
        for qf in query_features:
            docfreq={t:sum(t in fs for fs in entry_features.values()) for t in qf}
            norm=sum(math.log(1+(len(entries)+1)/(1+docfreq[x])) for x in qf) if qf else 0
            lexical_maps.append({key:sum(math.log(1+(len(entries)+1)/(1+docfreq[x]))
                                         for x in (qf & fs))/max(.001,norm)
                                 for key,fs in entry_features.items()})
        weights=settings.semantic_weight+settings.recency_weight+settings.importance_weight+settings.mention_weight
        for e in entries:
            key=(e['kind'],e['id'])
            scores=[]
            for k in range(len(queries)):
                lexical=lexical_maps[k][key]
                semantic=max(0.,embedding_maps[k][key]) if embedding_maps is not None else lexical
                scores.append(.78*semantic+.22*lexical if embedding_maps is not None else lexical)
            relevance=(1-context_weight)*scores[0]+(context_weight*scores[1] if len(scores)>1 else 0)
            meta=e['metadata']
            dt=_elapsed_days(meta.get('last_mentioned_at') or e['event_time'])
            recency=math.exp(-dt/30) if dt is not None else .35
            imp=meta.get('importance_override')
            imp=float(imp) if imp is not None else e['importance']
            pins=bool(meta.get('pinned'))
            mentions=min(1.,math.log1p(int(meta.get('mention_count') or 0))/math.log1p(20))
            association=self._session_scores.get(key,(-1,0))
            activation=.08 if association[0]>=turn else 0.
            base=(settings.semantic_weight*relevance+settings.recency_weight*recency+
                  settings.importance_weight*imp+settings.mention_weight*mentions)/weights
            e['score']=base+(.12 if pins else 0)+activation
            e['relevance']=round(relevance,4)
            e['recency']=round(recency,4)
            e['pinned']=pins
            e['importance']=imp
            e['components']={'current':round(scores[0],4),'context':round(scores[1],4) if len(scores)>1 else None,
                             'blended_relevance':round(relevance,4),'recency':round(recency,4),
                             'importance':round(imp,4),'mentions':round(mentions,4),
                             'pinned_bonus':.12 if pins else 0.,'session_bonus':activation,
                             'total':round(e['score'],4)}
        # High relevance first, then a bounded random associative card. Never
        # treat randomness as factual verification.
        entries.sort(key=lambda e:(-e['score'],e['kind'],e['id']))
        eps=[e for e in entries if e['kind']=='episode']
        facts=[e for e in entries if e['kind']!='episode']
        with self._lock:
            can_spontaneous = turn-self._last_spontaneous > settings.recall_cooldown_turns
        rng=random.Random(seed)
        spontaneous=(len(eps)>1 and can_spontaneous and
                     rng.random()<settings.spontaneous_recall_probability)
        directory=[];cand=[];expanded=[];selected=[]
        core,core_source=self.core_text()
        # Core is meant to travel together. The core_budget_tokens is a SOFT
        # preferred allocation: borrow from Archive space when the entire Core
        # still fits the hard memory_budget_tokens. Do not silently sacrifice
        # the AI Core just because a long manual Core was placed first.
        raw_core=core.strip()
        core_needed=estimated_tokens(raw_core)
        core_hard_limit=max(0,settings.memory_budget_tokens-100)
        core_truncated=core_needed>core_hard_limit
        core=_slice_budget(raw_core,core_hard_limit) if core_truncated else raw_core
        core_over_target=core_needed>settings.core_budget_tokens
        remaining=max(0,settings.memory_budget_tokens-estimated_tokens(core)-100)
        directory_limit=min(settings.directory_budget_tokens,remaining)
        card_count=min(settings.episode_cards_target,settings.episode_cards_max)
        # Threshold is on actual relevance, not total score (which also includes
        # recency and salience). An optional minimum may admit lower relevance.
        main_eps=[e for e in eps if e['relevance'] >= settings.episode_relevance_threshold]
        if len(main_eps)<settings.episode_cards_min:
            main_eps=eps[:settings.episode_cards_min]
        card_candidates=list(main_eps[:card_count])
        random_selected=[]
        if spontaneous and settings.spontaneous_recall_count and len(eps)>len(card_candidates):
            # Explore non-top but still meaningfully connected cards. Random
            # additions are distinct, budgeted, and never override the top cards.
            current={e['id'] for e in card_candidates}
            pool=[e for e in eps if e['id'] not in current and e['relevance']>0]
            max_extra=min(settings.spontaneous_recall_count,
                          max(0,settings.episode_cards_max-len(card_candidates)),len(pool))
            if max_extra:
                # Prefer less salient candidates, while allowing different seeds
                # to generate different associative memories.
                random_selected=rng.sample(pool,max_extra)
                card_candidates+=random_selected
        # The budget is the primary bound, not the number of cards.
        card_lines=[]
        for e in card_candidates:
            line=f"- [{e['id']}] [{e['time_label']}] {e['title']}：{e['content'][:85]}"
            if estimated_tokens('\n'.join(card_lines+[line]))>directory_limit:break
            card_lines.append(line);directory.append(e)
        used=estimated_tokens('\n'.join(card_lines));remaining-=used
        # A few weakly associated titles make voluntary recollection possible
        # even for a new topic. Never auto-expand them or promote to Core.
        tool_suggestions=[];tool_lines=[]
        if settings.episode_tool_enabled and settings.episode_tool_extra_cards and settings.episode_tool_max_items:
            regular={e['id'] for e in directory}
            pool=[e for e in eps if e['id'] not in regular]
            for e in rng.sample(pool,min(len(pool),settings.episode_tool_extra_cards)):
                line=f"- [{e['id']}] [{e['time_label']}] {e['title']}"
                room=max(0,directory_limit-used-estimated_tokens('\n'.join(tool_lines+[line])))
                if room<=0:break
                tool_suggestions.append(e);tool_lines.append(line)
            remaining-=estimated_tokens('\n'.join(tool_lines))
        # Explicit information questions can receive focused details. Casual
        # conversation only gets cards and short facts to avoid forced nostalgia.
        detail_request=bool(re.search(r'还记得|之前|当时|以前|怎么回事|为什么|那次|详细|原话|逐字|经历|回忆|工具结果|工具执行',question))
        selected_eps=[]
        if detail_request:
            # prefer strong relevance; low-scoring/irrelevant episodes aren't
            # expanded merely to fill the token budget.
            selected_eps=[e for e in eps if e['relevance']>=settings.detail_relevance_threshold][:settings.episode_detail_max]
        detail_limit=min(settings.detail_budget_tokens,remaining)
        detail_lines=[]
        for e in selected_eps:
            line=f"- [{e['id']}] [{e['time_label']}] {e['title']}：{e['detail'][:1800]}"
            if estimated_tokens('\n'.join(detail_lines+[line]))>detail_limit:continue
            detail_lines.append(line);expanded.append(e)
        remaining-=estimated_tokens('\n'.join(detail_lines))
        # Verbatim/actual tool questions: source excerpts are clearly labeled.
        exact_request=bool(re.search(r'原话|逐字|哪句话|说过什么|怎么说的|工具结果|工具执行',question))
        if exact_request and settings.raw_excerpt_limit and expanded:
            db=self._db()
            for e in expanded[:1]:
                source=db.get_episode(e['id'],core=self.core,include_raw=True).get('raw_dialogue',[])
                focus=_features(question)-_features('原话逐字哪句话说过什么工具结果工具执行')
                source.sort(key=lambda r:-len(focus & _features(r.get('content',''))))
                raw=[]
                for r in source[:settings.raw_excerpt_limit]:
                    role=r.get('role','unknown')
                    message=str(r.get('content',''))[:420]
                    mark='证据' if r.get('cited_as_evidence') else '邻近背景'
                    line=f"- [{r.get('id','')}; {role}; {mark}; {('来源消息时间（北京时间）：'+r['event_time_local']) if r.get('event_time_local') else '时间未知'}] {message}"
                    if estimated_tokens('\n'.join(detail_lines+raw+[line]))>settings.detail_budget_tokens:break
                    raw.append(line)
                if raw:
                    detail_lines+=['原始记录摘录（非事件摘要，可能需要逐条核实）：']+raw
                    break
        fact_lines=[]
        # Always permit a few pinned items when they are relevant, but don't
        # flood chat with old unprompted relationship statements.
        for e in facts:
            if len(cand)>=settings.cand_limit:break
            if (e['relevance'] < settings.cand_relevance_threshold
                    and len(cand)>=settings.cand_min
                    and not (e['pinned'] and e['relevance']>0)):continue
            line=f"- [{e['id']}] [{e['time_label']}] {e['content'][:440]}"
            if estimated_tokens('\n'.join(fact_lines+[line]))>remaining:break
            fact_lines.append(line);cand.append(e)
        blocks=[]
        if card_lines:blocks.append('历史事件目录（必要时参考，不要求主动提及）：\n'+'\n'.join(card_lines))
        if tool_lines:blocks.append('自由联想的其他旧事（只提供标题；若想了解才主动展开）：\n'+'\n'.join(tool_lines))
        if fact_lines:blocks.append('本轮相关的长期事实（可能过时）：\n'+'\n'.join(fact_lines))
        if detail_lines:blocks.append('选中事件的详细经过：\n'+'\n'.join(detail_lines))
        body='\n\n'.join(blocks)
        # Expose the same strict memory budget to every caller. Token counts are
        # conservative estimates; reserve headroom for section names and metadata.
        remaining_total=max(0,settings.memory_budget_tokens-estimated_tokens(core))
        body=_slice_budget(body,remaining_total)
        selected=[
                   {'kind':e['kind'],'id':e['id'],'score':round(e['score'],4),'relevance':e['relevance'],
                    'pinned':e['pinned'],'title':e['title'],'time_label':e['time_label'],'components':e['components']} for e in (directory+cand+expanded)]
        return {'question':question,'query_current':question,'query_context':context_query,
                '_mention_search':{e['kind']+':'+e['id']:e['search'] for e in (directory+cand+expanded)},
                'context_weight':context_weight,'memory_text':body,'core_text':core,'core_source':core_source,
                'core_truncated':core_truncated,'core_over_target':core_over_target,
                'core_requested_tokens':core_needed,
                'cards':[{'id':e['id'],'title':e['title'],'time_label':e['time_label'],'score':round(e['score'],4)} for e in directory],
                'cand':[{'id':e['id'],'content':e['content'],'time_label':e['time_label']} for e in cand],
                'details':[{'id':e['id'],'title':e['title'],'time_label':e['time_label']} for e in expanded],
                'selected':selected,'estimated_tokens':estimated_tokens(core)+estimated_tokens(body),
                'memory_budget_tokens':settings.memory_budget_tokens,'backend':backend,
                'spontaneous':bool(random_selected),'spontaneous_cards':[e['id'] for e in random_selected
                    if any(c['id']==e['id'] for c in directory)],
                'tool_available_episode_ids':[e['id'] for e in directory+tool_suggestions if e['id'] in body] if settings.episode_tool_enabled else [],
                'tool_candidate_cards':[{'id':e['id'],'title':e['title'],'time_label':e['time_label']} for e in tool_suggestions if e['id'] in body],
                'tool_expansions':[],
                'selection_policy':{'episode_threshold':settings.episode_relevance_threshold,
                    'cand_threshold':settings.cand_relevance_threshold,'detail_threshold':settings.detail_relevance_threshold,
                    'episode_min':settings.episode_cards_min,'cand_min':settings.cand_min,
                    'episode_count':len(directory),'cand_count':len(cand)},
                'seed':seed,'settings':asdict(settings)}

    def expand_episode_tool(self, requested_ids, prepared, *, max_items=None, budget_tokens=None):
        """Model-selected episode detail, limited to this turn's offered cards.

        No arbitrary IDs, no raw chat, no file access, no network, no mutation.
        Returns only currently active episodes (revocations take effect now).
        """
        settings=self.get_settings()
        limit=min(4,settings.episode_tool_max_items,max_items if max_items is not None else 4)
        max_budget=min(settings.episode_tool_budget_tokens,
                       max(0,settings.memory_budget_tokens-prepared.get('estimated_tokens',0)),
                       max(0,settings.max_input_tokens-prepared.get('estimated_total_input_tokens',0)-1000),
                       budget_tokens if budget_tokens is not None else settings.episode_tool_budget_tokens)
        allowed=set(prepared.get('tool_available_episode_ids',[]))
        if not isinstance(requested_ids,list):requested_ids=[]
        seen=set();output=[];used=0
        with self._db().connect() as con:
            for episode_id in requested_ids:
                if len(seen)>=limit:break
                if not isinstance(episode_id,str) or episode_id in seen or episode_id not in allowed:continue
                seen.add(episode_id)
                row=con.execute('''SELECT id,title,detail,time_start_utc,time_end_utc FROM extracted_episodes
                    WHERE id=? AND review_status IN ('approved','auto_active')''',(episode_id,)).fetchone()
                if not row or not row['detail']:continue
                time_label=_memory_time_label({'time_start':row['time_start_utc'],
                                               'time_end':row['time_end_utc']})
                head=f"[{row['id']}] [{time_label}] {row['title']}："
                room=max_budget-used-estimated_tokens(head)-50
                if room<=0:break
                detail=_slice_budget(row['detail'],min(room,1200))
                if not detail:break
                output.append({'id':episode_id,'title':row['title'],'time_label':time_label,'detail':detail})
                used+=estimated_tokens(head+detail)+50
        prepared.setdefault('tool_expansions',[]).extend({'id':r['id'],'tokens':estimated_tokens(r['detail'])} for r in output)
        return {'episodes':output,'expanded_count':len(output),'token_estimate':used,
                'note':'时间表示来源对话记录时间（北京时间），不必然等于事件真实发生时间；这些是历史摘要，可能不完整；当前工具不读取原始聊天。'}

    def commit(self, prepared, *, success=True):
        if not success:return
        stamp=now();selected={(r['kind'],r['id']) for r in prepared.get('selected',[])}
        selected.update(('episode',e['id']) for e in prepared.get('tool_expansions',[]))
        with self._lock:
            self._turn+=1
            if prepared.get('spontaneous'):self._last_spontaneous=self._turn
            # Activation is for *focused* memories, not for every card merely
            # visible in a broad directory (which would create feedback loops).
            focused={(r['kind'],r['id']) for r in prepared.get('selected',[])
                     if r['kind']!='episode'}
            focused.update(('episode',e['id']) for e in prepared.get('details',[]))
            focused.update(('episode',e['id']) for e in prepared.get('tool_expansions',[]))
            focused.update(('episode',e['id']) for e in prepared.get('cards',[])[:3])
            for key in focused:
                self._session_scores[key]=(self._turn+self.get_settings().session_activation_turns,1)
        if selected:
            db=self._db()
            with db.connect() as con:
                for kind,item_id in selected:
                    con.execute('''INSERT INTO retrieval_metadata(kind,item_id,recall_count,last_recalled_at)
                     VALUES(?,?,1,?) ON CONFLICT(kind,item_id) DO UPDATE SET recall_count=recall_count+1,
                     last_recalled_at=excluded.last_recalled_at''',(kind,item_id,stamp))
                    row=next((r for r in prepared.get('selected',[]) if r['kind']==kind and r['id']==item_id),{})
                    # Never count retrieval itself as an actual mention. Only
                    # a strong literal topic anchor in the USER'S current text
                    # may reinforce durable mention history. Still a heuristic.
                    if _explicit_topic_mention(prepared.get('question',''),
                                               prepared.get('_mention_search',{}).get(kind+':'+item_id,'')):
                        con.execute('''UPDATE retrieval_metadata SET mention_count=mention_count+1,
                          last_mentioned_at=? WHERE kind=? AND item_id=?''',(stamp,kind,item_id))
        self.log(prepared)

    def log(self,prepared):
        path=self.data_dir/'logs'/'memory_recall.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        row={'at':now(),'question':str(prepared.get('question',''))[:300],
             'backend':prepared.get('backend'),'query_context':prepared.get('query_context','')[:600],
             'context_weight':prepared.get('context_weight',0),'tool_expansions':prepared.get('tool_expansions',[]),
             'estimated_tokens':prepared.get('estimated_tokens',0),
             'estimated_total_input_tokens':prepared.get('estimated_total_input_tokens'),
             'recent_messages_injected':prepared.get('recent_messages_injected'),
             'memory_budget_tokens':prepared.get('memory_budget_tokens'),
             'core_truncated':prepared.get('core_truncated',False),
             'core_requested_tokens':prepared.get('core_requested_tokens'),
             'spontaneous':prepared.get('spontaneous',False),
             'spontaneous_cards':prepared.get('spontaneous_cards',[]),
             'selection_policy':prepared.get('selection_policy',{}),'seed':prepared.get('seed'),
             'selected':prepared.get('selected',[])}
        with self._lock:
            if path.exists() and path.stat().st_size>5_000_000:
                path.replace(path.with_name('memory_recall.previous.jsonl'))
            with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')

    def logs(self,limit=30):
        path=self.data_dir/'logs'/'memory_recall.jsonl'
        if not path.exists():return []
        lines=path.read_text(encoding='utf-8').splitlines()[-min(100,int(limit)):]
        result=[]
        for line in reversed(lines):
            try:result.append(json.loads(line))
            except ValueError:continue
        return result
