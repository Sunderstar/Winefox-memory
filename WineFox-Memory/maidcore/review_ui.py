"""Local, dependency-free WineFox memory workbench (v0.3a.4).

Security and data boundary:
- HTTP server binds ONLY to 127.0.0.1, never a LAN/public address.
- All mutations require an unpredictable per-server header token and exact
  same-origin checks. No permissive CORS, cookies, or arbitrary file paths.
- Original chat JSONL and persona are READ-ONLY in this module.
- Edited test transcripts are immutable snapshots in SQLite, not edits to raw.
- UI experiments do not consume processed_records chronological checkpoints.
- Explicit auto-archive opt-in can publish unconfirmed memories; human can revoke.
"""
from __future__ import annotations

import json
import re
import secrets
import threading
import traceback
import datetime
import uuid
import webbrowser
from dataclasses import replace
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from .store import MaidCore
from .llm import ChatCompletionsClient, load_memory_config
from .long_memory import CATEGORIES, SUBJECTS, norm, canonical, sha256, now
from .improved_extraction import historical_records, known_memories_for_batch, run_extraction, VERSION
from .memory_verification import VerifiedMemoryDatabase, PLAN_STATES
from .retrieval import MemoryRetriever
from .core_curator import CoreCurator
from .auto_archive import AutoArchiveManager


class MemoryWorkbench:
    """Service layer also usable directly by tests without running HTTP."""
    def __init__(self, data_dir: str | Path, env_file: str | Path):
        self.core = MaidCore(data_dir)
        self.db = VerifiedMemoryDatabase(self.core.root / "memory.db")
        self.retriever = MemoryRetriever(self.core)
        self.curator = CoreCurator(self.retriever)
        self.auto_archive=AutoArchiveManager(self.core,env_file,db=self.db)
        self._core_jobs = {}
        self._core_job_lock = threading.RLock()
        self._core_busy = False
        self._bulk_reviewing = False
        self.env_file = Path(env_file)
        self._jobs: dict[str, dict] = {}
        self._job_lock = threading.RLock()
        self._request_slots = threading.BoundedSemaphore(3)
        self._error_lock = threading.Lock()
        self.error_log = self.core.root / 'logs' / 'memory_workbench_errors.jsonl'
        self.error_log.parent.mkdir(parents=True, exist_ok=True)
        with self.db.connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS ui_samples (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL,
                    records_json TEXT NOT NULL, known_json TEXT NOT NULL,
                    source_ids_json TEXT NOT NULL, sample_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ui_sample_runs (
                    sample_id TEXT NOT NULL, run_id TEXT NOT NULL UNIQUE,
                    batch_key TEXT NOT NULL UNIQUE, model TEXT NOT NULL,
                    effort TEXT NOT NULL, mode TEXT NOT NULL,
                    verification INTEGER NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY(sample_id,run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_ui_sample_runs ON ui_sample_runs(sample_id);
                CREATE TABLE IF NOT EXISTS ui_edit_history (
                    id TEXT PRIMARY KEY, object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL, before_json TEXT NOT NULL,
                    after_json TEXT NOT NULL, edited_at TEXT NOT NULL
                );
            """)
            cols={r[1] for r in con.execute("PRAGMA table_info(ui_sample_runs)")}
            if 'variant_label' not in cols:
                con.execute("ALTER TABLE ui_sample_runs ADD COLUMN variant_label TEXT NOT NULL DEFAULT ''")

    def record_error(self, context, exc, **info):
        """Persistent, private diagnostics. Never log prompts, messages or keys."""
        detail=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))
        # Defense in depth: avoid leaking an API token if a provider echoes it.
        detail=re.sub(r'(?i)(sk-[A-Za-z0-9_-]{12,})','[API_KEY_REDACTED]',detail)
        record={"timestamp":now(),"context":str(context)[:100],
                "type":type(exc).__name__,"message":str(exc)[:2000],
                "traceback":detail[-12000:],"info":{k:str(v)[:120] for k,v in info.items()}}
        with self._error_lock:
            if self.error_log.exists() and self.error_log.stat().st_size>5_000_000:
                previous=self.error_log.with_suffix('.previous.jsonl')
                self.error_log.replace(previous)
            with self.error_log.open('a',encoding='utf-8') as fp:
                fp.write(canonical(record)+'\n')
        return {"error_log":str(self.error_log),"message":record['message']}

    def recent_errors(self, limit=30):
        if not self.error_log.exists():return []
        with self._error_lock:
            lines=self.error_log.read_text(encoding='utf-8').splitlines()
        result=[]
        for line in lines[-min(max(int(limit),1),100):]:
            try:result.append(json.loads(line))
            except ValueError:continue
        return list(reversed(result))

    def list_records(self, *, source="chat", start=0, count=40):
        if source not in {"chat", "autonomous"}:
            raise ValueError("来源只能为 chat / autonomous")
        if not 0 <= start <= 1000000 or not 1 <= count <= 80:
            raise ValueError("起点为非负整数；单页最多 80 条")
        records = historical_records(self.core, source)
        processed = self.db.processed_record_ids(source)
        next_index = next((n for n, r in enumerate(records) if r['id'] not in processed), len(records))
        return {
            "source": source, "total": len(records), "start": start,
            "next_unprocessed_index": next_index,
            "items": [{**r, "index": i, "processed": r["id"] in processed}
                      for i,r in enumerate(records[start:start+count],start)],
            "note": "预览显示将传给模型的文本；超过 4500 字符的原消息被标记 truncated_for_api，原始 JSONL 不变。"
        }

    def create_sample(self, *, source="chat", title="", entries=None):
        if source not in {"chat", "autonomous"} or not isinstance(entries, list):
            raise ValueError("样本来源或条目不合规")
        if not 1 <= len(entries) <= 80:
            raise ValueError("每份样本需要勾选 1～80 条消息")
        originals = historical_records(self.core, source)
        lookup = {r["id"]: (i,r) for i,r in enumerate(originals)}
        seen=set(); records=[]; indices=[]
        for e in entries:
            if not isinstance(e,dict) or e.get("id") not in lookup:
                raise ValueError("样本中存在无效的原始消息 ID")
            if e["id"] in seen:
                raise ValueError("一份样本不能重复选择同一条消息")
            seen.add(e["id"])
            pos,original=lookup[e["id"]]
            content=e.get("content")
            if not isinstance(content,str) or not content.strip() or len(content)>4500:
                raise ValueError("选中消息的内容需为 1～4500 字符，不能留空")
            item={**original,"content":content, "original_index":pos,
                  "edited_for_experiment":content!=original["content"]}
            records.append(item)
            indices.append(pos)
        if indices != sorted(indices):
            raise ValueError("选择的消息必须保持原始时间顺序")
        if sum(len(r["content"]) for r in records)>30000:
            raise ValueError("总正文超过 30000 字符，请减少选中消息")
        title=title.strip()[:100] or f"{source} · {indices[0]+1}～{indices[-1]+1}"
        sample_id="sample_"+uuid.uuid4().hex[:16]
        batch=self._batch_from_items(sample_id,source,records,indices)
        # Freeze existing-memory hints NOW; later A/B variants cannot see
        # candidates newly produced by earlier variants in the same comparison.
        known=known_memories_for_batch(self.db,batch)
        fingerprint=sha256({"source":source,"records":records,"known":known})
        with self.db.connect() as con:
            con.execute("""INSERT INTO ui_samples VALUES (?,?,?,?,?,?,?,?)""",
                       (sample_id,title,source,canonical(records),canonical(known),
                        canonical([r["id"] for r in records]),fingerprint,now()))
        return {"id":sample_id,"title":title,"count":len(records),"characters":batch["characters"],
                "sample_hash":fingerprint,"known_count":len(known),
                "edited_count":sum(r["edited_for_experiment"] for r in records)}

    @staticmethod
    def _batch_from_items(sample_id, source, items, indices):
        return {"key":sha256({"sample":sample_id,"new_attempt":uuid.uuid4().hex}),
                "source":source,"items":json.loads(canonical(items)),"context_before":[],
                "start_index":min(indices),"end_index":max(indices)+1,
                "selected_indices":indices,"characters":sum(len(r["content"]) for r in items),
                "experiment_only":True,"experiment_sample_id":sample_id}

    def samples(self,limit=100):
        with self.db.connect() as con:
            rows=con.execute("""SELECT s.id,s.title,s.source,s.sample_hash,s.created_at,
                   (SELECT COUNT(*) FROM ui_sample_runs r WHERE r.sample_id=s.id) AS run_count,
                   s.source_ids_json FROM ui_samples s ORDER BY s.rowid DESC LIMIT ?""",
                             (min(max(1,int(limit)),300),)).fetchall()
        return [{**{k:x[k] for k in x.keys() if k!="source_ids_json"},
                 "message_count":len(json.loads(x["source_ids_json"]))} for x in rows]

    def sample(self, sample_id):
        with self.db.connect() as con:
            row=con.execute("SELECT * FROM ui_samples WHERE id=?",(sample_id,)).fetchone()
        if row is None: raise KeyError("样本 ID 不存在")
        obj=dict(row)
        obj["records"]=json.loads(obj.pop("records_json"))
        obj["known_memories"]=json.loads(obj.pop("known_json"))
        obj["source_ids"]=json.loads(obj.pop("source_ids_json"))
        return obj

    def runs(self, limit=100):
        all_runs=self.db.list_runs(min(max(1,int(limit)),300))
        with self.db.connect() as con:
            links={r["run_id"]:dict(r) for r in con.execute(
                """SELECT r.*,s.title AS sample_title,s.sample_hash FROM ui_sample_runs r
                   JOIN ui_samples s ON s.id=r.sample_id""")}
        results=[]
        for r in all_runs:
            exp=links.get(r['id'])
            label=(exp['sample_title']+(' · '+exp['variant_label'] if exp.get('variant_label') else ' · '+r['id'][-6:])) if exp else ('常规提炼 · '+r['id'][-6:])
            results.append({**r,"experiment":exp,"display_title":label})
        return results

    def run_sample(self, sample_id, *, model, effort="high",mode="two-pass",verify=False,
                   max_tokens=16384, timeout=240.0, client_factory=None, variant_label=""):
        sample=self.sample(sample_id)
        variant_label=str(variant_label or '').strip()[:80]
        if mode not in {"one-pass","two-pass"}:raise ValueError("提炼模式不合规")
        if effort not in {"none","low","high","max"}:raise ValueError("思考强度不合规")
        if not re.fullmatch(r"[A-Za-z0-9_./:-]{2,100}",model or ""):
            raise ValueError("模型名称包含无效字符")
        max_tokens=int(max_tokens);timeout=float(timeout)
        if not 2000<=max_tokens<=100000 or not 10<=timeout<=600:
            raise ValueError("Token 预算或超时超出允许范围")
        idx=[r["original_index"] for r in sample["records"]]
        batch=self._batch_from_items(sample_id,sample["source"],sample["records"],idx)
        # Never silently advance the ordinary chronological processed checkpoint.
        assert batch['experiment_only'] is True
        # Validate .env, but NEVER expose API key in UI responses or run logs.
        cfg=replace(load_memory_config(self.env_file),model=model,
                    reasoning_effort=effort,max_tokens=max_tokens,timeout=timeout)
        client=client_factory(cfg) if client_factory else ChatCompletionsClient(cfg)
        try:
            result=run_extraction(self.db,batch,client,mode=mode,verify=bool(verify),
                                  known_memories_override=sample["known_memories"])
        finally:
            # Even failures may have a logged extraction run.
            with self.db.connect() as con:
                rec=con.execute("SELECT id FROM extraction_runs WHERE batch_key=?",(batch['key'],)).fetchone()
                if rec is not None:
                    con.execute("""INSERT OR IGNORE INTO ui_sample_runs
                      (sample_id,run_id,batch_key,model,effort,mode,verification,created_at,variant_label)
                      VALUES (?,?,?,?,?,?,?,?,?)""",
                        (sample_id,rec["id"],batch['key'],model,effort,mode,int(bool(verify)),now(),variant_label))
        return {**result,"sample_id":sample_id,"sample_hash":sample["sample_hash"]}

    def start_run_jobs(self, options_list):
        """Queue independent runs on one frozen sample; at most 3 API calls in flight."""
        if not isinstance(options_list,list) or not 1<=len(options_list)<=6:
            raise ValueError("每次提交须包含 1～6 组模型配置")
        sample_ids={o.get("sample_id") for o in options_list if isinstance(o,dict)}
        if len(sample_ids)!=1 or len(sample_ids)==0:
            raise ValueError("并行对照必须选择同一份冻结样本")
        sample_id=next(iter(sample_ids))
        self.sample(sample_id)
        with self._job_lock:
            pending=sum(j['status'] in {'queued','running'} for j in self._jobs.values())
            if pending+len(options_list)>9:raise ValueError("当前待执行任务较多，请先查看任务结果")
            created=[]
            for opts in options_list:
                job_id='job_'+uuid.uuid4().hex[:12]
                job={"id":job_id,"status":"queued","sample_id":sample_id,
                     "message":"正在排队","variant_label":str(opts.get('variant_label',''))[:80]}
                self._jobs[job_id]=job
                created.append((job_id,dict(opts)))
        for job_id,options in created:
            def run_job(job_id=job_id,options=options):
                with self._request_slots:
                    with self._job_lock:
                        self._jobs[job_id].update(status='running',message='正在调用模型')
                    try:
                        result=self.run_sample(sample_id,model=options.get('model',''),
                            effort=options.get('effort','high'),mode=options.get('mode','two-pass'),
                            verify=options.get('verify',False),max_tokens=options.get('max_tokens',16384),
                            timeout=options.get('timeout',240),variant_label=options.get('variant_label',''))
                        if result.get('verification_error'):
                            self.record_error('verification_nonfatal', ValueError(result['verification_error']),
                                              job_id=job_id,run_id=result.get('run_id',''))
                        with self._job_lock:
                            self._jobs[job_id].update(status='succeeded',result=result,
                                message=('提炼完成，但核验失败；候选已保留' if result.get('verification_error') else '已完成'))
                    except Exception as exc:
                        logged=self.record_error('run_sample',exc,job_id=job_id,sample_id=sample_id)
                        with self._job_lock:
                            self._jobs[job_id].update(status='failed',error=logged['message'],
                                log_path=logged['error_log'],message='完整信息已写入错误日志')
            threading.Thread(target=run_job,daemon=True,name=job_id).start()
        return {"jobs":[{"job_id":i,"status":"queued"} for i,_ in created],
                "concurrency_limit":3,"sample_id":sample_id}

    def start_run_job(self,options):
        result=self.start_run_jobs([options])
        return result['jobs'][0]

    def job(self,job_id):
        with self._job_lock:
            item=self._jobs.get(job_id)
            if item is None: raise KeyError("任务编号不存在或服务器已重启；可在运行历史中查看结果")
            return dict(item)

    def run(self,run_id):
        result=self.db.get_run(run_id)
        with self.db.connect() as con:
            link=con.execute("SELECT * FROM ui_sample_runs WHERE run_id=?",(run_id,)).fetchone()
        exp=dict(link) if link else None
        if exp:
            exp['sample_title']=self.sample(exp['sample_id'])['title']
        result["experiment"]=exp
        result['display_title']=(exp['sample_title']+(' · '+exp.get('variant_label','') if exp.get('variant_label') else ' · '+run_id[-6:])) if exp else '常规提炼 · '+run_id[-6:]
        return result

    def compare_sample(self,sample_id):
        sample=self.sample(sample_id)
        runs=[r for r in self.runs(300) if r.get("experiment") and r['experiment']['sample_id']==sample_id]
        data=[]
        for r in runs:
            if r['status']!='succeeded':continue
            full=self.run(r['id'])
            # Compare what MODELS originally returned, not candidate/episode drafts
            # manually edited after extraction. This preserves a fair ablation.
            try:
                from .improved_extraction import _parse_object
                raw_episode=full['stage1_raw'] if r['mode']=='two-pass' else full['model_raw']
                original_episodes=_parse_object(raw_episode,'episodes')
                original_candidates=_parse_object(full['model_raw'],'memories')
                orig_event_titles=[x.get('title','') for x in original_episodes]
                orig_candidate_texts=[x.get('content','') for x in original_candidates]
            except (ValueError,KeyError,TypeError,AttributeError):
                orig_event_titles=[e['title'] for e in full['episodes']]
                orig_candidate_texts=[c['content'] for c in full['candidates']]
            data.append({"run_id":r['id'],"display_title":r['display_title'],"model":r['model'],"mode":r['mode'],
                         "effort":r['experiment']['effort'],"verify":bool(r['experiment']['verification']),
                         "episodes":len(orig_event_titles),"candidates":len(orig_candidate_texts),
                         "usage":full.get('usage',[]),
                         "candidate_texts":orig_candidate_texts,
                         "episode_titles":orig_event_titles})
        return {"sample":{k:sample[k] for k in ("id","title","sample_hash")},
                "same_frozen_input":True,"runs":data,
                "warning":"数量和 Token 并不能代替人工准确率；请对同一原始证据评分。"}

    def export_run(self, run_id, format='json'):
        """Downloadable, user-readable result; raw transcripts remain optional."""
        run=self.run(run_id)
        if format not in {'json','md','full'}:raise ValueError('导出格式只能 json/md/full')
        sample=run.get('experiment') or {}
        result={"display_title":run['display_title'],"sample_name":sample.get('sample_title'),
                "run_id":run_id,"model":run['model'],"mode":run['mode'],
                "model_options":run.get('model_options'),"status":run['status'],
                "input_message_count":run.get('input_payload',{}).get('message_count'),
                "episodes":run['episodes'],"candidates":run['candidates'],
                "verification":run.get('verification'),
                "verification_failure":run.get('verification_failure'),
                "validation_issues":run.get('validation_issues',[]),
                "usage":run.get('usage'),"error_text":run.get('error_text')}
        if format=='full':
            result['input_payload']=run['input_payload']
            result['model_raw']=run.get('model_raw')
            result['stage1_raw']=run.get('stage1_raw')
        if format!='md':
            return canonical(result).encode('utf-8'),'application/json; charset=utf-8',f'winefox_{run_id}_{format}.json'
        parts=[f"# {run['display_title']}",f"运行：{run_id}  |  模型：{run['model']}  |  模式：{run['mode']}",
               f"状态：{run['status']}  |  输入消息：{result['input_message_count']}"]
        if result['verification_failure']:
            parts.append('\n## 核验失败\n'+str(result['verification_failure']['error_text']))
        if result['validation_issues']:
            parts.append('\n## 模型输出校验／隔离记录\n```json\n'
                         +canonical(result['validation_issues'])+'\n```')
        parts.append('\n## Episode（情景事件）')
        for n,e in enumerate(run['episodes'],1):
            parts.append(f"\n### {n}. {e['title']}  [{e['review_status']}]\n\n{e['summary']}\n\n{e['detail']}")
            for p in e.get('plans',[]):parts.append(f"计划 [{p['status']}]：{p['content']}")
        parts.append('\n## Cand（候选长期记忆）')
        for n,c in enumerate(run['candidates'],1):
            parts.append(f"\n### {n}. [{c['status']}] {c['category']}  ({c['id']})\n\n{c['content']}")
        if run.get('usage'):
            parts.append('\n## API Token 记录')
            parts.append('| 阶段 | 输入 Token | 输出 Token | 思考 Token |')
            parts.append('|---|---:|---:|---:|')
            for u in run['usage']:
                parts.append('| {} | {} | {} | {} |'.format(
                    u.get('stage','?'),u.get('prompt_tokens','—'),
                    u.get('completion_tokens','—'),u.get('reasoning_tokens','—')))
        if run.get('verification'):
            parts.append('\n## 核验报告\n```json\n'+canonical(run['verification']['report'])+'\n```')
        return ('\n\n'.join(parts)+'\n').encode('utf-8'),'text/markdown; charset=utf-8',f'winefox_{run_id}.md'

    def review_inbox(self, *, limit=300):
        """Real Archive's unconfirmed items, excluding frozen-sample experiments.

        The experiment workbench still lets owners explicitly review/promote
        experimental material. A daily inbox must not silently mix in that data.
        """
        limit=max(1,min(int(limit),1000))
        fact_condition="""c.status IN ('pending','auto_active') AND
            COALESCE(json_extract(r.input_payload,'$.experiment_only'),0)=0 AND s.run_id IS NULL"""
        episode_condition="""e.review_status IN ('pending','auto_active') AND
            COALESCE(json_extract(r.input_payload,'$.experiment_only'),0)=0 AND s.run_id IS NULL"""
        fact_from="""FROM candidates c
            LEFT JOIN candidate_audit a ON a.candidate_id=c.id
            LEFT JOIN extraction_runs r ON r.id=a.run_id
            LEFT JOIN ui_sample_runs s ON s.run_id=a.run_id"""
        episode_from="""FROM extracted_episodes e
            LEFT JOIN extraction_runs r ON r.id=e.run_id
            LEFT JOIN ui_sample_runs s ON s.run_id=e.run_id"""
        with self.db.connect() as con:
            candidates=[dict(x) for x in con.execute(f"""SELECT c.id,c.content,c.category,
                c.status,c.accepted_memory_id,c.created_at {fact_from}
                WHERE {fact_condition} ORDER BY c.rowid DESC LIMIT ?""",(limit,))]
            episodes=[dict(x) for x in con.execute(f"""SELECT e.id,e.title,e.summary,
                e.review_status,e.run_id {episode_from}
                WHERE {episode_condition} ORDER BY e.rowid DESC LIMIT ?""",(limit,))]
            counts={
                'candidates':con.execute(f'SELECT COUNT(*) {fact_from} WHERE {fact_condition}').fetchone()[0],
                'extracted_episodes':con.execute(f'SELECT COUNT(*) {episode_from} WHERE {episode_condition}').fetchone()[0],
                'pending_candidates':con.execute(f'SELECT COUNT(*) {fact_from} WHERE {fact_condition} AND c.status=\'pending\'').fetchone()[0],
                'auto_active_candidates':con.execute(f'SELECT COUNT(*) {fact_from} WHERE {fact_condition} AND c.status=\'auto_active\'').fetchone()[0],
                'pending_episodes':con.execute(f'SELECT COUNT(*) {episode_from} WHERE {episode_condition} AND e.review_status=\'pending\'').fetchone()[0],
                'auto_active_episodes':con.execute(f'SELECT COUNT(*) {episode_from} WHERE {episode_condition} AND e.review_status=\'auto_active\'').fetchone()[0]
            }
        return {'candidates':candidates,'episodes':episodes,'counts':counts,
                'note':'仅列真实 Archive；冻结样本实验留在 /experiment。auto_active 可参与日常回忆，仍可人工确认/拒绝。'}

    def get_candidate(self,id):
        c=self.db.candidate_evidence(id,core=self.core)
        with self.db.connect() as con:
            row=con.execute("SELECT run_id,warnings FROM candidate_audit WHERE candidate_id=?",(id,)).fetchone()
        c['run_id']=row['run_id'] if row else None
        c['audit_warnings']=json.loads(row['warnings']) if row else []
        with self.db.connect() as con:
            c['edited_after_model']=bool(con.execute("SELECT 1 FROM ui_edit_history WHERE object_type='candidate' AND object_id=? LIMIT 1",(id,)).fetchone())
        if row:
            # The original verification report is indexed by candidate order
            # within the immutable extraction run; use IDs to avoid drift.
            v=self.db.verification_for_run(row['run_id'])
            if v:
                original=self.db.get_run(row['run_id'])['candidates']
                idx=next((n for n,r in enumerate(original) if r['id']==id),None)
                if idx is not None:
                    c['verification']=v['report'].get('memories',{}).get(str(idx))
        return c

    def edit_candidate(self, id, fields):
        allowed={"content","category","subject","fact_key","importance","confidence"}
        if not isinstance(fields,dict) or not fields or set(fields)-allowed:
            raise ValueError("无效的候选编辑字段")
        with self.db.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            old=con.execute("SELECT * FROM candidates WHERE id=?",(id,)).fetchone()
            if not old:raise KeyError("找不到候选")
            if old['status'] not in ('pending','auto_active'):raise ValueError("只能修改待审核或自动归档候选")
            changes={}
            for k,v in fields.items():
                if k=='content':
                    if not isinstance(v,str) or not 3<=len(v.strip())<=650:
                        raise ValueError("候选内容必须为 3～650 字符")
                    changes[k]=v.strip()
                elif k=='category':
                    if v not in CATEGORIES:raise ValueError("category 不合法")
                    changes[k]=v
                elif k=='subject':
                    if v not in SUBJECTS:raise ValueError("subject 不合法")
                    changes[k]=v
                elif k=='fact_key':
                    if not isinstance(v,str) or (v and not re.fullmatch(r"[a-z0-9_.:-]{1,120}",v)):
                        raise ValueError("fact_key 格式不合法")
                    changes[k]=v
                else:
                    try: f=float(v)
                    except (ValueError,TypeError): raise ValueError(f"{k} 必须为 0～1 数字")
                    if not 0<=f<=1:raise ValueError(f"{k} 范围必须为 0～1")
                    changes[k]=f
            # A live auto-archive memory must change together with the candidate.
            # Otherwise the UI would claim an edit while retrieval still uses the old fact.
            if old['status']=='auto_active' and old['accepted_memory_id']:
                mid=old['accepted_memory_id']
                if 'fact_key' in changes and changes['fact_key']:
                    collision=con.execute("SELECT id FROM memories WHERE fact_key=? AND status='active' AND id<>?",(changes['fact_key'],mid)).fetchone()
                    if collision: raise ValueError('事实键已被其他有效记忆占用')
                con.execute("UPDATE memories SET "+", ".join(k+"=?" for k in changes)+",updated_at=? WHERE id=?",
                    (*changes.values(),now(),mid))
            con.execute("UPDATE candidates SET "+", ".join(k+"=?" for k in changes)+" WHERE id=?",(*changes.values(),id))
            con.execute("INSERT INTO ui_edit_history VALUES (?,?,?,?,?,?)",
                        ("edit_"+uuid.uuid4().hex[:16],"candidate",id,
                         canonical({k:old[k] for k in changes}),canonical(changes),now()))
        return self.get_candidate(id)

    def edit_episode(self,id,fields):
        allowed={"title","summary","detail","event_status"}
        if not isinstance(fields,dict) or not fields or set(fields)-allowed:
            raise ValueError("无效的事件编辑字段")
        limits={"title":120,"summary":450,"detail":1500}
        with self.db.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            old=con.execute("SELECT * FROM extracted_episodes WHERE id=?",(id,)).fetchone()
            if not old:raise KeyError("事件不存在")
            if old['review_status'] not in ('pending','auto_active'):raise ValueError("只能修改待审核或自动归档事件")
            changes={}
            for k,v in fields.items():
                if k in limits:
                    if not isinstance(v,str) or not 2<=len(v.strip())<=limits[k]:
                        raise ValueError(f"{k} 字数应在 2～{limits[k]} 之间")
                    changes[k]=v.strip()
                else:
                    if v not in {"occurred","discussion","expression","uncertain"}:
                        raise ValueError("event_status 不合法")
                    changes[k]=v
            con.execute("UPDATE extracted_episodes SET "+", ".join(k+"=?" for k in changes)+" WHERE id=?",(*changes.values(),id))
            con.execute("INSERT INTO ui_edit_history VALUES (?,?,?,?,?,?)",
                        ("edit_"+uuid.uuid4().hex[:16],"episode",id,
                         canonical({k:old[k] for k in changes}),canonical(changes),now()))
        return self.db.get_episode(id)

    def edit_plan(self,id,*,content,status):
        if status not in PLAN_STATES or not isinstance(content,str) or not 3<=len(content.strip())<=500:
            raise ValueError("计划内容或状态无效")
        with self.db.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            old=con.execute("""SELECT p.*,e.review_status FROM episode_plans p
                 JOIN extracted_episodes e ON e.id=p.episode_id WHERE p.id=?""",(id,)).fetchone()
            if not old:raise KeyError("计划不存在")
            if old['review_status'] not in ('pending','auto_active'):raise ValueError("只能编辑待审核或自动归档事件下的计划")
            if old['status'] != status:
                # Preserve a log even for edits before episode approval.
                con.execute("""INSERT INTO plan_state_history
                  (id,plan_id,old_status,new_status,reason,evidence_ids,changed_at)
                  VALUES (?,?,?,?,?,?,?)""",
                  ("hist_"+uuid.uuid4().hex[:16],id,old['status'],status,
                   '人工在审核界面调整草稿计划状态',canonical([]),now()))
            con.execute("UPDATE episode_plans SET content=?,status=? WHERE id=?",(content.strip(),status,id))
            con.execute("INSERT INTO ui_edit_history VALUES (?,?,?,?,?,?)",
                        ("edit_"+uuid.uuid4().hex[:16],"plan",id,
                         canonical({"content":old['content'],"status":old['status']}),
                         canonical({"content":content.strip(),"status":status}),now()))
        return {"id":id,"content":content.strip(),"status":status}

    def _source_was_edited(self,object_type,id):
        with self.db.connect() as con:
            if object_type=='candidate':
                row=con.execute("""SELECT r.input_payload FROM candidate_audit a
                  JOIN extraction_runs r ON r.id=a.run_id WHERE a.candidate_id=?""",(id,)).fetchone()
            else:
                row=con.execute("""SELECT r.input_payload FROM extracted_episodes e
                  JOIN extraction_runs r ON r.id=e.run_id WHERE e.id=?""",(id,)).fetchone()
        if not row:return False
        payload=json.loads(row[0])
        return bool(payload.get('experiment_sample_id') and
                    any(m.get('edited_for_experiment') for m in payload.get('messages',[])))

    def _auto_core_if_needed(self):
        if self._bulk_reviewing or not self.retriever.get_settings().core_auto_enabled:
            return None
        if self.curator.preview()['pending_count'] < self.retriever.get_settings().core_auto_min_new:
            return None
        return self.start_core_job(force=False,automatic=True)

    def start_core_job(self, *, force=False, automatic=False):
        """No API in HTTP request thread; jobs are serialized, errors persist."""
        with self._core_job_lock:
            if self._core_busy:
                return {'status':'already_running','job':next((v for v in self._core_jobs.values()
                    if v['status'] in ('running','queued')),None)}
            self._core_busy=True
            job_id='corejob_'+uuid.uuid4().hex[:16]
            self._core_jobs[job_id]={'id':job_id,'status':'queued','automatic':automatic,
                                     'created_at':now(),'result':None,'error':None}
        def worker():
            with self._core_job_lock:
                self._core_jobs[job_id]['status']='running'
            try:
                result=self.curator.run(env_file=self.env_file,force=force)
                with self._core_job_lock:
                    self._core_jobs[job_id].update({'status':'succeeded' if result['status']=='succeeded' else 'skipped',
                                                    'result':result,'finished_at':now()})
            except Exception as exc:
                err=self.record_error('core_maintenance',exc,job_id=job_id)
                with self._core_job_lock:
                    self._core_jobs[job_id].update({'status':'failed','error':err['message'],
                                                    'error_log':err['error_log'],'finished_at':now()})
            finally:
                with self._core_job_lock:
                    self._core_busy=False
        threading.Thread(target=worker,name='winefox-core-'+job_id,daemon=True).start()
        return {'status':'queued','job_id':job_id}

    def core_job(self,id):
        with self._core_job_lock:
            job=self._core_jobs.get(id)
        if job is None:raise KeyError('找不到 Core 整理任务')
        return dict(job)

    def decide_candidate(self,id,approve, *, note="",replace_id=None,acknowledge_edited_source=False):
        if approve and not acknowledge_edited_source and self._source_was_edited('candidate',id):
            raise ValueError('本条来自经人工修改的实验对话；须先明确确认其与真实历史的关系')
        result=(self.db.approve(id,replace_id=replace_id) if approve else self.db.reject(id,reason=note))
        if approve:self._auto_core_if_needed()
        return result

    def decide_episode(self,id,approve,*,note="",acknowledge_edited_source=False):
        if approve and not acknowledge_edited_source and self._source_was_edited('episode',id):
            raise ValueError('此事件引用了经人工修改的实验对话；须确认真实性后才能批准')
        result=self.db.review_episode(id,approve=approve,note=note)
        if approve:self._auto_core_if_needed()
        return result

    def bulk_review(self,actions):
        if not isinstance(actions,list) or len(actions)>100:raise ValueError("审核数量不合规")
        result=[]
        self._bulk_reviewing=True
        try:
            for x in actions:
                try:
                    if x.get('type')=='candidate':
                        if type(x.get('approve')) is not bool:raise ValueError('审核决定必须为布尔值')
                        response=self.decide_candidate(x['id'],x['approve'],note=x.get('note',''),replace_id=x.get('replace_id'),acknowledge_edited_source=x.get('acknowledge_edited_source',False))
                    elif x.get('type')=='episode':
                        if type(x.get('approve')) is not bool:raise ValueError('审核决定必须为布尔值')
                        response=self.decide_episode(x['id'],x['approve'],note=x.get('note',''),acknowledge_edited_source=x.get('acknowledge_edited_source',False))
                    else:raise ValueError("审核类型无效")
                    result.append({"id":x['id'],"ok":True,"status":response.get('status',response.get('review_status','done'))})
                except Exception as exc:
                    result.append({"id":x.get('id'),"ok":False,"error":str(exc)})
        finally:
            self._bulk_reviewing=False
        if any(x['ok'] for x in result):self._auto_core_if_needed()
        return result

    def recall(self,question,level="summary"):
        return self.db.recall_path(self.core,question,level=level,limit=5,raw_limit=40)

    def stats(self):
        return {"data_path":str(self.core.root),"original_chat":str(self.core.chat_file),
                "original_autonomous":str(self.core.autonomous_file),
                "memory_db":str(self.db.path),"memory_stats":self.db.stats(),
                "source_totals":{"chat":len(historical_records(self.core,'chat')),
                                 "autonomous":len(historical_records(self.core,'autonomous'))}}


def create_server(workbench:MemoryWorkbench,host='127.0.0.1',port=8765):
    if host!='127.0.0.1':raise ValueError("安全限制：仅允许绑定 127.0.0.1")
    token=secrets.token_urlsafe(32)
    # HTML is shipped with project, no third-party CDN or external assets.
    page_file=Path(__file__).resolve().parents[1]/'ui'/'index.html'
    if not page_file.is_file():raise FileNotFoundError("找不到 UI 页面 ui/index.html")
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,fmt,*args):
            # Do not print raw request bodies, conversations or secrets.
            pass
        def _host_ok(self):
            value=self.headers.get('Host','')
            return value in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        def _reply(self,status,obj,ctype='application/json; charset=utf-8',filename=None):
            data=(obj if isinstance(obj,bytes) else json.dumps(obj,ensure_ascii=False).encode('utf-8')
                  if ctype.startswith('application/json') else obj)
            self.send_response(status)
            self.send_header('Content-Type',ctype)
            self.send_header('Cache-Control','no-store')
            if filename:
                self.send_header('Content-Disposition', 'attachment; filename="'+filename+'"')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('X-Frame-Options','DENY')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self'; frame-ancestors 'none'")
            self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        def _route(self,method):
            path=urlsplit(self.path);q=parse_qs(path.query)
            def param(k,default=''):
                return q.get(k,[default])[0]
            if method=='GET':
                if path.path in ('/','/experiment'):
                    selected=page_file if path.path=='/' else page_file.with_name('experiment.html')
                    content=selected.read_text(encoding='utf-8').replace('__WINEFOX_TOKEN__',token)
                    return content.encode('utf-8'),'text/html; charset=utf-8'
                if path.path=='/api/diagnostics/download':
                    from .diagnostics import build_diagnostic_report
                    report=build_diagnostic_report(workbench.core.root,page_file.parents[1])
                    return (json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'), 'application/json; charset=utf-8', 'winefox-diagnostic.json'
                if path.path=='/api/bootstrap':
                    return {"stats":workbench.stats(),"samples":workbench.samples(),"runs":workbench.runs(),"token":None}
                if path.path=='/api/records':
                    return workbench.list_records(source=param('source','chat'),start=int(param('start','0')),count=int(param('count','40')))
                if path.path=='/api/samples':return workbench.samples()
                if path.path=='/api/sample':return workbench.sample(param('id'))
                if path.path=='/api/runs':return workbench.runs()
                if path.path=='/api/run':return workbench.run(param('id'))
                if path.path=='/api/compare':return workbench.compare_sample(param('sample_id'))
                if path.path=='/api/job':return workbench.job(param('id'))
                if path.path=='/api/jobs':return sorted((dict(v) for v in workbench._jobs.values()),key=lambda x:x['id'])
                if path.path=='/api/errors':return {'entries':workbench.recent_errors(), 'file':str(workbench.error_log)}
                if path.path=='/api/errors/download':
                    content=workbench.error_log.read_bytes() if workbench.error_log.exists() else b''
                    return content,'application/x-ndjson; charset=utf-8','winefox_errors.jsonl'
                if path.path=='/api/export/run':return workbench.export_run(param('id'),param('format','json'))
                if path.path=='/api/candidate':return workbench.get_candidate(param('id'))
                if path.path=='/api/episode':
                    eid=param('id')
                    result=workbench.db.get_episode(eid,core=workbench.core,include_raw=True)
                    with workbench.db.connect() as con:
                        result['edited_after_model']=bool(con.execute("SELECT 1 FROM ui_edit_history WHERE object_type='episode' AND object_id=? LIMIT 1",(eid,)).fetchone())
                    return result
                if path.path=='/api/review/inbox':return workbench.review_inbox(limit=int(param('limit','300')))
                if path.path=='/api/memories':return workbench.db.all_memories()
                if path.path=='/api/plans':return workbench.db.plan_list(limit=200)
                if path.path=='/api/retrieval/settings':return workbench.retriever.get_settings().__dict__
                if path.path=='/api/retrieval/core':
                    content,source=workbench.retriever.core_text()
                    manual,manual_src=workbench.retriever.manual_core_text()
                    return {'content':content,'source':source,'manual_content':manual,
                            'manual_source':manual_src,'auto_entries':workbench.curator.entries(),
                            'versions':workbench.retriever.core_versions()}
                if path.path=='/api/retrieval/logs':return workbench.retriever.logs(limit=40)
                if path.path=='/api/retrieval/embedding':return workbench.retriever.embedding_index_stats()
                if path.path=='/api/auto-archive/preview':return workbench.auto_archive.plan()
                if path.path=='/api/auto-archive/settings':return workbench.auto_archive.settings().__dict__
                if path.path=='/api/auto-archive/jobs':return workbench.auto_archive.jobs()
                if path.path=='/api/archive/publish-preview':
                    return workbench.auto_archive.publish_pending(dry_run=True,limit=1000)
                if path.path=='/api/core-auto/preview':return workbench.curator.preview()
                if path.path=='/api/core-auto/history':return workbench.curator.history()
                if path.path=='/api/core-auto/job':return workbench.core_job(param('id'))
                if path.path=='/api/retrieval/items':
                    all_items=workbench.retriever._entries()
                    return [{'id':e['id'],'kind':e['kind'],'title':e['title'],
                             'preview':e['content'][:160],'importance':e['importance'],
                             'pinned':e['metadata'].get('pinned',0),
                             'importance_override':e['metadata'].get('importance_override')}
                             for e in all_items[-500:]]
                if path.path=='/api/retrieval/metadata':
                    return workbench.retriever.metadata(param('kind'),param('id'))
                raise KeyError('页面或接口不存在')
            # POST body handling happens before reaching here.
            body=self.body
            if path.path=='/api/sample/create':return workbench.create_sample(**body)
            if path.path=='/api/run/start':return workbench.start_run_job(body)
            if path.path=='/api/run/batch':return workbench.start_run_jobs(body.get('variants'))
            if path.path=='/api/candidate/edit':return workbench.edit_candidate(body['id'],body['fields'])
            if path.path=='/api/episode/edit':return workbench.edit_episode(body['id'],body['fields'])
            if path.path=='/api/plan/edit':return workbench.edit_plan(body['id'],content=body['content'],status=body['status'])
            if path.path=='/api/candidate/review':return workbench.decide_candidate(body['id'],body['approve'],note=body.get('note',''),replace_id=body.get('replace_id'),acknowledge_edited_source=body.get('acknowledge_edited_source',False))
            if path.path=='/api/episode/review':return workbench.decide_episode(body['id'],body['approve'],note=body.get('note',''),acknowledge_edited_source=body.get('acknowledge_edited_source',False))
            if path.path=='/api/reviews/bulk':return workbench.bulk_review(body.get('actions',[]))
            if path.path=='/api/plan/update':return workbench.db.plan_update(workbench.core,body['id'],status=body['status'],reason=body['note'])
            if path.path=='/api/recall':return workbench.recall(body['question'],body.get('level','summary'))
            if path.path=='/api/retrieval/settings':return workbench.retriever.update_settings(body.get('changes',{}))
            if path.path=='/api/auto-archive/settings':return workbench.auto_archive.update_settings(body.get('changes',{}))
            if path.path=='/api/auto-archive/start':return workbench.auto_archive.start_background()
            if path.path=='/api/archive/publish-all':
                if body.get('confirm') is not True:raise ValueError('批量启用需要明确确认')
                return workbench.auto_archive.publish_pending(dry_run=False,limit=1000)
            if path.path=='/api/core-auto/run':return workbench.start_core_job(force=bool(body.get('force',False)))
            if path.path=='/api/core-auto/restore':return workbench.curator.restore(body['id'],snapshot=body.get('snapshot','before'))
            if path.path=='/api/retrieval/core':return workbench.retriever.save_core_text(body['content'])
            if path.path=='/api/retrieval/core/restore':return workbench.retriever.restore_core(body['version'])
            if path.path=='/api/retrieval/preview':
                q=body['question']
                if not isinstance(q,str) or not 1<=len(q.strip())<=2000:raise ValueError('预览问题需 1～2000 字符')
                _context, result = workbench.core.build_messages(q,
                    retriever=workbench.retriever,return_retrieval=True,
                    retrieval_seed=body.get('seed',0))
                return result
            if path.path=='/api/retrieval/metadata':
                return workbench.retriever.set_metadata(body['kind'],body['id'],
                    pinned=body['pinned'],importance_override=body.get('importance_override'))
            raise KeyError('接口不存在')
        def do_GET(self):
            if not self._host_ok():return self._reply(403,{"error":"Host 非 localhost"})
            try:
                payload=self._route('GET')
                if isinstance(payload,tuple):return self._reply(200,*payload)
                return self._reply(200,payload)
            except (ValueError,KeyError,TypeError) as e:
                return self._reply(400,{"error":str(e)})
            except Exception as e:
                logged=workbench.record_error('GET '+self.path,e)
                return self._reply(500,{"error":"服务器内部错误："+logged['message'],"log_file":logged['error_log']})
        def do_POST(self):
            if not self._host_ok():return self._reply(403,{"error":"Host 非 localhost"})
            origin=self.headers.get('Origin','')
            if origin not in {f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}:
                return self._reply(403,{"error":"只允许从本地 UI 页面提交"})
            if self.headers.get('X-WineFox-Token')!=token:
                return self._reply(403,{"error":"本地会话令牌无效"})
            if self.headers.get('Content-Type','').split(';')[0].strip()!='application/json':
                return self._reply(415,{"error":"只接收 JSON 请求"})
            try:
                n=int(self.headers.get('Content-Length','0'))
                if n<=0 or n>800000:return self._reply(413,{"error":"请求大小超限"})
                self.body=json.loads(self.rfile.read(n))
                if not isinstance(self.body,dict):raise ValueError('请求正文必须是对象')
                response=self._route('POST')
                return self._reply(200,response)
            except (ValueError,KeyError,TypeError) as e:
                return self._reply(400,{"error":str(e)})
            except Exception as e:
                logged=workbench.record_error('POST '+self.path,e)
                return self._reply(500,{"error":logged['message'],"log_file":logged['error_log']})
    server=ThreadingHTTPServer((host,port),Handler)
    server.daemon_threads=True
    return server


def serve_ui(data_dir,env_file,port=8765,open_browser=True):
    wb=MemoryWorkbench(data_dir,env_file)
    httpd=create_server(wb,port=port)
    address=f'http://127.0.0.1:{httpd.server_port}/'
    print(f'酒狐本地记忆工作台：{address}')
    print(f'数据目录：{wb.core.root}；按 Ctrl+C 关闭；服务器仅绑定本机 127.0.0.1。')
    if open_browser:
        webbrowser.open(address)
    try:httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:print('\n工作台已关闭，历史记录与审核结果仍保存在本地。')
    finally:httpd.server_close()
