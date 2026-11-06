"""Opt-in, durable auto-extraction into *reversible* usable memory.

Raw JSONL remains authoritative. The scheduling checkpoint is not a deletion.
The worker runs in a separate process so a one-shot terminal session may exit
without abandoning a paid API request. Normal fixed-sample experiments are NEVER
published. No API request happens until the owner explicitly enables the feature
or invokes the --run command.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import os
import sqlite3
import subprocess
import sys
import traceback
import re
import uuid

from .long_memory import now, norm
from .memory_verification import VerifiedMemoryDatabase
from .improved_extraction import build_batches, historical_records, run_extraction
from .llm import APIConfig, ChatCompletionsClient, _read_env


@dataclass
class AutoArchiveSettings:
    enabled: bool = False
    trigger_messages: int = 24  # eligible UNPROCESSED messages, not the total archive size
    keep_recent_messages: int = 4
    max_messages: int = 40
    max_chars: int = 8000
    mode: str = 'two-pass'
    verify: bool = False  # an expensive separate model call; optional

    @classmethod
    def load(cls, data_dir):
        path=Path(data_dir)/'auto_archive_settings.json'
        if not path.exists():return cls()
        return cls.validate(json.loads(path.read_text(encoding='utf-8')))

    @classmethod
    def validate(cls, changes):
        if not isinstance(changes,dict):raise ValueError('自动归档配置必须是 JSON 对象')
        values=asdict(cls())
        unknown=set(changes)-set(values)
        if unknown:raise ValueError('未知自动归档配置：'+','.join(sorted(unknown)))
        values.update(changes)
        for k,v in values.items():
            if k in ('enabled','verify'):
                if type(v) is not bool:raise ValueError(k+' 必须是布尔值')
            elif k=='mode':
                if v not in ('one-pass','two-pass'):raise ValueError('mode 必须是 one-pass/two-pass')
            elif type(v) is not int:
                raise ValueError(k+' 必须是整数')
        if not 4<=values['trigger_messages']<=10000:raise ValueError('trigger_messages 范围 4～10000')
        if not 0<=values['keep_recent_messages']<=5000:raise ValueError('keep_recent_messages 范围 0～5000')
        if not 4<=values['max_messages']<=80:raise ValueError('max_messages 范围 4～80')
        if not 1200<=values['max_chars']<=30000:raise ValueError('max_chars 范围 1200～30000')
        return cls(**values)

    def save(self,data_dir):
        p=Path(data_dir)/'auto_archive_settings.json';p.parent.mkdir(parents=True,exist_ok=True)
        temp=p.with_suffix('.json.tmp')
        temp.write_text(json.dumps(asdict(self),ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(p)
        return {'path':str(p),'settings':asdict(self)}


def archive_model_config(env_file):
    """Dedicated inexpensive default; user's older MEMORY_MODEL=pro is untouched."""
    base=APIConfig.load(env_file)
    env=_read_env(Path(env_file))
    # Mirror APIConfig's supported environment overrides, without exposing key.
    model=env.get('AUTO_ARCHIVE_MODEL','deepseek-flash').strip() or 'deepseek-flash'
    effort=env.get('AUTO_ARCHIVE_REASONING_EFFORT','high').strip() or 'high'
    if effort not in ('none','low','high','max'):
        raise ValueError('AUTO_ARCHIVE_REASONING_EFFORT 必须为 none/low/high/max')
    try:
        tokens=int(env.get('AUTO_ARCHIVE_MAX_TOKENS','16384'))
        timeout=float(env.get('AUTO_ARCHIVE_TIMEOUT','240'))
    except ValueError as exc:
        raise ValueError('AUTO_ARCHIVE_MAX_TOKENS / AUTO_ARCHIVE_TIMEOUT 必须是数字') from exc
    if not 2000<=tokens<=100000 or not 10<=timeout<=600:
        raise ValueError('自动归档模型 Token 或 Timeout 配置超出范围')
    return replace(base,model=model,reasoning_effort=effort,max_tokens=tokens,
                   timeout=timeout,temperature=0.2)


class AutoArchiveManager:
    def __init__(self, core, env_file=None, db=None):
        self.core=core
        self.db=db or VerifiedMemoryDatabase(core.root/'memory.db')
        self.env_file=Path(env_file) if env_file else Path(__file__).resolve().parents[1]/'.env'
        with self.db.connect() as con:
            con.executescript('''
                CREATE TABLE IF NOT EXISTS auto_archive_jobs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL,
                    started_at TEXT, finished_at TEXT, run_id TEXT,
                    message_count INTEGER, candidate_count INTEGER, episode_count INTEGER,
                    auto_candidates INTEGER, auto_episodes INTEGER, error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_auto_archive_jobs_status
                    ON auto_archive_jobs(status);
            ''')
            cols={r[1] for r in con.execute('PRAGMA table_info(memories)')}
            if 'approval_source' not in cols:
                con.execute("ALTER TABLE memories ADD COLUMN approval_source TEXT NOT NULL DEFAULT 'human'")

    def settings(self):return AutoArchiveSettings.load(self.core.root)

    def update_settings(self,changes):
        old=asdict(self.settings())
        return AutoArchiveSettings.validate({**old,**changes}).save(self.core.root)

    def plan(self,*,force=False):
        cfg=self.settings()
        rows=historical_records(self.core,'chat')
        # Keep the most recent messages untouched, so the active conversation
        # does not abruptly get compressed mid-turn.
        stop=max(0,len(rows)-cfg.keep_recent_messages)
        eligible=rows[:stop]
        processed=self.db.processed_record_ids('chat')
        unprocessed=[r for r in eligible if r['id'] not in processed]
        info={'enabled':cfg.enabled,'total_messages':len(rows),
              'unprocessed_eligible':len(unprocessed),'keep_recent_messages':cfg.keep_recent_messages,
              'trigger_messages':cfg.trigger_messages,'ready':False,
              'batch':None,'mode':cfg.mode,'verification':cfg.verify}
        if not (force or cfg.enabled) or len(unprocessed)<(4 if force else cfg.trigger_messages):return info
        batches=build_batches(self.core,'chat',stop=stop,exclude_ids=processed,
                              max_chars=cfg.max_chars,max_messages=cfg.max_messages)
        if not batches:return info
        b=batches[0]
        if len(b['items'])<4:return info
        info['ready']=True
        info['batch']={'key':b['key'],'message_count':len(b['items']),
                       'characters':b['characters'],'start':b['start_index'],
                       'end':b['end_index'],'first_id':b['items'][0]['id'],
                       'last_id':b['items'][-1]['id']}
        return info

    def jobs(self,limit=30):
        with self.db.connect() as con:
            return [dict(x) for x in con.execute('SELECT * FROM auto_archive_jobs ORDER BY rowid DESC LIMIT ?',
                                                    (max(1,min(int(limit),100)),))]

    def _new_job(self):
        stamp=now();job_id='archivejob_'+uuid.uuid4().hex[:16]
        with self.db.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            # A crashed worker is recoverable. No eager API retry while it is fresh.
            cutoff=(datetime.now(timezone.utc)-timedelta(minutes=30)).isoformat(timespec='seconds')
            con.execute("UPDATE auto_archive_jobs SET status='interrupted',finished_at=?,error='超过30分钟未完成，需人工检查运行日志' WHERE status IN ('queued','running') AND created_at<?",(stamp,cutoff))
            existing=con.execute("SELECT id FROM auto_archive_jobs WHERE status IN ('queued','running') LIMIT 1").fetchone()
            if existing:return None,existing[0]
            con.execute('INSERT INTO auto_archive_jobs(id,status,created_at) VALUES(?,?,?)',(job_id,'queued',stamp))
        return job_id,None

    def start_background(self):
        status=self.plan()
        if not status['ready']:return {'status':'not_ready','preview':status}
        job_id,other=self._new_job()
        if other:return {'status':'already_running','job_id':other}
        main=Path(__file__).resolve().parents[1]/'main.py'
        cmd=[sys.executable,str(main),'--data',str(self.core.root),'memory-auto-archive',
             '--worker-job',job_id,'--env-file',str(self.env_file)]
        try:
            flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
            # Devnull: all failures go to SQLite job table; never print API keys.
            subprocess.Popen(cmd,cwd=str(main.parent),stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                creationflags=flags,close_fds=True)
        except Exception as exc:
            with self.db.connect() as con:
                con.execute("UPDATE auto_archive_jobs SET status='failed',finished_at=?,error=? WHERE id=?",
                            (now(),'无法启动本地归档工作进程：'+str(exc)[:400],job_id))
            raise
        return {'status':'queued','job_id':job_id,'preview':status}

    def run_once(self,*,client=None,force=False,job_id=None):
        """Run at most one batch; caller's explicit --run OR previously queued worker.

        A single failed API request never consumes processed_records. The model's
        raw reply stays in extraction_runs after failures, as in v0.3a.5.1.
        """
        if job_id:
            with self.db.connect() as con:
                con.execute('BEGIN IMMEDIATE')
                row=con.execute('SELECT status FROM auto_archive_jobs WHERE id=?',(job_id,)).fetchone()
                if not row or row[0]!='queued':raise ValueError('不存在待执行的自动归档任务')
                con.execute("UPDATE auto_archive_jobs SET status='running',started_at=? WHERE id=?",(now(),job_id))
        else:
            job_id,other=self._new_job()
            if other:return {'status':'already_running','job_id':other}
            with self.db.connect() as con:
                con.execute("UPDATE auto_archive_jobs SET status='running',started_at=? WHERE id=?",(now(),job_id))
        try:
            cfg=self.settings()
            preview=self.plan(force=force)
            if not preview['ready']:
                with self.db.connect() as con:
                    con.execute("UPDATE auto_archive_jobs SET status='skipped',finished_at=? WHERE id=?",(now(),job_id))
                return {'status':'skipped','job_id':job_id,'preview':preview}
            # Rebuild batch from current source/processed checkpoints. Another
            # process may have processed earlier records since planning.
            records=historical_records(self.core,'chat')
            stop=max(0,len(records)-cfg.keep_recent_messages)
            batches=build_batches(self.core,'chat',stop=stop,max_chars=cfg.max_chars,
                max_messages=cfg.max_messages,exclude_ids=self.db.processed_record_ids('chat'))
            batch=next((x for x in batches if len(x['items'])>=4),None)
            if batch is None:
                raise ValueError('没有可用的完整归档批次')
            if client is None: client=ChatCompletionsClient(archive_model_config(self.env_file))
            result=run_extraction(self.db,batch,client,mode=cfg.mode,verify=cfg.verify)
            if result.get('skipped'):raise ValueError('批次已被其他运行消耗，未自动发布')
            live=self.publish_run(result['run_id'])
            # Independent failure domain: an optional Core update must never
            # roll back valid Archive memory or force a second extraction.
            core_result={'status':'disabled'}
            try:
                from .retrieval import MemoryRetriever
                from .core_curator import CoreCurator
                retriever=MemoryRetriever(self.core)
                if retriever.get_settings().core_auto_enabled:
                    core_result=CoreCurator(retriever).run(env_file=self.env_file)
            except Exception as exc:
                core_result={'status':'failed','error':str(exc)[:400]}
            with self.db.connect() as con:
                con.execute('''UPDATE auto_archive_jobs SET status='succeeded',finished_at=?,run_id=?,
                    message_count=?,candidate_count=?,episode_count=?,auto_candidates=?,auto_episodes=? WHERE id=?''',
                    (now(),result['run_id'],len(batch['items']),result['created'],result['episodes'],
                     live['auto_candidates'],live['auto_episodes'],job_id))
            return {'status':'succeeded','job_id':job_id,'run_id':result['run_id'],**live,
                    'usage':result.get('usage',[]),'core_maintenance':core_result}
        except Exception as exc:
            with self.db.connect() as con:
                con.execute('UPDATE auto_archive_jobs SET status=?,finished_at=?,error=? WHERE id=?',
                            ('failed',now(),str(exc)[:2000],job_id))
            log=self.core.root/'logs'/'memory_auto_archive_errors.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
            detail=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))[-12000:]
            detail=re.sub(r'(?i)sk-[a-z0-9_-]{12,}', '[API_KEY_REDACTED]',detail)
            try:
                with log.open('a',encoding='utf-8') as fp:
                    fp.write(json.dumps({'at':now(),'job_id':job_id,'type':type(exc).__name__,
                       'error':str(exc)[:2000],'traceback':detail},ensure_ascii=False)+'\n')
            except OSError:pass
            raise

    @staticmethod
    def _has_table(con, table):
        return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone() is not None

    def _trusted_run(self, con, run, originals):
        """Require a successful ordinary extraction from the unchanged local chat.

        The API input must match our authoritative JSONL, not simply contain
        plausible-looking IDs. Edited/frozen experiments never become live.
        """
        if run['status'] != 'succeeded' or run['source'] != 'chat':
            return False, '仅正常 chat 的成功提炼可以自动发布'
        try:
            payload=json.loads(run['input_payload'])
            msgs=payload['messages']
        except (ValueError,KeyError,TypeError):
            return False, '提炼运行缺少有效来源消息'
        if (payload.get('experiment_only') or payload.get('experiment_sample_id') or
                payload.get('selected_indices') or payload.get('reprocess_trial')):
            return False, '冻结实验、编辑样本及重新测试不参与自动发布'
        if self._has_table(con,'ui_sample_runs') and con.execute(
                'SELECT 1 FROM ui_sample_runs WHERE run_id=?',(run['id'],)).fetchone():
            return False, '冻结样本实验禁止自动发布'
        if not isinstance(msgs,list) or not msgs:
            return False, '没有真实来源消息'
        # Standard extraction keys are deterministic. A run using the same
        # messages with an altered key is a re-test/variant, not a normal archive.
        # Restrict the check to the known schema to avoid excluding legacy data
        # that used a different batch-key algorithm.
        if run['prompt_version']=='episodes-v2.2':
            from .long_memory import sha256
            expected=sha256({'version':run['prompt_version'],'source':'chat',
                             'ids':[(m['id'],sha256(m['content'])) for m in msgs
                                    if isinstance(m,dict) and isinstance(m.get('id'),str)
                                    and isinstance(m.get('content'),str)]})
            if expected!=run['batch_key']:
                return False, '批次键不匹配：可能是重新提炼或实验运行'
        seen=set()
        for m in msgs:
            if not isinstance(m,dict) or not isinstance(m.get('id'),str):
                return False, '来源格式无效'
            rid=m['id']
            if rid in seen or rid not in originals:
                return False, '来源 ID 重复或不属于真实聊天'
            seen.add(rid)
            # A legitimate extraction may truncate long messages to the
            # same 4500 chars as historical_records(), but may not rewrite them.
            raw=originals[rid]
            if m.get('content')!=raw['content'] or m.get('role')!=raw['role'] or m.get('platform')!=raw['platform']:
                return False, '提炼原文与当前真实档案不一致'
        return True, ''

    def _publish_one(self,con,run,originals):
        valid,reason=self._trusted_run(con,run,originals)
        if not valid:
            return {'run_id':run['id'],'status':'excluded','reason':reason,
                    'auto_candidates':0,'auto_episodes':0,'duplicates_skipped':0,
                    'fact_key_conflicts_pending':0,'held_candidates':0,'held_episodes':0}
        payload=json.loads(run['input_payload'])
        valid_ids={r['id'] for r in payload['messages']}
        stamp=now()
        ec=cc=duplicates=conflicts=heldc=helde=0
        bad_episode_local=set()
        try:
            issues=json.loads(run['validation_issues_json'] or '[]')
            for issue in issues:
                if issue.get('kind') in ('episode','plan') and issue.get('action') in ('evidence_filtered','quarantined'):
                    if issue.get('local_id'):bad_episode_local.add(issue['local_id'])
        except (ValueError,TypeError,KeyError):
            # Unknown validation data is not evidence of safety.
            return {'run_id':run['id'],'status':'excluded','reason':'校验日志无法解析',
                    'auto_candidates':0,'auto_episodes':0,'duplicates_skipped':0,
                    'fact_key_conflicts_pending':0,'held_candidates':0,'held_episodes':0}
        audit_present=self._has_table(con,'episode_verification_audit')
        eps=con.execute('SELECT id,local_id,evidence_ids,review_status FROM extracted_episodes WHERE run_id=?',(run['id'],)).fetchall()
        for ep in eps:
            if ep['review_status']!='pending':continue
            try:ids=json.loads(ep['evidence_ids'])
            except (ValueError,TypeError):ids=[]
            if (not ids or any(i not in valid_ids for i in ids) or ep['local_id'] in bad_episode_local):
                helde+=1;continue
            if audit_present:
                audit=con.execute('SELECT verdict FROM episode_verification_audit WHERE episode_id=?',(ep['id'],)).fetchone()
                if audit and audit['verdict']!='supported':
                    helde+=1;continue
            con.execute("UPDATE extracted_episodes SET review_status='auto_active',review_note=? WHERE id=?",
                        ('AI 自动发布；仍可事后复核',ep['id']))
            ec+=1
        candidates=con.execute("""SELECT c.*,a.warnings FROM candidates c
            JOIN candidate_audit a ON a.candidate_id=c.id WHERE a.run_id=? ORDER BY c.rowid""",(run['id'],)).fetchall()
        for c in candidates:
            if c['status']!='pending':continue
            try:ids=json.loads(c['evidence_ids']);warnings=json.loads(c['warnings'])
            except (ValueError,TypeError):ids=[];warnings=['错误审查信息']
            if not ids or any(i not in valid_ids for i in ids) or warnings:
                heldc+=1;continue
            if c['fact_key']:
                old=con.execute("SELECT id,content FROM memories WHERE fact_key=? AND status='active'",(c['fact_key'],)).fetchone()
                if old and norm(old['content'])!=norm(c['content']):
                    conflicts+=1;heldc+=1;continue
            else:old=None
            same=next((x for x in con.execute("SELECT id,content FROM memories WHERE status='active' AND category=?",(c['category'],))
                       if norm(x['content'])==norm(c['content'])),None)
            if same or old:
                duplicates+=1
                con.execute("UPDATE candidates SET status='duplicate',review_note=? WHERE id=?",
                            ('自动发布：与现存有效记忆重复',c['id']))
                continue
            mid='memdb_'+uuid.uuid4().hex[:16]
            con.execute("""INSERT INTO memories
                (id,content,category,subject,fact_key,importance,confidence,evidence_ids,evidence_class,
                 platform,event_time,origin_candidate_id,status,replaces_id,created_at,updated_at,
                 approval_source,plan_status,temporal_validity)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (mid,c['content'],c['category'],c['subject'],c['fact_key'],c['importance'],c['confidence'],
                 c['evidence_ids'],c['evidence_class'],c['platform'],c['event_time'],c['id'],
                 'active',None,stamp,stamp,'auto',c['plan_status'],c['temporal_validity']))
            con.execute("UPDATE candidates SET status='auto_active',accepted_memory_id=?,review_note=? WHERE id=?",
                        (mid,'AI 自动发布；仍可事后复核',c['id']))
            cc+=1
        return {'run_id':run['id'],'publish_status':'published','auto_candidates':cc,'auto_episodes':ec,
                'duplicates_skipped':duplicates,'fact_key_conflicts_pending':conflicts,
                'held_candidates':heldc,'held_episodes':helde}

    def publish_run(self,run_id):
        """Publish an ordinary extraction once, without another LLM call."""
        originals={r['id']:r for r in historical_records(self.core,'chat')}
        with self.db.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            run=con.execute('SELECT * FROM extraction_runs WHERE id=?',(run_id,)).fetchone()
            if not run or run['status']!='succeeded':raise ValueError('只能发布已成功的提炼运行')
            result=self._publish_one(con,run,originals)
            if result.get('status')=='excluded':raise ValueError(result['reason'])
        return result

    def publish_pending(self,*,dry_run=True,limit=1000):
        """Bulk backfill from real successful runs; preview is truly read-only.

        Reuses precisely the same publication checks as live auto_archive.
        No API request, no edits to chat_history.jsonl or persona.
        """
        if type(limit) is not int or not 1<=limit<=10000:
            raise ValueError('limit 范围 1～10000')
        originals={r['id']:r for r in historical_records(self.core,'chat')}
        with self.db.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            # Both tables may exist only after workbench initialization.
            runs=con.execute("""SELECT r.* FROM extraction_runs r WHERE r.status='succeeded' AND r.source='chat'
                AND (EXISTS(SELECT 1 FROM extracted_episodes e WHERE e.run_id=r.id AND e.review_status='pending')
                  OR EXISTS(SELECT 1 FROM candidate_audit a JOIN candidates c ON c.id=a.candidate_id
                            WHERE a.run_id=r.id AND c.status='pending'))
                ORDER BY r.rowid ASC LIMIT ?""",(limit,)).fetchall()
            if dry_run:con.execute('SAVEPOINT winefox_preview')
            results=[self._publish_one(con,r,originals) for r in runs]
            totals={name:sum(x.get(name,0) for x in results) for name in
                    ('auto_candidates','auto_episodes','duplicates_skipped','fact_key_conflicts_pending','held_candidates','held_episodes')}
            if dry_run:
                con.execute('ROLLBACK TO SAVEPOINT winefox_preview')
                con.execute('RELEASE SAVEPOINT winefox_preview')
        return {'mode':'preview' if dry_run else 'applied','run_count':len(runs),**totals,
                'excluded_runs':sum(r.get('status')=='excluded' for r in results),
                'runs':results[:100], 'more_run_details':max(0,len(results)-100),
                'note':'仅正常且来源可追溯的 chat 结果；实验、已拒绝、证据异常不发布；不会调用大模型。'}
