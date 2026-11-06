"""v0.3a.2: reviewed episodic recall, plan lifecycle, timestamp and provenance.

This is an independent Python implementation. The factual archive remains the
existing JSONL file; older SQLite schemas are migrated additively. No network
requests, automatic approvals, or persona modifications occur in this module.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .improved_extraction import InspectableMemoryDatabase
from .long_memory import canonical, now

LOCAL_TZ = "Asia/Shanghai"
EVENT_STATES = {"occurred", "discussion", "expression", "uncertain"}
PLAN_STATES = {"proposed", "planned", "ongoing", "completed", "cancelled", "uncertain"}
REVIEW_STATES = {"pending", "auto_active", "approved", "rejected"}


def utc_iso(value):
    """Return unambiguous UTC date-time; unknown or naive times stay unknown."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return None
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    except (ValueError, OverflowError):
        return None


def local_iso(value, tz: str = LOCAL_TZ):
    value = utc_iso(value)
    if value is None:
        return None
    if tz != LOCAL_TZ:
        raise ValueError("当前仅内置北京时间 Asia/Shanghai，其他时区需自行配置")
    # 2026 Chinese timestamps use UTC+08:00; avoid requiring tzdata on Windows.
    return datetime.fromisoformat(value).astimezone(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def evidence_kind(role, source=""):
    if role == "user":
        return "user_report"
    if role == "tool":
        return "tool_output_unverified"
    if role == "autonomous" or source == "tlm_self_talk_archive":
        return "winefox_autonomous_expression"
    if role == "assistant":
        return "winefox_reply"
    return "other_record"


def _tokens(query):
    words = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", query.casefold())
    terms = set()
    for w in words:
        if re.fullmatch(r"[\u4e00-\u9fff]+", w):
            terms.update(w[i:i+n] for n in (2, 3) for i in range(max(0,len(w)-n+1)))
        elif len(w)>=3:
            terms.add(w)
    return terms


def _score(query, text):
    terms = _tokens(query)
    text = text.casefold()
    return sum((2 if len(t)>=3 else 1) for t in terms if t in text)


def _record_index(core):
    records = core.chat_history() + core.autonomous_history()
    return {r["id"]: r for r in records if r.get("id")}


class EpisodeMemoryDatabase(InspectableMemoryDatabase):
    """Schema migration works on existing v0.3a / v0.3a.1 databases.

    Episode records require a separate review before they can appear in chat.
    Candidate approvals alone do not expose an unreviewed episode summary.
    """
    def __init__(self, path):
        super().__init__(path)
        with self.connect() as con:
            cols = {x[1] for x in con.execute("PRAGMA table_info(extracted_episodes)")}
            migrating_legacy_events = "event_status" not in cols
            for name, sql_type, default in [
                ("event_status", "TEXT", "'uncertain'"),
                ("review_status", "TEXT", "'pending'"),
                ("time_start_utc", "TEXT", "NULL"),
                ("time_end_utc", "TEXT", "NULL"),
                ("time_zone", "TEXT", "'Asia/Shanghai'"),
                ("reviewed_at", "TEXT", "NULL"),
                ("review_note", "TEXT", "NULL"),
            ]:
                if name not in cols:
                    con.execute(f"ALTER TABLE extracted_episodes ADD COLUMN {name} {sql_type} DEFAULT {default}")
            for table in ("candidates", "memories"):
                col_names = {x[1] for x in con.execute(f"PRAGMA table_info({table})")}
                if "plan_status" not in col_names:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN plan_status TEXT")
            con.executescript("""
                CREATE TABLE IF NOT EXISTS episode_plans (
                    id TEXT PRIMARY KEY, episode_id TEXT NOT NULL,
                    content TEXT NOT NULL, status TEXT NOT NULL,
                    evidence_ids TEXT NOT NULL, created_at TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES extracted_episodes(id)
                );
                CREATE INDEX IF NOT EXISTS idx_episode_review ON extracted_episodes(review_status);
                CREATE TABLE IF NOT EXISTS candidate_evidence_sources (
                    candidate_id TEXT NOT NULL, record_id TEXT NOT NULL,
                    role TEXT NOT NULL, evidence_kind TEXT NOT NULL, platform TEXT NOT NULL,
                    PRIMARY KEY(candidate_id, record_id)
                );
            """)
            # Legacy status strings described a mix of event and plan status. Preserve
            # original state and infer a conservative separate event status only.
            if migrating_legacy_events:
                con.execute("""UPDATE extracted_episodes SET event_status=CASE
                      WHEN state='completed' THEN 'occurred'
                      WHEN state='expression' THEN 'expression'
                      WHEN state IN ('planned','ongoing') THEN 'discussion'
                      ELSE 'uncertain' END""")
            # Recover original times, not inferred dates from the model's prose.
            legacy = con.execute("""SELECT e.id,e.run_id,e.evidence_ids
                 FROM extracted_episodes e WHERE e.time_start_utc IS NULL""").fetchall()
            for ep in legacy:
                run = con.execute("SELECT input_payload FROM extraction_runs WHERE id=?",(ep["run_id"],)).fetchone()
                if not run:
                    continue
                payload = json.loads(run[0]); ids = set(json.loads(ep["evidence_ids"]))
                dates = sorted(filter(None,(utc_iso(r.get("event_time")) for r in payload.get("messages",[]) if r.get("id") in ids)))
                if dates:
                    con.execute("UPDATE extracted_episodes SET time_start_utc=?,time_end_utc=? WHERE id=?",
                                (dates[0],dates[-1],ep["id"]))
            # Old event.state=planned/ongoing used a mixed status. Preserve the
            # original field but do not invent a completed task. This placeholder
            # explicitly asks for human verification of the legacy plan.
            legacy_plan_rows=con.execute("""SELECT e.id,e.state,e.evidence_ids FROM extracted_episodes e
                JOIN extraction_runs r ON r.id=e.run_id
                WHERE e.state IN ('planned','ongoing') AND r.prompt_version!='episodes-v2.2'""").fetchall()
            for e in legacy_plan_rows:
                known=con.execute("SELECT 1 FROM episode_plans WHERE episode_id=? LIMIT 1",(e["id"],)).fetchone()
                if not known:
                    con.execute("""INSERT INTO episode_plans
                      (id,episode_id,content,status,evidence_ids,created_at)
                      VALUES (?,?,?,?,?,?)""",
                      ("plan_"+uuid.uuid4().hex[:16],e["id"],
                       "（旧版数据）这段事件可能包含未完成的事项，请根据原始对话核实", "uncertain",
                       e["evidence_ids"],now()))
            # Backfill source classification from input payload when possible.
            rows = con.execute("""SELECT c.id,c.evidence_ids,a.run_id FROM candidates c
                JOIN candidate_audit a ON a.candidate_id=c.id""").fetchall()
            for item in rows:
                run = con.execute("SELECT input_payload FROM extraction_runs WHERE id=?", (item["run_id"],)).fetchone()
                if not run:
                    continue
                index = {r.get("id"):r for r in json.loads(run[0]).get("messages",[])}
                for mid in json.loads(item["evidence_ids"]):
                    r = index.get(mid)
                    if r:
                        kind = evidence_kind(r.get("role"))
                        con.execute("""INSERT OR IGNORE INTO candidate_evidence_sources
                            (candidate_id,record_id,role,evidence_kind,platform)
                            VALUES (?,?,?,?,?)""",(item["id"],mid,r.get("role","unknown"),kind,r.get("platform","unknown")))

    def save_reviewed_extraction(self, batch, model, episodes, memories, flags, run_id):
        result = super().save_reviewed_extraction(batch, model, episodes, memories, flags, run_id)
        eligible = {r["id"]: r for r in batch["items"]}
        with self.connect() as con:
            rows = con.execute("SELECT id,local_id FROM extracted_episodes WHERE run_id=?",(run_id,)).fetchall()
            loc_to_id = {r["local_id"]:r["id"] for r in rows}
            for ep in episodes:
                eid=loc_to_id[ep["id"]]
                times = sorted(filter(None, (utc_iso(eligible[i].get("event_time")) for i in ep["evidence_ids"])))
                status = ep.get("event_status") or {"completed":"occurred","planned":"discussion","ongoing":"discussion","expression":"expression"}.get(ep.get("status"),"uncertain")
                if status not in EVENT_STATES: status="uncertain"
                con.execute("""UPDATE extracted_episodes SET event_status=?,review_status='pending',
                    time_start_utc=?,time_end_utc=?,time_zone=? WHERE id=?""",
                    (status,times[0] if times else None,times[-1] if times else None,LOCAL_TZ,eid))
                for p in ep.get("plans",[]):
                    # _validate_outputs checks plan evidence is part of current batch.
                    con.execute("""INSERT INTO episode_plans
                      (id,episode_id,content,status,evidence_ids,created_at)
                      VALUES (?,?,?,?,?,?)""",("plan_"+uuid.uuid4().hex[:16],eid,p["content"],p["status"],canonical(p["evidence_ids"]),now()))
                # Old prompt's combined state is only a hint, not proof of execution.
                if not ep.get("plans") and ep.get("status") in {"planned","ongoing"}:
                    con.execute("""INSERT INTO episode_plans
                      (id,episode_id,content,status,evidence_ids,created_at) VALUES (?,?,?,?,?,?)""",
                      ("plan_"+uuid.uuid4().hex[:16],eid,"（旧版推断）可能存在未完成事项，请查看原始事件后确认",
                       "uncertain",canonical(ep["evidence_ids"]),now()))
            cand = con.execute("SELECT id,content,fact_key FROM candidates WHERE batch_key=?",(batch["key"],)).fetchall()
            c_map = {(r["content"],r["fact_key"]):r["id"] for r in cand}
            for m in memories:
                cid=c_map.get((m["content"].strip(),m.get("fact_key") or ""))
                if cid:
                    ps=m.get("plan_status") if m["category"]=="plan" else None
                    if ps not in PLAN_STATES: ps="uncertain" if m["category"]=="plan" else None
                    con.execute("UPDATE candidates SET plan_status=? WHERE id=?",(ps,cid))
                    for mid in dict.fromkeys(m["evidence_ids"]):
                        r=eligible[mid]
                        con.execute("""INSERT OR IGNORE INTO candidate_evidence_sources
                          (candidate_id,record_id,role,evidence_kind,platform) VALUES (?,?,?,?,?)""",
                          (cid,mid,r["role"],evidence_kind(r["role"]),r["platform"]))
        return result

    def approve(self, candidate_id, *, edit=None, replace_id=None):
        result = super().approve(candidate_id, edit=edit, replace_id=replace_id)
        if result.get("id") and result.get("status")=="active":
            with self.connect() as con:
                con.execute("UPDATE memories SET plan_status=(SELECT plan_status FROM candidates WHERE id=?) WHERE id=?",(candidate_id,result["id"]))
        return result

    def get_run(self, run_id):
        result=super().get_run(run_id)
        result["episodes"]=[self.get_episode(e["id"]) for e in result["episodes"]]
        return result

    def get_episode(self, episode_id, *, core=None, include_raw=False):
        with self.connect() as con:
            r=con.execute("SELECT * FROM extracted_episodes WHERE id=?",(episode_id,)).fetchone()
            if not r: raise KeyError(f"没有找到情景事件 {episode_id}")
            obj=dict(r)
            obj["evidence_ids"]=json.loads(obj["evidence_ids"])
            obj["time_start_local"]=local_iso(obj.get("time_start_utc"),obj["time_zone"])
            obj["time_end_local"]=local_iso(obj.get("time_end_utc"),obj["time_zone"])
            obj["plans"]=[{**dict(x),"evidence_ids":json.loads(x["evidence_ids"])} for x in con.execute("SELECT * FROM episode_plans WHERE episode_id=? ORDER BY rowid",(episode_id,))]
            run=con.execute("SELECT input_payload FROM extraction_runs WHERE id=?",(obj["run_id"],)).fetchone()
            sources={r.get("id"):r for r in (json.loads(run[0]).get("messages",[]) if run else [])}
            obj["evidence_sources"]=[{"id":i,"role":sources.get(i,{}).get("role","unknown"),
                "kind":evidence_kind(sources.get(i,{}).get("role")),
                "platform":sources.get(i,{}).get("platform","unknown")} for i in obj["evidence_ids"]]
            # Legacy prose could use UTC's calendar date without converting to
            # Asia/Shanghai. Warn for review instead of silently rewriting text.
            obj["audit_warnings"]=[]
            start_date=(obj["time_start_local"] or "")[:10]
            end_date=(obj["time_end_local"] or "")[:10]
            prose_dates=set(re.findall(r"20\d{2}-\d{2}-\d{2}",obj["detail"]+" "+obj["summary"]))
            if start_date and start_date==end_date and any(d!=start_date for d in prose_dates):
                obj["audit_warnings"].append("摘要中的日期与原始 UTC 转换后的北京时间不一致，请审核修改。")
            obj["linked_candidates"]=[dict(x) for x in con.execute("""SELECT c.id,c.status,c.accepted_memory_id,c.content FROM candidates c
                JOIN candidate_episode_links l ON l.candidate_id=c.id WHERE l.episode_id=?""",(episode_id,))]
        if core and include_raw:
            originals=_record_index(core)
            with self.connect() as con:
                run=con.execute("SELECT input_payload FROM extraction_runs WHERE id=?",(obj["run_id"],)).fetchone()
            input_payload=json.loads(run[0]) if run else {}
            source_records=input_payload.get("messages",[])
            if input_payload.get("experiment_sample_id"):
                for item in source_records:
                    originals[item["id"]]={**item,"created_at":item.get("event_time")}
            idx={r.get("id"):n for n,r in enumerate(source_records)}
            positions=[idx[i] for i in obj["evidence_ids"] if i in idx]
            # Include uncited messages between first and last evidence, in order,
            # clearly marked. This is a context span, NOT a new proof of a claim.
            if positions:
                left,right=min(positions),max(positions)
                # If the final cited message is a user request, preserve its
                # immediate assistant/tool response as contextual raw dialogue.
                # No adjacent turn is promoted to supporting evidence.
                while right+1<len(source_records) and right+1<=max(positions)+3:
                    following=source_records[right+1]
                    if following.get("role")=="user" or following.get("platform")!=source_records[right].get("platform"):
                        break
                    right+=1
                span=[r["id"] for r in source_records[left:right+1]]
            else:
                span=obj["evidence_ids"]
            obj["raw_scope"]="source_span_between_cited_evidence"
            obj["raw_dialogue"]=[{"id":i,"role":originals[i].get("role", "autonomous"),
                                   "platform":originals[i].get("platform","unknown"),"content":originals[i].get("content"),
                                   "cited_as_evidence":i in obj["evidence_ids"],
                                   "event_time_utc":utc_iso(originals[i].get("created_at")),
                                   "event_time_local":local_iso(originals[i].get("created_at")),
                                   "game_time":originals[i].get("game_time"),
                                   "edited_for_experiment":bool(originals[i].get("edited_for_experiment",False))}
                                  if i in originals else {"id":i,"missing":True,"cited_as_evidence":i in obj["evidence_ids"]}
                                  for i in span]
        return obj

    def episode_pending(self, limit=50):
        with self.connect() as con:
            ids=[r[0] for r in con.execute("SELECT id FROM extracted_episodes WHERE review_status IN ('pending','auto_active') ORDER BY rowid LIMIT ?",(max(0,limit),))]
        return [self.get_episode(i) for i in ids]

    def review_episode(self, episode_id, *, approve:bool, summary=None, detail=None, note=""):
        if summary is not None and not 2<=len(summary.strip())<=450: raise ValueError("摘要字数需在 2～450 内")
        if detail is not None and not 2<=len(detail.strip())<=1500: raise ValueError("详情字数需在 2～1500 内")
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row=con.execute("SELECT review_status FROM extracted_episodes WHERE id=?",(episode_id,)).fetchone()
            if not row: raise KeyError(f"情景事件不存在：{episode_id}")
            if row[0] not in ("pending","auto_active"): raise ValueError("只能审核 pending/auto_active 事件")
            con.execute("""UPDATE extracted_episodes SET review_status=?,summary=COALESCE(?,summary),
                detail=COALESCE(?,detail),reviewed_at=?,review_note=? WHERE id=?""",
                ("approved" if approve else "rejected",summary.strip() if summary else None,
                 detail.strip() if detail else None,now(),note[:500],episode_id))
        return self.get_episode(episode_id)

    def candidate_evidence(self, candidate_id, *, core=None):
        candidate=self.get_candidate(candidate_id)
        with self.connect() as con:
            roles=[dict(r) for r in con.execute("SELECT record_id,role,evidence_kind,platform FROM candidate_evidence_sources WHERE candidate_id=?",(candidate_id,))]
        index=_record_index(core) if core else {}
        if core:
            with self.connect() as con:
                run=con.execute("""SELECT r.input_payload FROM candidate_audit a
                    JOIN extraction_runs r ON r.id=a.run_id
                    WHERE a.candidate_id=?""",(candidate_id,)).fetchone()
            payload=json.loads(run[0]) if run else {}
            if payload.get("experiment_sample_id"):
                index.update({m["id"]:m for m in payload.get("messages",[])})
        by_id={r["record_id"]:r for r in roles}
        result=[]
        for rid in candidate["evidence_ids"]:
            meta=by_id.get(rid,{})
            source=index.get(rid,{})
            result.append({"id":rid,"role":meta.get("role",source.get("role","unknown")),
                           "kind":meta.get("evidence_kind",evidence_kind(source.get("role"))),
                           "platform":meta.get("platform",source.get("platform","unknown")),
                           "content":source.get("content") if core else None})
        candidate["plan_status"]=candidate.get("plan_status")
        candidate["evidence_sources"] = result
        # A mixed-source record must not be labelled simply user_statement.
        kinds={r["kind"] for r in result}
        candidate["source_summary"]=(next(iter(kinds)) if len(kinds)==1 else "mixed:"+",".join(sorted(kinds)))
        return candidate

    def recall_path(self, core, question, *, level="summary", limit=3, raw_limit=12):
        """Approved memory -> separately approved episode -> original source span.

        Query may match a memory OR its episode; non-reviewed events never appear.
        """
        if level not in {"summary","detail","raw"}: raise ValueError("level 只能是 summary/detail/raw")
        if limit<=0: return {"question":question,"level":level,"items":[]}
        with self.connect() as con:
            linked=con.execute("""SELECT e.id AS episode_id, m.id AS memory_id, m.content,
                m.importance FROM extracted_episodes e
                JOIN candidate_episode_links l ON l.episode_id=e.id
                JOIN memories m ON m.origin_candidate_id=l.candidate_id
                WHERE e.review_status IN ('approved','auto_active') AND m.status='active'""").fetchall()
        rank=[]
        for row in linked:
            ep=self.get_episode(row["episode_id"])
            score=_score(question,row["content"]+" "+ep["title"]+" "+ep["summary"]+" "+ep["detail"])
            if score>0:
                rank.append((score,float(row["importance"]),dict(row),ep))
        # A separately approved episode can be found even if none of its
        # atomic candidate facts were approved; no pending event is exposed.
        with self.connect() as con:
            solo = con.execute("SELECT id FROM extracted_episodes WHERE review_status IN ('approved','auto_active')").fetchall()
        covered={m["episode_id"] for _,_,m,_ in rank}
        for r in solo:
            if r["id"] in covered: continue
            ep=self.get_episode(r["id"])
            score=_score(question,ep["title"]+" "+ep["summary"]+" "+ep["detail"])
            if score>0:
                rank.append((score,0.0,{"episode_id":ep["id"],"memory_id":None,"content":None},ep))
        rank.sort(key=lambda row:(-row[0],-row[1]))
        results=[]; seen=set()
        for score,imp,m,ep in rank:
            if ep["id"] in seen:continue
            seen.add(ep["id"])
            entry={"memory_id":m["memory_id"],"memory":m["content"],"episode_id":ep["id"],
                   "episode_title":ep["title"],"episode_summary":ep["summary"],
                   "event_status":ep["event_status"],"plans":ep["plans"],
                   "time_start_local":ep["time_start_local"],"time_end_local":ep["time_end_local"],
                   "time_zone":ep["time_zone"],"evidence_count":len(ep["evidence_ids"])}
            if level in {"detail","raw"}:entry["episode_detail"]=ep["detail"]
            if level=="raw":
                raw=self.get_episode(ep["id"],core=core,include_raw=True)["raw_dialogue"]
                entry["raw_dialogue"]=raw[:max(0,raw_limit)]
                entry["raw_truncated"]=len(raw)>max(0,raw_limit)
            results.append(entry)
            if len(results)>=limit:break
        return {"question":question,"level":level,"items":results}

    def compare_runs(self, first_id, second_id):
        first=self.get_run(first_id);second=self.get_run(second_id)
        a=[r["id"] for r in first["input_payload"].get("messages",[])]
        b=[r["id"] for r in second["input_payload"].get("messages",[])]
        def summary(run):
            cs=run["candidates"];eps=run["episodes"]
            return {"run_id":run["id"],"mode":run["mode"],"status":run["status"],
                    "model":run["model"],"prompt_version":run["prompt_version"],
                    "message_count":len(run["input_payload"].get("messages",[])),
                    "episodes":len(eps),"candidates":len(cs),
                    "mean_candidate_chars":round(sum(len(c["content"]) for c in cs)/len(cs),1) if cs else 0,
                    "uses_two_model_calls":run["mode"]=="two-pass"}
        return {"same_message_ids_in_order":a==b,
                "comparable":a==b and first["mode"]!=second["mode"],
                "first":summary(first),"second":summary(second),
                "warning":"数量和长度不能证明准确率；仍需核实证据、漏记和事实归因。"
                if a==b else "两次输入不完全相同，不能把候选数量差异归因于提炼模式。"}

    def context_for_chat(self, core, question, *, budget_chars=2600):
        # User explicitly asks for exact wording/details -> expand to reviewed episode.
        level="raw" if re.search(r"原话|逐字|怎么说的|说了什么",question) else ("detail" if re.search(r"细节|具体经过|当时发生|详细|讲讲那次",question) else "summary")
        path=self.recall_path(core,question,level=level,limit=3,raw_limit=5)
        chunks=[]
        for item in path["items"]:
            short=(f"- 已审核记忆：{item['memory']}\n  " if item.get("memory") else "- ")
            chunk=short+f"已审核事件【{item['episode_title']}】（状态:{item['event_status']}）：{item['episode_summary']}"
            if item.get("plans"):
                chunk+="\n  相关历史计划（不代表当前状态仍有效）："+"；".join(f"{p['content']}（记录状态:{p['status']}）" for p in item["plans"])
            if item.get("episode_detail"):
                chunk+="\n  事件详情："+item["episode_detail"]
            if item.get("raw_dialogue"):
                chunk+="\n  原始对话："+" | ".join(f"{r.get('role','?')}:{r.get('content','')[:250]}" for r in item["raw_dialogue"])
            if sum(len(x) for x in chunks)+len(chunk)>budget_chars: break
            chunks.append(chunk)
        return "\n".join(chunks)
