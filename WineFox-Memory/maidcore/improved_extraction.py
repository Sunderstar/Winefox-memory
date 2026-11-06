"""v0.3a.1: inspectable, rerunnable, episode-aware candidate extraction.

Independent Python implementation. Design inspiration (NOT copied source):
Yuuyuko-uu/dsh-fish-memory https://github.com/Yuuyuko-uu/dsh-fish-memory
and ARCHeON999/Companion-AI-Memory-Context-Token-Optimization.
Respect original upstream licences if substantial code is later ported.

This module never approves memory, modifies persona, deletes dialogue or makes
network calls without an explicit call to run_extraction from the CLI.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from pathlib import Path
from dataclasses import replace
from typing import Any

from .long_memory import MemoryDatabase, canonical, sha256, now, norm, CATEGORIES, SUBJECTS

VERSION = "episodes-v2.2"
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
STATUS_VALUES = {"completed", "planned", "ongoing", "expression", "uncertain"}


def read_prompt(name: str) -> str:
    names = {"one": "one_pass_v2.txt", "events": "event_stage_v2.txt", "memories": "memory_stage_v2.txt"}
    if name not in names:
        raise ValueError("未知提示词类型")
    return (PROMPTS_DIR / names[name]).read_text(encoding="utf-8").strip()


def historical_records(core, source: str) -> list[dict]:
    if source not in {"chat", "autonomous"}:
        raise ValueError("source 只能是 chat 或 autonomous")
    all_rows = core.chat_history() if source == "chat" else core.autonomous_history()
    result = []
    for row in all_rows:
        rid, content = row.get("id"), row.get("content")
        if not isinstance(rid, str) or not isinstance(content, str) or not content.strip():
            continue
        role = row.get("role", "autonomous") if source == "chat" else "autonomous"
        if role == "system":
            continue
        if role not in {"user", "assistant", "autonomous", "tool"}:
            role = "unknown"
        # This cap applies ONLY to the external API payload; original JSONL is untouched.
        cap = 4500
        result.append({"id": rid, "role": role,
                       "content": content[:cap], "truncated_for_api": len(content) > cap,
                       "platform": str(row.get("platform") or "unknown"),
                       "event_time": row.get("created_at"), "game_time": row.get("game_time")})
    return result


def _units(records: list[dict]) -> list[list[dict]]:
    """Keep a user message together with subsequent assistant/tool messages.

    Platform changes start a new unit; `game_time` is not treated as wall clock.
    """
    units: list[list[dict]] = []
    for r in records:
        if not units or r["role"] == "user" or r["platform"] != units[-1][-1]["platform"]:
            units.append([r])
        else:
            units[-1].append(r)
    return units


def build_batches(core, source: str, *, max_chars: int = 8000,
                  max_messages: int = 40, exclude_ids: set[str] | None = None,
                  start: int | None = None, stop: int | None = None,
                  latest_first: bool = False) -> list[dict]:
    """Batch whole user/reply units, bounded by payload size, preserving order.

    Context lookback is excluded from eligible IDs and batch checkpoints.
    Filtering processed records still preserves the original transcript order.
    """
    if not 1200 <= max_chars <= 30000:
        raise ValueError("--max-chars 必须介于 1200~30000")
    if not 4 <= max_messages <= 80:
        raise ValueError("--max-messages 必须介于 4~80")
    all_records = historical_records(core, source)
    if start is not None and start < 0:
        raise ValueError("--start 不能为负")
    if stop is not None and stop < 0:
        raise ValueError("--stop 不能为负")
    if start is not None and stop is not None and stop <= start:
        raise ValueError("--stop 必须大于 --start")
    excluded = exclude_ids or set()
    index_map = {r["id"]: i for i, r in enumerate(all_records)}
    active = [r for i, r in enumerate(all_records) if
              (start is None or i >= start) and (stop is None or i < stop)
              and r["id"] not in excluded]
    units = _units(active)
    packed: list[dict] = []
    current: list[dict] = []
    current_chars = 0
    current_platform = None

    def flush():
        nonlocal current, current_chars, current_platform
        if not current:
            return
        first_pos = index_map[current[0]["id"]]
        # One turn immediately before the batch supplies context, not evidence.
        prior = [r for r in all_records[:first_pos] if r["platform"] == current[0]["platform"]]
        previous_unit = _units(prior)[-1] if prior else []
        before = previous_unit[-8:]
        key = sha256({"version": VERSION, "source": source,
                      "ids": [(r["id"], sha256(r["content"])) for r in current]})
        packed.append({"key": key, "source": source, "items": current,
                       "context_before": before,
                       "start_index": first_pos, "end_index": index_map[current[-1]["id"]] + 1,
                       "characters": sum(len(r["content"]) for r in current),
                       "oversized_unit": len(current) > max_messages or current_chars > max_chars})
        current, current_chars, current_platform = [], 0, None

    for unit in units:
        unit_chars = sum(len(r["content"]) for r in unit)
        platform = unit[0]["platform"]
        if current and (current_platform != platform or
                        current_chars + unit_chars > max_chars or
                        len(current) + len(unit) > max_messages):
            flush()
        current.extend(unit)
        current_chars += unit_chars
        current_platform = platform
        # Rare oversized single unit: preserve it, flag it rather than silently truncating.
        if current_chars >= max_chars or len(current) >= max_messages:
            flush()
    flush()
    return list(reversed(packed)) if latest_first else packed


def _parse_object(raw: str, key: str) -> list[dict]:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, count=1, flags=re.I)
        clean = re.sub(r"\s*```\s*$", "", clean, count=1)
    try:
        obj = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError("模型未返回合法 JSON：批次未标记为已处理") from e
    if not isinstance(obj, dict) or not isinstance(obj.get(key), list):
        raise ValueError(f"返回格式需要 {key} 数组")
    return obj[key]


def _validate_outputs(batch: dict, episodes: list, memories: list, *,
                      validation_issues: list[dict] | None = None) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    """Validate each model-produced item without losing unrelated valid items.

    Invalid references must NEVER enter the approved-evidence index. With at
    least one valid reference a candidate survives with a conspicuous warning;
    with none it is quarantined in the run's diagnostic record. The original
    LLM output is always retained separately, so filtering is auditable.
    """
    issues = validation_issues if validation_issues is not None else []
    eligible = {r["id"]: r for r in batch["items"]}
    if len(episodes) > 12 or len(memories) > 16:
        raise ValueError("超过最多 12 个事件 / 16 条记忆，请减小批次或调整提示词")
    known_ids = set()
    eps = []
    for ep in episodes:
        if not isinstance(ep, dict):
            raise ValueError("事件必须是对象")
        eid = ep.get("id")
        if not isinstance(eid, str) or not re.fullmatch(r"e[1-9][0-9]{0,2}", eid) or eid in known_ids:
            raise ValueError("事件 ID 应为不重复的 e1、e2 等")
        known_ids.add(eid)
        ids = ep.get("evidence_ids")
        if not isinstance(ids,list):
            issues.append({"kind":"episode","index":len(eps),"local_id":eid,
                           "action":"quarantined","reason":"事件证据 ID 不是数组","original":ep})
            continue
        good_ids=list(dict.fromkeys(x for x in ids if isinstance(x,str) and x in eligible))[:12]
        bad_ids=[x for x in ids if not isinstance(x,str) or x not in eligible]
        if bad_ids or len(ids)>12:
            issues.append({"kind":"episode","index":len(eps),"local_id":eid,
                           "action":"evidence_filtered" if good_ids else "quarantined",
                           "reason":"存在批次外或超出上限的事件证据 ID",
                           "invalid_evidence_ids":bad_ids,"valid_evidence_ids":good_ids,
                           "original":ep if not good_ids else None})
        if not good_ids:
            continue
        ids=good_ids
        status = ep.get("status")
        if status is None and ep.get("event_status") in {"occurred","discussion","expression","uncertain"}:
            status={"occurred":"completed","discussion":"ongoing","expression":"expression","uncertain":"uncertain"}[ep["event_status"]]
        if status not in STATUS_VALUES:
            raise ValueError("事件状态必须为 completed/planned/ongoing/expression/uncertain")
        event_status=ep.get("event_status") or {"completed":"occurred","planned":"discussion", "ongoing":"discussion", "expression":"expression"}.get(status,"uncertain")
        if event_status not in {"occurred","discussion","expression","uncertain"}:
            raise ValueError("event_status 不合法")
        plans=ep.get("plans",[])
        if not isinstance(plans,list) or len(plans)>8:
            raise ValueError("plans 需要是最多 8 项数组")
        valid_plans=[]
        for plan in plans:
            if not isinstance(plan,dict) or plan.get("status") not in {"proposed","planned","ongoing","completed","cancelled","uncertain"}:
                raise ValueError("计划状态或格式不合规")
            content=str(plan.get("content", "")).strip()
            pids=plan.get("evidence_ids",[])
            good_pids=(list(dict.fromkeys(x for x in pids if isinstance(x,str) and x in eligible))[:8]
                       if isinstance(pids,list) else [])
            bad_pids=([x for x in pids if not isinstance(x,str) or x not in eligible]
                      if isinstance(pids,list) else ["<not-a-list>"])
            if not 3<=len(content)<=500 or not good_pids or not set(good_pids).intersection(ids):
                issues.append({"kind":"plan","index":len(valid_plans),"local_id":eid,
                               "action":"quarantined","reason":"计划内容或证据与事件不匹配", "original":plan})
                continue
            if bad_pids or (isinstance(pids,list) and len(pids)>8):
                issues.append({"kind":"plan","index":len(valid_plans),"local_id":eid,
                               "action":"evidence_filtered","reason":"计划引用了批次外证据 ID",
                               "invalid_evidence_ids":bad_pids,"valid_evidence_ids":good_pids})
            valid_plans.append({"content":content,"status":plan["status"],"evidence_ids":good_pids})
        ep2 = {"id": eid, "title": str(ep.get("title", "")).strip(),
               "detail": str(ep.get("detail", "")).strip(),
               "summary": str(ep.get("summary", "")).strip(),
               "status": status, "event_status":event_status,"plans":valid_plans,
               "evidence_ids": list(dict.fromkeys(ids))}
        if any(not 2 <= len(ep2[f]) <= limit for f,limit in [("title",120),("detail",1500),("summary",450)]):
            raise ValueError("事件标题、详情或摘要长度不合规")
        eps.append(ep2)
    warnings: dict[str, list[str]] = {}
    cleaned = []
    for idx, mem in enumerate(memories):
        if not isinstance(mem, dict):
            issues.append({"kind":"candidate","index":idx,"action":"quarantined",
                           "reason":"候选不是 JSON 对象","original":mem})
            continue
        linked = mem.get("episode_ids", [])
        if not isinstance(linked,list):
            linked=[]
        bad_links=[i for i in linked if not isinstance(i,str) or i not in {e['id'] for e in eps}]
        linked=[i for i in linked if isinstance(i,str) and i in {e['id'] for e in eps}]
        if bad_links:
            issues.append({"kind":"candidate","index":idx,"action":"link_filtered",
                           "reason":"候选引用了不存在或已隔离的 Episode",
                           "invalid_episode_ids":bad_links})
        m2 = {k: mem.get(k) for k in ("content", "category", "subject", "fact_key", "importance", "confidence", "evidence_ids")}
        if m2["category"] not in CATEGORIES or m2["subject"] not in SUBJECTS:
            issues.append({"kind":"candidate","index":idx,"action":"quarantined",
                           "reason":"候选类别或主体无效","original":mem})
            continue
        try:
            key=m2.get('fact_key') or ''
            content=m2.get('content')
            importance=float(m2.get('importance'))
            confidence=float(m2.get('confidence'))
            valid_shape=(isinstance(content,str) and 3<=len(content.strip())<=650
                         and isinstance(key,str) and len(key)<=120
                         and (not key or bool(re.fullmatch(r'[a-z0-9_.:-]+',key)))
                         and 0<=importance<=1 and 0<=confidence<=1)
        except (ValueError,TypeError):
            valid_shape=False
        if not valid_shape:
            issues.append({"kind":"candidate","index":idx,"action":"quarantined",
                           "reason":"候选正文、事实键或评分不合法","original":mem})
            continue
        ids = m2["evidence_ids"]
        if not isinstance(ids,list):
            issues.append({"kind":"candidate","index":idx,"action":"quarantined",
                           "reason":"候选证据不是数组","original":mem})
            continue
        valid_ids=list(dict.fromkeys(i for i in ids if isinstance(i,str) and i in eligible))[:8]
        invalid_ids=[i for i in ids if not isinstance(i,str) or i not in eligible]
        if invalid_ids or len(ids)>8:
            issues.append({"kind":"candidate","index":idx,
                           "action":"evidence_filtered" if valid_ids else "quarantined",
                           "reason":"候选引用了批次以外的消息 ID 或超出引用数量上限",
                           "invalid_evidence_ids":invalid_ids,"valid_evidence_ids":valid_ids,
                           "original":mem if not valid_ids else None})
        if not valid_ids:
            continue
        ids=valid_ids
        m2['evidence_ids']=ids
        c = str(m2.get("content") or "")
        flag = []
        if invalid_ids or len(mem['evidence_ids'])>8:
            flag.append("原模型引用了批次外/过量证据；已经过滤无效引用，批准前请核对原始对话")
        if bad_links:
            flag.append("原模型关联了不存在/已隔离的情景事件；已移除无效关联")
        roles = {eligible[i]["role"] for i in ids}
        if roles <= {"assistant", "autonomous"} and m2["category"] != "self_expression":
            flag.append("仅酒狐表达为证据，不能证明客观事件；建议人工核查")
        if m2["category"] == "event" and m2.get("fact_key"):
            flag.append("事件不应占用可更新事实键；已清空 fact_key")
            m2["fact_key"] = ""
        m2['importance']=importance
        m2['confidence']=confidence
        if m2["category"] == "plan" and re.search(r"已经完成|成功完成|完成了|实现了", c):
            flag.append("计划类候选出现完成式表述，请复核")
        if linked and not set(ids).intersection(set().union(*(set(e["evidence_ids"]) for e in eps if e["id"] in linked))):
            flag.append("候选与关联事件未共享证据，请复核")
        if any(eligible[i].get("truncated_for_api") for i in ids):
            flag.append("来源原文对 API 截断，需在原始 JSONL 中查看完整内容")
        warnings[str(len(cleaned))] = flag
        if m2["category"] == "plan":
            ps=mem.get("plan_status", "uncertain")
            if ps not in {"proposed","planned","ongoing","completed","cancelled","uncertain"}:
                flag.append("计划状态无效；已降级为 uncertain，需人工核对")
                ps="uncertain"
            m2["plan_status"]=ps
        else:
            m2["plan_status"]=None
        m2["episode_ids"] = list(dict.fromkeys(linked))
        cleaned.append(m2)
    # Cross-check fragmentary extraction: multiple event memories pointing to
    # the same episode are not necessarily wrong, but deserve human review.
    for eid in known_ids:
        indexes = [i for i,m in enumerate(cleaned)
                   if eid in m["episode_ids"] and m["category"] == "event"]
        if len(indexes) > 1:
            for i in indexes:
                warnings[str(i)].append("同一事件产生多条事件型记忆，建议合并重复细节")
    # Conservative duplicate hints. Never silently merge or discard evidence.
    def grams(s):
        s = re.sub(r"\s+", "", s.casefold())
        return {s[i:i+2] for i in range(max(0,len(s)-1))}
    for i in range(len(cleaned)):
        for j in range(i):
            if cleaned[i]["category"] != cleaned[j]["category"]:
                continue
            a,b=grams(cleaned[i]["content"]),grams(cleaned[j]["content"])
            if a and b and len(a&b)/len(a|b)>=0.76:
                warnings[str(i)].append("与本批另一条候选文字高度相似，请检查重复")
                warnings[str(j)].append("与本批另一条候选文字高度相似，请检查重复")
    return eps, cleaned, warnings


class InspectableMemoryDatabase(MemoryDatabase):
    """Add run logs and episode tables without changing existing v0.3a tables."""
    def __init__(self, path):
        super().__init__(path)
        with self.connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS extraction_runs (
                    id TEXT PRIMARY KEY, batch_key TEXT NOT NULL,
                    prompt_version TEXT NOT NULL, prompt_sha TEXT NOT NULL,
                    source TEXT NOT NULL, model TEXT NOT NULL, mode TEXT NOT NULL,
                    status TEXT NOT NULL, input_payload TEXT NOT NULL,
                    system_prompt TEXT NOT NULL, stage1_prompt TEXT,
                    stage1_raw TEXT, model_raw TEXT, error_text TEXT,
                    created_at TEXT NOT NULL, finished_at TEXT,
                    candidate_count INTEGER NOT NULL DEFAULT 0);
                CREATE INDEX IF NOT EXISTS idx_run_created ON extraction_runs(created_at);
                CREATE TABLE IF NOT EXISTS extracted_episodes (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, local_id TEXT NOT NULL,
                    title TEXT NOT NULL, detail TEXT NOT NULL, summary TEXT NOT NULL,
                    state TEXT NOT NULL, evidence_ids TEXT NOT NULL,
                    UNIQUE(run_id, local_id));
                CREATE TABLE IF NOT EXISTS candidate_episode_links (
                    candidate_id TEXT NOT NULL, episode_id TEXT NOT NULL,
                    PRIMARY KEY(candidate_id, episode_id));
                CREATE TABLE IF NOT EXISTS candidate_audit (
                    candidate_id TEXT PRIMARY KEY, warnings TEXT NOT NULL,
                    run_id TEXT NOT NULL);
            """)
            columns = {row[1] for row in con.execute("PRAGMA table_info(extraction_runs)")}
            if "usage_json" not in columns:
                con.execute("ALTER TABLE extraction_runs ADD COLUMN usage_json TEXT")
            if "model_options_json" not in columns:
                con.execute("ALTER TABLE extraction_runs ADD COLUMN model_options_json TEXT")
            if "validation_issues_json" not in columns:
                con.execute("ALTER TABLE extraction_runs ADD COLUMN validation_issues_json TEXT")

    def record_validation_issues(self, run_id, issues):
        """Persist quarantined raw proposals and every filtered reference for review."""
        with self.connect() as con:
            con.execute("UPDATE extraction_runs SET validation_issues_json=? WHERE id=?",
                        (canonical(issues),run_id))

    def start_run(self, batch, *, model, mode, prompt, payload, extra_prompt=None, model_options=None):
        rid = "run_" + uuid.uuid4().hex[:16]
        with self.connect() as con:
            con.execute("""INSERT INTO extraction_runs
              (id,batch_key,prompt_version,prompt_sha,source,model,mode,status,
               input_payload,system_prompt,stage1_prompt,created_at,model_options_json)
              VALUES (?,?,?,?,?,?,?,'running',?,?,?,?,?)""",
              (rid,batch["key"], VERSION, sha256([prompt, extra_prompt]),batch["source"],
               model,mode,canonical(payload),prompt,extra_prompt,now(),canonical(model_options or {})))
        return rid

    def finish_run(self, run_id, *, status, stage1_raw=None, model_raw=None, error=None, count=0, usage=None):
        with self.connect() as con:
            con.execute("""UPDATE extraction_runs SET status=?,stage1_raw=?,model_raw=?,
                           error_text=?,finished_at=?,candidate_count=?,usage_json=? WHERE id=?""",
                        (status,stage1_raw,model_raw,str(error)[:1200] if error else None,
                         now(),count,canonical(usage or []),run_id))

    def save_reviewed_extraction(self, batch, model, episodes, memories, flags, run_id):
        # Use existing v0.3a candidate/processed-record schema to preserve compatibility.
        to_save = [{k:v for k,v in mem.items() if k not in {"episode_ids", "plan_status"}} for mem in memories]
        result = self.save_candidates(batch, model, to_save)
        # The old schema includes a UNIQUE(batch_key,content,fact_key); map by index-safe content.
        with self.connect() as con:
            candidates = con.execute("SELECT id,content,fact_key FROM candidates WHERE batch_key=?", (batch["key"],)).fetchall()
            by_key = {(c["content"], c["fact_key"]): c["id"] for c in candidates}
            local_to_db = {}
            for ep in episodes:
                db_id = "ep_" + uuid.uuid4().hex[:16]
                local_to_db[ep["id"]] = db_id
                con.execute("""INSERT INTO extracted_episodes
                  (id,run_id,local_id,title,detail,summary,state,evidence_ids)
                  VALUES (?,?,?,?,?,?,?,?)""",
                  (db_id,run_id,ep["id"],ep["title"],ep["detail"],ep["summary"],ep["status"],canonical(ep["evidence_ids"])))
            for idx, mem in enumerate(memories):
                cid = by_key.get((mem["content"].strip(),mem.get("fact_key") or ""))
                if not cid:
                    continue
                con.execute("INSERT OR REPLACE INTO candidate_audit VALUES (?,?,?)",(cid,canonical(flags[str(idx)]),run_id))
                for eid in mem["episode_ids"]:
                    con.execute("INSERT OR IGNORE INTO candidate_episode_links VALUES (?,?)", (cid, local_to_db[eid]))
        return result

    def list_runs(self, limit=30):
        with self.connect() as con:
            return [dict(r) for r in con.execute("""SELECT id,batch_key,prompt_version,prompt_sha,
              source,model,mode,status,created_at,finished_at,candidate_count,error_text
              FROM extraction_runs ORDER BY rowid DESC LIMIT ?""", (limit,))]

    def get_run(self, run_id):
        with self.connect() as con:
            row = con.execute("SELECT * FROM extraction_runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"找不到提炼运行 {run_id}")
            events = [dict(r) for r in con.execute("SELECT * FROM extracted_episodes WHERE run_id=? ORDER BY rowid", (run_id,))]
            for e in events:
                e["evidence_ids"] = json.loads(e["evidence_ids"])
            cand = [self._record_from_row(r) for r in con.execute("SELECT * FROM candidates WHERE batch_key=? ORDER BY rowid", (row["batch_key"],))]
            for c in cand:
                a = con.execute("SELECT warnings FROM candidate_audit WHERE candidate_id=?",(c["id"],)).fetchone()
                c["audit_warnings"] = json.loads(a[0]) if a else []
                c["episode_ids"] = [r[0] for r in con.execute(
                    "SELECT e.local_id FROM candidate_episode_links l "
                    "JOIN extracted_episodes e ON e.id=l.episode_id WHERE l.candidate_id=?",
                    (c["id"],))]
            obj = dict(row)
            obj["input_payload"] = json.loads(obj["input_payload"])
            obj["usage"] = json.loads(obj.pop("usage_json") or "[]")
            obj["model_options"] = json.loads(obj.pop("model_options_json") or "{}")
            obj["validation_issues"] = json.loads(obj.pop("validation_issues_json") or "[]")
            obj["episodes"] = events
            obj["candidates"] = cand
            return obj

    def pending_with_audit(self, limit=100):
        items = self.pending(limit)
        with self.connect() as con:
            for c in items:
                r = con.execute("SELECT warnings,run_id FROM candidate_audit WHERE candidate_id=?",(c["id"],)).fetchone()
                c["audit_warnings"] = json.loads(r[0]) if r else []
                c["extraction_run_id"] = r[1] if r else None
                c["episode_ids"] = [x[0] for x in con.execute(
                    "SELECT e.local_id FROM candidate_episode_links l "
                    "JOIN extracted_episodes e ON e.id=l.episode_id WHERE l.candidate_id=?",
                    (c["id"],))]
        return items


def _payload(batch, known_memories):
    from .episode_memory import local_iso
    def localize(r):
        return {**r,"event_time_local": local_iso(r.get("event_time"))}
    return {"source":batch["source"], "message_count":len(batch["items"]),
            "display_timezone":"Asia/Shanghai",
            "context_before":[localize(r) for r in batch["context_before"]],
            "messages":[localize(r) for r in batch["items"]],
            "known_memories":known_memories,
            "notes":"context_before 仅用于理解背景，不允许作为新记忆的独立证据。",
            "experiment_sample_id":batch.get("experiment_sample_id"),
            "experiment_only":bool(batch.get("experiment_only",False)),
            "selected_indices":batch.get("selected_indices",[]),
            "reprocess_trial":bool(batch.get("original_batch_key"))}


def known_memories_for_batch(db, batch):
    """Freeze duplicate-awareness notes so A/B runs see identical context."""
    query = " ".join(r["content"][:150] for r in batch["items"] if r["role"] == "user")[:1800]
    notes = [{"content":m["content"][:260], "fact_key":m.get("fact_key") or "", "status":"approved"}
             for m in db.retrieve(query, limit=8)]
    for m in db.pending(limit=30):
        if any(norm(m["content"]) == norm(x["content"]) for x in notes):
            continue
        notes.append({"content":m["content"][:260], "fact_key":m.get("fact_key") or "", "status":"pending"})
        if len(notes) >= 12:
            break
    return notes


def run_extraction(db: InspectableMemoryDatabase, batch: dict, client, *, mode="one-pass", verify=False, verifier_client=None, known_memories_override=None):
    """Explicit 1- or 2-call extraction with full local logs and human review.

    Note: model confidence alone never approves a memory. Reprocess requires a
    different batch key; checkpoints remain intact.
    """
    if mode not in {"one-pass", "two-pass"}:
        raise ValueError("mode 只能是 one-pass 或 two-pass")
    if verify and not hasattr(db,"store_verification"):
        raise ValueError("独立核验需要 VerifiedMemoryDatabase；旧数据库接口不会被直接修改")
    if db.already_processed(batch["key"]):
        return {"skipped": True, "created": 0, "reason": "batch_already_processed"}
    notes = (known_memories_for_batch(db,batch) if known_memories_override is None
             else json.loads(canonical(known_memories_override)))
    payload = _payload(batch,notes)
    p1 = read_prompt("one" if mode == "one-pass" else "events")
    p2 = read_prompt("memories") if mode == "two-pass" else None
    run_id = db.start_run(batch,model=client.config.model,mode=mode,prompt=p1,payload=payload,
                          extra_prompt=p2,model_options={
                              "reasoning_effort":getattr(client.config,"reasoning_effort",None),
                              "max_tokens":getattr(client.config,"max_tokens",None),
                              "timeout":getattr(client.config,"timeout",None)})
    raw1,raw2 = None,None
    usage = []
    def save_usage(stage):
        if getattr(client,"last_usage",None):
            usage.append({"stage":stage, **client.last_usage})
    try:
        if mode == "one-pass":
            raw2 = client.complete([{"role":"system","content":p1},
                                    {"role":"user","content":"分析以下记录，只输出指定 JSON：\n"+canonical(payload)}])
            save_usage("one_pass")
            episodes = _parse_object(raw2,"episodes")
            memories = _parse_object(raw2,"memories")
        else:
            raw1 = client.complete([{"role":"system","content":p1},
                                    {"role":"user","content":"整理事件，只输出指定 JSON：\n"+canonical(payload)}])
            save_usage("episodes")
            episodes = _parse_object(raw1,"episodes")
            raw2 = client.complete([{"role":"system","content":p2},
                                    {"role":"user","content":"提取记忆，只输出指定 JSON：\n"+
                                     canonical({**payload,"episodes":episodes})}])
            save_usage("memories")
            memories = _parse_object(raw2,"memories")
        validation_issues=[]
        eps,mems,warnings = _validate_outputs(batch,episodes,memories,
                                              validation_issues=validation_issues)
        db.record_validation_issues(run_id,validation_issues)
        if not eps and not mems and (episodes or memories):
            raise ValueError("所有候选与事件均未通过校验；原始模型输出和逐项错误已经保存，请检查后重试")
        if verify:
            # Old SQLite intentionally stores one candidate for each unique
            # (batch, content, fact_key). Ambiguous repeated indexes cannot be
            # safely linked to reviewer verdicts; reject before writing.
            unique_keys={(m["content"].strip(),m.get("fact_key") or "") for m in mems}
            if len(unique_keys)!=len(mems):
                raise ValueError("同批出现完全重复的候选；为避免核验结果错位，未保存任何候选")
        # Extraction and verification have separate failure domains. A schema
        # violation or API error in stage 3 must not destroy valid stages 1/2.
        verified = None
        verification_error = None
        verification_failure_raw = None
        if verify:
            try:
                from .memory_verification import verify_proposals
                verified = verify_proposals(payload, eps, mems, verifier_client or client)
                if verified.get("usage"):
                    usage.append({"stage":"verification", **verified["usage"]})
            except Exception as exc:
                verification_error = str(exc)
                verification_failure_raw = getattr(exc,'raw',None)
                if getattr((verifier_client or client),'last_usage',None):
                    usage.append({"stage":"verification_failed",**(verifier_client or client).last_usage})
        result = db.save_reviewed_extraction(batch,client.config.model,eps,mems,warnings,run_id)
        if verified is not None:
            try:
                db.store_verification(run_id, verified)
            except Exception as exc:
                verification_error = "核验存储失败：" + str(exc)
                verified = None
        if verification_error is not None and hasattr(db,"record_verification_failure"):
            db.record_verification_failure(run_id, verification_error, raw=verification_failure_raw)
        db.finish_run(run_id,status="succeeded",stage1_raw=raw1,model_raw=raw2,
                      count=result["created"],usage=usage)
        return {**result,"run_id":run_id,"episodes":len(eps),"verified":bool(verified),"verification_issues":(sum(f["verdict"]!="supported" for f in verified["report"]["memories"].values()) if verified else None),"warnings":sum(len(x) for x in warnings.values()),"usage":usage,
                "verification_error":verification_error,
                "validation_issues":validation_issues,
                "quarantined_candidates":sum(i['kind']=='candidate' and i['action']=='quarantined' for i in validation_issues)}
    except Exception as exc:
        # Retain errors locally for debugging; do not expose API key or request headers.
        db.finish_run(run_id,status="failed",stage1_raw=raw1,model_raw=raw2,error=str(exc),usage=usage)
        raise


def make_review_sheet(db: InspectableMemoryDatabase, run_id: str):
    """Human grading scaffold; blanks are intentionally not scored automatically."""
    result=db.get_run(run_id)
    return {"run_id":run_id,"notes":"为每条 candidate 填写 supported、valuable、not_duplicate、event_complete 四个 true/false；漏记的内容写入 missed_memories。空值不计分。",
            "items":[{"candidate_id":c["id"],"content":c["content"],"evidence_ids":c["evidence_ids"],
                      "supported":None,"valuable":None,"not_duplicate":None,"event_complete":None,
                      "review_note":""} for c in result["candidates"]],"missed_memories":[]}


def grade_review_sheet(sheet: dict):
    metrics = {}
    for name in ("supported","valuable","not_duplicate","event_complete"):
        rated=[i[name] for i in sheet.get("items",[]) if isinstance(i.get(name),bool)]
        metrics[name]={"rated":len(rated),"passed":sum(rated),
                       "fraction":(round(sum(rated)/len(rated),3) if rated else None)}
    misses = sheet.get("missed_memories", [])
    if not isinstance(misses,list):
        raise ValueError("missed_memories 应是数组")
    return {"run_id":sheet.get("run_id"),"counts":{"candidates":len(sheet.get("items",[])),"manually_flagged_misses":len(misses)},
            "metrics":metrics, "note":"这些是人工标注指标；未标注项和未发掘的隐性漏记不在分母中。"}
