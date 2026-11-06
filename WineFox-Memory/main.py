"""Command-line interface to Wine Fox's local persona and memory core.

Run: python main.py --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from maidcore import MaidCore
from maidcore.llm import APIConfig, ChatCompletionsClient, ConfigError, LLMError, WineFoxChatSession, load_memory_config
from maidcore.long_memory import MemoryDatabase, make_batches, extract_batch
from maidcore.memory_verification import VerifiedMemoryDatabase, verify_existing, TEMPORAL, PLAN_STATES
from maidcore.episode_memory import EpisodeMemoryDatabase
from maidcore.improved_extraction import (InspectableMemoryDatabase, build_batches, run_extraction,
    make_review_sheet, grade_review_sheet, sha256, VERSION)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="酒狐 AI v0.3b.4 · 上下文感知召回与自主 Episode 阅读")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR, help="数据目录，默认 ./data")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("stats", help="查看文件与记忆统计")
    diagnostic=commands.add_parser("diagnose", help="离线导出可供 GitHub Issues 分享的隐私保护诊断 JSON")
    diagnostic.add_argument("--output",type=Path,default=None,help="可选导出路径；默认 ./data/diagnostics/ 下自动创建")
    commands.add_parser("check-install",help="离线检查 Python、界面和提示词文件（无需 API Key）")
    commands.add_parser("persona", help="输出当前人格文档")
    commands.add_parser("memory", help="输出长期记忆文档")
    commands.add_parser("memories", help="列出逐条记忆")
    commands.add_parser("files", help="列出最常用的可编辑文件")
    check = commands.add_parser("check-config", help="检查 API 配置，但不联网、不显示密钥")
    check.add_argument("--env-file", type=Path, default=PROJECT_DIR / ".env")
    converse = commands.add_parser("chat", help="在终端与酒狐实时聊天（会调用 API）")
    converse.add_argument("--env-file", type=Path, default=PROJECT_DIR / ".env")
    converse.add_argument("--once", metavar="MESSAGE", help="只发送一条消息，获得回复后退出")

    imp = commands.add_parser("import-maid", help="从 Minecraft .maid 档案导入（只读源文件）")
    imp.add_argument("file", type=Path)
    imp.add_argument("--persona-yml", type=Path, default=PROJECT_DIR / "defaults" / "winefox.yml")
    impzip = commands.add_parser("import-zip", help="从此前导出的 ZIP 导入")
    impzip.add_argument("file", type=Path)

    update = commands.add_parser("set-persona", help="将 UTF-8 文件内容写成人格文档")
    update.add_argument("file", type=Path)
    update = commands.add_parser("set-memory-doc", help="将 UTF-8 文件内容写成记忆文档")
    update.add_argument("file", type=Path)

    remember = commands.add_parser("remember", help="添加一条带来源的结构化记忆")
    remember.add_argument("text")
    remember.add_argument("--category", choices=["fact", "preference", "event", "plan", "relationship", "other"], default="fact")
    remember.add_argument("--source", default="manual")
    remember.add_argument("--tags", default="", help="用逗号分隔标签")

    edit = commands.add_parser("edit-memory", help="编辑已有结构化记忆")
    edit.add_argument("id")
    edit.add_argument("text")
    delete = commands.add_parser("forget", help="删除一条结构化记忆（不删除原始聊天）")
    delete.add_argument("id")

    chat = commands.add_parser("add-chat", help="添加一条用户或女仆聊天")
    chat.add_argument("role", choices=["user", "assistant"])
    chat.add_argument("text")
    chat.add_argument("--platform", default="terminal", help="可选 minecraft / qq / terminal")
    talk = commands.add_parser("add-self-talk", help="添加一条自主发言")
    talk.add_argument("text")
    talk.add_argument("--platform", default="terminal")

    hist = commands.add_parser("history", help="查看最近聊天")
    hist.add_argument("--limit", type=int, default=10)
    hist = commands.add_parser("self-talk", help="查看最近自主发言")
    hist.add_argument("--limit", type=int, default=10)
    search = commands.add_parser("search", help="全文关键字查询（不需要大模型）")
    search.add_argument("keyword")
    search.add_argument("--limit", type=int, default=10)
    recall = commands.add_parser("recall", help="自然语言查询关联片段（中英文简单离线词组匹配）")
    recall.add_argument("question")
    recall.add_argument("--limit", type=int, default=8)
    ctx = commands.add_parser("context", help="预览下一次模型请求会使用的上下文（不会真正调用 API）")
    ctx.add_argument("message")
    ctx.add_argument("--recent", type=int, default=8)
    ctx.add_argument("--recall-limit", type=int, default=8)
    cfgcmd = commands.add_parser("memory-retrieval-settings",help="查看当前的记忆预算和召回参数（不联网）")
    prev = commands.add_parser("memory-retrieval-preview",help="预览下轮会选中的 Cand/Episode（不联网）")
    prev.add_argument("question")
    prev.add_argument("--seed",type=int,default=0,help="固定随机种子，便于做 A/B 对照")
    commands.add_parser("memory-core-preview",help="显示新增已批准来源和 AI Core（不调用 API）")
    auto=commands.add_parser("memory-core-maintain",help="一次整理新增批准的 Cand/Episode；--run 才调用模型")
    auto.add_argument("--run",action="store_true",help="确认发起一次付费 Core 整理")
    auto.add_argument("--env-file",type=Path,default=PROJECT_DIR / ".env")
    commands.add_parser("memory-auto-settings",help="查看自动归档配置（不调用 API）")
    commands.add_parser("memory-auto-status",help="查看自动归档阈值、未处理消息与任务历史")
    autos=commands.add_parser("memory-auto-archive",help="手动运行一批自动归档；不加 --run 仅预览")
    autos.add_argument("--run",action="store_true",help="确认现在调用 API，处理一批符合条件的未提炼消息")
    autos.add_argument("--worker-job",default=None,help=argparse.SUPPRESS)
    autos.add_argument("--env-file",type=Path,default=PROJECT_DIR / ".env")
    release=commands.add_parser("memory-auto-publish",help="修复已成功提炼但未发布的自动 Archive（不调用模型）")
    release.add_argument("run_id",nargs="?",help="单次成功提炼运行 ID；批量使用 --all")
    release.add_argument("--all",action="store_true",help="扫描真实历史中尚未发布的成功提炼")
    release.add_argument("--apply",action="store_true",help="批量执行发布；默认只预览，不修改数据库")
    release.add_argument("--limit",type=int,default=1000,help="最多扫描的成功运行数，默认 1000")
    acfg=commands.add_parser("memory-auto-config",help="通过命令行开启或关闭自动归档并设置阈值")
    toggle=acfg.add_mutually_exclusive_group()
    toggle.add_argument("--enable",action="store_true")
    toggle.add_argument("--disable",action="store_true")
    acfg.add_argument("--trigger-messages",type=int,default=None)
    acfg.add_argument("--keep-recent-messages",type=int,default=None)
    acfg.add_argument("--max-messages",type=int,default=None)
    acfg.add_argument("--max-chars",type=int,default=None)
    commands.add_parser("memory-core-history",help="列出 Core 自动整理的版本及 Token 用量")
    idx=commands.add_parser("memory-embedding-stats",help="查看已持久化的 Embedding 索引")
    commands.add_parser("memory-check-config", help="查看独立记忆整理模型配置（不联网、不显示密钥）")
    ui=commands.add_parser("ui", help="打开本地浏览器记忆工作台：原文勾选/编辑、冻结样本、模型对照、批量审核")
    ui.add_argument("--port",type=int,default=8765,help="仅本机访问的端口，默认 8765")
    ui.add_argument("--no-browser",action="store_true",help="不自动打开浏览器，打印链接")
    ui.add_argument("--env-file",type=Path,default=PROJECT_DIR / ".env")
    e = commands.add_parser("memory-extract", help="预览或运行提炼；普通成功结果默认自动可用，事后可审核")
    e.add_argument("--source", choices=["chat", "autonomous", "all"], default="chat")
    e.add_argument("--batch-size", type=int, default=None, help="兼容旧命令：每批消息数上限（现在更推荐 --max-messages）")
    e.add_argument("--max-messages", type=int, default=40, help="每批最多消息数，默认 40")
    e.add_argument("--max-chars", type=int, default=8000, help="每批原文内容字符预算，默认 8000")
    e.add_argument("--latest-first", action="store_true", help="优先处理最新的未提炼聊天")
    e.add_argument("--start", type=int, default=None, help="按非空记录索引选择起点（0 开始，包含）")
    e.add_argument("--stop", type=int, default=None, help="按非空记录索引选择终点（不包含）")
    e.add_argument("--reprocess", action="store_true", help="明确重新提炼已处理记录（必须同时指定 --start 和 --stop；不会自动发布）")
    e.add_argument("--no-auto-publish", action="store_true",help="本次普通提炼只生成 pending，不自动发布")
    e.add_argument("--mode", choices=["one-pass", "two-pass"], default="one-pass",
                   help="一轮请求返回事件+记忆，或两轮请求先总结事件再提炼（费用更高）")
    verf = e.add_mutually_exclusive_group()
    verf.add_argument("--verify", dest="verify", action="store_true", help="增加第三次独立核验调用（仅 --run 才联网）")
    verf.add_argument("--no-verify", dest="verify", action="store_false", help="跳过第三次核验；默认 Two-pass 启用核验")
    e.set_defaults(verify=None)
    e.add_argument("--max-batches", type=int, default=1, help="本次最多处理多少批；防止意外产生大量费用")
    e.add_argument("--run", action="store_true", help="显式联网并调用 API；不提供此参数仅预览")
    e.add_argument("--env-file", type=Path, default=PROJECT_DIR / ".env")
    vr = commands.add_parser("memory-verify",help="对已有提炼运行单独做第三阶段事实核验，不重复提炼")
    vr.add_argument("run_id")
    vr.add_argument("--run",action="store_true",help="明确调用独立核验模型；不加只预览")
    vr.add_argument("--env-file",type=Path,default=PROJECT_DIR / ".env")
    plans=commands.add_parser("memory-plan-list",help="列出事件中的计划和当前状态")
    plans.add_argument("--status",choices=sorted(PLAN_STATES),default=None)
    plans.add_argument("--limit",type=int,default=60)
    pe=commands.add_parser("memory-plan-update",help="人工更改计划状态，并记录变更证据与原因")
    pe.add_argument("plan_id")
    pe.add_argument("--status",choices=sorted(PLAN_STATES),required=True)
    pe.add_argument("--note",required=True)
    pe.add_argument("--evidence-id",action="append",default=[])
    ph=commands.add_parser("memory-plan-history",help="查看计划状态修改历史")
    ph.add_argument("plan_id")
    mv=commands.add_parser("memory-validity-set",help="修改已批准事实的时间有效性标签")
    mv.add_argument("memory_id")
    mv.add_argument("--status",choices=sorted(TEMPORAL),required=True)
    mv.add_argument("--note",required=True)
    runs = commands.add_parser("memory-batches", help="显示实际提炼运行（包括失败与提示词版本）")
    runs.add_argument("--limit", type=int, default=20)
    show = commands.add_parser("memory-batch-show", help="查看一次提炼的完整输入、提示词、事件和模型原始输出")
    show.add_argument("run_id")
    evx = commands.add_parser("memory-eval-export", help="导出供人工评分的 JSON 表，不调用 API")
    evx.add_argument("run_id")
    evx.add_argument("--output", type=Path, default=None)
    evv = commands.add_parser("memory-eval", help="对手工填写的评估表计算指标，不调用 API")
    evv.add_argument("file", type=Path)
    commands.add_parser("memory-pending", help="查看待审核候选记忆（不调用 API）")
    ev = commands.add_parser("memory-evidence", help="查看某条候选记忆所引用的原始对话")
    ev.add_argument("id")
    a = commands.add_parser("memory-approve", help="人工审核通过候选记忆")
    a.add_argument("id")
    a.add_argument("--edit", default=None, help="在通过前修改候选内容")
    a.add_argument("--replace", default=None, help="有同事实键冲突时明确指定旧记忆 ID")
    r = commands.add_parser("memory-reject", help="拒绝候选记忆")
    r.add_argument("id")
    r.add_argument("--reason", default="")
    commands.add_parser("memory-db-list", help="列出已经审核通过的长期记忆")
    commands.add_parser("memory-db-stats", help="查看 SQLite 长期记忆统计")
    mctx = commands.add_parser("memory-db-recall", help="检索已批准的 SQLite 长期记忆")
    mctx.add_argument("question")
    eps = commands.add_parser("memory-episode-pending", help="待审核事件；仅通过后的事件可用于聊天")
    eps.add_argument("--limit",type=int,default=50)
    detail = commands.add_parser("memory-episode-show", help="查看事件详情和原始证据对话")
    detail.add_argument("id")
    detail.add_argument("--raw",action="store_true")
    review = commands.add_parser("memory-episode-review", help="独立审核情景事件")
    review.add_argument("id")
    action=review.add_mutually_exclusive_group(required=True)
    action.add_argument("--approve",action="store_true")
    action.add_argument("--reject",action="store_true")
    review.add_argument("--summary",default=None,help="审核前修改摘要")
    review.add_argument("--detail",default=None,help="审核前修改详情")
    review.add_argument("--note",default="")
    cmp = commands.add_parser("memory-compare-runs",help="比较同一输入的 One-pass 和 Two-pass 记录")
    cmp.add_argument("first_run_id")
    cmp.add_argument("second_run_id")
    path = commands.add_parser("memory-recall-path", help="三级召回：已批短记忆→已批事件→原始对话")
    path.add_argument("question")
    path.add_argument("--level",choices=["summary","detail","raw"],default="summary")
    path.add_argument("--limit",type=int,default=3)
    path.add_argument("--raw-limit",type=int,default=24)
    return parser.parse_args()


def run_chat(store: MaidCore, config: APIConfig, one_message: str | None = None, env_file=None) -> None:
    session = WineFoxChatSession(store, ChatCompletionsClient(config), platform="terminal", env_file=env_file)
    if one_message is not None:
        print("酒狐：" + session.reply(one_message))
        return
    print("酒狐 AI v0.3a.3 · 终端聊天")
    print("已加载人格和记忆。/help 查看命令，/exit 退出。")
    while True:
        try:
            line = input("\n你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n聊天结束，已成功回复的对话保存在本地。")
            break
        if not line:
            continue
        if line.lower() in {"/exit", "/quit", "/q"}:
            print("聊天结束。")
            break
        if line == "/help":
            print("输入聊天内容直接对话；/persona 查看人格；/memory 查看摘要；/stats 统计；/exit 退出。")
            print("要编辑人格或记忆，可直接修改 data/persona.md 和 data/memory.md，下一轮即生效。")
            continue
        if line == "/persona":
            print(store.read_persona())
            continue
        if line == "/memory":
            print(store.read_memory_document())
            continue
        if line == "/stats":
            print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
            continue
        try:
            print("酒狐：" + session.reply(line))
        except (LLMError, ConfigError, ValueError) as exc:
            print(f"请求失败：{exc}（此轮没有写入聊天历史，可以重试）")


def main():
    args = parse_args()
    command = args.command
    if command == "ui":
        from maidcore.review_ui import serve_ui
        serve_ui(store.root,args.env_file,port=args.port,open_browser=not args.no_browser)
        return
    if command == "diagnose":
        from maidcore.diagnostics import write_diagnostic_report
        dest=write_diagnostic_report(args.data,PROJECT_DIR,args.output)
        print("诊断报告已生成（不包含聊天原文、人格、Core、API Key、路径或 ID）：")
        print(dest)
        print("发布 GitHub Issue 前仍请亲自打开检查报告内容。")
        return
    if command == "check-install":
        import sys
        files={"ui/index.html":PROJECT_DIR / 'ui' / 'index.html',
               "ui/experiment.html":PROJECT_DIR / 'ui' / 'experiment.html',
               "prompts/event_stage_v2.txt":PROJECT_DIR / 'prompts' / 'event_stage_v2.txt',
               "prompts/memory_stage_v2.txt":PROJECT_DIR / 'prompts' / 'memory_stage_v2.txt'}
        result={"python_ok":sys.version_info>=(3,10),
                "python_version":".".join(map(str,sys.version_info[:3])),
                "files":{key:path.is_file() for key,path in files.items()},
                "api_key_needed_to_check":False,
                "ready":sys.version_info>=(3,10) and all(path.is_file() for path in files.values())}
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    # Diagnostic and installation checks above are strictly read-only.
    # Do not create a default persona or data directory when only checking.
    store = MaidCore(args.data)
    if command == "check-config":
        result = APIConfig.load(args.env_file).public_info()
    elif command == "memory-check-config":
        result = load_memory_config(args.env_file).public_info()
    elif command == "memory-retrieval-settings":
        from dataclasses import asdict
        from maidcore.retrieval import MemoryRetriever
        result = asdict(MemoryRetriever(store).get_settings())
    elif command == "memory-core-preview":
        from maidcore.retrieval import MemoryRetriever
        from maidcore.core_curator import CoreCurator
        result = CoreCurator(MemoryRetriever(store)).preview()
    elif command == "memory-core-history":
        from maidcore.retrieval import MemoryRetriever
        from maidcore.core_curator import CoreCurator
        result = CoreCurator(MemoryRetriever(store)).history()
    elif command == "memory-embedding-stats":
        from maidcore.retrieval import MemoryRetriever
        result = MemoryRetriever(store).embedding_index_stats()
    elif command == "memory-core-maintain":
        from maidcore.retrieval import MemoryRetriever
        from maidcore.core_curator import CoreCurator
        curator = CoreCurator(MemoryRetriever(store))
        result = curator.run(env_file=args.env_file,force=True) if args.run else curator.preview()
    elif command in ("memory-auto-settings","memory-auto-status","memory-auto-archive","memory-auto-config","memory-auto-publish"):
        from maidcore.auto_archive import AutoArchiveManager
        manager=AutoArchiveManager(store,env_file=getattr(args,'env_file',None))
        if command=="memory-auto-settings":
            result=manager.settings().__dict__
        elif command=="memory-auto-publish":
            if args.all:
                if args.run_id:raise ValueError("--all 不能同时传入 run_id")
                result=manager.publish_pending(dry_run=not args.apply,limit=args.limit)
            else:
                if args.apply or not args.run_id:
                    raise ValueError("单次发布请提供 run_id；全库发布使用 --all --apply")
                result=manager.publish_run(args.run_id)
        elif command=="memory-auto-status":
            result={"preview":manager.plan(),"settings":manager.settings().__dict__,"jobs":manager.jobs()}
        elif command=="memory-auto-config":
            changes={}
            if args.enable:changes['enabled']=True
            if args.disable:changes['enabled']=False
            for k in ('trigger_messages','keep_recent_messages','max_messages','max_chars'):
                v=getattr(args,k)
                if v is not None:changes[k]=v
            result=manager.update_settings(changes)
        elif args.worker_job:
            result=manager.run_once(job_id=args.worker_job)
        elif args.run:
            result=manager.run_once(force=True)
        else:
            result=manager.plan(force=True)
    elif command == "memory-retrieval-preview":
        from maidcore.retrieval import MemoryRetriever
        _context,result = store.build_messages(args.question,
            retriever=MemoryRetriever(store),return_retrieval=True,retrieval_seed=args.seed)
    elif command == "chat":
        config = APIConfig.load(args.env_file)
        run_chat(store, config, args.once,env_file=args.env_file)
        return
    elif command == "memory-extract":
        if args.max_batches <= 0:
            raise ValueError("--max-batches 必须为正整数")
        if args.reprocess and (args.start is None or args.stop is None):
            raise ValueError("--reprocess 必须明确指定 --start 与 --stop，避免重新处理整份历史")
        db = VerifiedMemoryDatabase(store.root / "memory.db")
        sources = ["chat", "autonomous"] if args.source == "all" else [args.source]
        max_msgs = args.batch_size if args.batch_size is not None else args.max_messages
        planned = []
        for source in sources:
            excluded = set() if args.reprocess else db.processed_record_ids(source)
            batches = build_batches(store,source,max_chars=args.max_chars,
                max_messages=max_msgs,exclude_ids=excluded,
                start=args.start,stop=args.stop,latest_first=args.latest_first)
            for batch in batches:
                if args.reprocess:
                    # An explicit retest is a NEW extraction, not an overwrite.
                    # Save the original content key as provenance.
                    batch["original_batch_key"] = batch["key"]
                    batch["key"] = sha256({"retake":batch["key"],"nonce":__import__('uuid').uuid4().hex})
                if not db.already_processed(batch["key"]):
                    planned.append((source,batch))
        if args.latest_first:
            planned.sort(key=lambda p: p[1]["start_index"],reverse=True)
        planned = planned[:args.max_batches]
        if not args.run:
            result = {"mode":"PREVIEW_ONLY_NO_API","pipeline":args.mode,
                "prompt_version":VERSION,"verification_will_run":(args.verify if args.verify is not None else args.mode=="two-pass"),"pending_batches_previewed":len(planned),
                "batches":[{"source":src,"message_count":len(b["items"]),
                    "char_count":b["characters"],"oversized_unit":b["oversized_unit"],
                    "start_index":b["start_index"],"end_index":b["end_index"],
                    "context_before_ids":[r["id"] for r in b["context_before"]],
                    "first_id":b["items"][0]["id"],"last_id":b["items"][-1]["id"]}
                    for src,b in planned],
                "to_run":"python main.py memory-extract --run --max-batches 1"}
        elif not planned:
            result = {"batches":0,"created":0,"message":"没有新的批次"}
        else:
            cfg = load_memory_config(args.env_file)
            client = ChatCompletionsClient(cfg)
            done = []
            for src,batch in planned:
                item=run_extraction(db,batch,client,mode=args.mode,
                    verify=(args.verify if args.verify is not None else args.mode=="two-pass"))
                if (src=='chat' and not args.reprocess and not args.no_auto_publish
                        and item.get('run_id') and not item.get('skipped')):
                    # Extraction is already durable: publication failure does not
                    # consume the paid result or cause a duplicate API request.
                    try:
                        from maidcore.auto_archive import AutoArchiveManager
                        item['publication']=AutoArchiveManager(store,db=db).publish_run(item['run_id'])
                    except Exception as exc:
                        item['publication']={'status':'failed','reason':str(exc),
                            'next':'py main.py memory-auto-publish --all --apply'}
                done.append({"source":src,"record_count":len(batch["items"]),**item})
            result = {"batches":len(done),"created":sum(i.get("created",0) for i in done),
                      "results":done,"next":"python main.py memory-batches"}
    elif command == "memory-verify":
        db=VerifiedMemoryDatabase(store.root / "memory.db")
        run=db.get_run(args.run_id)
        if not args.run:
            result={"mode":"PREVIEW_ONLY_NO_API","run_id":run["id"],"run_status":run["status"],
                    "already_verified":run["verification"] is not None,"episode_count":len(run["episodes"]),
                    "candidate_count":len(run["candidates"]),
                    "next":f"python main.py memory-verify {args.run_id} --run"}
        else:
            cfg=load_memory_config(args.env_file)
            result=verify_existing(db,args.run_id,ChatCompletionsClient(cfg))
    elif command == "memory-plan-list":
        result=VerifiedMemoryDatabase(store.root / "memory.db").plan_list(status=args.status,limit=args.limit)
    elif command == "memory-plan-update":
        result=VerifiedMemoryDatabase(store.root / "memory.db").plan_update(store,args.plan_id,status=args.status,
            reason=args.note,evidence_ids=args.evidence_id)
    elif command == "memory-plan-history":
        result=VerifiedMemoryDatabase(store.root / "memory.db").plan_history(args.plan_id)
    elif command == "memory-validity-set":
        result=VerifiedMemoryDatabase(store.root / "memory.db").set_memory_validity(args.memory_id,
            validity=args.status,note=args.note)
    elif command == "memory-batches":
        result = VerifiedMemoryDatabase(store.root / "memory.db").list_runs(args.limit)
    elif command == "memory-batch-show":
        result = VerifiedMemoryDatabase(store.root / "memory.db").get_run(args.run_id)
    elif command == "memory-eval-export":
        db = VerifiedMemoryDatabase(store.root / "memory.db")
        result = make_review_sheet(db,args.run_id)
        dest = (args.output or (store.root / (args.run_id + "_review.json"))).resolve()
        if dest.exists():
            raise ValueError("评测文件已存在，为避免覆盖请指定不同路径")
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        result = {"saved_review_template":str(dest),"items":len(result["items"]),
                  "next":f"python main.py memory-eval {dest}"}
    elif command == "memory-eval":
        result = grade_review_sheet(json.loads(args.file.read_text(encoding="utf-8")))
    elif command == "memory-pending":
        result = VerifiedMemoryDatabase(store.root / "memory.db").pending_with_audit()
    elif command == "memory-evidence":
        db = VerifiedMemoryDatabase(store.root / "memory.db")
        candidate = db.get_candidate(args.id)
        originals = {r.get("id"):r for r in (store.chat_history()+store.autonomous_history())}
        result = {"candidate":candidate,
                  "evidence":[originals[i] for i in candidate["evidence_ids"] if i in originals],
                  "missing_ids":[i for i in candidate["evidence_ids"] if i not in originals],
                  "note":"这里只显示本候选引用的证据；完整批次请 memory-batches + memory-batch-show"}
    elif command == "memory-approve":
        result = VerifiedMemoryDatabase(store.root / "memory.db").approve(args.id, edit=args.edit, replace_id=args.replace)
    elif command == "memory-reject":
        result = VerifiedMemoryDatabase(store.root / "memory.db").reject(args.id, reason=args.reason)
    elif command == "memory-db-list":
        result = VerifiedMemoryDatabase(store.root / "memory.db").all_memories()
    elif command == "memory-db-stats":
        result = VerifiedMemoryDatabase(store.root / "memory.db").stats()
    elif command == "memory-db-recall":
        result = VerifiedMemoryDatabase(store.root / "memory.db").retrieve(args.question)
    elif command == "memory-episode-pending":
        result = VerifiedMemoryDatabase(store.root / "memory.db").episode_pending(args.limit)
    elif command == "memory-episode-show":
        result = VerifiedMemoryDatabase(store.root / "memory.db").get_episode(args.id,core=store,include_raw=args.raw)
    elif command == "memory-episode-review":
        result = VerifiedMemoryDatabase(store.root / "memory.db").review_episode(
            args.id,approve=args.approve,summary=args.summary,detail=args.detail,note=args.note)
    elif command == "memory-compare-runs":
        result = VerifiedMemoryDatabase(store.root / "memory.db").compare_runs(
            args.first_run_id,args.second_run_id)
    elif command == "memory-recall-path":
        result = VerifiedMemoryDatabase(store.root / "memory.db").recall_path(
            store,args.question,level=args.level,limit=args.limit,raw_limit=args.raw_limit)
    elif command == "stats":
        result = store.stats()
        if (store.root / "memory.db").exists():
            result["reviewed_memory_db"] = VerifiedMemoryDatabase(store.root / "memory.db").stats()
    elif command == "persona":
        result = store.read_persona()
    elif command == "memory":
        result = store.read_memory_document()
    elif command == "memories":
        result = store.list_memories()
    elif command == "files":
        result = {"人格": str(store.persona_file), "记忆文档": str(store.memory_document_file), "单条记忆": str(store.memories_file), "普通聊天": str(store.chat_file), "自主发言": str(store.autonomous_file), "档案信息": str(store.info_file)}
    elif command == "import-maid":
        result = store.import_maid(args.file, default_persona_yml=args.persona_yml if args.persona_yml.exists() else None)
    elif command == "import-zip":
        result = store.import_export_zip(args.file)
    elif command == "set-persona":
        store.write_persona(args.file.read_text(encoding="utf-8"))
        result = f"已更新 {store.persona_file}"
    elif command == "set-memory-doc":
        store.write_memory_document(args.file.read_text(encoding="utf-8"))
        result = f"已更新 {store.memory_document_file}"
    elif command == "remember":
        result = store.remember(args.text, category=args.category, source=args.source, tags=[s.strip() for s in args.tags.split(",") if s.strip()])
    elif command == "edit-memory":
        result = store.update_memory(args.id, args.text)
    elif command == "forget":
        store.forget(args.id)
        result = "已删除该条结构化记忆。原始对话未修改。"
    elif command == "add-chat":
        result = store.append_chat(args.role, args.text, platform=args.platform)
        if args.role=='assistant':
            try:
                from maidcore.auto_archive import AutoArchiveManager
                mgr=AutoArchiveManager(store)
                if mgr.settings().enabled:
                    result['auto_archive']=mgr.start_background()
            except Exception as exc:
                result['archive_warning']=str(exc)[:400]
    elif command == "add-self-talk":
        result = store.append_autonomous(args.text, platform=args.platform)
    elif command == "history":
        result = store.chat_history(args.limit)
    elif command == "self-talk":
        result = store.autonomous_history(args.limit)
    elif command == "search":
        result = store.search(args.keyword, args.limit)
    elif command == "recall":
        result = store.recall(args.question, args.limit)
    elif command == "context":
        result = store.build_messages(args.message, recent=args.recent, recall_limit=args.recall_limit)
    else:
        raise RuntimeError("Unknown command")
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError, LLMError) as e:
        raise SystemExit(f"错误：{e}") from e
