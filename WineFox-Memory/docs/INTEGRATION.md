# 开发者接入指南：接口、扩展和对接流程

当前仓库提供**可导入的 Python 类、命令行和本地工作台 HTTP 接口**。QQ 和 Minecraft 在线收发未内置；下文示例是可供外部适配器调用的记忆能力，不表示已经具备登录、截图、操作世界等工具。

## 1. 核心模块总览

| 路径 | 入口 | 职责 |
|---|---|---|
| `maidcore/store.py` | `MaidCore(data_dir)` | 建立用户私有目录、Persona、JSONL、Legacy 记忆、构造对话请求 |
| `maidcore/llm.py` | `APIConfig.load()`、`ChatCompletionsClient`、`WineFoxChatSession` | API 配置、请求、工具调用续轮、成功后落盘 |
| `maidcore/retrieval.py` | `MemoryRetriever`、`RetrievalSettings` | 召回、评分、向量缓存、手动 Core、Episode 详情权限 |
| `maidcore/improved_extraction.py` | `build_batches`、`run_extraction` | 一步/两步提炼、证据校验、运行审计 |
| `maidcore/episode_memory.py` | `EpisodeMemoryDatabase` | Episode、计划状态、来源关联 |
| `maidcore/memory_verification.py` | `VerifiedMemoryDatabase` | 可选核验与状态修订 |
| `maidcore/auto_archive.py` | `AutoArchiveManager` | 自动提炼触发、单批运行、自动/批量发布 |
| `maidcore/core_curator.py` | `CoreCurator` | AI Core 自动筛选与版本历史 |
| `maidcore/review_ui.py` | `MemoryWorkbench` | HTTP 工作台的业务逻辑与本地安全检查 |
| `maidcore/nbt.py` | `read_maid_file` | 只读解析 `.maid` NBT 格式 |

`main.py` 是最简单且稳定的入口，所有命令通过 `py main.py --help` 检索。底层 Python 类目前属于**源码接口**，没有额外发布为 PyPI SDK，也没有单独的长期 API 兼容性承诺；集成时建议锁定 Git release/commit。

## 2. Python：最小可运行示例（不调用云模型）

```python
from pathlib import Path
from maidcore import MaidCore
from maidcore.retrieval import MemoryRetriever

core = MaidCore(Path('data'))  # 自动建立私有目录
core.write_persona('你是一位友善的虚构助手，表达自然，有不确定时承认不确定。')
core.append_chat('user', '今天我参观了天文馆', platform='terminal')
core.append_chat('assistant', '看到什么展览了？', platform='terminal')

retriever = MemoryRetriever(core)
prepared = retriever.prepare('上次去天文馆的事')
print(prepared['cards'])          # Archive 若仍为空，这里通常为空
print(prepared['backend'])        # lexical fallback 或本地 embedding
```

注意 `append_chat` 只是写入记录；**不会自动创造可靠的 Episode/Cand**。调用 LLM 提炼、导入已有记忆或之后的自动 Archive 才会产生结构化长期记忆。

## 3. Python：让外部聊天客户端共用一套数据

```python
from pathlib import Path
from maidcore import MaidCore
from maidcore.llm import APIConfig, ChatCompletionsClient, WineFoxChatSession

core = MaidCore(Path('data'))
config = APIConfig.load('.env')
client = ChatCompletionsClient(config)
session = WineFoxChatSession(core, client, platform='terminal', env_file='.env')

answer = session.reply('你记得我们以前讨论的书吗？')  # 会发起 API 请求
print(answer)
```

未来的 QQ/游戏适配器可以：① 接收平台消息并转换为纯文本，② 调用会话层生成回复，③ 将回复通过平台 SDK 发送出去。`platform` 可以标记为 `qq` 或 `minecraft`，但这只是当前会话的**来源标签**。正式集成还应校验用户身份、群聊权限、消息去重、并发、平台指令和外部工具权限。

对未来多平台共用同一 `data/` 的设计建议：实现一个**单独的记忆服务进程**或写入队列，其他适配器只发消息请求。当前的 `MaidCore` 会重写 JSONL 文件，**不适合多个 QQ/Minecraft 进程同时直接写同一个文件**。

## 4. 只使用记忆召回，不使用聊天模型

```python
from pathlib import Path
from maidcore import MaidCore
from maidcore.retrieval import MemoryRetriever

core = MaidCore(Path('data'))
retriever = MemoryRetriever(core)
prepared = retriever.prepare(
    '之前我们是不是也讨论过这件事？',
    context_query='用户刚才谈到读书，助手询问了小说结局。',
    fixed_seed=42,
)
print(prepared['memory_text'])
print(prepared['selected'])  # 评分分量与条目 ID
```

`context_query` 是近期对话的简短话题。若采用 `core.build_messages(..., return_retrieval=True)`，程序会自动从近期双方原始消息构造它。

获取可配置默认设置：

```python
from maidcore.retrieval import RetrievalSettings
settings = RetrievalSettings.load('data')
print(settings.memory_budget_tokens)
```

只改变必要参数，其余保留默认：

```python
retriever.update_settings({'episode_cards_target': 32, 'context_query_messages': 6})
```

配置将保存到私有 `data/memory_retrieval.json`，不应提交到公共仓库。

## 5. Episode 详情工具的权限边界

`ChatCompletionsClient.complete_with_episode_tool(...)` 由聊天会话层自动使用。工具名为 `expand_episode`，参数格式：

```json
{"episode_ids": ["ep_example_a", "ep_example_b"]}
```

仅允许模型查询当前 `prepared['tool_available_episode_ids']` 列出的 Episode。运行时还会二次检查数据库是否仍是 `approved`/`auto_active`，限制数量、文本长度和剩余 Token。用户可以在召回设置中关闭该功能。**不能任意检索旧聊天、执行游戏动作或访问操作系统**。

## 6. 命令行接口参考

| 任务 | 示例（从项目根目录执行） | 是否调用模型 |
|---|---|---|
| 初始化、统计 | `py main.py stats` | 否 |
| 配置验证 | `py main.py check-config` | 否 |
| 打开工作台 | `py main.py ui` | 启动否；点击付费操作另计 |
| 聊天 | `py main.py chat` | 是 |
| 添加平台标记聊天 | `py main.py add-chat user "你好" --platform qq` | 否 |
| 人格文件导入 | `py main.py set-persona my_persona.md` | 否 |
| Minecraft 档案导入 | `py main.py import-maid my_character.maid` | 否 |
| 召回预览 | `py main.py memory-retrieval-preview "之前的旅行"` | 否 |
| 自动归档状态 | `py main.py memory-auto-status` | 否 |
| 手动预览提炼 | `py main.py memory-extract --mode two-pass --no-verify` | 否 |
| 执行两步提炼 | `py main.py memory-extract --mode two-pass --no-verify --run` | 是 |
| 批量发布预览 | `py main.py memory-auto-publish --all` | 否 |
| 批量发布执行 | `py main.py memory-auto-publish --all --apply` | 否 |
| 评分/索引状态 | `py main.py memory-embedding-stats` | 否 |

命令行输出常包含实际私人聊天或已存记忆，**不要把终端输出截图或 JSON 报告不经检查就上传**。

## 7. 本地工作台 HTTP 接口（仅供本地浏览器使用）

服务默认绑定 `127.0.0.1:8765`，路径前缀为 `/api/`，主要入口：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/review/inbox` | 未确认 Cand/Episode，含 `pending` 与 `auto_active` |
| `GET` | `/api/retrieval/settings` | 当前检索参数 |
| `POST` | `/api/retrieval/settings` | 提交 `{"changes": {"cand_limit": 12}}` |
| `POST` | `/api/retrieval/preview` | 预览本轮将使用的记忆与评分 |
| `GET` | `/api/retrieval/embedding` | 本地向量索引状态 |
| `GET` | `/api/auto-archive/preview` | 可压缩消息数量与预览 |
| `GET` | `/api/archive/publish-preview` | 不写库的批量发布模拟 |
| `POST` | `/api/archive/publish-all` | `{"confirm": true}`，批量执行发布 |
| `POST` | `/api/reviews/bulk` | 工作台批量审核 |
| `GET` | `/api/retrieval/logs` | 最近召回日志（可能含私人信息） |

浏览器工作台中的**写操作需要启动时生成的会话 token 与同源检查**。请使用网页界面，不要假设直接向 HTTP 端口 POST 就可操作。该 token 不等于对公网用户的账号鉴权；当前实现不是通用远程 REST API。对接其他程序建议优先调用 Python 类，或单独实现有权限控制的服务层。

## 8. 建议的后续插件接口

若以后开发真正的 QQ / Minecraft / 读图工具，可将其拆成三个边界：

```text
外部适配器：平台收消息 / 发消息 / 权限校验
              ↓
会话层：按用户选择的角色读取 Persona、Core、近期上下文
              ↓
长期记忆层：检索、写归档、标记平台、追溯证据、审核
```

不要把外部工具返回值直接作为指令执行，尤其不要让聊天原文控制文件路径、可执行命令或所有者权限；将工具结果视为不可信来源数据。
