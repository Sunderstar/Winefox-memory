# 技术说明：WineFox Memory 记忆架构（v0.3b.4）

本文描述**当前仓库代码已实现的机制**，并明确哪些是经验性排序/近似估计，不把设计目标描述成已经验证的认知科学事实。

## 1. 设计边界

目标不是把所有历史发送给大模型，而是将所有历史**本地完整保留**，每轮仅选择有限 Core、Episode、Cand 与近期聊天，以控制输入 Token 和维持角色长期交流连续性。SQLite 保存结构化记忆、来源证据、索引、审核和处理进度，JSONL 保留原始消息。

多平台可以共享同一 `data/` 记忆库，但**目前没有生产级多进程写锁或消息队列**，不应让多个进程同时直接修改同一份 JSONL。正式部署跨平台适配器时，应串行化消息写入或增加中心服务。

## 2. 数据流

```text
用户 / 终端 / 未来的 QQ 或游戏适配器
                    │
                    ▼
             原始对话 JSONL ─────────────────────────┐
                    │                               │
     已处理 ID / 近期保留窗口                       │ 完整原文长期保留
                    ▼                               │
           历史分批、来源证据                       │
                    ▼                               │
    (可选 one-pass / 默认建议 two-pass)             │
        事件抽取 → 短事实提取 → 可选独立核验          │
            │             │                         │
            ▼             ▼                         │
     Episode（情景）    Cand（候选事实）            │
            └─────────┬───┘                         │
                      ▼                             │
        自动发布 / 人工复核 / 拒绝 / 修订            │
                      │                             │
             Archive SQLite                          │
                      │                             │
   核心记忆 Core ◄── AI Core 筛选（可选、付费）      │
                      │                             │
  当前问题 + 最近双方话题 → 本地语义/词组召回       │
                      │                             │
  Core + 选中 Episode 卡 + Cand + 近期聊天 ◄────────┘
                      │
                      ▼
                Chat API 请求
                      │
             可选 expand_episode
              仅本轮合法事件 ID
                      │
              模型生成最终回复
                      │
            成功后写回用户和助手消息
```

## 3. 持久数据：为何同时有 Cand 与 Episode？

| 层次 | 存储 | 内容 | 典型用途 |
|---|---|---|---|
| Raw | `chat_history.jsonl`、`autonomous_history.jsonl` | 所有聊天消息与自主发言，含角色、平台、ID、时间 | 来源审计、重新提炼、原话追溯 |
| Episode | SQLite `extracted_episodes` | `title`、`summary`、`detail`、状态、时间、证据和计划 | 事件目录卡、详细经历、自主回忆 |
| Cand / Memory | SQLite `candidates` 与 `memories` | 独立可理解的短事实、关联证据、类别、重要性 | 快速检索稳定事实和约定 |
| Core | `core_memory.md` 与 SQLite `core_auto_entries` | 少量持续影响角色的稳定认识 | 每轮优先注入（仍受整体 Token 硬预算） |
| Persona | `persona.md` | 人格与语言风格（不是事件记忆） | 保持角色表达方式与行为边界 |

**Episode** 是叙事单元，处理“发生了什么、前因后果、结果”；**Cand** 是从事件或原话中提取的可直接复用的事实句，处理“某个约定、偏好、计划是什么”。一个 Episode 可以关联多个 Cand；同一个 Cand 可在来源中引用 Episode。它们会增加本地存储量，但目的是根据当前话题注入**卡片或短事实**，而不是每次同时注入全部详情。重复的 Core/Cand/Episode 仍可能在某一轮上下文中出现，当前没有实现完美去重。

## 4. 提炼、校验和发布

- `improved_extraction.py`：将不重复的未处理原始消息按对话单元、长度与平台分批，生成提炼提示词；`two-pass` 先识别 Episode，再依据原始消息与 Episode 抽取 Cand。`one-pass` 一次返回两类对象。
- `prompts/`：事件识别、候选提取、单步提取与可选核验的提示词模板。输入对话被明确视为**资料而非可执行指令**。
- `memory_verification.py`：可选检查引用证据、归因、时间和计划状态；不能保证每条事实都真实。
- `auto_archive.py`：自动归档默认关闭，开启后按 `trigger_messages` 和 `keep_recent_messages` 选择待处理区；成功后将符合证据条件的正常来源标记为 `auto_active`。
- 对普通手动 `memory-extract --run --source chat`，合规结果也可自动发布；实验冻结样本、异常证据、冲突候选继续隔离。`approved` 是人工确认；`rejected` 无法被自动恢复；`pending` 未发布；`auto_active` 是未人工确认但可用于召回。
- SQLite `processed_records` 防止同一原始 ID 被正常后台工作反复处理。重测实验可以显式指定 `--reprocess`，不会覆盖原有候选。
- **原始聊天不会因为压缩而删除**。运行日志和提炼原始模型响应可能包含完整私人对话，也必须保存在私有 `data/` 下。

## 5. 召回机制与具体评分

默认通过 `MemoryRetriever.prepare()` 对已发布的 Archive 记录筛选：

1. 使用当前问题 `q`，以及最近 `context_query_messages` 条由用户和助手共同构成的简短话题 `c`（默认 6 条、不超过 380 字）。
2. 分别计算 Embedding / 词组相关度：`R(q,m)` 与 `R(c,m)`。
3. 一般问题按 `0.75 R(q,m) + 0.25 R(c,m)`，省略性追问按 `0.30 R(q,m) + 0.70 R(c,m)` 合并；具体权重可调整。
4. 将合并的相关度 `R_i` 与近期性、重要性、重复谈及等组成综合分数，排序并按阈值、数量和 Token 预算截断。

若本地 BGE 可用，基础相关度为：

\[
R(x,m)=0.78\max(0,\operatorname{cos}(e_x,e_m))+0.22L(x,m)
\]

`L(x,m)` 是基于中文二元/三元字组和英文词语的逆文档频率加权匹配；未安装或没有缓存模型时 `R=L`。软件不会在每轮聊天时隐式下载模型权重。

综合排序：

\[
S_i=\frac{w_sR_i+w_tT_i+w_iI_i+w_mM_i}{w_s+w_t+w_i+w_m}
+0.12\,\mathbf 1_{\rm pinned}+0.08\,\mathbf 1_{\rm activated}
\]

默认 `w_s=0.65, w_t=0.20, w_i=0.15, w_m=0`：

- `T_i=exp(-Δt_i/30)`，`Δt` 单位为天，若时间信息不可用则用 `0.35`；最近实际谈及会改变时间。
- `I_i` 是 Cand 的提炼重要性或用户覆盖值；Episode 目前通常用默认的重要性，并非模型独立评估的情绪强度。
- `M_i=min(1,ln(1+n_i)/ln(21))`，`n_i=mention_count`；默认权重为 0。现在只在用户当轮文字具有明确共同话题片段时才进行保守更新，而不是“被检索就+1”。
- Pinned 加 0.12 分、会话激活加 0.08 分，**二者不是强制注入**；仍受相关度与预算控制。

Episode 目录、Cand 和 Detail 分别有阈值与数目上限；**阈值检查的是相关性 `R_i`，不是综合分数 `S_i`**。最低卡片数可让弱相关记忆有机会进入目录，但仍不能突破 Token 预算。

## 6. 类人联想与主动展开

- 每轮以配置的 `spontaneous_recall_probability`（默认 0.08）尝试额外附加少量非最高排名的 Episode 卡；有冷却轮数和目录预算限制。
- 除常规目录以外，可以随机提供最多几条**弱关联旧事件标题**，不预先注入详情，允许自主联想。
- 当用户明确问“以前、当时、具体经过”等，程序仍可依照相关性和预算自动展开少量 Episode。
- 模型如主动请求 `expand_episode`，仅能从**当轮提供的候选 ID** 读取本地 SQLite 中 `approved` / `auto_active` 的详情；默认最多 4 条，另设 Token 上限。不会读取任意文件、修改记忆、访问网页或自行读取原始聊天全文。
- **延迟代价**：模型无需工具时只有一次模型请求；要求展开则通常要增加一次请求。思考模型的 `reasoning_content` 仅在本轮续请求中传递，不保存到历史。

随机联想和重要性不是心理学意义上经过实验验证的人类记忆模型，仅是工程性启发式，当前也没有实现无限联想图。

## 7. Token 预算与计算开销

1. Core 优先分配，但若超过建议 `core_budget_tokens` 而未超长期记忆硬预算，会占用部分 Archive 预算。真正超出总预算时仍会截断并报告。
2. Episode 目录卡优先使用 `directory_budget_tokens`，Cand 和 Detail 按剩余预算选取。
3. 近期原始聊天通过 `recent_chat_messages`、`recent_chat_budget_tokens` 截断；对话与长期记忆共同受 `max_input_tokens` 限制。
4. 本项目的 Token 估算使用基于字符长度的近似公式，而不是 API 服务商的精确 tokenizer；实用时需预留余量。
5. BGE Embedding 缓存在 `embedding_vectors`，用模型、条目 ID、正文哈希判定是否可复用。新内容首次向量化需要 CPU 时间，SQLite BLOB 存储不要求外置向量数据库。

## 8. AI Core 与人工 Core

- `data/core_memory.md`（人工编辑）优先于旧的 `memory.md`，可留空或由用户自己写。
- `core_auto_entries` 保存模型根据新发表的 Cand/Episode 选择的重要共同经历，支持版本审计与恢复；程序不会将临时口头表达永久改写成 `persona.md`。
- 自动 Core 默认关闭，需要用户显式开启。它是额外的**付费模型整理**，不会因为预览或打开网页而自动执行。

## 9. 开放接口、可扩展性与当前局限

业务入口：`MaidCore`（数据读写）、`MemoryRetriever`（检索和详情工具）、`WineFoxChatSession`（模型聊天）、`AutoArchiveManager`（后台提炼）、`CoreCurator`（AI Core）、`MemoryWorkbench`（本地 UI）。更完整的代码与调用示例见 [INTEGRATION.md](INTEGRATION.md)。

当前不存在内建 QQ 收发适配器、生产级服务化鉴权、多进程数据库/JSONL 并发协调，也不保证不同模型会稳定调用主动回忆工具。首次部署不建议把 HTTP 工作台绑定到公网，真实对话数据及所有输出运行记录不得发布。
