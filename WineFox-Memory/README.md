# WineFox Memory / 酒狐长期记忆核心

**社区测试版 v0.3b.4-community.1** · 以收集可复现的错误与改进建议为主；不自动收集或上传私人数据。


一个**本地优先、跨平台可扩展**的 Python AI 角色长期记忆实验项目。提供原始聊天归档、情景事件（Episode）、短事实（Cand）、核心记忆（Core）、语义/词组检索、受限自主回忆，以及可视化记忆审核工作台。

> 本仓库是**公开清理版**。不包含开发者的 API Key、个人角色人格、任何真实聊天或自发言、个人记忆数据库、运行日志或 Minecraft 存档。首次使用时，你需要配置自己的角色和 API。
>
> 本项目**目前提供终端对话、记忆核心与本地网页工作台**，并提供 Python 接口供 QQ、Minecraft 或其他客户端接入；**不包含可直接运行的 QQ 机器人或实时 Minecraft 控制插件**。

## 反馈和离线诊断（欢迎参与测试）

没有 API Key 也可以先运行完整的**合成数据**离线演示：

```powershell
py main.py check-install
py tools/demo_offline.py
```

如果遇到安装、自动提炼或记忆召回问题，可以通过以下任意方式生成**只含统计信息**的报告：

```powershell
py main.py diagnose
```

也可双击 `diagnose_windows.bat`，或在本地网页工作台点击「下载诊断报告」。报告**默认不包含原始聊天、人格卡、Core、密钥或日志正文**，也**不会自动联网或上传**；请在提交之前自行打开查看。详细步骤请阅读 [docs/FEEDBACK.md](docs/FEEDBACK.md)，使用 [GitHub Issues 模板](.github/ISSUE_TEMPLATE/) 反馈，并尽量使用[虚构对话样本](examples/fictional_conversation.json)代替真实聊天。

公开包包含 GitHub Actions 离线测试任务和用于贡献测试的 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 1. 环境与安装

- 推荐：Windows 10/11、Python **3.11 或 3.12**（代码语法最低要求为 Python 3.10）。其他操作系统也可以运行命令行和本地网页界面。
- 基础功能只使用 Python 标准库：**无需 pip install**，不需要 Docker、数据库服务器或浏览器插件；SQLite 随 Python 自带。
- 需要一个兼容 OpenAI Chat Completions 格式的模型 API 账户与 Key（如配置中默认的 DeepSeek 地址）。真实模型名、思考参数请以你的服务商支持情况为准。
- 可选：本地 BGE 语义向量检索，需要另装 `sentence-transformers` 和 NumPy；不安装时可回退为词组检索。

在项目目录打开 PowerShell，先检查：

```powershell
py --version
py main.py --help
```

若没有 `py`，可用 `python` 替代。建议首次安装使用独立虚拟环境，但不是必需：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe main.py --help
```

**注意：** 本版不附带任何角色私有数据。`data/` 会在首次执行命令后自动建立，并被 `.gitignore` 排除。不要从作者的私人备份或旧 release 直接复制 `data/` 到 GitHub。

## 2. 首次配置：只要两处

**① 配置自己的 API（项目根目录 `.env`）**

```powershell
Copy-Item .env.example .env
notepad .env
```

打开 `.env`，在 `API_KEY=` 后填写你自己的密钥。`BASE_URL` 和 `MODEL` 有示例默认值；如果使用其他 API 服务，改为其官方提供的地址和模型名。其他参数**先不用改**。

```ini
API_KEY=在自己的电脑上填写_不要提交GitHub
BASE_URL=https://api.deepseek.com
MODEL=deepseek-flash
REASONING_EFFORT=none
```

这段是填写示意，不是密钥。`MODEL` 不一定是所有 API 服务商均支持的模型名。对不支持 `REASONING_EFFORT` 的接口，将其留空；若工具调用报 400，可在记忆工作台关闭 Episode 主动展开。

**② 配置自己的角色人格（`data/persona.md`）**

第一次运行：

```powershell
py main.py stats
notepad data\persona.md
```

在 `persona.md` 中写下角色名字、表达风格、稳定的性格设定。可以采用简短格式：

```markdown
# 人格设定
你是一位喜欢旅行和阅读的虚构伙伴，名字由使用者自行设定。
你可以自然地表达疑问、偏好和不确定性。
不要宣称自己已经执行了不存在的现实操作。
```

这是一个**合成示例**，不是作者的原始人物人格。没有必要预先创建聊天历史、SQLite 或 Core，它们会按需生成。

验证配置但不显示 API Key、不调用 API：

```powershell
py main.py check-config
```

## 3. 开始使用

**终端聊天（会调用模型 API）**：

```powershell
py main.py chat
```

也可 Windows 双击 `start_chat_windows.bat`，或只发一条：

```powershell
py main.py chat --once "你好，今天想聊聊什么？"
```

**记忆工作台（启动本身不调用模型）**：

```powershell
py main.py ui
```

浏览器打开 `http://127.0.0.1:8765/`。首页是日常使用界面：自动归档、未人工确认的 Cand/Episode、召回预算与预览、Core、记忆权重；`/experiment` 是用于冻结样本、对比不同模型和压缩模式的高级实验界面。

### 离线检查和记忆操作

```powershell
py main.py stats
py main.py history --limit 20
py main.py memory-retrieval-settings
py main.py memory-retrieval-preview "我们以前聊过旅行吗？"
py main.py memory-auto-status
py main.py memory-embedding-stats
```

**手动写入与跨平台消息导入**：

```powershell
py main.py add-chat user "我今天读完了一本小说" --platform qq
py main.py add-chat assistant "那本小说的结局怎么样？" --platform qq
py main.py remember "用户喜欢科幻小说" --category preference
```

这里的 `--platform qq` **只是给本地记录贴上来源标签**，并不代表程序已经连接 QQ。真正的 QQ 收发接口要由外部适配器实现。

**从 Minecraft 导出档案中导入**（可选）：

```powershell
py main.py import-maid "C:\路径\your-character.maid"
```

导入只读取 `.maid`，不会修改游戏存档。导入后请检查 `data/persona.md` / `data/chat_history.jsonl` 是否是你期待的内容。`defaults/winefox.yml` 是中性兼容模板，**没有原作者个人的角色设定**；如需完整原角色设定，请自行提供有权使用的人格文件。

## 4. 记忆是怎样自动产生的？

一次聊天成功后，`data/chat_history.jsonl` 会保留本轮用户与助手的原文。长期记忆压缩会从**尚未处理**的聊天中提取 Episode（完整经历）和 Cand（简短稳定信息），不会删除聊天原文。

正常后台自动压缩初始为**关闭**，防止新用户首次安装就产生预料之外的费用。要自动运行，可以到工作台开启，或者：

```powershell
py main.py memory-auto-config --enable --trigger-messages 24 --keep-recent-messages 20
py main.py memory-auto-status
```

即使不启用后台任务，也可以先预览，再明确选择发起一次模型请求：

```powershell
py main.py memory-extract --source chat --mode two-pass --no-verify
py main.py memory-extract --source chat --mode two-pass --no-verify --run --max-batches 1
```

`two-pass` 会先提炼 Episode，再提炼 Cand；可选第三次核验。`--run` 才执行、可能产生 API 费用。普通聊天来源的安全提炼结果默认会自动发布为 `auto_active`，**无需逐条人工审批才能召回**；事后仍可审核、修改或拒绝。

如果以前手动提炼过但还没发布，可以预览和批量启用已有候选（不额外调用模型）：

```powershell
py main.py memory-auto-publish --all
py main.py memory-auto-publish --all --apply
```

实验样本、来源异常、已拒绝或证据核验不通过的结果不会被无条件启用。

## 5. 本地向量检索（完全可选）

基础版无需外部包；只用词组相关性也可运行。如果需要更丰富的语义匹配：

```powershell
py -m pip install -r requirements-embedding.txt
# 使用上述 .venv 虚拟环境的人，请改用：
# .\.venv\Scripts\python.exe -m pip install -r requirements-embedding.txt
```

正常聊天过程**不会自动下载模型权重**。首次使用可单独下载官方公开模型到当前用户的 Hugging Face 缓存（执行这一命令需要联网）：

```powershell
py -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
```

然后在工作台“召回设置”中确认 `semantic_backend=auto` 和模型名 `BAAI/bge-small-zh-v1.5`，并检查：

```powershell
py main.py memory-embedding-stats
```

向量只是一份按需缓存，存放在本地 SQLite 中；换模型或修改正文后会按内容哈希失效并重新计算。首次下载和模型加载可能比较慢。如果不想下载，保持默认 `auto`，缺包或模型未缓存时回退词组匹配；也可以把后端明确设为 `lexical`。

## 6. 数据保存在哪里？

| 文件 | 内容 | 是否允许公开 |
|---|---|---|
| `.env` | 模型 API Key 和个人配置 | **禁止** |
| `data/persona.md`、`data/persona_source.yml` | 个人角色设定与导入来源 | **禁止** |
| `data/chat_history.jsonl`、`autonomous_history.jsonl` | 全部聊天/自发言 | **禁止** |
| `data/memory.db` | Episode、Cand、Core、向量、提炼输入输出及审核日志 | **禁止** |
| `data/core_memory.md`、`data/memory.md`、`data/profile.json` | 核心关系记忆、旧摘要、个人档案 | **禁止** |
| `data/logs/`、`data/core_memory_history/` | 召回记录、报错、历史快照 | **禁止** |
| `data/memory_retrieval.json`、`data/auto_archive_settings.json` | 可调设置（也可能暴露用户偏好） | 建议保持私有 |
| `defaults/winefox.yml` | 本发行版的**中性示例** | 可以 |

此发行版的 `data/` **不存在**，只在用户首次启动后本地建立。不要把运行过的项目目录不经清理直接再次压成公开 ZIP。

## 7. 开发者接口与文档

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：记忆架构、压缩管线、召回公式、自动发布、冷热机制与开销。
- [docs/INTEGRATION.md](docs/INTEGRATION.md)：Python API、命令行、HTTP 工作台接口及如何接入 Minecraft / QQ 等客户端。
- [docs/PRIVACY_AND_RELEASE.md](docs/PRIVACY_AND_RELEASE.md)：隐私边界、密钥保护、发布前检查。
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)：参考项目与授权说明。
- [LICENSE](LICENSE)：本公开发行包的 MIT 授权。发布者须确认自己拥有各源文件和素材的再授权权利。

项目目前没有独立的互联网服务端、跨进程事务队列或生产级多用户隔离。**本地工作台只监听 localhost**；不要把它直接暴露到公网。代码可作为自有客户端的记忆后台，但需要外部平台适配、权限检查与并发处理。

## 8. 常见问题

**启动后只有空记忆？** 正常。公开版没有作者的私人聊天和 Core。开始对话、导入自己的历史后才会有内容。

**为什么工作台没有触发归档？** 默认自动归档关闭；开启后还要满足 `trigger_messages` 和 `keep_recent_messages`。压缩保留窗口与下一轮直接携带的近期对话窗口是不同参数。

**为什么已生成的记忆没出现在回答里？** 首先检查状态是否 `auto_active` / `approved`，然后检查当前查询相关度和 Token 预算。被保存、具备召回资格和实际进入本轮模型上下文是三件不同的事。

**没有向量库还能用？** 可以。词组检索、完整聊天归档、SQLite、Core 和 Two-pass 都不依赖 `sentence-transformers`。

**能直接连 QQ 吗？** 现在不能。程序支持 `platform="qq"` 作为记录标签、并暴露 Python 接口供你自己开发 QQ 适配器；此仓库不附带 QQ 登录和消息收发实现。

**如何保护密钥？** 只存到本机 `.env` 或安全的环境变量。任何已经被公开上传过的真实 Key 都应立即在服务商后台作废并重新创建；从 GitHub 删除文件不等于删除历史提交中的密钥。
