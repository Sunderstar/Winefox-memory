# 用户反馈与隐私保护诊断

## 一键报告

- Windows：双击 `diagnose_windows.bat`。
- 终端：`python main.py diagnose`。
- 本地工作台：点击“下载诊断报告”，保存 `winefox-diagnostic.json`。

命令行报告默认写入 `data/diagnostics/`（位于 `.gitignore` 中），不会发送到网络。网页版在浏览器本地下载，不保存到服务器上的额外数据库。

### 包含的数据（固定白名单）

- WineFox 版本、Python/SQLite 版本和操作系统大类（Windows/Linux/macOS）；
- 可选包的公开版本号；
- 是否存在人格或 API 配置文件（**不读文件内容**）；
- 原始历史的消息**行数**；
- SQLite 已知记忆表的**行数与状态枚举计数**；
- 少量数值召回／自动归档设置；
- 最近最多 100 条召回日志的聚合：输入 Token 估计均值、Core 截断轮数、Episode 工具调用轮数。

### 不包含的数据

**绝不输出** API Key、环境变量、主机名、用户名、绝对路径、任意记忆 ID、聊天原文、助手回复、人格卡、Core 正文、提炼提示词、模型原始输出、错误消息或堆栈内容。

诊断数据仍然可能透露使用量与偏好设置，所以请在公开分享之前手工查看文件。生成报告不要求 API Key，不调用模型、不打开互联网、不修改 SQLite。不会有隐式上传或后台收集。

## 推荐报告方式

1. 先尝试 `python tools/demo_offline.py`，看看合成记忆是否能被提炼、发布并召回。
2. 运行 `python main.py check-install`，检查 Python 版本与文件结构。
3. 如果是自己的长期聊天发生问题，先用**改写过的虚构示例**尝试复现。
4. 在 GitHub Issues 选择“程序故障”或“记忆表现”模板，提供最短复现步骤及审阅过的诊断报告。
5. 不愿公开的人可以只填复现步骤和软件版本，不必上传任何个人数据。

**不要上传** `data/memory.db`、`chat_history.jsonl`、`data/logs/`、`.env`、`persona.md`、`core_memory.md`、模型原始输出、冻结样本运行详情以及完整上下文预览。它们都可能包含私人文本。
