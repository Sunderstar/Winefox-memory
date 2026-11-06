# Memory source timestamps — v0.3b.4.1

本文件为社区版 WineFox Memory 的时间戳功能更新说明。此次升级只修改检索展示及其测试，保留现有 GitHub README、社区诊断功能和 Issue 模板。

## 发生了什么改变？

- Episode 目录、自动展开的 Episode Detail、自由联想候选和 `expand_episode` 的返回值都增加 **time_label**。
- Cand 优先使用 `memories.event_time`，若缺失，则尝试通过 `origin_candidate_id → candidate_audit → extraction_runs.input_payload.messages` 和 `evidence_ids` 恢复来源对话的时间范围。
- 缺失来源时间但有收录时间时，明确显示“仅知收录时间”；无时区时间不做猜测。
- 时间展示统一采用北京时间 UTC+08:00，使用代码里已有的 `local_iso()`。
- 时间标签只出现在提供给模型的文本或工具响应中，不参与语义检索和原有的相关性评分。
- `store.py` 的系统提示词说明：**消息被记录的时刻不等于现实或游戏事件发生的时刻**。

## 输出示例（虚构数据）

```text
历史事件目录（必要时参考，不要求主动提及）：
- [ep_demo] [来源对话时间（北京时间）：2026-10-08T09:00:00+08:00] 参观自然博物馆：参观化石展览……

本轮相关的长期事实（可能过时）：
- [memdb_demo] [仅知收录时间（非事件发生时间，北京时间）：2026-10-09T18:00:00+08:00] 用户喜欢天文展览。
```

`expand_episode` 读取批准/自动启用的 Episode，输出 `{episodes:[{id,title,time_label,detail}], expanded_count, ...}`；未批准的 Episode 依然不可读取。

## 安装和验证

1. 关闭聊天终端、网页工作台和后台自动归档，备份 `.env` 和整个 `data/`。
2. 将本社区兼容补丁从 **GitHub 项目根目录**解压覆盖。它刻意**不含 `README.md`**，因此不会覆盖自行编辑的仓库首页。
3. 不需要迁移 SQLite 表，不需要重新提炼已有 Episode/Cand。首次检索会即时读取已有时间证据。
4. 执行：

```powershell
py -m unittest discover -s tests -q
py main.py check-install
py main.py memory-retrieval-preview "你还记得我们参观自然博物馆吗？"
```

## 性能与局限

- 旧 Cand 的时间跨度恢复需要按批读取之前的 `extraction_runs.input_payload`。在大量历史批次下，这种即时重建会增加检索前的 SQLite/JSON 开销；本次没有引入持续化缓存或数据库迁移。
- 时间标签占用上下文 Token，预算不足时可能减少本轮选中的记忆数量。
- 旧资料缺少时间戳时会明确显示未知或收录时间，不会凭空推断事件日期。
- 测试仅验证本地逻辑及兼容性；没有联网调用真实模型 API。

## 文件说明

覆盖文件：`maidcore/retrieval.py`、`maidcore/store.py`。
新增文件：`tests/test_timestamps_v0341.py`、`docs/TIMESTAMP_MEMORY.md`。
其他社区测试功能、网页工作台、用户 README、诊断报告和 `.github` 文件均保持原样。
