# WineFox Memory v0.3b.4-community.1

这是 v0.3b.4 公开清理版上的社区测试更新，专注提高外部用户安装、反馈与复现问题的便利性，不改变记忆检索与自动归档的核心算法。

- 新增 `python main.py check-install`：无需 API Key、无需生成私人数据；只检测 Python 版本和重要代码资源。
- 新增 `python main.py diagnose`：仅导出白名单计数、运行环境及少量召回统计，不包含聊天正文、人格、Core、API Key、错误日志、文件路径、记忆 ID，不联网，不自动上传。
- 新增 Windows 一键诊断脚本 `diagnose_windows.bat` 和工作台安全诊断下载链接。
- 新增 GitHub Issues 模板（程序故障、记忆表现）、PR 模板、CONTRIBUTING 和 Actions 多平台离线测试。
- 新增 `examples/fictional_conversation.json` 和 `python tools/demo_offline.py`，无需 API Key、真实聊天或下载模型即可测试提炼、自动发布和召回。
- 新增 `tools/public_release_check.py` 用于检查发布 ZIP 中的隐私目录/密钥形态；此检查不能代替人工审查。
- 公开版自动测试：20 项合成/隐私测试；本地模拟，不代表真实提供商 API 的性能和兼容性。

## 安装/升级

全新用户请下载完整 ZIP 并阅读 README。已有 `v0.3b.4_public_clean` 用户可以关闭聊天/工作台、备份 `data/` 和 `.env`，将升级补丁相对项目根目录覆盖；补丁不包含私人数据。

**正式发布前需人工确认 MIT 授权及第三方组件的授权兼容性。**
