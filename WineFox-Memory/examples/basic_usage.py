"""Synthetic, offline demo. No API key, Minecraft world or private data required."""
from pathlib import Path
from tempfile import TemporaryDirectory

from maidcore import MaidCore
from maidcore.retrieval import MemoryRetriever


def main():
    with TemporaryDirectory(prefix="winefox_demo_") as tmp:
        core = MaidCore(Path(tmp))
        core.write_persona("你是一位友善的虚构角色，可以自然交流。")
        core.append_chat("user", "今天我参观了博物馆。", platform="terminal")
        core.append_chat("assistant", "看到了什么特别的展品？", platform="terminal")
        r = MemoryRetriever(core)
        result = r.prepare("我们聊过博物馆吗？", context_query="用户参观过博物馆。")
        print("原始聊天条数：", len(core.chat_history()))
        print("当前选中长期记忆：", len(result.get("selected", [])))
        print("检索后端：", result.get("backend"))
        print("提示：原始聊天存在不代表自动生成了长期 Cand / Episode。")


if __name__ == "__main__":
    main()
