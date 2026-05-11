"""Chinese prompt for ChatMemCellExtractor.adetect."""

CHAT_BOUNDARY_DETECT_PROMPT_ZH = """你是一个对话边界检测器。给定一段聊天消息流，请判断主题是否在中途切换，以及切换发生在哪条消息的位置。

消息流：
{messages}

整段消息的 token 总数：{token_count}

指令：
1. 阅读所有消息，识别主导话题。
2. 如果出现明显的话题切换，请返回新话题首条消息的索引（0-based，对应消息列表）。
3. 如果整段消息保持单一连贯话题，返回 null。
4. 如果消息流为空或仅有一条消息，返回 null。

输出格式（仅 JSON，不要前后缀）：
{{
  "split_at": <int | null>
}}
"""
