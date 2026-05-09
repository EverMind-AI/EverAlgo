"""Chinese prompt for EpisodeExtractor.aextract."""

EPISODE_EXTRACT_PROMPT_ZH = """你是一名情景记忆生成专家。给定一段对话切片（MemCell），请提取结构化的 Episode 记忆。

对话内容：
{memcell_text}

对话时间戳（Unix epoch 毫秒）：{timestamp}

指令：
1. 识别每个独立的情景事件 —— 一段完整的「发生了什么」轨迹，包含参与者、地点、时间、行为、结果。
2. 将对话形式转换为第三人称叙述。
3. 保留人名、日期、地点、决策、情感。
4. 用对话时间戳作为情景的时间锚点。
5. 为每个 Episode 生成唯一 id（如 "ep_<random>"）。
6. 从对话中选取稳定的 owner_id（如不明则默认 "u_default"）。

输出格式（仅 JSON，不带前后缀）：
{{
  "episodes": [
    {{
      "id": "<string>",
      "owner_id": "<string>",
      "episode": "<叙述文本>",
      "timestamp": <int>
    }}
  ]
}}

注意：parent_type 和 parent_id 由调用方自动补齐，不要输出。
"""
