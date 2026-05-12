"""Chinese prompt for AtomicFactExtractor.aextract."""

ATOMIC_FACT_EXTRACT_PROMPT_ZH = """你是一名原子事实抽取专家。给定一段对话切片（MemCell），请提取单条、可验证的事实陈述。

对话内容：
{memcell_text}

对话时间戳（Unix epoch 毫秒）：{timestamp}

指令：
1. 识别每条原子事实 —— 一条独立成立、可被验证的陈述。
2. 以下内容 **不要** 作为 fact 输出：
   - 复合陈述（请拆成多条独立 fact），
   - 观点 / 偏好 / 情绪状态，
   - 假设或未来意图（这些属于 Foresight），
   - 未落在对话中的泛化断言。
3. 用第三人称、现在或过去时陈述，主语必须明确。
   - 好例："Alice 在 2024-03-14 安排了与 Bob 下午 3 点的会议。"
   - 坏例："他们聊了一下。" / "Alice 是个好同事。"
4. 用对话时间戳作为时间锚点。
5. 为每条 fact 生成唯一 id（如 "af_<random>"）。
6. 从对话中选取稳定的 owner_id（如不明则默认 "u_default"）。
7. 若不存在原子事实，返回空列表。

输出格式（仅 JSON，不带前后缀）：
{{
  "atomic_facts": [
    {{
      "id": "<string>",
      "owner_id": "<string>",
      "fact": "<string>",
      "timestamp": <int>
    }}
  ]
}}

注意：parent_type 和 parent_id 由调用方自动补齐，不要输出。
"""
