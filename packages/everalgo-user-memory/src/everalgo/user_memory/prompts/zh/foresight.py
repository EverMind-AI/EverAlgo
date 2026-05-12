"""Chinese prompt for ForesightExtractor.aextract."""

FORESIGHT_EXTRACT_PROMPT_ZH = """你是一名前瞻预测专家。给定一段对话切片（MemCell），请从参与者的发言中提取被预示的未来事件或承诺。

对话内容：
{memcell_text}

对话时间戳（Unix epoch 毫秒）：{timestamp}

指令：
1. 识别每个被预示的未来事件 —— 明确承诺（"我会在周五前完成 X"）、隐含计划（"下周该 review Y"）、开放意图（"希望本季度发布 Z"）。
2. 对每条 foresight，记录：
   - foresight：用第三人称概括所预示的事件。
   - evidence：触发该预示的原对话片段（一段简短引用或转述）。
3. 用对话时间戳作为时间锚点；不要凭空捏造具体未来时间戳。
4. 为每条 foresight 生成唯一 id（如 "fs_<random>"）。
5. 从对话中选取稳定的 owner_id（如不明则默认 "u_default"）。
6. 若不存在预示事件，返回空列表。

输出格式（仅 JSON，不带前后缀）：
{{
  "foresights": [
    {{
      "id": "<string>",
      "owner_id": "<string>",
      "foresight": "<string>",
      "evidence": "<string>",
      "timestamp": <int>
    }}
  ]
}}

注意：parent_type 和 parent_id 由调用方自动补齐，不要输出。
"""
