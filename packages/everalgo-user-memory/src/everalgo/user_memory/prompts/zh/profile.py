"""Chinese prompt for ProfileExtractor.aextract."""

PROFILE_EXTRACT_PROMPT_ZH = """你是一名用户画像合成专家。给定当前对话切片（MemCell）以及同一用户历史 MemCell 簇的摘要，请合成一份长期用户画像快照。

当前对话内容：
{current_memcell_text}

历史对话簇（按时间排序的摘要）：
{cluster_summaries}

当前时间戳（Unix epoch 毫秒）：{timestamp}

指令：
1. 合成 **长期、稳定的用户特征** —— 不是一次性事件，也不是预期承诺。
   - 在范围内：兴趣、习惯、沟通风格、反复出现的偏好、技能、常谈话题、决策模式。
   - 不在范围内：离散事件（"Alice 安排了一场会议"）、单次承诺（"用户会在周五前发草稿"）。
2. 必填字段：
   - id：稳定的唯一标识（如 "pf_<owner_id>" 或随机 "pf_<random>"）。
   - owner_id：本画像描述的用户（如不明则默认 "u_default"）。
   - summary：一段叙述式画像总结（3-6 句话）。
   - timestamp：使用上方给出的当前时间戳。
3. 选填字段 —— 当历史簇中有证据支持时，可作为额外 JSON key 输出；否则完全省略（不要输出空占位）：
   - interests：list[str]
   - habits：list[str]
   - preferences：dict[str, str]
   - hard_skills：list[str]
   - communication_style：str
   - decision_patterns：list[str]
4. 若历史簇为空，仅基于当前 MemCell 合成画像，并在 summary 中说明证据有限。

输出格式（仅 JSON，不带前后缀）：
{{
  "id": "<string>",
  "owner_id": "<string>",
  "summary": "<一段叙述式画像总结>",
  "timestamp": <int>
}}

你可以在证据支持的前提下追加 interests / habits / preferences / hard_skills / communication_style / decision_patterns 等额外字段。不要输出 parent_id 或 parent_type —— Profile 是用户级聚合，不针对单个 MemCell。
"""
