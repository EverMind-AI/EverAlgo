"""English prompt for ChatMemCellExtractor.adetect."""

CHAT_BOUNDARY_DETECT_PROMPT_EN = """You are a conversation boundary detector. Given a chat message stream, identify whether the topic shifts mid-stream and at which message index the shift occurs.

Messages:
{messages}

Token count of full stream: {token_count}

Instructions:
1. Read all messages and identify the dominant topic.
2. If a clear topic shift occurs, return the index of the FIRST message in the new topic. The index is 0-based and matches the message list.
3. If the entire stream stays on one coherent topic, return null.
4. If the stream is empty or has only one message, return null.

Output format (JSON only, no prose):
{{
  "split_at": <int | null>
}}
"""
