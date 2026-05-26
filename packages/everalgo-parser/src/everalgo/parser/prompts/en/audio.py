"""Audio parser prompt."""

PROMPT_FOR_AUDIO = """This is a voice message sent by a user in a chat application. Transcribe it word by word.

CRITICAL INSTRUCTIONS:
- Output ONLY the transcribed text directly. Do NOT include any preamble or postscript.
- Maintain the same language as the audio. If the audio is in Chinese, output in Simplified Chinese.
- If any of the following situations occur, return '##UNKNOWN':
  1. No audio file is provided
  2. The audio file contains no content
  3. The audio file is corrupted or cannot be parsed"""
