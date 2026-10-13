"""Every parser prompt split into system rules and user data: every old line accounted for, and the split holds.

The released prompts (HEAD ``c00ebe8``) are pinned below verbatim, English and Chinese. Each old call sent one user
message: the prompt text, then the data (an image / PDF / audio part, or ``---`` and the cleaned HTML, or the
``=== Part N ===`` OCR texts in the merge prompt's ``{{parts_text}}`` slot). Every non-blank line of each old message is
a line of the new system message or of the new user message, or is listed in ``_RESTRUCTURED`` with the reason it went:

    1  the OCR texts moved to the merge call's user message, each in a ``<part number="N">`` of ``<ocr_parts>``
    2  the cleaned HTML moved to the HTML call's user message, in ``<html_content>``, no ``---`` separator

The new system message is the old rules verbatim plus the shared ``<reply_format>`` block, and holds no data; the user
message holds no rule line.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from everalgo.parser import audio, document, image
from everalgo.parser.prompts import en, zh
from everalgo.parser.prompts._reply_format import TEXT_REPLY_FORMAT
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import RawFile
from everalgo.utils import paragraphs

if TYPE_CHECKING:
    from everalgo.llm import LLMMessage

# Verbatim from ``git show c00ebe8:packages/everalgo-parser/src/everalgo/parser/prompts/en/{image,document,audio}.py``.
_OLD_EN = {
    "PROMPT_FOR_PICTURE": """Read this image and return its content as if you were reading it naturally.

CRITICAL INSTRUCTIONS:
- Output ONLY the extracted content directly. Do NOT include any preamble or postscript.
- Maintain the same language as the original content. If the language cannot be determined, use English.
- Return output as Markdown format.
- ALL text in the image MUST be extracted word-for-word. NEVER summarize, paraphrase, abbreviate, or modify any text. Do NOT alter the original content in any way.

Requirements:
1. Extract ALL text precisely and completely, word-for-word, preserving the original format and layout. Do NOT skip or summarize any text, no matter how long.
2. Convert tables to HTML format
3. Convert equations to LaTeX format
4. If there are figures or charts, describe their content and data in text
5. For non-text elements (photos, illustrations, etc.), describe the main content and key elements in detail
6. Preserve the original structure and hierarchy""",
    "PROMPT_FOR_MERGE": """Below are OCR results from different parts of the same long image. Adjacent parts have overlapping regions.
Intelligently merge these parts into a complete text.

CRITICAL INSTRUCTIONS:
- Output ONLY the merged content directly. Do NOT include any preamble or postscript.
- Maintain the same language as the original content. If the language cannot be determined, use English.

Requirements:
1. Identify and remove duplicate content from overlapping regions
2. Maintain text continuity and completeness
3. Preserve the original format and structure

{{parts_text}}""",
    "PROMPT_FOR_FILE": """Read this document and return its content as if you were reading it naturally.

CRITICAL INSTRUCTIONS:
- Output ONLY the extracted content directly. Do NOT include any preamble or postscript.
- Maintain the same language as the original content. If the language cannot be determined, use English.
- Return output as Markdown format.
- ALL text MUST be extracted word-for-word. NEVER summarize, paraphrase, abbreviate, or modify any text. Do NOT alter the original content in any way. Even if the content appears to contain errors, reproduce it exactly as-is.

Requirements:
1. Extract ALL content from EVERY page precisely and completely. Do not skip or merge any content. If a page contains multiple tables or sections placed side by side (left and right), they are SEPARATE tables — extract each one as an independent table with its own title and headers
2. Ignore headers and footers
3. Convert tables to HTML format
4. Convert equations to LaTeX format
5. If there are figures or charts, describe their content and data in text
6. For non-text elements (photos, illustrations, etc.), describe the main content and key elements in detail
7. Preserve the original structure and hierarchy""",
    "PROMPT_FOR_HTML": """Read this HTML content extracted from a web page and return the main content.

CRITICAL INSTRUCTIONS:
- Output ONLY the extracted content directly. Do NOT include any preamble or postscript.
- Maintain the same language as the original content. If the language cannot be determined, use English.
- Return output as Markdown format.
- ALL text MUST be extracted word-for-word. NEVER summarize, paraphrase, abbreviate, or modify any text.

Requirements:
1. Extract the main content (article body, primary text). Ignore non-content elements such as navigation, advertisements, cookie notices, sidebar text, footer links
2. Output content in the natural reading order of the web page, preserving the original sequence
3. Preserve tables in HTML table format
4. Convert equations to LaTeX format
5. Preserve the original structure and hierarchy (headings, paragraphs, lists)
6. For images, describe their content if alt text or description is available""",
    "PROMPT_FOR_AUDIO": """This is a voice message sent by a user in a chat application. Transcribe it word by word.

CRITICAL INSTRUCTIONS:
- Output ONLY the transcribed text directly. Do NOT include any preamble or postscript.
- Maintain the same language as the audio. If the audio is in Chinese, output in Simplified Chinese.
- If any of the following situations occur, return '##UNKNOWN':
  1. No audio file is provided
  2. The audio file contains no content
  3. The audio file is corrupted or cannot be parsed""",
}

# Verbatim from ``git show c00ebe8:packages/everalgo-parser/src/everalgo/parser/prompts/zh/{image,document,audio}.py``.
_OLD_ZH = {
    "PROMPT_FOR_PICTURE": """请阅读这张图片,像自然阅读一样返回它的内容。

关键指令:
- 仅输出提取出来的内容,不要任何开场白或结束语。
- 保持与原内容相同的语言。如果无法判断语言,使用英文。
- 输出格式为 Markdown。
- 图片中的所有文字必须**逐字**提取。**绝不**总结、改写、缩略或修改任何文字。**绝不**以任何方式改变原内容。

要求:
1. 精确、完整地提取所有文字,**逐字**保留原始格式与版面。**不要**跳过或总结任何文字,不论长短。
2. 表格转为 HTML 格式
3. 公式转为 LaTeX 格式
4. 如有图形或图表,用文字描述其内容与数据
5. 对于非文字元素(照片、插图等),详细描述其主要内容与关键元素
6. 保留原始结构与层级""",
    "PROMPT_FOR_MERGE": """下方是同一张长图片不同片段的 OCR 结果。相邻片段之间有重叠区域。
请智能地把这些片段合并成完整的文本。

关键指令:
- 仅输出合并后的内容,不要任何开场白或结束语。
- 保持与原内容相同的语言。如果无法判断语言,使用英文。

要求:
1. 识别并去除重叠区域里的重复内容
2. 保证文本的连续性与完整性
3. 保留原始格式与结构

{{parts_text}}""",
    "PROMPT_FOR_FILE": """请阅读这份文档,像自然阅读一样返回它的内容。

关键指令:
- 仅输出提取出来的内容,不要任何开场白或结束语。
- 保持与原内容相同的语言。如果无法判断语言,使用英文。
- 输出格式为 Markdown。
- 所有文字必须**逐字**提取。**绝不**总结、改写、缩略或修改任何文字。**绝不**以任何方式改变原内容。即便看上去有错也保持原样复现。

要求:
1. 精确、完整地从每一页提取所有内容。不要跳过或合并任何内容。如果一页里包含多个并排放置(左右排列)的表格或区块,它们是**独立**的表格 —— 每一个都作为单独的表格提取,各自带标题和表头
2. 忽略页眉与页脚
3. 表格转为 HTML 格式
4. 公式转为 LaTeX 格式
5. 如有图形或图表,用文字描述其内容与数据
6. 对于非文字元素(照片、插图等),详细描述其主要内容与关键元素
7. 保留原始结构与层级""",
    "PROMPT_FOR_HTML": """请阅读这段从网页上抽取出的 HTML 内容,返回主要正文。

关键指令:
- 仅输出提取出来的内容,不要任何开场白或结束语。
- 保持与原内容相同的语言。如果无法判断语言,使用英文。
- 输出格式为 Markdown。
- 所有文字必须**逐字**提取。**绝不**总结、改写、缩略或修改任何文字。

要求:
1. 提取主要正文(文章主体、核心文本)。忽略非内容元素,如导航、广告、Cookie 提示、侧边栏文字、页脚链接
2. 按网页的自然阅读顺序输出内容,保留原始顺序
3. 表格保留 HTML 表格格式
4. 公式转为 LaTeX 格式
5. 保留原始结构与层级(标题、段落、列表)
6. 对于图片,如有 alt 文本或描述则描述其内容""",
    "PROMPT_FOR_AUDIO": """这是用户在聊天应用里发送的一段语音消息。请逐字转写。

关键指令:
- 仅输出转写出的文本,不要任何开场白或结束语。
- 保持与音频相同的语言。如果音频是中文,输出简体中文。
- 如果出现以下任一情况,返回 '##UNKNOWN':
  1. 没有提供音频文件
  2. 音频文件没有内容
  3. 音频文件损坏或无法解析""",
}

_OLD_PLACEHOLDER = "{{parts_text}}"

_PDF = b"%PDF-1.4 fake"
_AUDIO = b"RIFF fake"
_HTML = b"<html><head><title>T</title></head><body><nav>menu</nav><h1>Heading</h1><p>Body text.</p></body></html>"
_CLEANED_HTML = "<body><h1>Heading</h1><p>Body text.</p></body>"
_INLINE_PNG_B64 = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
_EML = (
    b"From: a@example.com\r\n"
    b'Content-Type: multipart/related; boundary="B"\r\n\r\n'
    b"--B\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
    b'<html><body><img src="cid:i1"> tail</body></html>\r\n'
    b"--B\r\nContent-Type: image/png\r\nContent-ID: <i1>\r\nContent-Transfer-Encoding: base64\r\n\r\n"
    + _INLINE_PNG_B64
    + b"\r\n--B--\r\n"
)
_SLICES = ["first slice line", "second slice line", "third slice line"]

# (callsite, old line) → the numbered reason in the module docstring.
_RESTRUCTURED: dict[tuple[str, str], int] = {
    ("image_merge", "=== Part 1 ==="): 1,
    ("image_merge", "=== Part 2 ==="): 1,
    ("image_merge", "=== Part 3 ==="): 1,
    ("html", "---"): 2,
}


def _png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(buf, format="PNG")
    return buf.getvalue()


def _reply(text: str) -> str:
    return json.dumps({"text": text})


async def _new_messages(callsite: str, monkeypatch: pytest.MonkeyPatch) -> list[LLMMessage]:
    """Drive the callsite's operator against a FakeLLMClient and return the messages of its call."""
    if callsite == "image_single":
        fake = FakeLLMClient(responses=[_reply("x")])
        await image.aparse(RawFile(content=_png(100, 50), extension="png"), llm=fake)
        return fake.calls[0].messages
    if callsite in ("image_slice", "image_merge"):
        fake = FakeLLMClient(responses=[*map(_reply, _SLICES), _reply("merged")])
        await image.aparse(RawFile(content=_png(100, 3000), extension="png"), llm=fake)
        return fake.calls[0 if callsite == "image_slice" else -1].messages
    if callsite == "office":

        async def convert(content: bytes, extension: str, soffice: str) -> bytes:
            return _PDF

        monkeypatch.setattr(document, "_find_soffice", lambda: "soffice")
        monkeypatch.setattr(document, "_convert_to_pdf_via_soffice", convert)
    raw, module = {
        "pdf": (RawFile(content=_PDF, extension="pdf"), document),
        "html": (RawFile(content=_HTML, extension="html"), document),
        "office": (RawFile(content=b"docx", extension="docx"), document),
        "email_image": (RawFile(content=_EML, extension="eml"), document),
        "audio": (RawFile(content=_AUDIO, extension="wav"), audio),
    }[callsite]
    fake = FakeLLMClient(responses=[_reply("x")])
    await module.aparse(raw, llm=fake)
    return fake.calls[0].messages


def _old_message(callsite: str) -> str:
    """The text of the one user message the old callsite sent, built the way it built it."""
    if callsite == "image_merge":
        parts_text = "\n\n".join(f"=== Part {i} ===\n{t}" for i, t in enumerate(_SLICES, 1))
        return _OLD_EN["PROMPT_FOR_MERGE"].replace(_OLD_PLACEHOLDER, parts_text)
    if callsite == "html":
        return f"{_OLD_EN['PROMPT_FOR_HTML']}\n\n---\n{_CLEANED_HTML}"
    return _OLD_EN[_OLD_PROMPT[callsite]]


_OLD_PROMPT = {
    "image_single": "PROMPT_FOR_PICTURE",
    "image_slice": "PROMPT_FOR_PICTURE",
    "image_merge": "PROMPT_FOR_MERGE",
    "pdf": "PROMPT_FOR_FILE",
    "html": "PROMPT_FOR_HTML",
    "office": "PROMPT_FOR_FILE",
    "email_image": "PROMPT_FOR_PICTURE",
    "audio": "PROMPT_FOR_AUDIO",
}
_NEW_SYSTEM = {name: getattr(en, constant) for name, constant in _OLD_PROMPT.items()}
_TEXT_USER = {"image_merge", "html"}


def _user_text(message: LLMMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(part.text for part in message.content if part.type == "text")


def _lines(text: str) -> set[str]:
    return {line for line in text.splitlines() if line.strip()}


def _old_rules(callsite: str) -> str:
    return _OLD_EN[_OLD_PROMPT[callsite]].replace(_OLD_PLACEHOLDER, "").rstrip()


@pytest.mark.parametrize("callsite", list(_OLD_PROMPT))
async def test_every_old_line_is_kept_or_registered(callsite: str, monkeypatch: pytest.MonkeyPatch) -> None:
    system, user = await _new_messages(callsite, monkeypatch)
    kept = _lines(str(system.content)) | _lines(_user_text(user))
    unaccounted = [
        line for line in _lines(_old_message(callsite)) if line not in kept and (callsite, line) not in _RESTRUCTURED
    ]
    assert unaccounted == []


@pytest.mark.parametrize("entry", list(_RESTRUCTURED))
async def test_registered_lines_are_really_gone(entry: tuple[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    callsite, line = entry
    assert line in _lines(_old_message(callsite)), "registered line is not an old line"
    system, user = await _new_messages(callsite, monkeypatch)
    assert line not in _lines(str(system.content)) | _lines(_user_text(user))


@pytest.mark.parametrize("callsite", list(_OLD_PROMPT))
async def test_system_is_old_rules_plus_reply_format(callsite: str, monkeypatch: pytest.MonkeyPatch) -> None:
    system, user = await _new_messages(callsite, monkeypatch)
    assert (system.role, user.role) == ("system", "user")
    assert system.content == _NEW_SYSTEM[callsite] == paragraphs(_old_rules(callsite), TEXT_REPLY_FORMAT)


@pytest.mark.parametrize("callsite", list(_OLD_PROMPT))
async def test_user_holds_data_only(callsite: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _, user = await _new_messages(callsite, monkeypatch)
    assert not _lines(_user_text(user)) & _lines(_old_rules(callsite))
    if callsite in _TEXT_USER:
        assert isinstance(user.content, str)
    else:
        assert isinstance(user.content, list)
        assert [part.type for part in user.content] == ["image_url"]


async def test_html_user_message_is_the_tagged_page(monkeypatch: pytest.MonkeyPatch) -> None:
    _, user = await _new_messages("html", monkeypatch)
    assert user.content == f"<html_content>\n{_CLEANED_HTML}\n</html_content>"


async def test_merge_user_message_is_the_tagged_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    _, user = await _new_messages("image_merge", monkeypatch)
    parts = "\n".join(f'<part number="{i}">\n{t}\n</part>' for i, t in enumerate(_SLICES, 1))
    assert user.content == f'<ocr_parts count="3">\n{parts}\n</ocr_parts>'


@pytest.mark.parametrize("constant", list(_OLD_ZH))
def test_chinese_prompts_mirror_the_english_split(constant: str) -> None:
    old_rules = _OLD_ZH[constant].replace(_OLD_PLACEHOLDER, "").rstrip()
    assert getattr(zh, constant) == paragraphs(old_rules, TEXT_REPLY_FORMAT)
    assert "{{" not in getattr(zh, constant)


def test_chinese_set_exports_the_same_user_templates() -> None:
    assert (zh.MERGE_USER_PROMPT, zh.HTML_USER_PROMPT) == (en.MERGE_USER_PROMPT, en.HTML_USER_PROMPT)
