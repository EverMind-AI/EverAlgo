# `render_prompt` helper — extract the `(prompt or DEFAULT).format(**fields)` pattern

> **Status:** Approved by BOSS on 2026-05-11. Implementation plan at `docs/superpowers/plans/2026-05-11-render-prompt-helper.md`.

## 1. 背景

EverAlgo 的 17 个 operator 函数签名统一接 `prompt: str | None = None`(boundary / user-memory / agent-memory / knowledge 全覆盖),其中目前 2 处实现都用了同一个 pattern:

```python
# packages/everalgo-boundary/src/everalgo/boundary/chat.py:56
rendered = (prompt or CHAT_BOUNDARY_DETECT_PROMPT_EN).format(
    messages=_format_messages_for_prompt(messages),
    token_count=count_tokens(_concat_messages(messages)),
)

# packages/everalgo-user-memory/src/everalgo/user_memory/episode.py:49
rendered = (prompt or EPISODE_EXTRACT_PROMPT_EN).format(
    memcell_text=_render_memcell_text(memcell),
    timestamp=memcell.timestamp,
)
```

15 处是 stub,等待实现 — 它们以后落地时都会复读这个 pattern。AGENTS.md §7 (Adding a New Algorithm Operator) 步骤 3 也明示这是 EverAlgo 算子写 prompt-driven LLM 调用的标准约定。

## 2. 决策

在 `everalgo-core/src/everalgo/prompts/render.py` 加一个 module-level 函数:

```python
def render_prompt(default: str, prompt: str | None, /, **fields: Any) -> str:
    """Render `prompt` with `fields`; if `prompt` is None, use `default`."""
    return (prompt or default).format(**fields)
```

调用方:

```python
from everalgo.prompts import render_prompt

rendered = render_prompt(
    CHAT_BOUNDARY_DETECT_PROMPT_EN,
    prompt,
    messages=_format_messages_for_prompt(messages),
    token_count=count_tokens(_concat_messages(messages)),
)
```

### 2.1 签名要点

- **`default` 在前 + positional-only(`/`)**:语义上 default 是函数本身的属性(operator 自带的 fallback),caller 的 `prompt` override 才是可变输入;positional-only 强迫调用方写出顺序,避免 `render_prompt(prompt=X, default=Y, ...)` 这种顺序错乱
- **`prompt` 也 positional-only**:跟 `default` 同列,避免混淆
- **`**fields`**:keyword-only 强迫 explicit naming,跟 `str.format(**fields)` 一一对应,模板里的 `{name}` 直接读 `fields["name"]`
- **`Any` for fields**:`str.format` 接受任何能 `__format__` 的对象,不做窄类型限制

### 2.2 模块定位

- 物理位置:`packages/everalgo-core/src/everalgo/prompts/render.py`
- 跟现有 `prompts/validator.py`(`check_placeholders` / `check_length`)同级,共属"prompt 基础设施"角色
- 通过 `prompts/__init__.py` 公开:`from everalgo.prompts.render import render_prompt; __all__ = ["render_prompt"]`

## 3. 范围

### 3.1 In Scope

| 文件 | 操作 |
|---|---|
| `packages/everalgo-core/src/everalgo/prompts/render.py` | 新建,~30 行(函数 + docstring) |
| `packages/everalgo-core/tests/prompts/test_render.py` | 新建,~50 行单元测试 |
| `packages/everalgo-core/src/everalgo/prompts/__init__.py` | 加 `from .render import render_prompt` 与 `__all__` |
| `packages/everalgo-boundary/src/everalgo/boundary/chat.py:55-58` | 替换为 `render_prompt(...)` 调用 |
| `packages/everalgo-user-memory/src/everalgo/user_memory/episode.py:48-51` | 替换为 `render_prompt(...)` 调用 |

### 3.2 Out of Scope

- **15 处 stub**:它们的实现是将来 work,本 MR 不动 stub 体内的 `raise NotImplementedError`
- **Prompt validator 集成**:不在 `render_prompt` 内部自动调用 `check_placeholders` / `check_length` — 校验是 prompt **导入期** fail-fast 的事情(validator.py docstring 已说明 "called at module import time after a prompt constant is defined"),不是 render 期事情;混进 render 反而违反单一职责
- **i18n 切换 / prompt logging / escape / async**:都不在本 MR;未来加这些时改一个地方就够,这就是抽出来的价值兑现
- **`AGENTS.md` §7 文档更新**:本 MR 不动 AGENTS;等抽好 + merge 后,后续 docs MR 把 §7 步骤 3 的伪代码示例从 `(prompt or DEFAULT).format(...)` 改成 `render_prompt(...)`,作为给新人的 reference impl

## 4. 单元测试

`tests/prompts/test_render.py` 至少覆盖 4 个 case:

1. `prompt is None` → 用 `default`,fields 正确填入
2. `prompt` 非 None → 用 `prompt`(忽略 `default`),fields 正确填入
3. 模板缺 fields 中的 placeholder 不报错(extras 静默忽略,符合 `str.format(**kwargs)` 行为)
4. fields 缺模板需要的 placeholder → 抛 `KeyError`(把这个 error 暴露给 caller,不吞)

不需要 mock LLM / 不需要 `asyncio_mode` — 纯函数测试。

## 5. 风险

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 现有 2 处替换语义偏差 | 低 | 静默错误 | 单元测试 +  `chat.py` `episode.py` 自身的 test 仍跑过 |
| 17 处 stub 将来不按此 pattern 实现 | 中 | 抽象失效 | AGENTS.md §7 reference impl 跟进(本 MR 外) |
| 模块循环依赖 | 低 | import error | `render_prompt` 不 import 任何 evercore-* 模块,纯 stdlib |
| `default` 跟 `prompt` 位置颠倒 | 低 | 调用方 bug | positional-only `/` 编译期 catch |

## 6. 验收

- `uv run pytest` 仍 167+ passed(新增至少 4 个 test)
- `uv run ruff check . / ruff format --check . / mypy .` 全绿
- 现有 2 处调用读起来更简洁(从 4 行 reduce 到 ~6 行 keyword-arg block)
- `from everalgo.prompts import render_prompt` 成功
- MR CI 5 个 job 全绿
