# LLM 契约

状态：终版设计，2026-09-23。本文是 [英文正本](llm-contract.md) 的中文对照版，章节逐一对应；两者有出入时以英文正本为准。

本文定义 EverAlgo 算子与宿主注入的 `LLMClient` 实现之间的契约。本文描述**契约是什么样**，不描述现有代码如何迁移过去。

立场：EverAlgo 是协议指定方，定义契约与一致性检查；client 怎么构造、怎么重试、怎么路由、默认值放哪，全是实现方的事。

本文的「必须 / 不得 / 应当 / 可以」对应英文正本按 [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119) 解释的 MUST / MUST NOT / SHOULD / MAY。

| 术语 | 含义 |
|---|---|
| 算子 | 调用 `LLMClient.achat` 的 EverAlgo 算法 |
| 实现方 | 宿主一侧的 `LLMClient` 实现 |
| 宿主 | 构造实现方、把它注入算子、接住算子抛出的异常的应用；实现方和宿主是同一方 |

---

## 1. 概述

### 1.1 一句话

一个方法、一次调用、多模态输入、**JSON 输出**。算子用一个 pydantic 模型声明它要什么结构，自己解析、自己兜底。

```
          算子（EverAlgo）                     实现方（宿主）
  ┌──────────────────────────────┐        ┌──────────────────────────┐
  │ prompt：写明输出 JSON        │        │ 怎么让模型产出这个结构   │
  │         + 可解析的示例       │──────► │ 怎么校验、怎么重试       │
  │ response_format：模型类      │  要求  │ 全是实现方自己的事       │
  │                              │        │                          │
  │ model_validate_json(text)    │◄────── │ 期望：交过来的           │
  │   失败 → LLMOutputError      │  文本  │   已经是合法结果         │
  │ 业务校验                     │        │                          │
  │   不过 → LLMOutputError      │        │                          │
  └──────────────────────────────┘        └──────────────────────────┘
     知道格式，不知道模型                    知道模型，不知道策略
```

### 1.2 契约怎么划线：看 EverAlgo 自己能不能发现

| EverAlgo 能不能发现 | 例子 | 契约怎么写 |
|---|---|---|
| **发现不了** | 显式 `temperature` 被默认值覆盖；图片被悄悄丢掉，模型可能对着空内容编出一段合法 JSON；上游标记了拒答，拒答内容却写在合法 JSON 字段里；上游报告了截断，正文却恰好是闭合的合法对象 | **硬性条款**（第 3 节），一致性套件必检 |
| **发现得了** | 输出不是合法 JSON、缺必填字段、顶层对象未闭合的截断 | **只作建议**（第 7 节）。算子解析失败会抛 `LLMOutputError` 兜住（第 4 节），但那时宿主只能重跑整个算子，成本最高；实现方在自己那层先处理掉最划算 |

---

## 2. 接口

### 2.1 签名

```python
async def achat(
    self,
    messages: list[LLMMessage],
    *,
    response_format: type[BaseModel],     # 必填
    temperature: float | None = None,
    max_tokens: int | None = None,
    **extra: Any,
) -> str          # 模型输出的文本；期望能被 response_format.model_validate_json 通过
```

### 2.2 `response_format`：一个 pydantic 模型

模型由 EverAlgo 定义，必须满足下面四条（由 EverAlgo 自己的单测保证，见 4.3）：

| 要求 | 为什么 |
|---|---|
| **顶层是 object**：`BaseModel` 子类，不接受 `RootModel` | 4.2 的截断拦截依赖这一条。`RootModel` 也是 `BaseModel` 子类（实测 `issubclass(RootModel, BaseModel) is True`），`RootModel[int]` 的 schema 顶层是 `integer`，截断的数字仍能解析 |
| **只写结构约束** | 实现方若用它校验，会在失败时重试；业务规则不通过时重试多半没用，白花预算 |
| **继承 `StrictOutputModel`（含嵌套子模型），字段不设默认值** | 让本地校验与 strict schema 接受同一批输入，见下 |
| **落在 strict 子集内**（8.2） | 否则实现方用不了原生 schema 约束这条最可靠的路 |

#### 默认值、类型转换、多余字段也是输出语义

实现方用原生 schema 约束时，发出去的是 EverAlgo 请求构造函数从模型生成的 strict schema：它把**全部字段设为必填**，并补 `additionalProperties: false`（第 8 节）。默认配置的 pydantic 模型在本地校验时判法不同。实测：

| 输入 | 默认模型（`count: int = 0`） | `strict=True, extra="forbid"`，不设默认值 | strict schema |
|---|---|---|---|
| `{}` | 接受，补成 `count=0` | 拒绝 | 拒绝（必填） |
| `{"count":"3"}` | 接受，转成整数 3 | 拒绝 | 拒绝（`type: integer`） |
| `{"count":3,"extra":true}` | 接受，忽略多余字段 | 拒绝 | 拒绝（`additionalProperties: false`） |
| `{"count":3}` | 接受 | 接受 | 接受 |

用默认模型的话，同一个算子在实现方走原生 schema 约束时，模型一定输出全部字段；走 JSON 模式或只靠 prompt 时，缺字段会被默认值悄悄补上、`"3"` 会被悄悄转成 3。**算子拿到的东西就取决于实现方选了哪种手段**，而手段由实现方自由选（3.2）。

所以模型这样写：

```python
class StrictOutputModel(BaseModel):
    """Base for every response_format model: config only."""

    model_config = ConfigDict(strict=True, extra="forbid")


class Fact(StrictOutputModel):
    content: str

class ReflectOutput(StrictOutputModel):
    facts: list[Fact]          # 子模型也继承基类，否则配置传不过来
```

- **子模型也必须继承。** pydantic 的配置只随继承传递，不随字段类型传递：官方文档说嵌套模型 "each model has its own 'configuration boundary'"（*Configuration* 一章「Configuration propagation」）；实测父模型设了 `forbid`、子模型没设时，子对象里的多余字段照样被接受。
- **字段不得设默认值。** 可以缺省的字段写成 `X | None`，由模型显式给 `null`（实测 strict 下不给键拒绝、给 `null` 接受，与 strict schema 一致）。基类管不了这一条，由 4.3 的单测检查。
- 确实需要默认值或宽松类型转换的，必须是有意为之，并在算子单测里覆盖「只靠 prompt」这条路径。

#### `StrictOutputModel` 的边界

| 规则 | 为什么 |
|---|---|
| **只放配置**：不放字段、方法、校验器 | 放了任何行为，它就成了多余的抽象层 |
| **只有一层**：所有模型直接继承它，不从它再派生中间基类 | 避免「父类一改，子类全跟着变」 |
| **不进协议**：放在 everalgo-core 供算子包 import，协议签名仍是 `type[BaseModel]` | 实现方不需要知道它 |

用父类共享配置是 pydantic 官方推荐的做法（*Configuration* 一章「Change behaviour globally」："you can create your own custom parent class with a custom configuration, as the configuration is inherited"）。宿主的代码库里已有同样写法：一个只放 `model_config` 的基类，被几个操作模型继承。Python 语言本身（PEP、官方文档）对此没有推荐，这属于库的约定；Django 的「Abstract base classes」是同一思路。

#### 结构约束与业务约束的界线

| 写进模型（结构约束） | 留在算子解析之后（业务约束） |
|---|---|
| 字段类型、必填、`Literal` 枚举、嵌套子模型 | 取值合不合理、结果是不是空、跨字段一致性、**内容是否完整**（比如 OCR 有没有漏掉原文，4.2） |
| 能被 JSON Schema 表达的 `Field` 约束（长度、`pattern`） | `field_validator` / `model_validator` 等自定义校验器 |

### 2.3 多模态输入

走 `LLMMessage.content`：接受 `str`，也接受 `list[ContentPart]`（text + image / pdf / audio 的 data URI），序列化后就是 OpenAI / OpenRouter 的 wire 形态。

### 2.4 命名

协议这一族统一 `LLM` 前缀：`LLMClient` / `LLMMessage` / `LLMOutputError`。client 怎么构造不属于协议，是实现方的事。EverAlgo 自带的参考实现 `OpenAICompatClient` 和 openai、anthropic 两家 SDK 的客户端一样，直接收关键字参数。返回值没有包装类型，就是 `str`，理由见 6.4。

`LLMMessage` 与领域类型 `everalgo.types.chat.ChatMessage` 是**两个不同的东西**：后者是对话记录，带 `id` / `timestamp` / `sender_id` / `sender_name`，`role` 只有 user 与 assistant；前者只有 `role` 与 `content`，`role` 含 system，是发给模型的一次请求。两者从不互相转换——边界检测这类算子拿领域消息渲染成文本，作为一条 user `LLMMessage` 的内容发出去。

**方法名叫 `achat`：叫 `chat` 不叫 `generate`，并带 `a` 前缀。** 入参就是聊天式的多条消息，`chat` 描述的正是这个形状；`generate` 的出处是 HuggingFace `model.generate`、Google `models.generate_content` 那类 prompt-in 接口，与本签名不符。主流 SDK 也无一例外：OpenAI 是 `chat.completions.create` / `.parse`（响应类型 `ChatCompletion`），Anthropic 是 `messages.create`（响应类型 `Message`），litellm 是 `completion()` / `acompletion()`（响应类型 `ModelResponse`）。类型名需要前缀是因为它们脱离上下文出现在 import 与注解里；方法名永远挂在 `LLMClient` 实例上，不存在这个问题。`a` 前缀遵循全仓「原生异步的算子名一律带 `a` 前缀」的规则，litellm 的 `completion()` / `acompletion()` 也是这样分的。旁边没有同步的 `chat`：协议是宿主要实现的接口，再配一个同步孪生会让实现负担翻倍；算子只在异步代码里调用它，同步桥在上一层、挂在算子上。

### 2.5 错误：契约只定义 EverAlgo 自己抛的那一个

```python
class LLMOutputError(Exception):
    ...   # 模型的回复不符合算子的要求（解析失败，或业务校验不过）
```

| 异常 | 谁抛 | 什么时候 |
|---|---|---|
| `LLMOutputError` | **算子** | 解析失败，或业务校验不过（4.1） |
| 实现方自己的异常 | 实现方 | 调用失败、参数满足不了、上游拒答或截断等。**类型由实现方定，契约不规定** |

**契约不替实现方定义异常。** 实现方和宿主是同一方：异常由实现方抛、由宿主接，EverAlgo 只是中间经过，从不按类型捕获它们。核对结果（截至 2026-09-23）：

| 核对项 | 结果 |
|---|---|
| 算子按类型捕获调用失败 | **零处**。仅有的 3 处降级（`everalgo.parser` 的文档与图片解析、`everalgo.user_memory` 的 episode 抽取）都是 `except Exception`，不看类型 |
| 宿主有没有依赖 EverAlgo 定义的调用异常 | 没有。生产宿主自己定义异常体系，从不 import EverAlgo 的调用异常 |

所以 EverAlgo 这边只需要守一条：**实现方抛的异常原样透传，不包装、不改类型**（4.3），宿主才能用自己的类型认出它。宿主看到的异常只有三种来源：

| 看到的异常 | 意思 |
|---|---|
| 自己实现方的类型 | 调用失败 |
| `LLMOutputError` | 模型的输出不合要求 |
| 其他 | EverAlgo 的 bug，或调用方传了非法参数 |

`LLMOutputError` 只表示「模型没按要求输出」，**是否重试由宿主自己决定**（第 5 节）。

**什么时候才需要协议定义调用异常：** 算子要区分「这次调用失败了」和「其他错误」并做不同处理时。那时它不认识宿主的类型，就需要协议给一个统一类型。现有的降级点用不到这种区分，所以不定义。

---

## 3. 实现方的硬性条款：EverAlgo 发现不了的事

### 3.1 显式值原样送达

传了值就用这个值；传 `None` 表示由实现方配置的默认决定；两边都没有则请求里省略该参数，交给服务端。**实现方的默认值不得覆盖调用方的显式值，无论那些默认存在哪里。**

**每一次**上游请求都如此，包括实现方自己发起的重试。尤其是**不得为了绕开截断而抬高 `max_tokens`**——调用方给的值就是它要的预算，抬高它是在替调用方改成本与延迟。

**为什么是硬性的：** 这些值被改掉以后，模型照样返回一段合法 JSON，EverAlgo 看不出任何异常。

### 3.2 不得静默丢弃或改写 `messages`

能送达就送达；确知送达不了（例如图片发给只收文本的模型）就抛错，类型由实现方定（2.5）。**不得静默丢弃。** 为适配上游做的**等价格式转换不算改写**：同一个语义在不同供应商的请求体里长得不一样，翻译是适配器的本职（反例见 6.6）。

**为什么是硬性的：** parser 的 OCR 调用里，图片被丢掉后，模型可能照样按 prompt 编出一段格式合法的 `{"text": "..."}`（这是推断，未实测）。那样算子的校验会全部通过，错误的内容直接入库，EverAlgo 没有任何办法察觉。

上游服务端静默忽略参数带来的不确定性**不在实现方责任内**：OpenAI 兼容服务端对不认识的参数默认忽略而不报错，契约**无权**要求服务端报错。

**`response_format` 不在这一条里。** 发什么格式参数、发不发，是实现方的手段，EverAlgo 不规定（建议见 7.2）。格式要求也就不存在「送达不了」的情况。

### 3.3 取消原样传播

实现方不得吞掉 `asyncio.CancelledError`（比如吞掉它继续重试）。

**为什么是硬性的：** 有的算子会并发发起多次调用（`everalgo.agent_memory.case`、`everalgo.rank.agentic`、`everalgo.rank.hybrid` 用了 `asyncio.gather` / `TaskGroup`），取消被吞掉会让上层的超时与取消失效，EverAlgo 察觉不到。

### 3.4 上游明确报告拒答时必须抛错

上游**明确标记**模型拒绝作答时，实现方必须抛错（类型由实现方定），不得把拒答文本或空内容当作结果返回。

**为什么是硬性的：** 拒答能不能被算子的解析发现，取决于它长什么样。实测（`model_validate_json`）：

| 拒答的样子 | 算子解析 |
|---|---|
| 纯文本「抱歉，我无法处理这个请求。」 | 拒绝 |
| 空内容 | 拒绝 |
| `{"text": "抱歉，我无法处理这个请求。"}` | **通过** |

第三种正是自由文本调用点要用的 `{"text": "..."}` 形状（6.1）。它是合法 JSON，结构也对，算子无从分辨。而拒答信号只有实现方看得到：

| 上游 | 拒答信号 | 出处 |
|---|---|---|
| OpenAI / OpenAI 兼容 | `message.refusal` 非空 | openai SDK 2.35.1 `openai/types/chat/chat_completion_message.py:63` |
| Anthropic | `stop_reason: "refusal"` | 官方文档 *Handling stop reasons*：Claude declined to respond. |

可行性：EverAlgo 的参考实现 `everalgo.llm.openai_compat` 看到 `refusal` 就抛错；已有生产宿主的适配器在 content 为空时抛错，覆盖了 API 拒答。

**这条只管上游明确标记的拒答。** 模型没被标记、只是在字段里用自然语言说「无法处理」，契约不要求实现方去猜，算子的通用解析也分辨不出。这是已知局限：能靠业务校验识别的由算子自己做，识别不了的就是识别不了。

### 3.5 上游明确报告截断或异常中止时，不得返回该次正文

上游**明确报告**这次生成被截断或异常中止时，实现方必须重试或抛错（类型由实现方定），**不得直接返回该次正文**。重试同样受 3.1 约束：不得抬高 `max_tokens`。

**结束原因缺失时，不要求实现方推断完整性。**

**为什么是硬性的：** 理由与 3.4 相同。截断能被算子发现，前提是对象没闭合（4.2）。下面这个响应的正文能通过解析：

```
content       = {"text": "部分内容"}
finish_reason = length
```

实测 `model_validate_json` 正常返回 `text='部分内容'`。正文恰好闭合的情况确实存在，例如模型刚写完右括号就撞上上限，或者实现方、网关对截断的 JSON 做了修复（补齐括号）——后者是推断，未实测。而结束原因只有实现方看得到。

| 上游 | 截断 / 异常中止的信号 | 出处 |
|---|---|---|
| OpenAI / OpenAI 兼容 | `finish_reason` 为 `length`、`content_filter` | openai SDK 2.35.1 `openai/types/chat/chat_completion.py:25` |
| vLLM 等兼容网关 | `finish_reason` 为 `abort`、`error` | 实际遇到过，已有生产宿主的适配器处理了这两种，见下 |
| Anthropic | `stop_reason` 为 `max_tokens`、`model_context_window_exceeded` | 官方文档 *Handling stop reasons* |

`content_filter` 表示上游因内容过滤停止生成，归在「异常中止」里。

可行性：已有生产宿主的适配器把 `length`、`abort`、`error`、`content_filter` 都当失败处理。

**这条不承诺内容完整，也不把结束原因交给算子。** 它只要求实现方处理自己看得到的明确信号；没有信号时，内容完整性归业务校验（4.2）。

---

## 4. 算子侧

### 4.1 算子怎么处理返回值

```python
text = await self._llm.achat(messages=[...], response_format=ReflectOutput)
output = ReflectOutput.model_validate_json(text)   # 失败 → 抛 LLMOutputError
check_business_rules(output)                       # 不过 → 抛 LLMOutputError
```

1. **总是自己解析。** 不假设实现方校验过。解析失败抛 `LLMOutputError`。
2. **做业务校验**（2.2 界线表的右列）。不过也抛 `LLMOutputError`。

算子**从不重试**（第 5 节），也**不需要 `finish_reason`**：上游报告的截断由实现方处理（3.5），顶层对象未闭合的截断会在第 1 步被拦下（4.2）。让算子去匹配 OpenAI 的 `length`、Anthropic 的 `max_tokens`，只会把 provider 知识泄漏进算法层。

### 4.2 解析只拦得住「对象没闭合」，拦不住「内容不完整」

**顶层 JSON 对象未闭合的截断由解析拦截；解析成功不保证内容完整，内容完整性由业务校验负责。**

前半句的依据：顶层固定为 object（2.2），在对象闭合之前截断，一定解析失败。实测（`model_validate_json`）：

- 一个 26 字符的对象，它的 25 个真前缀**全部**校验不过；
- 按 JSON 值类型穷举全部真前缀：object、array、string、`true`、`null` 的任何截断都不合法，**只有顶层是数字时**截断仍合法（`12345` 截成 `123`）。2.2 排除了顶层非 object，这条路也就堵上了。

所以不管实现方做不做校验、结束原因报不报，**语法上没写完的 JSON** 到不了业务逻辑。上游报告了截断、但正文恰好闭合的情况，由 3.5 在实现方那一层拦下。

后半句的依据：真前缀实验只证明「对象闭合之前截断会失败」，证明不了内容完整。`{"text": "只提取了部分内容"}` 是一个闭合的、合法的对象，它有没有漏掉原文，解析看不出来。这类判断属于业务校验：能判断的由算子判断，判断不了的就是已知局限。

### 4.3 算子侧义务

这些是 EverAlgo 自己要守的规矩。实现方看不到策略内容，所以**它们由 EverAlgo 自己的单测检查，不进实现方的一致性套件**（第 9 节）。

| 义务 | 为什么 | EverAlgo 单测怎么查 |
|---|---|---|
| **prompt 写明输出 JSON，并给可解析的示例** | 模型要靠 prompt 才知道该输出什么；实现方若不发格式参数，prompt 就是唯一的约束。发 JSON 模式时也必须写：openai SDK 2.35.1 的参数说明（`openai/types/beta/assistant_create_params.py:93-99`）要求另外在消息里让模型输出 JSON，否则模型可能输出 "unending stream of whitespace" 直到撞上 token 上限 | 渲染后的 prompt 含格式说明；示例能被对应模型 `model_validate_json` 通过 |
| **模型只写结构约束** | 2.2 | 模型上没有自定义校验器 |
| **模型（含嵌套子模型）继承 `StrictOutputModel`，字段不设默认值；例外须有意为之** | 2.2：否则算子拿到的东西随实现方的手段变化 | 遍历每个 `response_format` 模型及其全部子模型，断言都是 `StrictOutputModel` 的子类、没有字段带默认值；有例外的算子，单测覆盖「只靠 prompt」路径 |
| **模型顶层是 object，且落在 strict 子集内** | 2.2 / 8.2 | 对每个 `response_format` 模型跑 `to_openai_response_format`，不抛错 |
| **自己解析，失败与业务校验不过都抛 `LLMOutputError`，不抛裸 `ValueError`，不自行重试** | 4.1 / 第 5 节 | 算子单测 |
| **实现方抛的异常原样透传，不包装、不改类型** | 宿主要靠自己的类型认出调用失败（2.5）。明确的降级点可以接住异常，但必须是有意为之 | 算子单测：替身抛一个自定义异常，断言它原样冒出来 |

示例：reflect 的两个 prompt（`everalgo.user_memory.prompts.en.reflect`）都写了 "Reply with one JSON object, nothing else."

---

## 5. 谁重试：算子从不重试，宿主按错误类型决定

| 层 | 做什么 | 失败时 |
|---|---|---|
| 实现方 | 硬性条款要求的处理（第 3 节）；建议在自己那层校验并重试（7.1） | 自己决定重试或抛错 |
| **算子** | 解析 + 业务校验 | 抛 `LLMOutputError`，**从不重试** |
| **宿主** | 按错误类型决定重试 / 死信 / 跳过 | —— |

算子没有重试机制，也不该有：它不知道预算、不知道成本、不知道这次调用在哪条链路上。换一次生成值不值得，只有宿主答得了。

**为什么要有 `LLMOutputError` 这个类型：** 算子解析失败若抛**裸 `ValueError`**，宿主就分不清「模型吐了垃圾」和「算子有 bug」，只能走 traceback 猜来源，再按「万一是我们自己的 bug」的保守口径处理，也就是当作不可重试。**后果是一次偶发的畸形输出被永久丢进死信**，而重试一次大概率就好了。

**但异常类型描述的是失败来源，不等于可恢复性：**

- `LLMOutputError` **不一定值得重试**——prompt 本身有问题、业务约束模型永远满足不了，重几次都一样；
- 裸 `ValueError` 也**不一定是算子 bug**——调用方传了非法参数同样会得到它。

所以契约只承诺**来源可辨**，重试与否由宿主结合预算、链路位置、已重试次数决定。

**宿主在这一层重试的代价最高**：它重试的是整个算子调用，要重读数据、重渲染 prompt，UPDATE 这类多遍算子还要重跑每一遍。这正是第 7 节建议实现方先在自己那层处理的原因。如果实现方内部也重试，两层次数会相乘，宿主定策略时要把这一点算进去。

---

## 6. 关键取舍与依据

### 6.1 为什么所有输出都必须是 JSON，不提供自由文本一级

**收益：**

- **职责分得干净。** 要不要 JSON、要什么结构，是策略 prompt 决定的，归 EverAlgo；模型支持什么手段，归实现方。只要输出统一是 JSON，宿主就完全不需要知道每个策略要什么格式。
- **算子能自己兜底。** 有结构才能校验。若允许自由文本输出，截断的文本看起来就是正常文本，算子无从发现；若允许顶层为任意 JSON 值，截断的数字仍能解析。输出统一为顶层 object 后，对象未闭合的截断一定解析失败（4.2）。内容是否完整仍要靠业务校验。

**代价：**

1. **本来就要自由文本的调用点要包一层。** 影响面（截至 2026-09-23 实测，非测试代码，按文件粗筛）：

   | 调用点类型 | 数量 | 怎么写 |
   |---|---|---|
   | 所在文件已经在解析 JSON | 26 | 声明对应的模型即可 |
   | 输出自由文本（`everalgo.parser` 的 image 3、document 4、audio 1） | 8 | 包成 `{"text": "..."}` 一类的模型 |
   | 已声明 schema（reflect） | 1 | 无 |

   自由文本放进 JSON 字符串后，换行和引号都要转义，token 会多一些。

2. **格式限制会不会影响推理质量，有争议。** Tam 等人，*Let Me Speak Freely?*（arXiv:2408.02442，2024-08-05）观察到格式限制下推理能力明显下降，格式越严格下降越多；.txt 团队的回应 *Say What You Mean* 认为那是 prompt 写法的问题，prompt 写好后结构化输出反而高 1–4 个百分点。常用的缓解办法是需要推理的字段放在结论字段前面，reflect 的 prompt 已经这么排。这一点要靠实验确认，不能靠文献下结论。

3. **动态键的 `dict[str, X]` 不能用**（8.2 实测 400），要改成 `list[子模型]`。

### 6.2 为什么 `response_format` 是 pydantic 模型，而不是 JSON Schema 字典

| | JSON Schema 字典 | pydantic 模型 |
|---|---|---|
| 转 strict schema | 每个算子自己转 | 实现方需要时调 EverAlgo 提供的、对应其 API 的请求构造函数（第 8 节） |
| 实现方想校验时 | 要另引入 `jsonschema` 库 | 直接 `model_validate_json` |
| 实现方与算子的**本地校验**一致吗 | 不一致：`int` 字段收到 `"3"`、出现多余字段时，schema 与默认 pydantic 判法相反（实测） | 一致，是同一个模型 |
| 算子写法 | `response_format=转换函数(M)` | `response_format=M` |

注意「本地校验一致」不等于「三种生成方式约束一致」：后者要求模型按 2.2 配置。

字典方案唯一的好处，是实现方不必自己写「pydantic 转 strict schema」那约 50 行递归改造。EverAlgo 的请求构造函数做了这套改造之后，这个好处就没有了。OpenAI SDK 的 `chat.completions.parse()` 也直接收模型类（pydantic `BaseModel`，dataclass 也接受；openai 2.35.1 `openai/lib/_parsing/_completions.py:270-277`）。everalgo-core 本来就依赖 pydantic，协议又是 Python 协议，不存在「实现方没有 pydantic」的情况。

### 6.3 调用面只留四项，`model` 不在其中

| 参数 | 拥有者 | 结论 | 依据 |
|---|---|---|---|
| `response_format` | **算子**——结构是策略 prompt 决定的 | 必填 | 6.1 |
| `max_tokens` | 宿主（预算）；算子知道自己输出形状时覆盖 | 可选 | everalgo-parser 对同一流水线两个阶段设了 8 000 与 32 000 两个预算，宿主的单一 scene 配置表达不了 |
| `temperature` | 宿主（与模型绑定的调优，模型是宿主选的） | 可选 | 35 个调用点（截至 2026-09-23）中 5 处有具体理由覆盖（如 OCR 合并写死 0.0 求确定性），其余交给宿主 |
| `model` | 宿主（scene 路由 + fallback 换模型） | **不在调用面** | 35 个调用点（截至 2026-09-23）**零处**传它；宿主适配器收下也只是丢掉 |

「实际服务的模型」确实需要被记录（fallback 会换模型、网关会路由），但**需要它的是宿主，不是算子**——宿主的 provider 自己就知道路由到了谁，计费也在那一层。所以它也不在返回值里，见 6.4。

### 6.4 返回就是 `str`，没有包装类型

逐个字段查过算子的真实读取（截至 2026-09-23）：

| 字段 | 算子读取次数 | 结论 | 理由 |
|---|---|---|---|
| `content` | 41 | **就是返回值本身** | 业务正文 |
| `model` | 9 | ❌ | 9 处**全在 everalgo-parser**，全都只是塞进 `ParsedContent.metadata`，而那两个键**除了 parser 自己的单测没有任何消费者**；benchmarks 零读取。归因是宿主的事，见 6.3 |
| `finish_reason` | 8（**零处分支**） | ❌ | 同样全在 parser 的 metadata 里。上游报告的截断由实现方处理（3.5），对象未闭合的截断由算子的解析拦下（4.2），没有决策价值 |
| `parsed` | 3 | ❌ | 算子自己 `model_validate_json` 就拿到精确类型，不用 cast |
| `usage` | **0** | ❌ | 实现方自己就在计费链路上，不必经由算子 |
| `raw` | **0** | ❌ | 同上 |

只剩一个字段，就不该有包装：**`achat()` 直接返回 `str`**。算子总要自己解析一次（4.1），实现方即使解析过，也不必把结果对象传回来。

唯一支持保留空壳的理由是演进——往 dataclass 加字段是向后兼容的，`str` 改成对象不是。但从上表看，算子将来会需要的东西也是零：`usage`、`finish_reason`、重试次数全是宿主的关切。真到了要加的那天，加一个新方法比一直留一个空壳干净。

**连带影响**：`ParsedContent.metadata` 没有 `model` 与 `finish_reason` 两个键。

### 6.5 `messages: list[LLMMessage]`，不是 `prompt: str`

「一串带角色的消息」是各家共有的形状，它天然装得下多模态、系统提示与 few-shot 的多轮表达；换成 `prompt: str` 就要再发明 `system` 与 `attachments` 两个参数，用两个自造机制绕开一个不需要绕的问题。

但**不要把它当成各家逐字段相同的 wire format**——细节差异真实存在，见 6.6，那正是适配器要翻译的东西。

### 6.6 为什么契约不规定 wire 形状：一个具体反例

Anthropic Messages API 与 OpenAI 的差异不是细枝末节（2026-09-22 查官方文档）：

| | OpenAI / OpenRouter | Anthropic Messages |
|---|---|---|
| 系统提示 | `messages` 里 `role="system"` 的一条 | **顶层 `system` 参数**，文档原话：there is no `"system"` role for input messages in the Messages API |
| 结构化输出 | `response_format` | **没有这个参数**，用 `output_config.format`：`{"type": "json_schema", "schema": {...}}` |
| 图片 / PDF | `image_url` / `file` content part | `image` / `document` content block |

一个 Anthropic 适配器把 `role="system"` 的消息提到顶层 `system`，是 3.2 允许的等价翻译。至于结构化输出走 `output_config.format` 还是只靠 prompt，是它自己的手段，契约不管。若契约写成「逐字转发 OpenAI 字段」，这个正确的实现反而处处违规。

---

## 7. 给实现方的建议（不属于契约）

以下都**不是**契约要求（属于「应当」级别的建议，非规范性），EverAlgo 不检查、也不依赖。之所以写出来，是因为按这样做，宿主的重试成本最低。

推荐的实现流程把第 3 节的硬性条款和本节的建议按先后排在一起，`OpenAICompatClient.achat` 就是照着这个流程一步步写的（7.4）。每一步都标明是「必须」（第 3 节）还是「建议」（本节）：

1. **构造请求，只构造一次**，每次尝试都原样发送它。
   - 算子显式传的 `temperature` / `max_tokens` 原样发送；只有传 `None` 时才用实现方自己的默认值，没有默认值就不带这个参数（必须，3.1）。
   - 每个 content part 都要送到，包括非文本内容；送不到就抛错（必须，3.2）。
   - 按模型支持的能力选择格式参数（建议，7.2）。
2. **发送请求，直到拿到通过全部检查的回复**，次数要有上限（建议，7.1、7.3）。每次尝试：
   1. **发送请求。** `asyncio.CancelledError` 原样往外抛（必须，3.3）。传输失败后要不要重发，由实现方自己决定。
   2. **看上游对这次生成的报告。** 报告拒答就抛错（必须，3.4）。被截断或异常中止的生成不能返回（必须，3.5）：`length`、`content_filter` 直接抛错，因为同一个请求一定会再撞上；`abort`、`error` 可以像传输失败一样重发（建议，7.3）。
   3. **用 `response_format` 调 `model_validate_json` 校验回复**，不通过就重新发送（建议，7.1）。

### 7.1 交给算子之前先校验

**理由：重试粒度。** 实现方发现输出不合格，就地重发一次调用即可，prompt 都不用重渲染；留给算子发现，宿主只能重跑整个算子调用（第 5 节）。

建议做法：

- 用**算子给的同一个模型** `model_validate_json` 校验，判法与算子完全一致；
- 不合格就重试，有上限；用尽后抛实现方自己的异常；
- 结束原因缺失时，以校验结果为准即可（有些网关会在内容可用的响应上省略结束原因，按校验判就不会误杀）。

上游明确标记的拒答、截断、异常中止不在这里：那是硬性条款（3.4、3.5）。

pydantic 的实际判定（实测，`model_validate_json`）：JSON 本身以外多出任何东西都会被拒，所以这类回复要重试，不交给算子：

| 输入 | 结果 |
|---|---|
| 完整 JSON | 通过 |
| 前后有空白 | 通过 |
| 尾部多出字符 | 拒绝 |
| 包在 ` ```json ` 围栏里 | 拒绝 |
| 前面有一句说明文字 | 拒绝 |

**反面写法：** 请求降了级，校验也跟着降级。

```python
json_mode = {"type": "json_object"}
text = await provider.achat(..., response_format=json_mode)
json.loads(text)      # 只确认是合法 JSON，没按模型校验，缺字段的回复也会交给算子
return text
```

这不违反契约——算子解析时照样会发现问题——但它把本该在实现方这层处理的失败，推到了重试成本最高的那一层。

### 7.2 发什么格式参数，按模型能力自己选

| 当前模型支持 | 可以发 |
|---|---|
| 原生 schema 约束 | strict schema，由对应 API 的构造函数生成（第 8 节），首次成功率最高 |
| 只支持 JSON 模式 | JSON 模式参数 |
| 都不支持 | 不发，只靠 prompt（4.3 保证 prompt 里写明了格式） |

选哪种会影响路由，要自己权衡。实测（2026-09-22，`z-ai/glm-5.3-flash`，32 个 endpoint）：

| 请求里带的格式参数 | 可用 endpoint |
|---|---|
| 不带 | 32 |
| JSON 模式 | 29 |
| 原生 schema 约束 | **24**（8 家不支持 structured outputs，含 Z.AI 官方） |

而且 OpenRouter 的软偏好看的是**取值**：同一实验里把路由钉向三家不支持 `response_format` 的 provider，不带或带 `{"type": "text"}` 时三次都落在目标 provider，带 JSON 模式时三次全被挤走。

**EverAlgo 自带的参考实现提供这个开关。** `OpenAICompatClient` 有一个构造参数 `response_format_mode: Literal["json_schema", "json_object", "prompt_only"]`，默认 `json_schema`。core 的工具函数 `everalgo.llm.to_openai_response_format(model, mode)` 把它转成 OpenAI Chat Completions 的 `response_format` 参数，任何 OpenAI 兼容的实现方都可以直接复用，不必自己再写一遍（8.4）。它是构造参数，所以同一个算子换一个取值就能对比三种手段，算子代码不用改。

取值叫 `prompt_only` 而不叫 `text`：它的意思是「不发格式参数，只靠 prompt」，模型输出**仍然是 JSON**。想让模型真的输出自由文本做对比，得换 prompt 和解析逻辑，那是 EverAlgo 这边的实验，不是 client 配置能做的（6.1）。

### 7.3 重试与观测

- 重试要有上限，否则「交出合法结果」会变成无限尝试；
- **同一个请求必然再撞上的，不重试**：`length` 和 `content_filter`。同一 prompt 重试**未必**同样截断，但既然不能抬高 `max_tokens`（3.1），同一个请求会撞上同一个上限、触发同一个过滤，重试就是用调用方的预算去赌。`abort` 和 `error` 不一样：那是服务端放弃了一次本可以完成的生成（抢占、上游抖动），像 5xx 一样重发是合理的。已有生产宿主的适配器也是这样划线：`length`、`content_filter` 直接抛错，`abort`、`error` 重试；
- 记录每次调用用了哪种格式手段、重试了几次。宿主要靠这些才看得出「这个 scene 配的模型平均要重试 3 次，该换了」，也才能算清两层重试相乘的预算（第 5 节）。

### 7.4 参考实现把以上建议全部做到

`OpenAICompatClient` 是 EverAlgo 官方提供、供其他实现方照着写的样板，所以第 3 节的硬性条款和本节的建议它全部做到：

| 建议 | `OpenAICompatClient` 怎么做 |
|---|---|
| 交给算子之前先校验（7.1） | 用算子给的同一个 `response_format` 模型校验每一次回复 |
| 有界重试（7.1、7.3） | 校验不过的回复最多重试 `invalid_reply_retries` 次（默认 1 次）；用尽后抛 `LLMError`，`__cause__` 是最后一次的 `ValidationError` |
| 每次尝试发同一个请求（3.1） | 请求只构造一次，所以每次尝试都原样带着调用方的 `max_tokens` |
| 同一个请求必然再撞上的，不重试（7.3） | 上游报告拒答、`length` 或 `content_filter` 时立即抛错；`abort`、`error` 在 `transport_retries` 次数内重发 |
| 结束原因缺失（7.1） | 只看校验结果 |
| 选择格式参数（7.2） | 由 `response_format_mode` 决定，经 `to_openai_response_format` 生成 |
| 记录手段与重试次数（7.3） | 每次校验失败都记一条 warning 日志，带上第几次尝试、模型和 `response_format_mode`；每次因偶发故障重发也记一条，带上原因、退避时长和第几次尝试 |

它直接用 httpx 发请求，不依赖任何厂商 SDK，所以自带客户端的宿主不会被 everalgo-core 连带装上 SDK。连接错误、超时、HTTP 408、409、429、5xx、返回 200 但没有 choices（OpenRouter 对部分上游拒绝就是这样答复的），或服务端中止的生成（`abort`、`error`），最多重发 `transport_retries` 次（默认 2 次），每次先按指数退避或上游的 `Retry-After` 等待；这组状态码和退避常量取自 openai SDK 2.35.1（`openai/_base_client.py`）。

一致性套件对它跑下来，必检和选检全部通过（第 9 节）。

---

## 8. 给实现方的工具函数

### 8.1 strict schema：所有构造函数的起点

每个构造函数（8.4）的 `json_schema` 档都从同一套改写出发，取 `model.model_json_schema()` 后做改造：每个已有 `type: "object"` 且**未声明** `additionalProperties` 的节点补 `additionalProperties: false`、`required` 填成全部 property 名，剥掉值为 `null` 的 `default`，展开带兄弟键的 `$ref`，把判别联合的 `oneOf` 改成 `anyOf` 并删掉 `discriminator`（8.2），并递归处理 `$defs` / `definitions` / `properties` / `items` / `anyOf` / `allOf`。

这些改写让它比默认配置的 pydantic 模型更严。只有模型按 2.2 配置时，它和本地校验才接受同一批输入。

这套改写是 `everalgo.llm.schema` 的私有函数，不对外导出：各家 API 包装 schema 的方式不同，每个构造函数又已经返回整个请求取值，实现方用不到裸 schema。还没有构造函数的 API，在补上构造函数（8.4）之前，实现方可以先取 `to_openai_response_format(model)["json_schema"]["schema"]`。

OpenAI 兼容的实现方这样组装：

```python
from everalgo.llm import to_openai_response_format

openai_response_format = to_openai_response_format(response_format, "json_schema")
# {"type": "json_schema",
#  "json_schema": {"name": response_format.__name__, "schema": <strict schema of response_format>, "strict": True}}
```

`strict: True` 的依据：OpenAI 文档原话 "Structured Outputs is a feature that ensures the model will always generate responses that adhere to your supplied JSON Schema"，并且 "We recommend always using Structured Outputs instead of JSON mode when possible"。

### 8.2 支持的子集

这套改写是机械的，本身不会把 strict 表达不了的结构修好。2026-09-22 用 `openai/gpt-5.4-mini` 实打验证（`strict: true`）：

| pydantic 构造 | 结果 |
|---|---|
| 扁平字段 | ✅ |
| `X \| None`（可空） | ✅ 渲染成 `anyOf` 含 `null` |
| 嵌套子模型（`$defs` + `$ref`） | ✅ |
| `list[子模型]` | ✅ |
| `Literal["a", "b"]` | ✅ |
| 带默认值的字段 | ✅（默认值被剥掉，字段仍进 `required`） |
| `Field(description=...)` 的子模型（产生带兄弟键的 `$ref`） | ✅ 展开后通过 |
| 自引用递归模型 | ✅ |
| 判别联合（`Field(discriminator=...)`） | ✅ 改写后：`oneOf` 改成 `anyOf`，删掉 `discriminator`；不在 2026-09-22 那批实测里，见下文 |
| **`dict[str, int]`（动态键）** | **❌ 400** |

`dict[str, int]` 的 pydantic schema 是 `{"type": "object", "additionalProperties": {"type": "integer"}}`。它**已经**声明了 `additionalProperties`，所以改写逻辑不会覆盖它——覆盖成 `false` 会把「允许任意键」改成「不允许任何键」，那是在改变模型含义。不覆盖则该节点不满足 strict，API 原话：

> Invalid schema for response_format 'WithDict': In context=(), 'required' is required to be supplied and to be an array including every key in properties. Extra required key 'counts' supplied.

所以动态键的 `dict` 不在支持范围内，改成 `list[子模型]`。

动态键的 `dict` 改不得，判别联合却可以改写而不改变模型含义。pydantic 把 `Annotated[A | B, Field(discriminator="kind")]` 渲染成 `{"oneOf": [...], "discriminator": {"propertyName": "kind", "mapping": {...}}}`，openai SDK 2.35.1 的 `to_strict_json_schema` 把这两个键原样透传。strict 模式不接受 `oneOf`：OpenAI Structured Outputs 文档的「Supported types」清单只列了 `anyOf`、没有 `oneOf`；openai-agents-python issue #474 贴出的 API 报错原话是 `In context=('properties', 'results', 'items'), 'oneOf' is not permitted.`。改名不改变 schema 接受哪些回复：每个臂都把判别字段固定为各自的 `const`，最多只有一个臂能匹配，所以 `anyOf` 接受的回复与 `oneOf` 相同；本地按模型校验（7.1）仍按判别字段选臂。`discriminator` 是 OpenAPI 的词汇，JSON Schema 校验从不读它（pydantic 源码自己也把它标为 OpenAPI 的东西），8.4 的四家 API 也都没有列出它，所以直接删掉，而不是挪进描述。

改写后的形态还没有发给真实 API 实调过。这一行的 ✅ 依据的是文档列出的 `anyOf` 支持和上面那条报错，不像其他行那样来自实调；带上它的版本发布前，应实调一次确认。

另外两条常见说法与实测不符：`pattern` **是被支持的**（同一批实测里含 `pattern` 的 schema 返回 200，限制主要涉及微调模型）；真正不被支持的组合关键字是 `allOf`——这套改写会把单元素 `allOf` 摊平，多元素 `allOf` 则会原样留下并在请求时被拒。

### 8.3 构建期检查

遇到不在子集内的结构，调用构造函数时就抛错，而不是产出一个到请求时才 400 的 schema。错误消息指出哪个字段、为什么不行、替代方案是什么：

| 情况 | 处理 |
|---|---|
| `additionalProperties: false` | 已经满足 strict，**保留，不报错** |
| `additionalProperties: true`（开放对象） | 抛错 |
| `additionalProperties` 是一个 schema（动态键 `dict[str, X]`） | 抛错，提示改成 `list[子模型]` |
| 多元素 `allOf` | 抛错（单元素会被摊平） |
| 同一节点上 `oneOf` 与 `anyOf` 并存（pydantic 不会产出） | 抛错：合并两者会改变该节点接受什么 |
| 顶层类型不是 `object`（如 `RootModel[int]`） | 抛错，违反 2.2 |

这些检查同时服务两边：算子的单测靠它确认模型落在子集内（4.3），实现方调用时也不会拿到一个注定 400 的 schema。

### 8.4 请求参数构造函数：`to_<api>_response_format`，每个 API 一个

每个构造函数都接收 `response_format` 模型和一个 `mode`，返回某一家 API 请求里需要的内容：

| 函数 | 返回 | `mode` |
|---|---|---|
| `to_openai_response_format` | `response_format` 的值（OpenAI Chat Completions 以及所有 OpenAI 兼容端点） | `json_schema` / `json_object` / `prompt_only` |
| `to_anthropic_response_format` | `output_config.format` 的值（Anthropic Messages） | `json_schema` / `prompt_only`：Anthropic 没有 JSON 模式这一级 |
| `to_gemini_response_format` | `generationConfig` 里的 `responseMimeType` 和 `responseJsonSchema` 两个条目（Gemini `generateContent`） | `json_schema` / `json_object` / `prompt_only` |
| `to_ollama_response_format` | `format` 的值（Ollama `/api/chat`、`/api/generate`）：schema 本体、`"json"` 或 `None` | `json_schema` / `json_object` / `prompt_only` |

`json_schema` 一律从 strict schema（8.1）出发。某家只接受 JSON Schema 的一部分时，构造函数只保留这家列出支持的关键字，**其余关键字一律挪进描述**（`描述\n\n{key: value, ...}`），模型仍然看得到：Anthropic 挪走数值范围、字符串长度、`pattern`、大于 1 的 `minItems` 和不在列表里的字符串格式；Gemini 挪走 `pattern`、字符串长度、`multipleOf` 和 `const`。OpenAI 和 Ollama 原样保留。`prompt_only` 表示不发这个字段。

| 选择 | 为什么 |
|---|---|
| 不支持的关键字挪进描述，而不是拒绝 | Anthropic 官方 SDK 就是这样屏蔽差异的（`transform_schema`，以及它文档里「How SDK transformation works」一节：剔除不支持的约束、写进描述、按原始 schema 校验回复），litellm 也照搬了这套做法。这正是契约自己的规则（3.2：请求可以降级，校验不能降级）：实现方仍按完整模型做本地校验（7.1），被挪走的约束照样在本地执行。至于上游接不接受剩下的部分，构造函数不做判断 |
| 维护「这家支持什么」的清单，而不是「不支持什么」 | 白名单过时了也是往安全方向失效：某家新支持的关键字仍被挪进描述，只是少一个上游约束；黑名单一旦漏掉某个新增的不支持关键字，就会照样发出去，可能收到 400 |
| 每个 API 一个函数，命名统一 | 一个实现方只对接一种 API，不会在运行时切换，`target` 参数对它没有用处；按 API 分开的函数能被 IDE 补全、返回类型和文档都只讲这一家，新增一家也不用改已有的函数。litellm 也是按提供方各自组织转换逻辑 |
| `mode` 只列出这家有的级别 | 向 Anthropic 要 JSON 模式在写代码时就是类型错误，而不是运行时的 `ValueError` |
| 返回请求里要放的内容，而不是要合并的字段 | 各家把这项设置放在不同的嵌套路径下，用 `dict.update` 浅合并嵌套字段会覆盖掉调用方放在里面的其他内容。Anthropic 就是实例：`output_config` 里还放着 `effort` |
| Gemini 返回两个条目 | Gemini 把这项设置拆成了 `generationConfig` 下两个并列的字段。两个都是平铺的键，不和其他条目重名，所以 `generation_config.update(...)` 覆盖不到别的内容。键名用 REST 的写法，`google-genai` SDK 也接受（它的模型用了驼峰别名并开启 `populate_by_name`） |

请求格式的依据（2026-09-23 核对）：

| API | 出处 |
|---|---|
| Anthropic | platform.claude.com/docs/en/build-with-claude/structured-outputs；anthropic SDK 0.75.0 `types/beta/beta_output_config_param.py`、`beta_json_output_format_param.py`、`lib/_parse/_transform.py` |
| Gemini | google-genai SDK 1.53.0 `types.py`（`GenerateContentConfig.response_mime_type`、`response_json_schema` 及其支持关键字清单） |
| Ollama | docs.ollama.com/capabilities/structured-outputs；github.com/ollama/ollama `docs/api.md` |

其他 API 的函数，等真有实现方需要时再加，加之前按那家的官方文档核对请求格式。

---

## 9. 一致性测试套件

契约只有文档是不够的。EverAlgo 作为协议指定方，交付物是三样：**契约、可执行的一致性检查、测试替身**。

这套检查面向**实现方**，只验第 3 节的硬性条款。4.3 的算子侧义务由 EverAlgo 自己的单测检查；第 7 节的建议另列为**选检**，实现方自愿跑，不通过不算违约。

### 9.1 测试工厂

3.1、3.2 要看**实际发出了什么**，3.4、3.5 要**注入上游的拒答与截断信号**，只拿一个 client 实例都验不了，所以实现方要提供一个工厂。契约只规定它的能力，不规定它怎么实现：

```python
class LLMClientTestHarness(Protocol):
    """实现方提供，用于跑一致性检查。"""

    def build(self, **defaults: Any) -> LLMClient:
        """按给定默认值造一个 client（用来验「显式值压过默认值」）。"""

    def queue_upstream(self, *responses: UpstreamStub) -> None:
        """排入上游要返回的东西：合格回复、截断、拒答、不合模型的回复。"""

    def sent_requests(self) -> list[EffectiveRequest]:
        """已发出的请求，归一化成语义视图。"""
```

`EffectiveRequest` 是**语义视图**而不是 wire 字节，只含 `temperature`、`max_tokens`、`messages` 三项。契约不规定 wire 形状（6.6），要求套件去解析各家 wire 等于把契约又绑回某一家。格式参数不在视图里：发什么是实现方的手段，契约不管（3.2）。

**视图必须从实际捕获的出站请求归一化得来，不得复制调用入参。** 照抄入参会让检查变成同义反复。实现方应当把捕获点放在**离线最近的地方**（HTTP transport 层、httpx 的 request hook 之类），而不是 `achat()` 方法入口。**这一点套件无法强制**：实现方偷懒直接返回入参，套件看不出来。所以它是**实现方的义务**，套件是辅助，不是证明。

### 9.2 必检（第 3 节硬性条款）

| 检查 | 对应条款 |
|---|---|
| 显式 `temperature` / `max_tokens` 到达上游，且不被实现的默认值覆盖 | 3.1 |
| 实现方内部重试时，每一次请求的 `max_tokens` 都没被抬高 | 3.1 |
| 传 `None` 时用实现的默认或省略，不得是字面 `None` | 3.1 |
| 非文本 `ContentPart` 确实出现在出站 `messages` 里；接不了时抛错（不限类型），不得静默丢弃 | 3.2 |
| 调用被取消时 `asyncio.CancelledError` 原样传播，不被吞掉 | 3.3 |
| 上游标记拒答（`refusal` 非空、`stop_reason: "refusal"` 这类）时抛错，即使拒答内容是合法 JSON；既不返回拒答文本也不返回空内容 | 3.4 |
| 上游报告截断或异常中止（`finish_reason` 为 `length` / `content_filter` 等）时重试或抛错，即使正文是闭合的合法对象；重试时 `max_tokens` 不变 | 3.5 |

### 9.3 选检（第 7 节建议）

| 检查 | 对应建议 |
|---|---|
| 上游返回不合模型的回复（说明文字、围栏、缺必填字段、对象未闭合的截断）时，不交给算子 | 7.1 |
| 结束原因缺失、内容合格时正常返回 | 7.1 |
| 重试次数有上限 | 7.3 |
