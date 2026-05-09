# ADR 012: LLM 抽象层架构 — 各家原生 SDK + Protocol + 双层路由 + 错误归一

## 状态

✅ **Accepted** — 2026-04-28（v0.30 制定，整合 6 轮调研）

## 背景

EverCore 算法库需调 LLM。多个交叉决策需统一沉淀（之前散落 v0.29 / v0.30 changelog）：

1. **底层 SDK 选择**：LiteLLM 统一底层 / 各家原生 SDK / aiohttp 自 wrap
2. **抽象基类**：ABC 还是 Protocol（与 [ADR 011](011-protocol-vs-abc.md) 协同）
3. **路由分层**：Scene 路由 vs Provider 路由职责划分
4. **错误层级**：统一 `LLMError` 抽象 vs 透传 SDK 错误
5. **重试 / fallback 责任**：算法库内做 vs 外推服务层

相关硬约束：
- **H3** 算法同学迭代速度
- **H4** 无状态接口
- **H6** v0.x 演化阶段

## 候选方案

### 维度 1：底层 SDK 路线

| 方案 | 描述 | 代表 |
|------|------|------|
| **P1. aiohttp 自 wrap** | 自实现 OpenAI compat HTTP 客户端 | memsys_opensource 现状 |
| **P2. 各家原生 SDK** | `openai.AsyncOpenAI` / `anthropic.AsyncAnthropic` / `boto3` 各自适配 | LlamaIndex / LangChain / AutoGen / Letta / Ragas / Instructor / mem0（B 派 7 项目）|
| **P3. LiteLLM 强绑** | LiteLLM 作为统一底层 | DSPy / Cognee（A 派 2 项目）|

### 维度 2：抽象基类

| 方案 | 描述 |
|------|------|
| ABC + abstractmethod | 4 家 B 派全用：LlamaIndex `BaseLLM` / LangChain `BaseChatModel` / AutoGen `ChatCompletionClient` / Letta `LLMClientBase` |
| **Protocol structural** | EverCore 选择，[ADR 011](011-protocol-vs-abc.md) 已论证 |

### 维度 3：路由分层

| 方案 | 描述 |
|------|------|
| 单层 scene → client | scene 直接映射 LLMClient（早期 v0.29 设计）|
| **双层 Scene → Config → Provider → Client** | Scene 路由（业务层）+ Provider 路由（实现层）分离 |
| 仅 Provider 路由 | EverCore 不做 scene，仅做 Provider 适配 |

### 维度 4：错误层级

| 方案 | 描述 | 代表 |
|------|------|------|
| A SDK 透传（无统一 LLMError）| 调用方适配多 SDK 错误 schema | LlamaIndex / AutoGen / smolagents（D 派）|
| B 完整 LLMError 抽象层 | 算法库归一为统一基类 + N 子类 | Letta（13 子类，B 派唯一）|
| C 平铺 1-2 个 | 仅特殊错误（如 ContextLength）| mem0 / CrewAI / DSPy / LangChain partner |
| **B 精简 + 混合多重继承** | LLMError + 7 子类 + 让原生 SDK catch 仍有效 | EverCore 选择（Letta + LangChain partner 启发） |

### 维度 5：重试 / fallback 责任

| 方案 | 描述 | 代表 |
|------|------|------|
| 库内重试 + fallback | 在 SDK 之上加薄重试 | DSPy 3 / LlamaIndex 3-10 / instructor 1（A 派）|
| 库外重试 | 仅依赖 SDK 默认重试，跨 Provider fallback 由调用方 | LangChain Core（核心不重试 + ChatOpenAI 透传 max_retries）|
| **B 同款** | EverCore 选择 | LangChain Core 同款 |

## 客观优劣分析

### P1. aiohttp 自 wrap

**优势**：无外部依赖；与 memsys_opensource 现状代码兼容；HTTP 层完全控制。

**劣势**：
- 业界 13 个 AI 算法库**零先例**采用此路线
- function calling / structured output / streaming / 新模型接入需自行跟进
- 自实现 retry / API key 轮换 / 错误归一**重新发明轮子**
- multi-provider 扩展（anthropic / bedrock / vllm）每家自实现工作量大

### P2. 各家原生 SDK

**优势**：
- **业界 7/13 项目主流**（B 派事实标准）
- SDK 内置默认 2 次重试（OpenAI/Anthropic `DEFAULT_MAX_RETRIES = 2`，hardcoded `_constants.py`，覆盖 connection / 408 / 409 / 429 / 5xx）
- 官方维护活跃（function calling / structured output / streaming 跟进官方进度）
- 错误层级完备（OpenAI 20 子类 / Anthropic 17 子类）
- OpenAI compat 协议入口已是事实标准（OpenRouter / vLLM / DeepSeek / Together 全支持）

**劣势**：
- 每加一家 provider 需加一个适配文件
- 多 SDK 错误 schema 不一致，需 EverCore 内部归一
- 多 SDK 安装依赖

### P3. LiteLLM 强绑

**优势**：100+ provider 一次接入；内置 retry / fallback；Router 产品化（load balance / cooldown 60s）。

**劣势**：
- **业界少数派**（仅 DSPy / Cognee 2 项目，孤例）
- **2026-03 供应链投毒事件**：`litellm 1.82.7/1.82.8` PyPI 包夹带 `litellm_init.pth` 凭据窃取脚本（在线 ~40 分钟才被隔离）
- **2026-04 续发安全 hardening**；CVE-2026-35029 / GHSA-69x8-hrgq-fjj8 proxy 高危漏洞；issues 800+ 长期居高
- **与 enterprise `EnterprisePipelineRouter` 职责重叠**——按 opensource vs enterprise 分工，多 provider 路由属 enterprise 范畴，强绑 LiteLLM Router 造成两层抽象冲突

## 对 EverCore 适配度评估

### P 派选择

| 优势 / 劣势 | EverCore 适配度 |
|------------|-----------------|
| P2 业界 7/13 主流 | ✅ **强需要**（避免 cargo cult + 与生态对齐）|
| P2 SDK 内置 2 次重试 | ✅ **强需要**（H4 无状态 + 不重新发明）|
| P2 错误层级完备 | ✅ 受益（EverCore 归一基础）|
| P2 多 SDK 错误归一成本 | ⚠️ 可 mitigate（adapter 出口集中映射）|
| P3 LiteLLM 100+ provider | ⚠️ 用不上（EverCore 主用 OpenAI / Anthropic / Bedrock 即可，OpenAI compat 已覆盖大部分）|
| P3 2026-03 供应链事件 | ❌ **强烈介意**（生产风险）|
| P3 与 enterprise 职责重叠 | ❌ 强烈介意（两层抽象冲突）|
| P1 aiohttp 自 wrap 业界零先例 | ❌ **强烈介意**（cargo cult 反向 + 维护成本）|
| P1 自实现 retry / 错误归一 | ❌ 强烈介意（重新发明轮子）|

→ **选 P2**

### 抽象基类（与 ADR 011 协同）

B 派 4/4 ABC 真实驱动：concrete mixin（LangChain ~40+ / LlamaIndex ~32 / Letta 11）+ 多继承胶水（`BaseComponent` / `RunnableSerializable` / `ComponentBase`）+ 实例化早失败 + 生态级 framework 需求。EverCore **一条都不命中**——是生态级 framework 才有的需求，library 级别工具无此压力。维持 [ADR 011](011-protocol-vs-abc.md) Protocol。

### 路由分层

业界主流 LLM 库 4/4 横跨 3 阵营都**不做 scene 路由**（DSPy `dspy.settings.lm` + LlamaIndex `Settings.llm` 算法库阵营 / LangChain `prompt | model | parser` chain 框架阵营 / AutoGen `client → agent` agent 框架阵营）——scene 是业务编排概念（"哪个步骤用哪个模型"），与算法本身无关。3 阵营都不做 → 反向印证 scene 路由不属任何阵营 LLM 库的职责。**Scene 路由剥离出 EverCore，归 EverOS**。

Provider 路由（`config.provider` → SDK 适配实现）是实现层职责，紧耦合算法库 SDK 适配代码，**保留 EverCore 内部**。Letta `LLMClient.create` `match-case` 同款。

→ **双层分离**：
- Scene 路由（业务层）→ EverOS `SceneRouter`
- Provider 路由（实现层）→ EverCore `evercore/llm/routing.py: build_client(config)`

### 错误层级

D 派（LlamaIndex / AutoGen 不定义）让多 SDK 错误穿透——调用方需 `except (openai.RateLimitError, anthropic.RateLimitError, botocore.ClientError)` 适配 3 套，**不优雅**。

B 派（仅 Letta 13 子类）按 HTTP 状态完整划分，但子类过多冗余。

C 派（mem0 / CrewAI 平铺 1-2 个）覆盖不全。

→ **B 精简 7 子类 + LangChain partner 混合多重继承**：
- 7 子类按 EverCore 实际处置策略（重试 / 不重试 / 裁剪输入重试）划分
- adapter 出口做映射（`_OpenAIRateLimit(LLMRateLimitError, openai.RateLimitError)` 多重继承），让原生 SDK catch + 语义 catch 双有效

### 重试 / fallback

业界 OpenAI / Anthropic SDK 默认 `DEFAULT_MAX_RETRIES = 2` 已成熟兜底；DSPy 3 / LlamaIndex 3-10 / instructor 1 在 SDK 之上叠加是早期 SDK 不可靠的历史包袱，现代叠加只放大延迟。

→ **EverCore opensource 算法层不加 retry**（LangChain Core 同款）。高可用编排（multi-key 轮转 / 跨 Provider fallback / 长时间退避 / 配额降级 / 租户级路由）**不属于开源版需求**——这些是部署侧 reliability 关注点，不影响算法本身 SOTA 复现，开源版不做。`LLMClient` Protocol 天然支持装饰器扩展，部署方有需要时可自行实现透明叠加。

## 决策

5 个决策综合：

### D1：底层 SDK 选 P2（各家原生 SDK）

- 主用 `openai.AsyncOpenAI` / `anthropic.AsyncAnthropic` / `boto3` AsyncBedrockRuntime
- `evercore/llm/providers/{openai_compat, anthropic, bedrock}.py` 各自薄适配
- `openai_compat.py` 单文件覆盖 OpenAI / OpenRouter / vLLM / DeepSeek / Azure（OpenAI compat 协议事实入口）
- **不强绑 LiteLLM**（与 enterprise `EnterprisePipelineRouter` 解耦 + 安全风险隔离）

### D2：抽象基类用 Protocol（不用 ABC）

- `LLMClient` Protocol `@runtime_checkable`
- 与 [ADR 011](011-protocol-vs-abc.md) 一致

### D3：双层路由分离

- **Scene 路由（业务层）剥离出 EverCore，归 EverOS**：业界主流 LLM 库 4/4 横跨 3 阵营（算法库 / chain 框架 / agent 框架）都不做 scene 路由
- **Provider 路由（实现层）保留 EverCore**：`build_client(config)` `match-case` 分发，Letta 同款

### D4：错误层级 LLMError + 7 子类 + 混合多重继承

```python
# evercore/llm/errors.py
class LLMError(Exception): ...
class LLMRateLimitError(LLMError): ...        # 429 → 退避重试 / fallback
class LLMTimeoutError(LLMError): ...          # 网络超时 → 退避重试
class LLMServerError(LLMError): ...           # 5xx → 退避重试
class LLMConnectionError(LLMError): ...       # network → 退避重试
class LLMAuthError(LLMError): ...             # 401 / 403 → 不可重试
class LLMBadRequestError(LLMError): ...       # 400 → 不可重试
class LLMContextLengthError(LLMBadRequestError): ...  # → 裁剪输入重试

# providers/openai_compat.py adapter 出口混合多重继承
class _OpenAIRateLimit(LLMRateLimitError, openai.RateLimitError): ...
class _OpenAITimeout(LLMTimeoutError, openai.APITimeoutError): ...

# 调用方既可 except LLMRateLimitError 也可 except openai.RateLimitError
```

### D5：算法层不加 retry，依赖 SDK 默认；高可用编排不属于开源版需求

- `OpenAI / Anthropic SDK DEFAULT_MAX_RETRIES = 2` 自动生效
- 高可用编排（multi-key 轮转 / 跨 Provider fallback / 长时间退避 / 配额降级 / 租户级路由）**开源版不做**——部署侧 reliability 关注点，不影响算法本身 SOTA 复现，不属于开源版需求
- opensource 调用失败直接抛 `LLMError`，由 caller 决定如何处理（dev / 测试场景常见做法是直接传播）
- `LLMClient` Protocol 天然支持装饰器扩展（[ADR 011](011-protocol-vs-abc.md)），部署方有需要时可自行实现透明叠加，上层算子调用形态不变
- LangChain Core 同款分层（核心不重试 + 依赖 provider SDK + 业务编排归调用方）

## 实施细节

### 子包结构

```
evercore/llm/
├── __init__.py        # facade: chat / stream / use / configure + re-export types/errors
├── types.py           # ChatMessage / ChatResponse / Usage / ToolCall / Chunk
├── client.py          # LLMClient Protocol (@runtime_checkable) + LLMConfig dataclass
├── errors.py          # LLMError + 7 子类（混合多重继承让原生 SDK catch 仍有效）
├── routing.py         # Provider 路由：build_client(config) → LLMClient
└── providers/
    ├── openai_compat.py   # openai.AsyncOpenAI（含 OpenRouter / vLLM / DeepSeek / Azure 兼容）
    ├── anthropic.py       # anthropic.AsyncAnthropic
    └── bedrock.py         # boto3 AsyncBedrockRuntime
```

### 核心接口

```python
# types.py
class MessageRole(str, Enum): SYSTEM=...; USER=...; ASSISTANT=...; TOOL=...
class ChatMessage(BaseModel):
    role: MessageRole
    content: str | list[ContentBlock]   # 多模态：TextBlock / ImageBlock
    tool_calls: list[ToolCall] | None
class Usage(BaseModel): prompt_tokens: int; completion_tokens: int; cache_read_tokens: int = 0
class ChatResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall]
    usage: Usage
    finish_reason: Literal["stop","length","tool_calls","content_filter"]
    raw: Any                            # 透传原始 SDK 响应给需要的 caller


# client.py
@dataclass(frozen=True)
class LLMConfig:
    provider: Literal["openai_compat", "anthropic", "bedrock"]
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    async def chat(self, messages: Sequence[ChatMessage], *, model: str,
                   tools: Sequence[ToolSchema] = (),
                   response_format: type[BaseModel] | None = None, **kw) -> ChatResponse: ...
    async def stream(self, messages: Sequence[ChatMessage], *, model: str,
                     **kw) -> AsyncIterator[ChatChunk]: ...


# routing.py — Provider 路由层
def build_client(config: LLMConfig) -> LLMClient:
    match config.provider:
        case "openai_compat": from .providers.openai_compat import build; return build(config)
        case "anthropic":     from .providers.anthropic     import build; return build(config)
        case "bedrock":       from .providers.bedrock       import build; return build(config)
        case _: raise ValueError(f"Unknown provider: {config.provider}")
```

## 行业实证印证

### LLM 调用底层（13 项目矩阵）

| 派 | 项目 | 数量 |
|----|------|------|
| **B 派各家原生 SDK** | LlamaIndex / LangChain partner / AutoGen / Letta / Ragas / Instructor / mem0 | **7（事实主流）** |
| A 派 LiteLLM 强绑 | DSPy / Cognee | 2 |
| D 派多适配器 | smolagents / CrewAI（原生优先 + LiteLLM 兜底） | 2 |
| C 派 aiohttp 自 wrap | memsys_opensource 现状 | **0 个明星项目（孤本）** |

### LiteLLM 工业级真实定位

| 指标 | 数据 |
|------|------|
| GitHub stars | 45k |
| PyPI 月下载 | 261.5M（vs OpenAI SDK 258.5M / langchain 233.1M）|
| 生产用户 | Stripe / Netflix / Google ADK / OpenHands |
| 风险 | 2026-03 供应链投毒（PyPI 包凭据窃取）/ 2026-04 安全 hardening / proxy CVE-2026-35029 / OOM issues |

→ 不是 toy，但有真实安全风险 + 与 enterprise EnterprisePipelineRouter 重叠

### Scene 路由不在 LLM 库（4/4 横跨 3 阵营实证）

| 项目 | 阵营 | 注入模式 |
|------|------|----------|
| **DSPy** | 算法库 | `dspy.settings.lm = lm` 全局 + `dspy.context(lm=...)` scoped 切换 |
| **LlamaIndex** | 算法库 | `Settings.llm = ...` 全局 + per-query override |
| **LangChain** | chain 框架 | chain 组合时传 model（`prompt \| model \| parser`）|
| **AutoGen** | agent 框架 | 用户构造 client 传给 agent |

**3 阵营无差别**——scene 路由（"哪个步骤用哪个模型"）不属任何阵营 LLM 库的职责，归业务编排层。

### LLMError 错误层级（5 派分布）

| 派 | 项目 | 数量 |
|----|------|------|
| A SDK 自带完整层级 | OpenAI 20 子类 / Anthropic 17 子类 | 3 |
| **B 算法库统一 LLMError 完整层级** | **Letta 13 子类（唯一）** | **1（B 派唯一）** |
| C 平铺 1-2 个 | mem0 / CrewAI / DSPy / LangChain partner | 4 |
| D 完全不定义 | LlamaIndex / AutoGen / smolagents | 3 |
| E 通用层级非 LLM 维度 | LangChain core / Instructor | 2 |

### B 派 ABC 真实驱动（4 家共性）

| 驱动 | 项目实证 |
|------|---------|
| concrete mixin（telemetry / cache / callback）| LangChain ~40+ / LlamaIndex ~32 / Letta 11 / AutoGen ~4 |
| 多继承胶水 | LlamaIndex `BaseComponent + DispatcherSpanMixin` / LangChain `RunnableSerializable` / AutoGen `ComponentBase` |
| 实例化早失败 | partner 生态运行时校验 |
| 生态级 framework | chain 编排 / yaml 配置 / instrumentation 注入 |

→ EverCore **一条不命中**（算法库无状态 + 不挂体系 + 不是 partner 生态 + library 不是 framework），Protocol 维持稳健。

## 后续演化触发条件

1. **某 provider 增加（如 Google Gemini / Mistral / 自部署 vLLM 特殊化）**：在 `providers/` 加新文件 + `routing.py: match-case` 加 case
2. **OpenAI / Anthropic SDK 弃用 / 大版本不兼容**：评估迁移成本，可能切回 LiteLLM
3. **LiteLLM 安全状况显著改善 + EverOS 不再用 EnterprisePipelineRouter**：可重新评估 P3 LiteLLM 路线
4. **EverCore 算法层需要内置 telemetry / cache / callback**：用 decorator / contextmanager 装饰具体实现，**不改 Protocol 为 ABC**
5. **新 provider 错误类型不在 7 子类覆盖范围**（如配额超出 `LLMQuotaExceededError`）：扩展子类
6. **Scene 路由是否真不在 EverCore**：若 EverOS 团队反馈 contextmanager wrap 5+ 场景代码冗长难维护，可重新评估"算子签名加可选 `llm` 参数"路径

## memsys_opensource 现状代码迁移清单

| 现状 | 迁移到 EverCore |
|------|---------------|
| `OpenAIProvider`（aiohttp 自实现 5 次重试）| `evercore.llm.providers.openai_compat`（`openai.AsyncOpenAI` 薄 wrap，**单 api_key**）|
| `ApiKeyRotator` multi-key 轮换 | **开源版不做**（不属于开源版需求，部署方有需要时可自行实现 LLMClient 装饰器叠加）|
| `_MAX_RETRIES = 5` | 删除（依赖 SDK 默认 2 次）|
| `FallbackLLMProvider`（库内 decorator）| **开源版不做**（跨 Provider fallback 不属于开源版需求，部署方有需要时可自行实现 LLMClient 装饰器叠加）|
| `LLMScene` enum + `_get_provider_for_scene` | 移到 EverOS 端 `SceneRouter` |
| `LLMProvider` Protocol（旧版）| `evercore.llm.client.LLMClient` Protocol（method 重命名 `generate` → `chat`）|
| OpenAI compat 协议直发 HTTP | `openai.AsyncOpenAI(base_url=..., api_key=...)` 享受 SDK 默认 2 次重试 |
| `model` whitelist env 校验 | 保留为 EverCore 业务定制（`providers/openai_compat.py` 内）|

## 相关 ADR

- [ADR 003 PEP 420 namespace](003-namespace-package-pep420.md) — `evercore.llm` 子包遵循 namespace 规范
- [ADR 004 providers 内嵌于 llm/](004-providers-nested-in-llm.md) — providers in-tree 不用 partner pip 包
- [ADR 008 re-export facade](008-re-export-vs-client-facade.md) — `evercore.configure(llm=...)` 全局注入而非 Client 类
- [ADR 010 sync/async 双接口](010-sync-async-dual-interface.md) — `chat` / `stream` async-first
- [ADR 011 Protocol vs ABC](011-protocol-vs-abc.md) — `LLMClient` 用 Protocol 不用 ABC
