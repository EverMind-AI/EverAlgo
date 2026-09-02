# 从 Episode 文本提取 Profile

状态：EverAlgo 已实现；可开始 EverOS 集成。

本文定义 EverOS 与 EverAlgo 之间的接口契约：从按时间排序的 generic 或 reflected Episode 叙事文本中，为一个指定用户提取 `Profile`。接口刻意只接收文本：EverOS 保留存储标识和编排元数据，EverAlgo 只接收叙事正文以及准确提取个人画像所必需的目标用户信息。

## 接口定义

```python
from collections.abc import Sequence

from asgiref.sync import async_to_sync

from everalgo.types import Profile
from everalgo.user_memory import OutputLanguage


class ProfileExtractor:
    async def aextract_from_episode_texts(
        self,
        episode_texts: Sequence[str],
        *,
        owner_id: str,
        timestamp: int,
        owner_name: str | None = None,
        old_profile: Profile | None = None,
        categories: Sequence[str] | None = None,
        prompt: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Profile:
        ...

    extract_from_episode_texts = async_to_sync(aextract_from_episode_texts)
```

`aextract_from_episode_texts` 是原生异步接口。`extract_from_episode_texts` 是供非事件循环调用方使用的同步桥接，不能在正在运行的事件循环中调用。

现有 [`ProfileExtractor.aextract`](../../packages/everalgo-user-memory/src/everalgo/user_memory/profile.py) 接口保持不变。两个接口分别服务不同输入：

| 接口 | 输入 | 目标用户定位方式 |
|---|---|---|
| `aextract` | 按时间排序的 `Sequence[MemCell]` | 根据结构化用户消息验证 `sender_id` |
| `aextract_from_episode_texts` | 按时间排序的 Episode 叙事 `Sequence[str]` | 从 `owner_name` 或 `owner_id` 确定一个目标引用，再对每条叙事执行校验 |

## 参数

### `episode_texts`

非空、按时间升序排列的 Episode 叙事正文列表，每一项都必须是非空白字符串。每项可以是 generic Episode，也可以是 reflected Episode。

每项文本都必须包含同一个已确定的目标引用。`owner_name` 非空时，EverAlgo 使用姓名；没有有效姓名时回退到 `owner_id`。

列表必须从最早到最新排序。由于本接口不接收 Episode ID，EverOS 必须在转换为字符串列表前完成 Episode 去重。

### `owner_id`

目标用户的权威标识。EverAlgo 必须将该值写入 `Profile.owner_id`，不得从模型输出中推断或替换此标识。

`owner_id` 不得为空。UPDATE 模式下，必须满足 `old_profile.owner_id == owner_id`。

### `owner_name`

用于在 Episode 叙事中定位目标用户的可选权威显示名称。EverOS 在有可信参与人姓名时传入；EverAlgo 不得从叙事文本中推断姓名。传入 `None` 或空白字符串时，EverAlgo 使用 `owner_id` 作为目标引用。

`owner_name` 只用于定位目标用户，返回 Profile 的归属始终由 `owner_id` 决定。

### `timestamp`

写入返回结果 `Profile.timestamp` 的 Unix 毫秒时间戳。EverOS 应传入去重后本批 Episode 的最大时间戳。

由于 `Sequence[str]` 不携带时间元数据，因此该参数必须显式提供。

### `old_profile`

- 传入 `None` 表示 INIT 模式，创建新的 Profile。
- 传入 `Profile` 表示 UPDATE 模式，在已有 Profile 上执行新增、更新和删除。

UPDATE 后继续沿用现有的透明压缩行为。

### `categories`

`explicit_info.category` 本次可用的完整分类快照。调用方必须在调用前组装完整快照；EverAlgo 不区分分类的来源或生命周期状态。

EverAlgo 会去除每个分类字符串两端的空白、忽略空白值，并按首次出现顺序去除完全相同的重复值；规范化后的 JSON 列表会不变地注入 INIT、UPDATE、COMPACT 和 REGROUP。`None` 与空列表都会渲染为 `[]`。直接传入字符串而不是字符串序列，或序列中包含非字符串值时，会在调用 LLM 前抛出 `TypeError`。

每个阶段处理显式事实时，都必须根据事实本身的语义，从该列表中选择最准确的匹配分类。该列表不是白名单：没有准确匹配项时，可以创建必要、简洁且语义准确的分类。分类复用和减少分类数量不得覆盖分类准确性。该列表不约束 `implicit_traits.trait`。

### `prompt`

可选的 prompt 覆盖参数。传入 `None` 时使用内置的 Episode 文本 Profile prompt。即使使用自定义 prompt，输入校验和输出归属规则仍然生效。

### `output_language`

期望的输出语言，可以传入 `OutputLanguage` 或等价的大小写不敏感字符串。传入 `None` 时，INIT 根据 Episode 叙事语言生成，UPDATE 保持已有 Profile 的语言。

## 目标用户校验

所有校验必须在第一次 LLM 调用前完成。EverAlgo 必须逐条校验 Episode 文本，只校验拼接后的整批文本是不够的。

EverAlgo 在校验文本前只确定一次目标引用：

```python
target_user = owner_name.strip() if owner_name and owner_name.strip() else owner_id.strip()
```

对每个 `episode_texts[index]` 执行：

1. 文本为空白时拒绝该输入。
2. 使用一致的规则规范化文本和 `target_user`。
3. 文本包含 `target_user` 的字面目标引用时通过。
4. 不包含目标引用时，拒绝整个调用并抛出 `ValueError`。

异常必须包含未通过校验的列表下标和目标用户参数，但不得在异常或日志中输出 Episode 正文。

错误示例：

```text
episode_texts[2] does not reference target user 'Alice'
```

该校验是确定性的安全保护，不是身份认证。它能够发现缺少目标用户或明显传错用户的情况，但不能区分两个同名用户。EverOS 仍然负责提供权威的 owner 信息和已经按 owner 划分的 Episode 批次。

## 提取行为

该接口只为传入的 `owner_id` 执行一次个人 Profile 提取。上游可以把同一个 generic Episode fanout 给多个 owner，但必须为每个 owner 分别调用本接口，因为每个人的画像解释不同，不能共享一次 Profile 结果。

返回结果必须满足：

- `Profile.owner_id == owner_id`。
- `Profile.timestamp == timestamp`。
- 只提取目标用户的事实和特征，不混入其他参与人画像。
- 在 INIT、UPDATE、COMPACT 和 REGROUP 中，事实正确性与 `explicit_info.category` 准确性并列为绝对最高优先级。
- 即使保留内容有利于召回率，也必须排除缺乏来源支持、归属错误、短期状态、会过期的计划、独立问题，以及对所有任职者普遍适用的团队或组织流程。
- INIT、UPDATE、合并和压缩语义与现有 `aextract` 一致。
- 校验或提取失败时不返回部分 Profile。

## Description 句式

每条 `explicit_info[].description` 和 `implicit_traits[].description` 必须使用简洁的无主语陈述句，直接描述 owner，不出现姓名或替代主语，也不得写成命令式祈使句。

正确示例：

- `主要使用 Python。`
- `偏好简洁、直接的回答。`

错误示例：

- `Alice 主要使用 Python。`——出现了姓名主语。
- `用户偏好简洁的回答。`——使用了通用替代主语。
- `使用 Python。`——把画像陈述改成了行动指令。

## Evidence 契约

Episode 文本路径无法访问原始对话，因此它的 evidence 规则与 `MemCell` 路径不同：

- `explicit_info[].evidence` 必须是可在输入 Episode 文本中核对的叙事摘录或忠实转述。
- evidence 不需要 Episode 编号或人为生成的标识。
- 除非输入 Episode 本身包含用户原话，否则 evidence 不得把叙事改写成用户直接引语。
- 模型不得伪造用户措辞、日期或说话人归属。
- `implicit_traits[].basis` 必须指出能在输入 Episode 中找到的信号，不得伪造用户引语。
- `evidence` 与 `basis` 都是单个 JSON 字符串，不是数组；允许忠实的叙事转述，但必须保留来源含义与归属。

EverOS 必须把这些证据视为 Episode 派生证据，而不是原始对话证据。

## 异常

出现以下情况时，接口必须在调用 LLM 前抛出 `ValueError`：

- `episode_texts` 为空；
- 任一 Episode 文本为空白；
- `owner_id` 为空白；
- 任一 Episode 文本不包含已确定的目标引用；
- `old_profile.owner_id` 与 `owner_id` 不一致；
- `output_language` 不受支持。

`categories` 不是字符串序列或包含非字符串值时，接口会在调用 LLM 前抛出 `TypeError`。

接口继续保留现有 Profile 提取异常：

- LLM 返回结果违反 Profile 结构时抛出 `ValueError`；
- LLM 返回内容不是可解析的 JSON 时抛出 `json.JSONDecodeError`；
- 注入的 LLM 客户端失败时抛出 `LLMError`。

输入校验失败属于调用方数据错误，使用相同输入重试没有意义。LLM 传输失败或模型输出格式错误可以沿用上游重试策略。

## EverOS 职责

调用 EverAlgo 前，EverOS 必须：

1. 选择属于目标 owner 范围的 Episode。
2. 根据上游 Episode ID 去重。
3. 按时间从最早到最新排序。
4. 只提取每个 Episode 的叙事正文。
5. 从参与人元数据中解析权威的 `owner_id`，并在有姓名时解析 `owner_name`。
6. 传入姓名时，确保每条叙事都包含准确的 `owner_name`；未传姓名时，确保每条叙事都包含准确的 `owner_id`。
7. 将最大 Episode 时间戳作为 `timestamp`。
8. 如果目标用户已有 Profile，加载后作为 `old_profile` 传入。
9. 组装并传入当前完整的 `explicit_info` 分类快照；当前没有可用分类时传入 `None`。
10. 持久化返回的 Profile 并处理重试；EverAlgo 保持无状态。

现有 EverOS Episode 事件已经携带 Episode ID、叙事文本、时间戳和 `owner_id`，但不携带 `owner_name`。上游集成应在参与人元数据有姓名时解析并传入；无法获得姓名时依赖 `owner_id` 回退。

## 异步调用示例

```python
from everalgo.user_memory import OutputLanguage, ProfileExtractor


episode_records = await episode_repository.list_for_owner(owner_id="user-123")
deduplicated = deduplicate_by_entry_id(episode_records)
ordered = sorted(deduplicated, key=lambda episode: episode.timestamp_ms)

profile = await ProfileExtractor(llm=llm).aextract_from_episode_texts(
    [episode.text for episode in ordered],
    owner_id="user-123",
    owner_name="Alice",
    timestamp=max(episode.timestamp_ms for episode in ordered),
    old_profile=existing_profile,
    categories=available_profile_categories,
    output_language=OutputLanguage.CHINESE,
)
```

如果希望由模型自行决定输出语言，可以省略 `output_language` 或显式传入 `None`：

```python
profile = await ProfileExtractor(llm=llm).aextract_from_episode_texts(
    episode_texts,
    owner_id=owner_id,
    owner_name=owner_name,
    timestamp=latest_timestamp_ms,
    old_profile=existing_profile,
    categories=None,
    output_language=None,
)
```

## 上游集成验收标准

EverOS 集成测试必须覆盖：

- 传入 `owner_name` 且每条文本都包含该姓名时，INIT 成功。
- 未传 `owner_name` 且每条文本都包含 `owner_id` 时，INIT 成功。
- `old_profile.owner_id` 与 `owner_id` 一致时，UPDATE 成功。
- `categories=None`、空列表、空白值和完全相同的重复值都有确定且已记录的渲染行为。
- 分类列表包含非字符串值时，在调用 LLM 前失败。
- INIT、UPDATE、COMPACT 和 REGROUP 收到同一份规范化分类快照。
- 有准确匹配项时按事实语义选用；没有准确匹配项时允许创建必要分类。
- 分类快照不约束 `implicit_traits.trait`。
- 任意一条文本缺少已确定的目标引用时，在调用 LLM 前失败。
- 已有 Profile 的 owner 不一致时，在调用 LLM 前失败。
- 构造文本列表前，已根据 Episode ID 完成去重。
- 文本按时间排序，且 `timestamp` 等于最新 Episode 的时间戳。
- 同一个 generic Episode fanout 给两个 owner 时，分别调用接口并得到归属不同的 Profile。
- evidence 能够追溯到 Episode 叙事，并且不包含伪造的用户引语。

## 为什么选择这个接口

使用 `Sequence[str]` 可以让 EverAlgo 与 EverOS 的持久化模型解耦，并符合无状态算法边界。`owner_id` 始终决定 Profile 归属，可选的 `owner_name` 用于在可能不包含 ID 的模型生成叙事中定位目标人物。提取前只确定一个目标引用，并强制每条叙事都包含它，可以在调用 LLM 前暴露输入错误，避免多参与人画像静默抽取到错误用户身上。单独接收当前分类快照，可以让分类策略仍由调用方管理，同时让无状态的提取与维护阶段应用同一套语义规则。
