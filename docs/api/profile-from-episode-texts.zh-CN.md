# 从 Episode 提取 Profile

状态：EverAlgo 已实现；可开始 EverOS 集成。

本文定义 EverOS 与 EverAlgo 之间的接口契约：从带观察时间的 generic 或 reflected Episode 中，为一个指定用户提取 `Profile`。EverOS 保留存储标识和编排元数据，并决定「哪些」Episode 对 profile 而言是新的；EverAlgo 接收这些 Episode 及其观察时间，维护 profile 的时间线，并决定每条 Episode「可以改什么」。

## 接口定义

```python
from collections.abc import Sequence

from asgiref.sync import async_to_sync

from everalgo.types import Episode, Profile
from everalgo.user_memory import OutputLanguage


class ProfileExtractor:
    async def aextract_from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        owner_id: str,
        owner_name: str | None = None,
        old_profile: Profile | None = None,
        categories: Sequence[str] | None = None,
        prompt: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Profile:
        ...

    extract_from_episodes = async_to_sync(aextract_from_episodes)
```

`aextract_from_episodes` 是原生异步接口。`extract_from_episodes` 是供非事件循环调用方使用的同步桥接，不能在正在运行的事件循环中调用。

现有 [`ProfileExtractor.aextract`](../../packages/everalgo-user-memory/src/everalgo/user_memory/profile.py) 接口保持不变。两个接口分别服务不同输入：

| 接口 | 输入 | 目标用户定位方式 |
|---|---|---|
| `aextract` | 按时间排序的 `Sequence[MemCell]` | 根据结构化用户消息验证 `sender_id` |
| `aextract_from_episodes` | 任意顺序的 `Sequence[Episode]` | 从 `owner_name` 或 `owner_id` 确定一个目标引用，再选择叙事包含该引用的 Episode |

## 参数

### `episodes`

非空的 `everalgo.types.Episode` 序列。每项的 `episode` 叙事必须非空白，`timestamp`（Unix 毫秒）是该叙事所述内容的观察时间。每项可以是 generic Episode，也可以是 reflected Episode。顺序不重要：EverAlgo 会在渲染前按 `timestamp` 对选中的 Episode 排序。

`owner_name` 非空时，EverAlgo 使用姓名；没有有效姓名时回退到 `owner_id`。叙事不包含已确定目标引用的 Episode 会被排除。至少需要一项包含目标引用。

只传入自上次调用以来、已存 profile 尚未消费过的 Episode（见 [EverOS 职责](#everos-职责)）。EverOS 必须在调用前完成 Episode 去重；本接口不接收存储标识。

### `owner_id`

目标用户的权威标识。EverAlgo 必须将该值写入 `Profile.owner_id`，不得从模型输出中推断或替换此标识。

`owner_id` 不得为空。UPDATE 模式下，必须满足 `old_profile.owner_id == owner_id`。

### `owner_name`

用于在 Episode 叙事中定位目标用户的可选权威显示名称。EverOS 在有可信参与人姓名时传入；EverAlgo 不得从叙事文本中推断姓名。传入 `None` 或空白字符串时，EverAlgo 使用 `owner_id` 作为目标引用。

`owner_name` 只用于定位目标用户，返回 Profile 的归属始终由 `owner_id` 决定。

### `old_profile`

- 传入 `None` 表示 INIT 模式，创建新的 Profile。
- 传入 `Profile` 表示 UPDATE 模式，在已有 Profile 上执行新增、更新和删除。

`old_profile.timestamp` 被读作已存 profile 所反映到的观察时间，见 [时间感知合并](#时间感知合并)。UPDATE 后继续沿用现有的透明压缩行为。

### `categories`

`explicit_info.category` 本次可用的完整分类快照。调用方必须在调用前组装完整快照；EverAlgo 不区分分类的来源或生命周期状态。

EverAlgo 会去除每个分类字符串两端的空白、忽略空白值，并按首次出现顺序去除完全相同的重复值；规范化后的 JSON 列表会不变地注入 INIT、UPDATE、COMPACT 和 REGROUP。`None` 与空列表都会渲染为 `[]`。直接传入字符串而不是字符串序列，或序列中包含非字符串值时，会在调用 LLM 前抛出 `TypeError`。

每个阶段处理显式事实时，都必须根据事实本身的语义，从该列表中选择最准确的匹配分类。该列表不是白名单：没有准确匹配项时，可以创建必要、简洁且语义准确的分类。分类复用和减少分类数量不得覆盖分类准确性。该列表不约束 `implicit_traits.trait`。

### `prompt`

可选的 prompt 覆盖参数。传入 `None` 时使用内置的 Episode Profile prompt。即使使用自定义 prompt，输入校验、输出归属和时间感知合并规则仍然生效，因为它们由代码强制执行。

### `output_language`

期望的输出语言，可以传入 `OutputLanguage` 或等价的大小写不敏感字符串。传入 `None` 时，INIT 根据 Episode 叙事语言生成，UPDATE 保持已有 Profile 的语言。

## 目标用户校验

所有校验和目标筛选必须在第一次 LLM 调用前完成。EverAlgo 必须逐条检查 Episode；只检查拼接后的整批文本是不够的，因为不含目标用户的正文不得进入提示词。

EverAlgo 在校验前只确定一次目标引用：

```python
target_user = owner_name.strip() if owner_name and owner_name.strip() else owner_id.strip()
```

对每个 `episodes[index]` 执行：

1. 该项不是 `Episode` 时以 `TypeError` 拒绝整个调用；叙事为空白时以 `ValueError` 拒绝。
2. 使用一致的规则规范化叙事和 `target_user`。
3. 叙事包含 `target_user` 的字面目标引用时保留该项。
4. 不包含目标引用时跳过该项，不把叙事传给 LLM。
5. 检查完全部序列后，仅在没有剩余项时抛出 `ValueError`。

结构校验异常必须包含无效项的列表下标；全部未命中异常必须包含目标用户参数。异常和日志都不得输出 Episode 正文。

错误示例：

```text
no episodes reference target user 'Alice'
```

该筛选是确定性的安全保护，不是身份认证。它能阻止未命中的叙事影响抽取，但无法区分两个同名用户。EverOS 仍负责提供权威的 owner 信息和已经按 owner 划分的 Episode 批次。

## 时间感知合并

Profile 是「这个人现在是什么样」的画像。EverOS 可能不按观察顺序投递 Episode——回填历史、设备同步旧会话——所以合并不能把每条新叙事都当成最新事实。EverAlgo 维护两个时间，并强制执行一条规则。

**两个时间**

- `Profile.timestamp` 是 profile 所反映到的最新观察时间。INIT 取选中 Episode 中最新的一条；UPDATE 取 `max(old_profile.timestamp, 本批最新 Episode)`。它永不倒退。
- 每条 `explicit_info` 和 `implicit_traits` 条目携带 `observed_at`（Unix 毫秒）：最后一次确立该条目 `description` 的叙事的观察时间。INIT 给所有条目打上本批最新时间；UPDATE 给每条新增条目和每条被改写 description 的条目打上本批最新时间，只改 grounding（`evidence` / `basis`）或分类的更新保留原时间；被 COMPACT 或 REGROUP 重写的条目取 `Profile.timestamp`。该字段出现之前写入的存量条目按 `old_profile.timestamp` 读取。

`observed_at` 只由 EverAlgo 写入。模型看到的是渲染在每条存量条目旁的日期，模型写进操作里的任何 `observed_at` 都会被丢弃。

**规则**

观察时间早于某条目 `observed_at` 的 Episode，是关于这个人的更老的知识。它可以新增尚未记录的事实，可以为已有条目补充证据，但不得改写该条目的 description，也不得删除它——更晚的 Episode 已经确立了当前状态。UPDATE 提示词写明了这条规则，`_apply_ops` 在代码里强制执行：本批时间早于目标条目的 delete 或改动 description 的 update 会被丢弃，并以 `reason=stale evidence` 记录日志。

**两遍处理**

规则按批判定，以本批最新时间为准。跨越 `old_profile.timestamp` 的一批因此会被切分：早于它的 Episode 先跑历史遍（只能补缺），晚于它的 Episode 再跑正常遍。实时摄入永远没有历史半批，不多付 LLM 调用；只有回填和乱序投递才会，且仅当同一批里两种都有时。

## 提取行为

该接口只为传入的 `owner_id` 执行一次个人 Profile 提取。上游可以把同一个 generic Episode fanout 给多个 owner，但必须为每个 owner 分别调用本接口，因为每个人的画像解释不同，不能共享一次 Profile 结果。

返回结果必须满足：

- `Profile.owner_id == owner_id`。
- INIT 时 `Profile.timestamp` 等于选中 Episode 中最新的 `timestamp`；UPDATE 时等于 `max(old_profile.timestamp, 选中的最新 Episode)`。
- 每条条目携带整数 `observed_at`，定义如上。
- 由更晚 Episode 确立的条目不会被更早的 Episode 改写或删除。
- 只提取目标用户的事实和特征，不混入其他参与人画像。
- 在 INIT、UPDATE、COMPACT 和 REGROUP 中，事实正确性与 `explicit_info.category` 准确性并列为绝对最高优先级。
- 即使保留内容有利于召回率，也必须排除缺乏来源支持、归属错误、短期状态、会过期的计划、独立问题，以及对所有任职者普遍适用的团队或组织流程。
- 隐性特质需要来自不同 Episode 叙事的两个独立且一致的信号；一条已存的 `explicit_info` 条目算作其中一个信号，因此一条新叙事加一条与之一致的已存事实即可成立。
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

Episode 路径无法访问原始对话，因此它的 evidence 规则与 `MemCell` 路径不同：

- `explicit_info[].evidence` 必须是可在输入 Episode 叙事中核对的摘录或忠实转述。
- evidence 不需要 Episode 编号或人为生成的标识。每条叙事以其观察时间 `[YYYY-MM-DD HH:MM:SS UTC]` 开头展示给模型；模型可以引用该日期，但不得杜撰其他日期。
- 除非输入 Episode 本身包含用户原话，否则 evidence 不得把叙事改写成用户直接引语。
- 模型不得伪造用户措辞、日期或说话人归属。
- `implicit_traits[].basis` 必须指出能在输入 Episode 叙事或已存 `explicit_info` 条目中找到的信号，不得伪造用户引语。
- `evidence` 与 `basis` 都是单个 JSON 字符串，不是数组；允许忠实的叙事转述，但必须保留来源含义与归属。

EverOS 必须把这些证据视为 Episode 派生证据，而不是原始对话证据。

## 异常

出现以下情况时，接口在调用 LLM 前抛出 `TypeError`：

- `episodes` 中某项不是 `Episode`；
- `categories` 不是字符串序列或包含非字符串值。

出现以下情况时，接口在调用 LLM 前抛出 `ValueError`：

- `episodes` 为空；
- 任一 Episode 叙事为空白；
- `owner_id` 为空白；
- 所有 Episode 叙事都不包含已确定的目标引用；
- `old_profile.owner_id` 与 `owner_id` 不一致；
- `output_language` 不受支持。

接口继续保留现有 Profile 提取异常：

- LLM 返回结果违反 Profile 结构时抛出 `ValueError`；
- LLM 返回内容不是可解析的 JSON 时抛出 `json.JSONDecodeError`；
- 注入的 LLM 客户端失败时抛出 `LLMError`。

输入校验失败属于调用方数据错误，使用相同输入重试没有意义。LLM 传输失败或模型输出格式错误可以沿用上游重试策略。

## EverOS 职责

调用 EverAlgo 前，EverOS 必须：

1. 选择属于目标 owner 范围、且已存 profile **尚未消费**的 Episode——例如按 profile 级水位线选取其后写入的那些。每次都传全量历史会把 prompt 重复花在 profile 已反映的事实上；只按观察时间取最新几条则会静默丢掉回填进来的更早 Episode。
2. 根据上游 Episode ID 去重。
3. 把每条映射为 `everalgo.types.Episode`，带上叙事和观察 `timestamp`。顺序无关。
4. 从参与人元数据中解析权威的 `owner_id`，并在有姓名时解析 `owner_name`。
5. 有准确姓名时传入 `owner_name`，否则依赖 `owner_id`；EverAlgo 会过滤不包含已确定值的叙事。
6. 如果目标用户已有 Profile，加载后作为 `old_profile` 传入，包含条目及其 `observed_at`。
7. 组装并传入当前完整的 `explicit_info` 分类快照；当前没有可用分类时传入 `None`。
8. 持久化返回的 Profile——包括 `timestamp` 和每条条目的 `observed_at`，下一次调用要读它们——并且只在持久化成功后推进水位线；EverAlgo 保持无状态。

现有 EverOS Episode 记录已经携带 Episode ID、叙事文本、观察时间戳和 `owner_id`，但不携带 `owner_name`。上游集成应在参与人元数据有姓名时解析并传入；无法获得姓名时依赖 `owner_id` 回退。

## 异步调用示例

```python
from everalgo.types import Episode
from everalgo.user_memory import OutputLanguage, ProfileExtractor


rows = await episode_repository.list_unconsumed_for_owner(owner_id="user-123", after=existing_profile_watermark)
episodes = [
    Episode(owner_id=row.owner_id, episode=row.text, subject=row.subject, summary=row.summary, timestamp=row.timestamp_ms)
    for row in deduplicate_by_entry_id(rows)
]

profile = await ProfileExtractor(llm=llm).aextract_from_episodes(
    episodes,
    owner_id="user-123",
    owner_name="Alice",
    old_profile=existing_profile,
    categories=available_profile_categories,
    output_language=OutputLanguage.CHINESE,
)
```

如果希望由模型自行决定输出语言，可以省略 `output_language` 或显式传入 `None`：

```python
profile = await ProfileExtractor(llm=llm).aextract_from_episodes(
    episodes,
    owner_id=owner_id,
    owner_name=owner_name,
    old_profile=existing_profile,
    categories=None,
    output_language=None,
)
```

## 上游集成验收标准

EverOS 集成测试必须覆盖：

- 传入 `owner_name` 且每条叙事都包含该姓名时，INIT 成功。
- 未传 `owner_name` 且每条叙事都包含 `owner_id` 时，INIT 成功。
- `old_profile.owner_id` 与 `owner_id` 一致时，UPDATE 成功。
- `categories=None`、空列表、空白值和完全相同的重复值都有确定且已记录的渲染行为。
- 分类列表包含非字符串值时，在调用 LLM 前失败。
- INIT、UPDATE、COMPACT 和 REGROUP 收到同一份规范化分类快照。
- 有准确匹配项时按事实语义选用；没有准确匹配项时允许创建必要分类。
- 分类快照不约束 `implicit_traits.trait`。
- 缺少已确定目标引用的叙事不会进入 LLM 提示词；命中的叙事按最早优先排列，每条位于自己的观察时间之下。
- 所有叙事都缺少已确定的目标引用时，在调用 LLM 前失败。
- 已有 Profile 的 owner 不一致时，在调用 LLM 前失败。
- 构造批次前，已根据 Episode ID 完成去重。
- 只传入已存 profile 尚未消费的 Episode，且水位线只在返回的 Profile 持久化后推进。
- 持久化的 Profile 能把 `timestamp` 和每条条目的 `observed_at` 原样带入下一次调用的 `old_profile`。
- 观察时间早于已存状态的 Episode 可以新增事实，但不改动由更晚 Episode 确立的条目 description，且 `Profile.timestamp` 不倒退。
- 同一个 generic Episode fanout 给两个 owner 时，分别调用接口并得到归属不同的 Profile。
- evidence 能够追溯到 Episode 叙事，并且不包含伪造的用户引语。

## 为什么选择这个接口

接收 `Sequence[Episode]` 而不是裸字符串，让每条叙事的观察时间与叙事绑定，合并才能区分「更老的知识」与「更新的状态」；core 里已有 `Episode` 类型，不需要新造。`owner_id` 始终决定 Profile 归属，可选的 `owner_name` 用于在可能不包含 ID 的模型生成叙事中定位目标人物。提取前只确定一个目标引用并逐条筛选，既能阻止无关叙事影响 Profile，也不会因为一条无关 Episode 丢弃仍可使用的整批输入；全部未命中时继续拒绝抽取，避免在没有 owner 证据时生成画像。时间规则放在代码里而不只放在提示词里，因为它是确定性的：模型即使忽视指令也无法把 profile 倒回过去。「哪些 Episode 是新的」由拥有存储和水位线的 EverOS 决定，「这些 Episode 可以改什么」由拥有合并逻辑的 EverAlgo 决定。单独接收当前分类快照，可以让分类策略仍由调用方管理，同时让无状态的提取与维护阶段应用同一套语义规则。
