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
        custom_instructions: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Profile:
        ...

    extract_from_episodes = async_to_sync(aextract_from_episodes)
```

`aextract_from_episodes` 是原生异步接口。`extract_from_episodes` 是供非事件循环调用方使用的同步桥接，不能在正在运行的事件循环中调用。

现有 [`ProfileExtractor.aextract`](../../packages/everalgo-user-memory/src/everalgo/user_memory/profile/_extractor.py) 接口保持不变。两个接口分别服务不同输入：

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

`old_profile.timestamp` 被读作已存 profile 所反映到的观察时间，见 [时间感知合并](#时间感知合并)。UPDATE 之后在同一次调用里把 profile 收回上限之内：总数超过 60 条先做 COMPACT，然后同一个分类名或特征名下超过 8 条的分组各做一次 REGROUP。做过 COMPACT 时，分组检查看的是 COMPACT 的结果，所以 COMPACT 留下的超限分组也会被重组。

两个桶里的条目都是 `everalgo.types` 中的类型化模型：

| 字段 | `explicit_info[]` — `ExplicitInfoItem` | `implicit_traits[]` — `ImplicitTraitItem` |
|---|---|---|
| `description` | 必填。事实本身，一句短句；其归一化形式即条目身份。 | 必填。规律本身，一句短句；身份规则相同。 |
| 标签 | `category: str \| None`，事实所属维度。 | `trait: str \| None`，特征标签。 |
| 依据 | `evidence: str \| None`，原文摘录或忠实转述。 | `basis: str \| None`，推断所依据的信号。 |
| `observed_at` | `int \| None`，epoch 毫秒；由 EverAlgo 写入，见 [时间感知合并](#时间感知合并)。 | 同左。 |
| 其他键 | 在每次合并中原样保留（`extra="allow"`），从不展示给模型。 | 同左。 |

宿主若以文档形式存储条目，用 `model_dump()` 导出（`model_dump(exclude_unset=True)` 不输出从未赋值的字段），下次通过 `Profile.model_validate` 传回。缺少 `description` 或不是对象的条目会在该校验中失败。

### `categories`

`explicit_info.category` 本次可用的完整分类快照。调用方必须在调用前组装完整快照；EverAlgo 不区分分类的来源或生命周期状态。

EverAlgo 会去除每个分类字符串两端的空白、忽略空白值，并按首次出现顺序去除完全相同的重复值；规范化后的 JSON 列表会不变地注入 INIT、UPDATE、COMPACT 和 REGROUP。`None` 与空列表都会渲染为 `[]`。直接传入字符串而不是字符串序列，或序列中包含非字符串值时，会在调用 LLM 前抛出 `TypeError`。

提示词把分类定义为事实所属的维度，按事实本身而不是它出现的那段叙事的主题命名，并要求模型优先从该列表中选；没有列表或列表中没有合适的分类时，模型自拟一个简短的维度名。因此该列表是偏好而非白名单，也不约束 `implicit_traits.trait`。

### `custom_instructions`

可选的宿主自定义指令，追加在本次调用经过的每个阶段——INIT 或 UPDATE，以及其后的 COMPACT 或 REGROUP——的内置规则之后。参数 `custom_instructions` 的文本以 `<additional_requirements>` 块进入 system 消息，放在最末、`<available_categories>` 之后，块首是固定引导语，说明这段文本叠加在内置规则之上、`<reply_format>` 规定的回复结构保持不变，且与内置规则冲突时以内置规则为准——它只能收紧规则，不能放宽。文本原样发送，不作为模板渲染，其中的花括号保持字面含义。

传入 `None` 或空白字符串时不追加任何内容，system 消息与内置版本完全一致。该参数不能替换或删除任何内置规则，输出结构保持固定；输入校验、输出归属和时间感知合并规则无论如何都由代码强制执行。

这段文本由宿主自己撰写，不是终端用户的输入。它会作为指令到达模型——这也是 `output_language` 只接受固定的语言名、不接受自由文本的原因——所以宿主要为转发进来的任何内容负责。把该块放在最后，可以让它之前的所有块继续作为共享的缓存前缀；只有该块本身随文本变化。

### `output_language`

期望的输出语言，可以传入 `OutputLanguage` 或等价的大小写不敏感字符串。传入 `None` 时，INIT 根据 Episode 叙事语言生成，UPDATE、COMPACT、REGROUP 沿用已有 Profile 多数内容所用的语言，所以已经混杂的 Profile 也会收敛到一种语言。指定语言后，每个阶段的所有字段都用它写，不管输入是什么语言，已存的内容也一样：它优先于用其他语言写成的存量 Profile，UPDATE 的提示词要求把用其他语言写成的存量条目改写成它，即使没有叙事提到那一条。

## 目标用户校验

所有校验和目标筛选必须在第一次 LLM 调用前完成。EverAlgo 必须逐条检查 Episode；只检查拼接后的整批文本是不够的，因为不含目标用户的正文不得进入提示词。

EverAlgo 在校验前只确定一次目标引用：

```python
target_user = owner_name.strip() if owner_name and owner_name.strip() else owner_id.strip()
```

对每个 `episodes[index]` 执行：

1. 该项不是 `Episode` 时以 `TypeError` 拒绝整个调用；叙事为空白时以 `InvalidInputError`（`ValueError` 子类，来自 `everalgo.utils`）拒绝。
2. 使用一致的规则规范化叙事和 `target_user`。
3. 叙事包含 `target_user` 的字面目标引用时保留该项。
4. 不包含目标引用时跳过该项，不把叙事传给 LLM。
5. 检查完全部序列后，仅在没有剩余项时抛出 `InvalidInputError`。

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

`observed_at` 只由 EverAlgo 写入。模型在每条存量条目的 JSON 里看到它，格式为精确到分钟的 UTC 时间（`2024-04-10 14:00 UTC`），与每段叙事的 `time` 属性同形；操作里写了 `observed_at` 的回复无法通过校验，抛出 `LLMOutputError`。

**规则**

观察时间早于某条目 `observed_at` 的 Episode，是关于这个人的更老的知识。它可以新增尚未记录的事实，可以为已有条目补充证据，但不得改写该条目的 description，也不得删除它——更晚的 Episode 已经确立了当前状态。UPDATE 提示词写明了这条规则——包括「一个已经有值的维度，它更早的值进那条条目的 evidence，不另立条目；description 可以在现值后面用一个短括号补上这段履历，但现值本身保持原样」——`_apply_ops` 在代码里强制执行：本批时间早于目标条目的 delete，以及不是「在旧 description 后面加一个括号」的改动 description 的 update，会被丢弃，并以 `reason=stale evidence` 记录日志。更早一批的 update 若新 description 是存量 description 后面接一个括号（忽略大小写与空白差异比较，存量末尾的 `.` `!` `?` `。` `！` `？` 不参与比较），例如在 `住在成都` 上写成 `住在成都（此前住在重庆）`，则会落地：采用它的 description 和 evidence，条目的 `observed_at` 不变，因为现值仍由更晚的 Episode 确立。

**两遍处理**

规则按批判定，以本批最新时间为准。跨越 `old_profile.timestamp` 的一批因此会被切分：早于它的 Episode 先跑历史遍（只能补缺），晚于它的 Episode 再跑正常遍。两遍发送同一份 system 消息；模型按它看到的时间逐条应用规则，切分只是为了让每一遍只有一个观察时间。实时摄入永远没有历史半批，不多付 LLM 调用；只有回填和乱序投递才会，且仅当同一批里两种都有时。

## 提取行为

该接口只为传入的 `owner_id` 执行一次个人 Profile 提取。上游可以把同一个 generic Episode fanout 给多个 owner，但必须为每个 owner 分别调用本接口，因为每个人的 Profile 解释不同，不能共享一次 Profile 结果。

返回结果必须满足：

- `Profile.owner_id == owner_id`。
- INIT 时 `Profile.timestamp` 等于选中 Episode 中最新的 `timestamp`；UPDATE 时等于 `max(old_profile.timestamp, 选中的最新 Episode)`。
- 每条条目携带整数 `observed_at`，定义如上。
- 由更晚 Episode 确立的条目不会被更早的 Episode 改写或删除。
- 只提取目标用户的事实和特征，不混入其他参与人的 Profile。
- 单个动作（或被改述成能力、技能、职责、兴趣、态度的单个动作）、一时的状态或此刻手头的工作、换谁坐在这个位子上都成立的规则、为某一个项目做过一次的决定、单个问题、助手或其他参与者说的话、关于 owner 以外任何人的事实（即使是 owner 自己说的，即使放在关于这段关系的条目里；关系只有作为 owner 自己生活里的长期组成部分才进，如伴侣、同住的人，且不带关于对方的任何东西，只剩「有某人」的空壳不成为条目）、单次的格式或操作要求，都不成为条目；归属、持久性或依据不确定的条目留在外面。
- `explicit_info` 条目是叙事明确归到 owner 身上的事实。`implicit_traits` 条目是没有哪一段叙事直接写出、但从多段叙事里看出的稳定模式，其 `basis` 点名具体信号——叙事记录的 owner 的选择或要求，而不是 owner 执行的操作。
- INIT、UPDATE、合并和压缩语义与现有 `aextract` 一致。
- 校验或提取失败时不返回部分 Profile。

INIT 与 UPDATE 的 system 消息在 `<never_in_the_profile>` 之后带一个 `<examples>` 块，用完整示例示范上面这些界线落在哪里：INIT 两个、UPDATE 一个，各按本阶段的回复形状，人物均为虚构。COMPACT 与 REGROUP 不带示例。

## Description 句式

每条 `explicit_info[].description` 和 `implicit_traits[].description` 必须使用简洁的无主语陈述句，直接描述 owner，不出现姓名或替代主语，也不得写成命令式祈使句。

正确示例：

- `主要使用 Python。`
- `偏好简洁、直接的回答。`

错误示例：

- `Alice 主要使用 Python。`——出现了姓名主语。
- `用户偏好简洁的回答。`——使用了通用替代主语。
- `使用 Python。`——把 profile 陈述改成了行动指令。

## Evidence 契约

Episode 路径无法访问原始对话，因此它的 evidence 规则与 `MemCell` 路径不同：

- `explicit_info[].evidence` 必须是可在输入 Episode 叙事中核对的摘录或忠实转述。
- evidence 不需要 Episode 编号或人为生成的标识。每条叙事以 `<narrative time="YYYY-MM-DD HH:MM UTC">…</narrative>` 的形式展示给模型，观察时间在属性里；模型可以引用该日期，但不得杜撰其他日期。叙事正文不做 XML 转义：只有 prompt 自身元素的闭合标签字面串（`</narrative>`、`</episode_narratives>`、`</item>`、`</explicit_info>`、`</implicit_traits>`、`</stored_profile>`、`</group>`，大小写不敏感）会把其中的 `<` 换成全角 `＜`（U+FF1C），叙事与已存条目同样处理，因此不会提前闭合元素。
- 除非输入 Episode 本身包含用户原话，否则 evidence 不得把叙事改写成用户直接引语。
- 模型不得伪造用户措辞、日期或说话人归属。
- `implicit_traits[].basis` 点名支撑该特质的信号——哪几段叙事里的哪些选择或要求——不得伪造用户引语；复述要求本身（「几个信号」）不算依据。
- `evidence` 与 `basis` 都是单个 JSON 字符串，不是数组；允许忠实的叙事转述，但必须保留来源含义与归属。

EverOS 必须把这些证据视为 Episode 派生证据，而不是原始对话证据。

## 异常

出现以下情况时，接口在调用 LLM 前抛出 `TypeError`：

- `episodes` 中某项不是 `Episode`；
- `categories` 不是字符串序列或包含非字符串值。

出现以下情况时，接口在调用 LLM 前抛出 `InvalidInputError`（`ValueError` 子类，来自 `everalgo.utils`）：

- `episodes` 为空；
- 任一 Episode 叙事为空白；
- `owner_id` 为空白；
- 所有 Episode 叙事都不包含已确定的目标引用；
- `old_profile.owner_id` 与 `owner_id` 不一致；
- `output_language` 不受支持。

LLM 的回复不是该阶段的合法输出时（不是可解析的 JSON；不符合 INIT、UPDATE、COMPACT、REGROUP 的结构；或条目形状与所属桶不符，即 UPDATE 操作的 `data` 与其 `type` 不一致、REGROUP 条目不是该组所属桶的形状），接口抛出 `LLMOutputError`（来自 `everalgo.llm`）。异常信息只指出每处校验失败的位置，不包含回复内容。

注入的 LLM 客户端抛出的异常原样透传。

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
- 缺少已确定目标引用的叙事不会进入 LLM 提示词；命中的叙事按最早优先排列，每条以属性携带自己的观察时间。
- 所有叙事都缺少已确定的目标引用时，在调用 LLM 前失败。
- 已有 Profile 的 owner 不一致时，在调用 LLM 前失败。
- 构造批次前，已根据 Episode ID 完成去重。
- 只传入已存 profile 尚未消费的 Episode，且水位线只在返回的 Profile 持久化后推进。
- 持久化的 Profile 能把 `timestamp` 和每条条目的 `observed_at` 原样带入下一次调用的 `old_profile`。
- 观察时间早于已存状态的 Episode 可以新增事实，但不改动由更晚 Episode 确立的条目 description，且 `Profile.timestamp` 不倒退。
- 同一个 generic Episode fanout 给两个 owner 时，分别调用接口并得到归属不同的 Profile。
- evidence 能够追溯到 Episode 叙事，并且不包含伪造的用户引语。
- 传入 `custom_instructions` 时，其内容由宿主撰写，从不转发终端用户输入。

## 为什么选择这个接口

接收 `Sequence[Episode]` 而不是裸字符串，让每条叙事的观察时间与叙事绑定，合并才能区分「更老的知识」与「更新的状态」；core 里已有 `Episode` 类型，不需要新造。`owner_id` 始终决定 Profile 归属，可选的 `owner_name` 用于在可能不包含 ID 的模型生成叙事中定位目标人物。提取前只确定一个目标引用并逐条筛选，既能阻止无关叙事影响 Profile，也不会因为一条无关 Episode 丢弃仍可使用的整批输入；全部未命中时继续拒绝抽取，避免在没有 owner 证据时生成 Profile。时间规则放在代码里而不只放在提示词里，因为它是确定性的：模型即使忽视指令也无法把 profile 倒回过去。「哪些 Episode 是新的」由拥有存储和水位线的 EverOS 决定，「这些 Episode 可以改什么」由拥有合并逻辑的 EverAlgo 决定。单独接收当前分类快照，可以让分类策略仍由调用方管理，同时让无状态的提取与维护阶段应用同一套语义规则。`custom_instructions` 采用追加而非替换：替换会连同规定输出结构、规定哪些内容能进入 profile 的规则一起丢掉，而追加的块让所有内置规则——以及共享的缓存前缀——保持不变。
