"""Each stage's reply model, the shape its prompt's output skeleton shows, and the LLM callsites held to it.

Each callsite makes one call and validates the reply against its stage's output model; there is no retry here,
the host decides (``docs/concepts/llm-contract.md`` §4, §5).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, WithJsonSchema

from everalgo.llm import LLMMessage, LLMOutputError, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

# The package's name rather than this module's: hosts collect the log contract under it.
logger = logging.getLogger("everalgo.user_memory.profile")


class _ExplicitItem(StrictOutputModel):
    """An ``explicit_info`` item as every stage writes it, grounding first: ``{"evidence", "description", "category"}``."""

    evidence: str
    description: str
    category: str


class _ImplicitTrait(StrictOutputModel):
    """An ``implicit_traits`` item as every stage writes it, grounding first: ``{"basis", "description", "trait"}``."""

    basis: str
    description: str
    trait: str


# Episode INIT and UPDATE items also carry the fields the item checks read (``_pending``): the model's ``pattern``
# flag, its ``quote`` and the ``regular`` words in it, and on an add an optional ``pending`` pointer. These fields, and ``type`` on ``confirm``,
# are optional — the one exception to the required-fields rule of the reply models (docs/concepts/llm-contract.md
# §2.2): the evaluated replies leave them out now and then, and a required field would fail the whole reply where the
# checks only drop the one item (a missing quote counts as not found). None of them is stored
# (``_merge._REPLY_ONLY_FIELDS``). COMPACT, REGROUP and the MemCell path keep the plain items.
#
# They stay ``Any`` to validation, so a stray value (an index written as "0", a list of quotes) reaches the checks,
# which drop the one item, instead of failing the reply. Their schema still declares the type the prompt asks for:
# strict structured outputs refuse an untyped value schema with a 400 ("schema must have a 'type' key").


def _nullable(schema: dict[str, Any]) -> WithJsonSchema:
    """The schema of an optional reply-only field: ``schema``, or ``null`` when the reply leaves it out."""
    return WithJsonSchema({"anyOf": [schema, {"type": "null"}]})


_Flag = Annotated[Any, _nullable({"type": "boolean"})]
_Index = Annotated[Any, _nullable({"type": "integer"})]
_Indices = Annotated[Any, _nullable({"type": "array", "items": {"type": "integer"}})]
_Text = Annotated[Any, _nullable({"type": "string"})]


class _EpisodeExplicitItem(StrictOutputModel):
    """An ``explicit_info`` item of an Episode INIT or UPDATE reply: the plain item plus the reply-only fields."""

    evidence: str
    description: str
    category: str
    pattern: _Flag = None
    pending: _Index = None
    quote: _Text = None
    regular: _Text = None


class _EpisodeImplicitTrait(StrictOutputModel):
    """An ``implicit_traits`` item of an Episode INIT or UPDATE reply: the plain item plus the reply-only fields."""

    basis: str
    description: str
    trait: str
    pattern: _Flag = None
    pending: _Index = None
    quote: _Text = None
    regular: _Text = None


class _ProfileItems(StrictOutputModel):
    """INIT reply: the ``{"explicit_info", "implicit_traits"}`` skeleton of both INIT prompts."""

    explicit_info: list[_ExplicitItem]
    implicit_traits: list[_ImplicitTrait]


class _CompactOutput(StrictOutputModel):
    """COMPACT reply as a rewrite of both buckets, without a note.

    The Episode path's COMPACT reply before it became operations (:class:`_CompactOpsOutput`); no stage requests
    it now, and the extractor's rewrite branch still accepts it.
    """

    explicit_info: list[_ExplicitItem]
    implicit_traits: list[_ImplicitTrait]


class _MemCellCompactOutput(StrictOutputModel):
    """COMPACT reply on the MemCell path: the rewritten buckets plus the ``compact_note`` its prompt asks for."""

    explicit_info: list[_ExplicitItem]
    implicit_traits: list[_ImplicitTrait]
    compact_note: str


class _RegroupOutput(StrictOutputModel):
    """REGROUP reply as the group's kept items, without a note.

    The Episode path's REGROUP reply before it became operations (:class:`_RegroupOpsOutput`); no stage requests
    it now, and the extractor's rewrite branch still accepts it.
    """

    items: list[_ExplicitItem | _ImplicitTrait]


class _MemCellRegroupOutput(StrictOutputModel):
    """REGROUP reply on the MemCell path: the group's kept items plus the ``regroup_note`` its prompt asks for."""

    items: list[_ExplicitItem | _ImplicitTrait]
    regroup_note: str


_Bucket = Literal["explicit_info", "implicit_traits"]
# The item shape each bucket holds; the business checks after parsing hold replies to it.
_BUCKET_SHAPES: dict[str, type[_ExplicitItem | _ImplicitTrait]] = {
    "explicit_info": _ExplicitItem,
    "implicit_traits": _ImplicitTrait,
}
# The Episode INIT and UPDATE item of each bucket (the same shape plus the reply-only fields).
_EPISODE_BUCKET_SHAPES: dict[str, type[_EpisodeExplicitItem | _EpisodeImplicitTrait]] = {
    "explicit_info": _EpisodeExplicitItem,
    "implicit_traits": _EpisodeImplicitTrait,
}


class _AddOp(StrictOutputModel):
    """UPDATE ``add`` operation: a new item for the bucket named by its ``type`` key."""

    action: Literal["add"]
    # The JSON key stays "type", as the UPDATE prompts show it; the field name avoids the builtin.
    bucket: _Bucket = Field(alias="type")
    data: _ExplicitItem | _ImplicitTrait


class _UpdateOp(StrictOutputModel):
    """UPDATE ``update`` operation: the item at ``index`` restated in full."""

    action: Literal["update"]
    bucket: _Bucket = Field(alias="type")
    index: int
    data: _ExplicitItem | _ImplicitTrait


class _DeleteOp(StrictOutputModel):
    """UPDATE ``delete`` operation: the item at ``index`` removed, with the model's ``reason``."""

    action: Literal["delete"]
    bucket: _Bucket = Field(alias="type")
    index: int
    reason: str


class _ForgetOp(StrictOutputModel):
    """A ``forget`` operation as the MemCell UPDATE reply would parse one: ``{"action", "topic", "evidence"}``.

    The MemCell prompt asks for none and keeps no do-not-record list; one that arrives is refused by the merge as an
    op without a ``type``. The Episode path's is :class:`_EpisodeForgetOp`.
    """

    action: Literal["forget"]
    topic: str
    evidence: str


# ``quote`` on a matter, like the reply-only fields of an item above, is optional so that a reply leaving it out loses
# the one matter (a missing quote counts as not found) instead of failing whole. ``evidence`` is what the reply carried
# before ``quote``; it is still accepted, and ignored.


class _EpisodeForgetOp(StrictOutputModel):
    """Episode UPDATE ``forget`` operation: this person asked that the matter ``topic`` not be recorded from now on.

    It names no stored item and has no ``type``. ``quote`` is the request in the narrative's words and ``pending`` the
    ``<pending_patterns>`` entries about the matter. The extractor adds the matter to the profile's do-not-record
    list when it finds the quote (``_do_not_record.enter_matters``); the stored items about it are removed by the
    reply's own deletes.
    """

    action: Literal["forget"]
    topic: str
    quote: _Text = None
    pending: _Indices = None
    evidence: _Text = None


class _AllowOp(StrictOutputModel):
    """Episode UPDATE ``allow`` operation: this person withdrew the request listed at index ``matter``.

    ``quote`` is the withdrawal in the narrative's words. With ``part`` only that part of the matter is allowed again
    and the rest stays listed (``_do_not_record.withdraw_matters``).
    """

    action: Literal["allow"]
    matter: _Index = None
    part: _Text = None
    quote: _Text = None


class _DoNotRecordEntry(StrictOutputModel):
    """A matter this person asked not to be recorded, as the Episode INIT reply lists it: ``{"topic", "quote"}``."""

    topic: str
    quote: _Text = None
    evidence: _Text = None


class _EpisodeInitOutput(StrictOutputModel):
    """INIT reply on the Episode path: the two lists plus the matters this person asked not to record.

    The MemCell path's INIT reply stays :class:`_ProfileItems`.
    """

    explicit_info: list[_EpisodeExplicitItem]
    implicit_traits: list[_EpisodeImplicitTrait]
    do_not_record: list[_DoNotRecordEntry]


class _EpisodeAddOp(StrictOutputModel):
    """Episode UPDATE ``add``: :class:`_AddOp` whose item carries the reply-only fields."""

    action: Literal["add"]
    bucket: _Bucket = Field(alias="type")
    data: _EpisodeExplicitItem | _EpisodeImplicitTrait


class _EpisodeUpdateOp(StrictOutputModel):
    """Episode UPDATE ``update``: :class:`_UpdateOp` whose item carries the reply-only fields."""

    action: Literal["update"]
    bucket: _Bucket = Field(alias="type")
    index: int
    data: _EpisodeExplicitItem | _EpisodeImplicitTrait


class _ConfirmOp(StrictOutputModel):
    """Episode UPDATE ``confirm`` operation: the ``<pending_patterns>`` entry at ``pending`` seen again, its item in full.

    The item checks (``_pending.check_update_ops``) turn it into an add into ``type`` (``explicit_info`` when it is
    left out) and take the entry off the pending list.
    """

    action: Literal["confirm"]
    bucket: _Bucket | None = Field(default=None, alias="type")
    pending: _Index = None
    data: _EpisodeExplicitItem | _EpisodeImplicitTrait


class _NoneOp(StrictOutputModel):
    """UPDATE ``none`` operation: the documented no-op."""

    action: Literal["none"]


# Shared by both UPDATE replies. Only the Episode prompt asks for ``forget``; on the MemCell path one would parse and
# then be refused by the merge as an op without a ``type``.
_Operation = Annotated[_AddOp | _UpdateOp | _DeleteOp | _ForgetOp | _NoneOp, Field(discriminator="action")]
# The Episode UPDATE reply also takes ``confirm`` and ``allow``; the MemCell prompt has no pending list and no
# do-not-record list.
_EpisodeOperation = Annotated[
    _EpisodeAddOp | _EpisodeUpdateOp | _DeleteOp | _EpisodeForgetOp | _AllowOp | _NoneOp | _ConfirmOp,
    Field(discriminator="action"),
]


class _UpdateOutput(StrictOutputModel):
    """UPDATE reply on the Episode path: the ``{"operations"}`` skeleton of its prompt."""

    operations: list[_EpisodeOperation]


class _MemCellUpdateOutput(StrictOutputModel):
    """UPDATE reply on the MemCell path: the operations plus the ``update_note`` its prompt asks for."""

    operations: list[_Operation]
    update_note: str


# The Episode path's COMPACT and REGROUP reply with operations on the items as rendered with their indices, so an
# item no operation names stays the stored object, its host keys and observed_at with it. The item models accept
# either bucket's shape; ``_compact_ops.apply_compact_ops`` refuses an item not shaped for the bucket it lands in.


class _MergeOp(StrictOutputModel):
    """COMPACT / REGROUP ``merge``: the items at ``indices`` become the one item in ``data``."""

    action: Literal["merge"]
    bucket: _Bucket = Field(alias="type")
    indices: list[int]
    data: _ExplicitItem | _ImplicitTrait


class _SplitOp(StrictOutputModel):
    """COMPACT / REGROUP ``split``: the item at ``index`` becomes the items in ``items``."""

    action: Literal["split"]
    bucket: _Bucket = Field(alias="type")
    index: int
    items: list[_ExplicitItem | _ImplicitTrait]


class _MoveOp(StrictOutputModel):
    """COMPACT ``move``: the trait at ``index`` becomes the explicit_info item in ``data``; REGROUP has no move."""

    action: Literal["move"]
    bucket: _Bucket = Field(alias="type")
    index: int
    data: _ExplicitItem | _ImplicitTrait


_CompactOperation = Annotated[_MergeOp | _SplitOp | _MoveOp | _UpdateOp | _DeleteOp, Field(discriminator="action")]
_RegroupOperation = Annotated[_MergeOp | _SplitOp | _UpdateOp | _DeleteOp, Field(discriminator="action")]


class _CompactOpsOutput(StrictOutputModel):
    """COMPACT reply on the Episode path: operations on the indexed stored items."""

    operations: list[_CompactOperation]


class _RegroupOpsOutput(StrictOutputModel):
    """REGROUP reply on the Episode path: operations on the group's indexed items."""

    operations: list[_RegroupOperation]


class _MatterMergeOp(StrictOutputModel):
    """One merge of the do-not-record list: the entries at ``drop`` are redundant beside the entry at ``keep``.

    ``relation`` says how: ``same`` or ``inside``. Its value is read after parsing
    (``_do_not_record.apply_merge_ops``), so a merge of any other kind is refused alone and the reply's other merges
    still apply.
    """

    keep: int
    drop: list[int]
    relation: str


class _MatterMergeOutput(StrictOutputModel):
    """Reply of the do-not-record merge call: the merges to make, an empty list when no entry is redundant."""

    operations: list[_MatterMergeOp]


# Each path validates against its own models, repeated rather than extended (docs/concepts/llm-contract.md §2.2).
_AnyUpdateOutput = _UpdateOutput | _MemCellUpdateOutput
_AnyCompactOutput = _CompactOutput | _MemCellCompactOutput | _CompactOpsOutput
_AnyRegroupOutput = _RegroupOutput | _MemCellRegroupOutput | _RegroupOpsOutput


async def _call_llm_for_profile_init(
    llm: LLMClient, messages: list[LLMMessage], *, reply: type[_ProfileItems] | type[_EpisodeInitOutput]
) -> dict[str, Any]:
    """Request an INIT reply and return it validated, dumped to a dict.

    Raises:
        LLMOutputError: If the reply is not a valid INIT output.
    """
    text = await llm.achat(messages=messages, response_format=reply)
    return parse_reply(reply, text, operator="profile init").model_dump(by_alias=True)


async def _call_llm_for_profile_update[ReplyT: _AnyUpdateOutput](
    llm: LLMClient, messages: list[LLMMessage], *, reply: type[ReplyT]
) -> dict[str, Any]:
    """Request an UPDATE reply and return it validated, dumped to a dict whose ``operations`` keep their ``type`` key.

    An add or update whose item is not shaped for the bucket its ``type`` names is dropped with a warning and the
    reply's other operations are kept. Models do write such an operation now and then (an update into
    ``implicit_traits`` carrying an ``explicit_info`` item), and failing the whole reply for it lost every other
    operation of the batch.

    Raises:
        LLMOutputError: If the reply is not a valid UPDATE output.
    """
    text = await llm.achat(messages=messages, response_format=reply)
    output = parse_reply(reply, text, operator="profile update")
    kept = [
        op
        for op in output.operations
        if not (
            isinstance(op, _AddOp | _UpdateOp | _EpisodeAddOp | _EpisodeUpdateOp)
            and not isinstance(op.data, (_BUCKET_SHAPES[op.bucket], _EPISODE_BUCKET_SHAPES[op.bucket]))
        )
    ]
    if len(kept) != len(output.operations):
        logger.warning(
            "profile update ops dropped: count=%d reason=item shaped for the other bucket",
            len(output.operations) - len(kept),
        )
        return output.model_copy(update={"operations": kept}).model_dump(by_alias=True)
    return output.model_dump(by_alias=True)


async def _call_llm_for_profile_compact[ReplyT: _AnyCompactOutput](
    llm: LLMClient, messages: list[LLMMessage], *, reply: type[ReplyT]
) -> dict[str, Any]:
    """Request a COMPACT reply and return it validated, dumped to a dict.

    Raises:
        LLMOutputError: If the reply is not a valid COMPACT output.
    """
    text = await llm.achat(messages=messages, response_format=reply)
    return parse_reply(reply, text, operator="profile compact").model_dump(by_alias=True)


async def _call_llm_for_profile_regroup[ReplyT: _AnyRegroupOutput](
    llm: LLMClient, messages: list[LLMMessage], *, reply: type[ReplyT], bucket: str
) -> dict[str, Any]:
    """Request a REGROUP reply for a group of ``bucket`` and return it validated, dumped to a dict.

    A rewrite reply (the MemCell path) is held to ``bucket`` here. An operations reply (the Episode path) is not: an
    operation on the other bucket or carrying the other bucket's item is refused, alone, when it is applied.

    Raises:
        LLMOutputError: If the reply is not a valid REGROUP output, or a rewritten item is not shaped for ``bucket``.
    """
    text = await llm.achat(messages=messages, response_format=reply)
    output = parse_reply(reply, text, operator="profile regroup")
    if not isinstance(output, _RegroupOpsOutput):
        _check_item_shapes(output, bucket=bucket)
    return output.model_dump(by_alias=True)


async def _call_llm_for_do_not_record_merge(llm: LLMClient, messages: list[LLMMessage]) -> list[dict[str, Any]]:
    """Request the do-not-record merge reply and return its operations, validated and dumped to dicts.

    Raises:
        LLMOutputError: If the reply is not a valid merge output.
    """
    text = await llm.achat(messages=messages, response_format=_MatterMergeOutput)
    output = parse_reply(_MatterMergeOutput, text, operator="profile do-not-record merge")
    return [op.model_dump() for op in output.operations]


def _check_item_shapes(output: _RegroupOutput | _MemCellRegroupOutput, *, bucket: str) -> None:
    """Reject a regrouped item that is not shaped for the group's bucket; regroup never moves an item across.

    Raises:
        LLMOutputError: Naming each mismatched item by position and the shape it carried.
    """
    expected = _BUCKET_SHAPES[bucket]
    mismatches = [
        f"items.{position}: a {type(item).__name__}"
        for position, item in enumerate(output.items)
        if not isinstance(item, expected)
    ]
    if mismatches:
        raise LLMOutputError(
            f"profile regroup reply for a {bucket} group holds items not shaped as {expected.__name__}: "
            f"{'; '.join(mismatches)}"
        )
