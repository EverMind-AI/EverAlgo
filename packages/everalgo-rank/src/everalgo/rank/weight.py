"""Weight tools — single-list / multi-list LR + LR coefficient supply."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.types import Candidate

__all__ = [
    "LRCoefs",
    "default_lr_coefs",
    "multi_field_weighting",
    "weighted_score",
]


def weighted_score(
    items: Sequence[Candidate],
    *,
    fields: dict[str, float],
    intercept: float = 0.0,
) -> list[Candidate]:
    """Single-list LR weighting — per-item sigmoid of a linear combination.

    For each item::

        logit = sum(metadata[k] * coef for k, coef in fields.items()) + intercept
        new_score = 1 / (1 + exp(-logit))

    Use cases:

    - **case facade** — ``base_weight=1.0, fields={"quality_score": w}`` so the
      fusion score stays in the logit and ``quality_score`` is an additive
      feature.
    - **skill facade** — ``base_weight=0.0, fields={"maturity_score": ws,
      "confidence": wc}`` so the fusion score drops out and only business
      fields drive the logit.

    Args:
        items: Candidates whose ``.score`` is the base feature.
        fields: ``{metadata_key: coef}`` linear-combo terms. Missing keys count
            as 0 — never raises.
        base_weight: Coefficient on ``.score``. ``1.0`` keeps the fusion order
            in the logit; ``0.0`` discards it.
        intercept: Logit bias. Default ``0.0``; callers with a trained intercept
            (e.g. from ``LRCoefs``) can pass it explicitly.

    Returns
    -------
        New list of Candidates with ``.score`` set to the sigmoid output.
        **Not** sorted; caller decides whether to sort.
    """
    out: list[Candidate] = []
    for item in items:
        logit = sum(_get_scalar(item.metadata, key) * coef for key, coef in fields.items()) + intercept
        prob = 1.0 / (1.0 + math.exp(-logit))
        out.append(item.model_copy(update={"score": prob}))
    return out


def multi_field_weighting(
    sources: dict[str, Sequence[Candidate]],
    *,
    weights: dict[str, float] | None = None,
    intercept: float = 0.0,
    coefs: LRCoefs | None = None,
) -> list[Candidate]:
    """Multi-source LR fusion — N ranked lists → single LR probability per doc.

    For each candidate id appearing in any source::

        logit = sum(score_in_source[name] * weights[name] for name in weights)
              + intercept
        prob  = 1 / (1 + exp(-logit))

    Two modes — selected by whether ``weights`` is supplied:

    - **Generic mode** (``weights`` is not ``None``): use caller-supplied
      ``weights`` / ``intercept`` directly. Use for ad-hoc multi-route fusion
      (e.g. emb + bm25 + recency with bespoke coefficients).
    - **LR-trained mode** (``weights is None``): use trained LR coefficients
      via ``_lrcoefs_to_weights(coefs)``. ``coefs=None`` resolves to
      ``default_lr_coefs()``; ``coefs=LRCoefs(...)`` overrides. Expects
      ``sources`` to contain keys ``"emb"`` and ``"bm25"``. Used internally by
      ``fusion.lr`` and ``fusion.cosine_to_lr_score``.

    Args:
        sources: ``{source_name: ranked_list}``. Each list is descending.
        weights: ``{source_name: coef}``. Sources not in weights contribute 0.
            Sources in weights but not in sources contribute 0.
        intercept: Logit bias for generic mode. Default ``0.0``.
        coefs: LR-trained coefficients for LR mode. Ignored if ``weights`` is
            supplied.

    Returns
    -------
        New list of Candidates, ``.score`` set to the sigmoid output, sorted
        descending by score.
    """
    if weights is None:
        weights, intercept = _lrcoefs_to_weights(coefs)

    by_id: dict[str, dict[str, float]] = {}
    doc_map: dict[str, Candidate] = {}

    for name, ranked in sources.items():
        for item in ranked:
            if not item.id:
                continue
            doc_map.setdefault(item.id, item)
            by_id.setdefault(item.id, {})[name] = item.score

    probs: list[tuple[str, float]] = []
    for doc_id, score_map in by_id.items():
        logit = intercept + sum(score_map.get(name, 0.0) * coef for name, coef in weights.items())
        prob = 1.0 / (1.0 + math.exp(-logit))
        probs.append((doc_id, prob))

    probs.sort(key=lambda kv: kv[1], reverse=True)
    return [doc_map[doc_id].model_copy(update={"score": prob}) for doc_id, prob in probs]


class LRCoefs(NamedTuple):
    """Logistic Regression training coefficients (emb + bm25 + intercept).

    These are the trained coefficients for the 2-source LR in ``fusion.lr`` /
    ``fusion.cosine_to_lr_score``. Defaults match the production line in
    ``memsys_enterprise/.../mrag_fusion.py``.

    Customise per call::

        from everalgo.rank import fusion, weight

        coefs = weight.LRCoefs(emb_coef=..., bm25_coef=..., intercept=...)
        fused = fusion.lr(emb_results, bm25_results, coefs=coefs)
    """

    emb_coef: float = 6.27473151675093
    bm25_coef: float = 0.09395183408310023
    intercept: float = -4.858095765012703


def default_lr_coefs() -> LRCoefs:
    """Return the current default LR coefficients.

    Implemented as a function (not a module-level constant) for two reasons:

    1. Future-proofing — when scene-specific defaults appear
       (``case_default_coefs`` etc.), a function can branch internally without
       breaking callers; a constant cannot.
    2. Monkey-patch friendly — callers can replace this function at startup,
       and ``fusion.lr`` / ``fusion.cosine_to_lr_score`` pick up the override
       automatically because they invoke it lazily when ``coefs=None``.
    """
    return LRCoefs()


# ─── Internal helpers ───────────────────────────────────────────────────────


def _lrcoefs_to_weights(coefs: LRCoefs | None) -> tuple[dict[str, float], float]:
    """Resolve ``LRCoefs`` to a ``(weights, intercept)`` pair for the 2-source LR.

    Centralises the ``default_lr_coefs()`` lookup so callers of the
    ``fusion.*`` family never touch the default-coefs function directly. Pass
    ``None`` to get the production defaults; pass a custom ``LRCoefs`` to
    override.
    """
    resolved = coefs or default_lr_coefs()
    return {"emb": resolved.emb_coef, "bm25": resolved.bm25_coef}, resolved.intercept


def _get_scalar(metadata: dict[str, Any], key: str) -> float:
    """Extract a scalar metadata value; treat missing / non-numeric as 0."""
    value = metadata.get(key, 0)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
