"""HTML visualisation for ``list[KnowledgeMemory]``, in memsys_enterprise viz layout.

Self-contained — vanilla HTML + inline CSS + a tiny inline JS block for
the click-to-scroll / click-legend-to-highlight interactions. No external
assets, no build step. Open the resulting file in any browser.

Sections (top to bottom):

1. Header — title + stats (N atom blocks / N topics / covered / uncovered).
2. Legend — colour swatch per topic, hierarchically indented with tree
   connectors. Click a row to highlight every atom that belongs to that
   topic in the Full Content section.
3. Topic summary panel — appears after a legend row is selected.
4. Block Map — nested coloured boxes. Each small square = 1 atom; the
   square's tooltip shows the atom text.
5. Full Content — every atom rendered in order, with a coloured left
   border indicating its leaf topic and a topic path header.

Used by ``run_one_doc.py --html``. Renderer lives under ``examples/``
because EverAlgo itself is stateless / does not emit user-facing
artefacts; rendering is caller territory.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from everalgo.types import KnowledgeMemory


# Golden-angle HSL palette — same recipe as memsys_enterprise so colours
# are stable and visually distinct even for ~30 sibling topics.
_GOLDEN_ANGLE = 137.508


def _hsl(i: int) -> str:
    """Generate an HSL color string from an index using the golden-angle palette."""
    h = (i * _GOLDEN_ANGLE) % 360
    s = 65 + (i % 3) * 10
    light = 45 + (i % 2) * 10
    return f"hsl({h:.0f},{s}%,{light}%)"


def _esc(s: str) -> str:
    """HTML-escape a string, defaulting to empty on None."""
    return html.escape(s or "")


# ── reconstruct a nested topic tree out of the flat KnowledgeMemory list ──


def _to_nested(km: KnowledgeMemory, by_index: dict[int, KnowledgeMemory]) -> dict[str, Any]:
    """Reconstruct a nested dict tree from a flat KnowledgeMemory node."""
    return {
        "topic": km.topic,
        "summary": km.summary,
        "block_refs": list(km.block_refs),
        "children": [_to_nested(by_index[i], by_index) for i in km.children_index],
    }


def _flat_with_depth(topics: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    """DFS-flatten nested topic dicts, returning ``(flat_topics, depths)``."""
    flat: list[dict[str, Any]] = []
    depths: list[int] = []

    def _walk(t: dict[str, Any], depth: int) -> None:
        flat.append({k: v for k, v in t.items() if k != "children"})
        depths.append(depth)
        for child in t.get("children", []):
            _walk(child, depth + 1)

    for t in topics:
        _walk(t, 0)
    return flat, depths


# ── pieces of the HTML body ──


def _legend_html(
    flat_topics: list[dict[str, Any]],
    depths: list[int],
    colors: list[str],
) -> str:
    items: list[str] = []
    for ti, tp in enumerate(flat_topics):
        c = colors[ti]
        depth = depths[ti]
        indent = f"margin-left:{depth * 24}px;" if depth > 0 else ""
        connector = ""
        if depth > 0:
            is_last = True
            for j in range(ti + 1, len(depths)):
                if depths[j] == depth:
                    is_last = False
                    break
                if depths[j] < depth:
                    break
            connector = "└─ " if is_last else "├─ "
        refs = tp.get("block_refs", [])
        items.append(
            f'<span class="legend-item" data-topic="{ti}" '
            f'title="{_esc(tp.get("summary", ""))}" style="{indent}">'
            f'<span class="legend-color" style="background:{c}"></span>'
            f'<span class="legend-label">{connector}{_esc(tp.get("topic", ""))}</span>'
            f'<span class="legend-count">({len(refs)} blocks)</span>'
            "</span>",
        )
    return "\n".join(items)


def _render_blocks_with_gaps(
    refs: list[int],
    color: str,
    block_map: dict[int, str],
    *,
    is_parent: bool = False,
) -> str:
    if not refs:
        return ""
    sorted_refs = sorted(refs)
    parts: list[str] = []
    prev: int | None = None
    for r in sorted_refs:
        if prev is not None and r > prev + 1:
            gap = r - prev - 1
            parts.append(
                f'<div class="gap" title="{gap} blocks skipped">··{gap}··</div>',
            )
        opacity = "opacity:0.35;border:1px dashed rgba(0,0,0,0.3);" if is_parent else ""
        text = _esc(block_map.get(r, "")[:80])
        parts.append(
            f'<div class="blk" style="background:{color};{opacity}" title="[{r}] {text}" data-block="{r}"></div>',
        )
        prev = r
    return "".join(parts)


def _render_nested_boxes(
    topics: list[dict[str, Any]],
    block_map: dict[int, str],
    colors: list[str],
) -> str:
    ti_counter = [0]

    def _walk(t: dict[str, Any]) -> str:
        ti = ti_counter[0]
        ti_counter[0] += 1
        c = colors[ti % len(colors)]
        refs = sorted(t.get("block_refs", []))
        children = t.get("children", [])
        tname = _esc(t.get("topic", ""))

        if not children:
            cells = _render_blocks_with_gaps(refs, c, block_map)
            return (
                f'<div class="leaf" style="border-color:{c}">'
                f'<div class="lbl" style="color:{c}">{tname} '
                f'<span class="cnt">({len(refs)})</span></div>'
                f'<div class="blks">{cells}</div></div>'
            )
        own = ""
        if refs:
            own = '<div class="own">' + _render_blocks_with_gaps(refs, c, block_map, is_parent=True) + "</div>"
        ch_html = "".join(_walk(ch) for ch in children)
        return (
            f'<div class="grp" style="border-color:{c}">'
            f'<div class="gl" style="background:{c}">{tname} '
            f'<span style="opacity:0.7">({len(refs)}+children)</span></div>'
            f'{own}<div class="ch">{ch_html}</div></div>'
        )

    return "".join(_walk(t) for t in topics)


def _build_block_to_topics(flat_topics: list[dict[str, Any]]) -> dict[int, list[int]]:
    """Map each block id to the list of topic indices that reference it."""
    out: dict[int, list[int]] = {}
    for ti, t in enumerate(flat_topics):
        for r in t.get("block_refs", []):
            out.setdefault(r, []).append(ti)
    return out


def _build_parent_chain(flat_topics: list[dict[str, Any]], depths: list[int]) -> dict[int, list[str]]:
    """Build ancestor topic-name chains for each flat topic index."""
    chain: dict[int, list[str]] = {}
    for ti, _topic in enumerate(flat_topics):
        ancestors: list[str] = []
        depth = depths[ti]
        for prev_ti in range(ti - 1, -1, -1):
            if depths[prev_ti] < depth:
                ancestors.append(flat_topics[prev_ti].get("topic", ""))
                depth = depths[prev_ti]
            if depth == 0:
                break
        ancestors.reverse()
        chain[ti] = ancestors
    return chain


def _block_path_tag(
    leaf_ti: int,
    flat_topics: list[dict[str, Any]],
    parent_chain: dict[int, list[str]],
    colors: list[str],
) -> tuple[str, str]:
    """Return ``(border-style, tag-html)`` for a block owned by ``leaf_ti``."""
    c = colors[leaf_ti]
    leaf_name = flat_topics[leaf_ti].get("topic", "")
    ancestors = parent_chain.get(leaf_ti, [])
    if ancestors:
        path = " &rsaquo; ".join([_esc(a) for a in ancestors] + [f"<b>{_esc(leaf_name)}</b>"])
    else:
        path = _esc(leaf_name)
    return f"border-left: 4px solid {c}", f'<span class="block-topic" style="color:{c}">{path}</span>'


def _content_html(
    flat_topics: list[dict[str, Any]],
    depths: list[int],
    colors: list[str],
    block_map: dict[int, str],
    max_block_id: int,
) -> str:
    block_to_topics = _build_block_to_topics(flat_topics)
    block_to_leaf: dict[int, int] = {}
    for idx in range(max_block_id + 1):
        tis = block_to_topics.get(idx, [])
        if tis:
            block_to_leaf[idx] = max(tis, key=lambda ti: depths[ti])
    parent_chain = _build_parent_chain(flat_topics, depths)

    rows: list[str] = []
    for idx in range(max_block_id + 1):
        text = block_map.get(idx, "")
        leaf_ti = block_to_leaf.get(idx)
        tis = block_to_topics.get(idx, [])
        if leaf_ti is not None:
            border, tag = _block_path_tag(leaf_ti, flat_topics, parent_chain, colors)
        elif tis:
            c = colors[tis[0]]
            names = ", ".join(flat_topics[ti].get("topic", "") for ti in tis)
            border = f"border-left: 4px solid {c}"
            tag = f'<span class="block-topic" style="color:{c}">{_esc(names)}</span>'
        else:
            border = "border-left: 4px solid #ddd"
            tag = '<span class="block-topic uncovered-label">uncovered</span>'

        rows.append(
            f'<div class="block-row" id="block-{idx}" style="{border}">'
            f'<div class="block-id">{idx}</div>'
            f'<div class="block-content">{tag}'
            f'<div class="block-text">{_esc(text)}</div>'
            "</div></div>",
        )
    return "\n".join(rows)


_CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f5f5; color: #333; padding: 24px; max-width: 1400px; margin: 0 auto; }
h1 { font-size: 20px; margin-bottom: 6px; }
.stats { font-size: 13px; color: #666; margin-bottom: 20px; }
.section-label { font-size: 12px; font-weight: 600; color: #888; text-transform: uppercase;
                 letter-spacing: 0.5px; margin: 16px 0 6px; }

/* Legend */
.legend { display: flex; flex-wrap: wrap; gap: 8px 16px; margin-bottom: 20px;
          padding: 12px; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.legend.hierarchical { flex-direction: column; gap: 2px; }
.legend-item { display: flex; align-items: center; gap: 5px; cursor: pointer;
              padding: 2px 6px; border-radius: 4px; transition: background 0.15s; }
.legend-item:hover { background: #f0f0f0; }
.legend-item.active { background: #e8e8e8; }
.legend-color { width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }
.legend-label { font-size: 13px; font-weight: 500; }
.legend-count { font-size: 11px; color: #999; }

/* Topic summary panel */
.topic-summary { padding: 10px 14px; background: #fff; border-radius: 8px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 12px;
                 font-size: 13px; line-height: 1.5; border-left: 4px solid #ccc; display: none; }
.topic-summary .ts-name { font-weight: 600; margin-bottom: 4px; }
.topic-summary .ts-text { color: #555; }

/* Nested box map */
.box-map { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-start;
           background: #fff; padding: 16px; border-radius: 8px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }
.grp { border: 2px solid; border-radius: 8px; padding: 6px; }
.gl { color: #fff; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 4px;
      margin-bottom: 6px; text-shadow: 0 1px 1px rgba(0,0,0,0.3); }
.ch { display: flex; flex-wrap: wrap; gap: 6px; }
.own { display: flex; flex-wrap: wrap; gap: 2px; margin-bottom: 6px; align-items: center; }
.leaf { border: 2px solid; border-radius: 6px; padding: 4px 6px; }
.lbl { font-size: 11px; font-weight: 600; margin-bottom: 3px; white-space: nowrap; }
.cnt { font-weight: 400; opacity: 0.6; }
.blks { display: flex; flex-wrap: wrap; gap: 2px; align-items: center; }
.blk { width: 16px; height: 16px; border-radius: 2px; cursor: pointer; flex-shrink: 0; position: relative; }
.blk:hover { opacity: 0.85; transform: scale(1.4); z-index: 10; }
.blk:hover::after { content: attr(data-block); position: absolute; bottom: 100%; left: 50%;
                    transform: translateX(-50%); background: #333; color: #fff; font-size: 10px;
                    padding: 1px 4px; border-radius: 3px; white-space: nowrap; pointer-events: none; }
.gap { font-size: 9px; color: #999; padding: 0 3px; white-space: nowrap; user-select: none; }

/* Content */
.content { background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
           overflow: hidden; }
.block-row { display: flex; padding: 8px 12px; border-bottom: 1px solid #f0f0f0;
            transition: background 0.15s, opacity 0.15s; }
.block-row:hover { background: #fafafa; }
.block-row.dimmed { opacity: 0.2; }
.block-row.highlighted { background: #fffde7; }
.block-id { width: 36px; flex-shrink: 0; font-size: 11px; color: #999;
           font-family: monospace; padding-top: 2px; text-align: right; margin-right: 10px; }
.block-content { flex: 1; min-width: 0; }
.block-topic { font-size: 11px; font-weight: 600; margin-bottom: 2px; display: inline-block; }
.uncovered-label { color: #ccc; }
.block-text { font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
"""


_JS_TEMPLATE = """\
// Click block square -> scroll to content
document.querySelectorAll('.blk').forEach(blk => {
    blk.addEventListener('click', () => {
        const bid = blk.dataset.block;
        if (bid) {
            const row = document.getElementById('block-' + bid);
            if (row) row.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    });
});

// Click legend -> show summary + highlight content
const topicMeta = __TOPIC_META__;
const topicRefs = __TOPIC_REFS__;
const summaryEl = document.getElementById('topic-summary');
let activeTopic = null;

document.querySelectorAll('.legend-item').forEach(item => {
    item.addEventListener('click', () => {
        const ti = parseInt(item.dataset.topic);
        if (activeTopic === ti) {
            activeTopic = null;
            item.classList.remove('active');
            summaryEl.style.display = 'none';
            document.querySelectorAll('.block-row').forEach(r => r.classList.remove('dimmed', 'highlighted'));
        } else {
            activeTopic = ti;
            document.querySelectorAll('.legend-item').forEach(l => l.classList.remove('active'));
            item.classList.add('active');
            const meta = topicMeta[ti];
            const c = item.querySelector('.legend-color').style.background;
            summaryEl.style.display = 'block';
            summaryEl.style.borderLeftColor = c;
            summaryEl.innerHTML = '<div class="ts-name">' + meta.name + '</div><div class="ts-text">' + meta.summary + '</div>';
            const refsSet = new Set(topicRefs[ti].map(String));
            document.querySelectorAll('.block-row').forEach(r => {
                const bid = r.id.replace('block-', '');
                if (refsSet.has(bid)) {
                    r.classList.remove('dimmed');
                    r.classList.add('highlighted');
                } else {
                    r.classList.add('dimmed');
                    r.classList.remove('highlighted');
                }
            });
        }
    });
});
"""


def _prepare_render_context(
    memories: Iterable[KnowledgeMemory],
    atoms: list[tuple[int, str]],
) -> dict[str, Any] | None:
    """Convert memories and atoms into all pre-rendered HTML fragments and metadata needed by the template."""
    nodes = list(memories)
    if not nodes:
        return None

    by_index = {km.topic_index: km for km in nodes}
    root = nodes[0]
    top_level: list[dict[str, Any]] = [_to_nested(by_index[i], by_index) for i in root.children_index]

    flat_topics, depths = _flat_with_depth(top_level)
    has_hierarchy = any(d > 0 for d in depths)
    colors = [_hsl(i) for i in range(len(flat_topics))]

    block_map = dict(atoms)
    max_block_id = max(block_map.keys()) if block_map else 0

    return {
        "root": root,
        "top_level": top_level,
        "flat_topics": flat_topics,
        "depths": depths,
        "has_hierarchy": has_hierarchy,
        "colors": colors,
        "block_map": block_map,
        "max_block_id": max_block_id,
        "legend_html": _legend_html(flat_topics, depths, colors),
        "nested_html": _render_nested_boxes(top_level, block_map, colors),
        "content_html": _content_html(flat_topics, depths, colors, block_map, max_block_id),
    }


def _render_uncovered_section(
    flat_topics: list[dict[str, Any]],
    max_block_id: int,
    block_map: dict[int, str],
) -> tuple[str, set[int], list[int]]:
    """Compute uncovered blocks and render the uncovered-summary box HTML."""
    covered: set[int] = set()
    for t in flat_topics:
        covered.update(t.get("block_refs", []))
    uncovered_ids = sorted(set(range(max_block_id + 1)) - covered)
    if not uncovered_ids:
        return "", covered, uncovered_ids
    cells = _render_blocks_with_gaps(uncovered_ids, "#ccc", block_map)
    uncovered_html = (
        f'<div class="leaf" style="border-color:#999">'
        f'<div class="lbl" style="color:#999">Uncovered ({len(uncovered_ids)})</div>'
        f'<div class="blks">{cells}</div></div>'
    )
    return uncovered_html, covered, uncovered_ids


def render_html(
    memories: Iterable[KnowledgeMemory],
    atoms: list[tuple[int, str]],
    *,
    source_label: str = "",
    model: str = "",
) -> str:
    """Render a full standalone HTML document mirroring the memsys_enterprise viz layout.

    Args:
        memories: DFS-ordered output of ``KnowledgeExtractor.aextract``. ``memories[0]`` must be the synthetic
            doc root (``depth=0``, ``parent_index=None``).
        atoms: Original atom blocks ``(id, text)`` from ``split_content_to_blocks``. Required so the visualisation
            can show the actual atom text inside the block map and content sections.
        source_label: Free-form label shown in the header (e.g. fixture path).
        model: LLM model identifier — accepted for caller-side parity but not rendered (header carries only the
            document-derived title + stats).

    Returns:
        Complete standalone HTML document string.
    """
    _ = model  # parity with run_one_doc; not rendered in the header.
    ctx = _prepare_render_context(memories, atoms)
    if ctx is None:
        return "<!DOCTYPE html><html><body><p>(no memories)</p></body></html>"

    flat_topics, block_map, max_block_id = ctx["flat_topics"], ctx["block_map"], ctx["max_block_id"]
    uncovered_html, covered, uncovered_ids = _render_uncovered_section(flat_topics, max_block_id, block_map)

    n_blocks = len(block_map)
    n_flat = len(flat_topics)
    n_covered = len(covered & set(range(max_block_id + 1)))
    title = _esc(ctx["root"].topic) or _esc(source_label) or "Untitled"

    topic_meta_json = json.dumps(
        {ti: {"name": t.get("topic", ""), "summary": t.get("summary", "")} for ti, t in enumerate(flat_topics)},
        ensure_ascii=False,
    )
    topic_refs_json = json.dumps({ti: t.get("block_refs", []) for ti, t in enumerate(flat_topics)})
    js = _JS_TEMPLATE.replace("__TOPIC_META__", topic_meta_json).replace("__TOPIC_REFS__", topic_refs_json)

    legend_cls = "legend hierarchical" if ctx["has_hierarchy"] else "legend"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Topic Segmentation: {title}</title>
<style>
{_CSS}</style>
</head>
<body>
<h1>Topic Segmentation: {title}</h1>
<div class="stats">{n_blocks} blocks, {n_flat} topics, {n_covered} covered, {len(uncovered_ids)} uncovered</div>

<div class="section-label">Legend</div>
<div class="{legend_cls}">
{ctx["legend_html"]}
</div>

<div id="topic-summary" class="topic-summary"></div>

<div class="section-label">Block Map (nested boxes = topic hierarchy, each square = 1 block)</div>
<div class="box-map">
{ctx["nested_html"]}
{uncovered_html}
</div>

<div class="section-label">Full Content</div>
<div class="content">
{ctx["content_html"]}
</div>

<script>
{js}
</script>
</body>
</html>
"""
