"""
Plain-text table and tree formatting utilities.

Provides reusable formatters for terminal output used by both
``ProfileResult.summary()`` and ``EvalResult.summary()``.
"""
from __future__ import annotations


# ===== Table Formatting =====

def format_table(
    headers: list[str],
    rows: list[list[str] | None],
    alignment: list[str] | None = None,
) -> str:
    """
    Render a plain-text table with box-drawing borders.

    Parameters
    ----------
    headers : list[str]
        Column header labels.
    rows : list[list[str]]
        Row data. Each inner list must have the same length as *headers*.
    alignment : list[str] or None
        Per-column alignment: ``"<"`` left, ``">"`` right, ``"^"`` center.
        Defaults to left-aligned for the first column and right-aligned for
        the rest (typical for label + numeric data).

    Returns
    -------
    str
        Multi-line formatted table string.
    """
    n_cols = len(headers)

    if alignment is None:
        alignment = ["<"] + [">"] * (n_cols - 1)

    # ─── Compute column widths ───
    widths = [len(h) for h in headers]
    for row in rows:
        if row is None:
            continue
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(cells: list[str]) -> str:
        parts: list[str] = []
        for cell, width, align in zip(cells, widths, alignment):
            parts.append(f"{cell:{align}{width}}")
        return "  │  ".join(parts)

    # ─── Build table ───
    lines: list[str] = []
    lines.append(_fmt_row(headers))

    separator_parts = ["─" * w for w in widths]
    lines.append("──┼──".join(separator_parts))

    separator = "──┼──".join(separator_parts)
    for row in rows:
        if row is None:
            lines.append(separator)
        else:
            lines.append(_fmt_row(row))

    return "\n".join(lines)


# ===== Tree Formatting =====

_BRANCH = "├── "
_CORNER = "└── "
_PIPE = "│   "
_SPACE = "    "


def format_tree_node(
    label: str,
    lines: list[str],
    prefix: str,
    is_last: bool,
) -> None:
    """
    Append a single tree node line with box-drawing connectors.

    Parameters
    ----------
    label : str
        Formatted text for this node (e.g. ``"forward: 0.1023s"``).
    lines : list[str]
        Accumulator list; the formatted line is appended here.
    prefix : str
        Indentation prefix inherited from parent nodes.
    is_last : bool
        Whether this is the last sibling at its level.

    Returns
    -------
    None
    """
    connector = _CORNER if is_last else _BRANCH
    lines.append(f"{prefix}{connector}{label}")


def child_prefix(prefix: str, is_last: bool) -> str:
    """
    Return the prefix string for children of a node.

    Parameters
    ----------
    prefix : str
        Current node's prefix.
    is_last : bool
        Whether the current node is the last sibling.

    Returns
    -------
    str
    """
    return prefix + (_SPACE if is_last else _PIPE)
