"""
Tree-formatting helpers for profiling entries.

Used by ``ProfileResult.detail()`` to render hierarchical timing and
memory reports as indented trees.
"""
from __future__ import annotations

from ..utils import format as _fmt
from . import _types


def format_timing(
    entry: _types.TimingEntry,
    lines: list[str],
    prefix: str,
    is_last: bool,
) -> None:
    """Recursively format a TimingEntry with tree decorations."""
    label = f"{entry.name}: {entry.duration_s:.4f}s"
    _fmt.format_tree_node(label, lines, prefix, is_last)
    new_prefix = _fmt.child_prefix(prefix, is_last)
    for i, child in enumerate(entry.children):
        format_timing(
            child, lines, new_prefix,
            is_last=(i == len(entry.children) - 1),
        )


def format_memory(
    entry: _types.MemoryEntry,
    lines: list[str],
    prefix: str,
    is_last: bool,
) -> None:
    """Recursively format a MemoryEntry with tree decorations."""
    label = (
        f"{entry.name}: "
        f"peak {entry.peak_mb:.1f} MiB, "
        f"delta {entry.delta_mb:+.1f} MiB"
    )
    _fmt.format_tree_node(label, lines, prefix, is_last)
    new_prefix = _fmt.child_prefix(prefix, is_last)
    for i, child in enumerate(entry.children):
        format_memory(
            child, lines, new_prefix,
            is_last=(i == len(entry.children) - 1),
        )
