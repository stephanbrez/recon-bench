"""
ProfileResult and supporting dataclasses for timing and memory reports.

Collects hierarchical timing and GPU memory measurements into a structured
report with a human-readable tree summary.
"""
from __future__ import annotations

import dataclasses

from ..utils import format as _fmt


@dataclasses.dataclass(slots=True)
class TimingEntry:
    """
    A single timing measurement, optionally containing nested sub-sections.

    Parameters
    ----------
    name : str
        Label for this timed section.
    duration_s : float
        Wall-clock duration in seconds.
    children : list[TimingEntry]
        Nested sub-sections measured within this section.
    """
    name: str
    duration_s: float
    children: list[TimingEntry] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class MemoryEntry:
    """
    A single GPU memory measurement for a profiled section.

    Parameters
    ----------
    name : str
        Label for this section.
    peak_mb : float
        Peak GPU memory allocated during this section, in MiB.
    delta_mb : float
        Net change in GPU memory (exit − entry), in MiB.
    children : list[MemoryEntry]
        Nested sub-sections measured within this section.
    """
    name: str
    peak_mb: float
    delta_mb: float
    children: list[MemoryEntry] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class ProfileResult:
    """
    Aggregated profiling report returned by ``evaluate(..., profile=True)``.

    Parameters
    ----------
    timing : list[TimingEntry]
        Top-level timing entries (tree roots).
    memory : list[MemoryEntry]
        Top-level GPU memory entries (tree roots).
    cuda_available : bool
        Whether CUDA was available during profiling. When False, all memory
        values are zero.
    """
    timing: list[TimingEntry]
    memory: list[MemoryEntry]
    cuda_available: bool

    def summary(self) -> str:
        """
        Return a human-readable tree report of timing and memory usage.

        Returns
        -------
        str
            Multi-line string formatted as an indented tree.
        """
        lines: list[str] = []

        # ─── Timing tree ───
        lines.append("⏱  Timing")
        for i, entry in enumerate(self.timing):
            is_last = i == len(self.timing) - 1
            _format_timing(entry, lines, prefix="", is_last=is_last)

        # ─── Memory tree ───
        lines.append("")
        if self.cuda_available:
            lines.append("🔋 GPU Memory")
            for i, entry in enumerate(self.memory):
                is_last = i == len(self.memory) - 1
                _format_memory(entry, lines, prefix="", is_last=is_last)
        else:
            lines.append("🔋 GPU Memory (CUDA not available)")

        return "\n".join(lines)


# ===== Formatting Helpers =====


def _format_timing(
    entry: TimingEntry,
    lines: list[str],
    prefix: str,
    is_last: bool,
) -> None:
    """Recursively format a TimingEntry with tree decorations."""
    label = f"{entry.name}: {entry.duration_s:.4f}s"
    _fmt.format_tree_node(label, lines, prefix, is_last)
    new_prefix = _fmt.child_prefix(prefix, is_last)
    for i, child in enumerate(entry.children):
        _format_timing(
            child, lines, new_prefix,
            is_last=(i == len(entry.children) - 1),
        )


def _format_memory(
    entry: MemoryEntry,
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
        _format_memory(
            child, lines, new_prefix,
            is_last=(i == len(entry.children) - 1),
        )
