"""
Profiling dataclasses for timing and memory reports.

Defines the structured types returned by the profiling subsystem.
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
        Return a compact table of total time and peak memory per top-level
        section.

        Returns
        -------
        str
            Multi-line formatted string.
        """
        sections: list[str] = []

        # ─── Timing table ───
        if self.timing:
            n = len(self.timing)
            total_s = sum(e.duration_s for e in self.timing)
            mean_s = total_s / n
            headers = ["Section", "Time (s)"]
            rows = [
                [e.name, f"{e.duration_s:.4f}"]
                for e in self.timing
            ]
            rows.append(None)
            rows.append(["Mean", f"{mean_s:.4f}"])
            rows.append(["Total", f"{total_s:.4f}"])
            sections.append("⏱  Timing")
            sections.append(_fmt.format_table(headers, rows))

        # ─── Memory table ───
        if self.cuda_available and self.memory:
            n = len(self.memory)
            mean_peak = sum(e.peak_mb for e in self.memory) / n
            mean_delta = sum(e.delta_mb for e in self.memory) / n
            headers = ["Section", "Peak (MiB)", "Delta (MiB)"]
            rows = [
                [e.name, f"{e.peak_mb:.1f}", f"{e.delta_mb:+.1f}"]
                for e in self.memory
            ]
            rows.append(None)
            rows.append(["Mean", f"{mean_peak:.1f}", f"{mean_delta:+.1f}"])
            sections.append("")
            sections.append("🔋 GPU Memory")
            sections.append(_fmt.format_table(headers, rows))
        elif not self.cuda_available:
            sections.append("")
            sections.append("🔋 GPU Memory (CUDA not available)")

        return "\n".join(sections)

    def detail(self) -> str:
        """
        Return a human-readable tree report of timing and memory usage,
        including all nested sub-sections.

        Returns
        -------
        str
            Multi-line string formatted as an indented tree.
        """
        from . import _format

        lines: list[str] = []

        # ─── Timing tree ───
        lines.append("⏱  Timing")
        for i, entry in enumerate(self.timing):
            is_last = i == len(self.timing) - 1
            _format.format_timing(entry, lines, prefix="", is_last=is_last)

        # ─── Memory tree ───
        lines.append("")
        if self.cuda_available:
            lines.append("🔋 GPU Memory")
            for i, entry in enumerate(self.memory):
                is_last = i == len(self.memory) - 1
                _format.format_memory(entry, lines, prefix="", is_last=is_last)
        else:
            lines.append("🔋 GPU Memory (CUDA not available)")

        return "\n".join(lines)
