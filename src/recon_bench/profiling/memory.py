"""
GPU memory tracker with hierarchical section support.

Records peak and delta CUDA memory usage per named section using
``torch.cuda`` memory statistics. Gracefully degrades to zero-valued
reports when CUDA is not available.
"""
from __future__ import annotations

import contextlib

import torch

from . import profile as _profile

_BYTES_PER_MIB = 1024 * 1024


class MemoryTracker:
    """
    Track GPU memory allocation per named section.

    Records peak memory usage and net allocation change for each section.
    Sections may be nested to capture fine-grained breakdowns.

    When CUDA is not available, all measurements report zero and
    ``cuda_available`` is False.

    Parameters
    ----------
    enabled : bool
        If False, ``section()`` becomes a no-op and ``get_report()`` returns
        an empty list. Allows toggling profiling without restructuring
        call sites. Default True.

    Examples
    --------
    >>> tracker = MemoryTracker()
    >>> with tracker.section("forward_pass"):
    ...     output = model(input_tensor)
    >>> report = tracker.get_report()
    """

    def __init__(self, enabled: bool = True) -> None:
        self._cuda_available = torch.cuda.is_available()
        self._enabled = enabled
        self._root_entries: list[_profile.MemoryEntry] = []
        self._stack: list[_profile.MemoryEntry] = []

    @property
    def cuda_available(self) -> bool:
        """Whether CUDA was available when this tracker was created."""
        return self._cuda_available

    @contextlib.contextmanager
    def section(self, name: str):
        """
        Track peak and delta GPU memory for a named section.

        Parameters
        ----------
        name : str
            Label for this memory section.

        Yields
        ------
        None
        """
        if not self._enabled:
            yield
            return

        entry = _profile.MemoryEntry(name=name, peak_mb=0.0, delta_mb=0.0)

        # ─── Attach to parent or root ───
        if self._stack:
            self._stack[-1].children.append(entry)
        else:
            self._root_entries.append(entry)

        self._stack.append(entry)

        if self._cuda_available:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()

        try:
            yield
        finally:
            if self._cuda_available:
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                mem_after = torch.cuda.memory_allocated()
                entry.peak_mb = peak / _BYTES_PER_MIB
                entry.delta_mb = (mem_after - mem_before) / _BYTES_PER_MIB

            self._stack.pop()

    def get_report(self) -> list[_profile.MemoryEntry]:
        """
        Return the collected memory tree.

        Returns
        -------
        list[MemoryEntry]
            Top-level memory entries. Each entry may contain nested children.
        """
        return self._root_entries
