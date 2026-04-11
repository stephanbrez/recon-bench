"""
Hierarchical wall-clock timer with CUDA synchronization.

Provides a nestable context manager that records per-section durations,
producing a tree of TimingEntry objects. GPU operations are synchronized
before each measurement so that async kernel launches don't skew results.
"""
from __future__ import annotations

import contextlib
import time

import torch

from . import _types as _profile


class Timer:
    """
    Hierarchical wall-clock timer with named sections.

    Sections may be nested arbitrarily deep. Each section records its own
    duration plus the durations of its children.

    Parameters
    ----------
    sync_cuda : bool
        If True and CUDA is available, call ``torch.cuda.synchronize()``
        before each timing measurement. Default True.
    enabled : bool
        If False, ``section()`` becomes a no-op and ``get_report()`` returns
        an empty list. Allows toggling profiling without restructuring
        call sites. Default True.

    Examples
    --------
    >>> timer = Timer()
    >>> with timer.section("load"):
    ...     data = load_something()
    >>> with timer.section("compute"):
    ...     result = compute(data)
    >>> print(timer.get_report())
    """

    def __init__(self, sync_cuda: bool = True, enabled: bool = True) -> None:
        self._sync_cuda = sync_cuda and torch.cuda.is_available()
        self._enabled = enabled
        self._root_entries: list[_profile.TimingEntry] = []
        self._stack: list[_profile.TimingEntry] = []

    @contextlib.contextmanager
    def section(self, name: str):
        """
        Time a named section. Nestable for sub-steps.

        Parameters
        ----------
        name : str
            Label for this timed section.

        Yields
        ------
        None
        """
        if not self._enabled:
            yield
            return

        if self._sync_cuda:
            torch.cuda.synchronize()

        entry = _profile.TimingEntry(name=name, duration_s=0.0)

        # ─── Attach to parent or root ───
        if self._stack:
            self._stack[-1].children.append(entry)
        else:
            self._root_entries.append(entry)

        self._stack.append(entry)
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._sync_cuda:
                torch.cuda.synchronize()
            entry.duration_s = time.perf_counter() - start
            self._stack.pop()

    def get_report(self) -> list[_profile.TimingEntry]:
        """
        Return the collected timing tree.

        Returns
        -------
        list[TimingEntry]
            Top-level timing entries. Each entry may contain nested children.
        """
        return self._root_entries
