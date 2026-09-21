"""Flag intervals: matching, stacking and automatic suggestions.

An interval selects a cell (baseline a-b, time t, channel c) when every set
bound contains the cell; NULL bounds are wildcards.  ``antenna_a`` alone
flags every baseline touching that antenna; both antenna endpoints select a
single unordered baseline.  Several flags may cover the same cell, and each
match keeps its own source and reason (stacking, never overwrite).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Flag:
    id: int | None
    source: str
    status: str
    reason: str
    antenna_a: str | None
    antenna_b: str | None
    t_start: int | None
    t_end: int | None
    ch_start: int | None
    ch_end: int | None

    @classmethod
    def from_row(cls, row: Any) -> "Flag":
        return cls(
            id=row["id"],
            source=row["source"],
            status=row["status"],
            reason=row["reason"],
            antenna_a=row["antenna_a"],
            antenna_b=row["antenna_b"],
            t_start=row["t_start"],
            t_end=row["t_end"],
            ch_start=row["ch_start"],
            ch_end=row["ch_end"],
        )

    def matches(self, a: str, b: str, t: int, ch: int) -> bool:
        names = {a, b}
        if self.antenna_a is not None and self.antenna_a not in names:
            return False
        if self.antenna_b is not None and self.antenna_b not in names:
            return False
        if self.t_start is not None and t < self.t_start:
            return False
        if self.t_end is not None and t > self.t_end:
            return False
        if self.ch_start is not None and ch < self.ch_start:
            return False
        if self.ch_end is not None and ch > self.ch_end:
            return False
        return True

    def as_tuple(self) -> tuple:
        return (
            self.source,
            self.antenna_a,
            self.antenna_b,
            self.t_start,
            self.t_end,
            self.ch_start,
            self.ch_end,
        )


def build_match_map(
    flags: Iterable[Flag],
    antennas: list[str],
    ntime: int,
    nchan: int,
    statuses: tuple[str, ...] = ("applied",),
) -> dict[tuple[str, str, int, int], list[Flag]]:
    """Stack every covering flag onto each cell, preserving source/reason."""
    match_map: dict[tuple[str, str, int, int], list[Flag]] = defaultdict(list)
    baselines = [
        (a, b)
        for i, a in enumerate(sorted(antennas))
        for b in sorted(antennas)[i + 1 :]
    ]
    active = [f for f in flags if f.status in statuses]
    for a, b in baselines:
        for t in range(ntime):
            for ch in range(nchan):
                for flag in active:
                    if flag.matches(a, b, t, ch):
                        match_map[(a, b, t, ch)].append(flag)
    return match_map


def wrap_phase(phase: np.ndarray | float) -> np.ndarray | float:
    """Wrap phases to (-pi, pi]; the branch cut is the negative real axis.

    ``arctan2(sin x, cos x)`` guarantees branch-safe wrapping even when x
    sits exactly on the negative real axis.
    """
    x = np.asarray(phase, dtype=float)
    return np.arctan2(np.sin(x), np.cos(x))


def _interval_rects(
    cells: set[tuple[int, int]], ntime: int, nchan: int
) -> list[tuple[int, int, int, int]]:
    """Group flagged (t, ch) cells into minimal maximal rectangles."""
    occupied = set(cells)
    rects: list[tuple[int, int, int, int]] = []
    while occupied:
        t0, ch0 = min(occupied)
        # extend along channel at t0
        ch1 = ch0
        while (t0, ch1 + 1) in occupied:
            ch1 += 1
        # extend the full-width strip along time while every column exists
        t1 = t0
        while all((t1 + 1, c) in occupied for c in range(ch0, ch1 + 1)):
            t1 += 1
        for t in range(t0, t1 + 1):
            for c in range(ch0, ch1 + 1):
                occupied.discard((t, c))
        rects.append((t0, t1, ch0, ch1))
    return rects


def suggest_flags(
    vis_rows: list[Any],
    flags: Iterable[Flag],
    ntime: int,
    nchan: int,
) -> list[dict[str, Any]]:
    """Deterministic auto-suggestions on currently unflagged cells.

    Three detectors, each carrying its own reason:
      1. quality bit set;
      2. narrow-band/time-local RFI via robust per-baseline amplitude MAD;
      3. antenna phase jump via robust residual of the time-differenced phase.
    Only cells not already covered by an applied flag are suggested, so
    re-running detection never stacks duplicates of existing coverage.
    """
    applied = [f for f in flags if f.status == "applied"]

    per_baseline: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    quality_cells: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for row in vis_rows:
        key = (row["antenna_a"], row["antenna_b"])
        slot = per_baseline.setdefault(
            key,
            {"amp": np.full((ntime, nchan), np.nan),
             "ph": np.full((ntime, nchan), np.nan),
             "mask": np.zeros((ntime, nchan), dtype=bool)},
        )
        t, ch = row["t_idx"], row["channel"]
        z = complex(row["re"], row["im"])
        slot["amp"][t, ch] = abs(z)
        slot["ph"][t, ch] = np.angle(z)
        covered = any(f.matches(key[0], key[1], t, ch) for f in applied)
        slot["mask"][t, ch] = covered
        if row["quality"] == 1 and not covered:
            quality_cells[key].add((t, ch))

    rfi_cells: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for key, slot in per_baseline.items():
        good = slot["amp"][~slot["mask"]]
        if good.size < 5:
            continue
        med = float(np.median(good))
        mad = float(np.median(np.abs(good - med)))
        threshold = med + max(8.0 * 1.4826 * mad, 0.5 * med)
        for t in range(ntime):
            for ch in range(nchan):
                if slot["mask"][t, ch]:
                    continue
                if slot["amp"][t, ch] > threshold:
                    rfi_cells[key].add((t, ch))

    jump_cells: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for key, slot in per_baseline.items():
        ph = slot["ph"]
        mask = slot["mask"]
        # time-differenced phase; wrapping keeps jumps across the branch cut
        dtph = wrap_phase(np.diff(ph, axis=0))
        present = ~mask
        diff_present = present[1:, :] & present[:-1, :]
        if diff_present.sum() < 5:
            continue
        values = dtph[diff_present]
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        scale = max(1.4826 * mad, 1e-3)
        for ti in range(ntime - 1):
            for ch in range(nchan):
                if not diff_present[ti, ch]:
                    continue
                if abs(float(dtph[ti, ch]) - med) > 5.0 * scale:
                    # flag the two samples after the discontinuity and their
                    # same-time spectral neighbours (local glitch)
                    for tj in (ti + 1, ti + 2):
                        if tj >= ntime:
                            continue
                        for c in range(max(0, ch - 1), min(nchan, ch + 2)):
                            if present[tj, c]:
                                jump_cells[key].add((tj, c))

    suggestions: list[dict[str, Any]] = []
    detectors = (
        ("质量位 quality=1", quality_cells),
        ("自动RFI：幅度超稳健阈值", rfi_cells),
        ("自动相位突跳：时间差分相位离群", jump_cells),
    )
    for reason, grouped in detectors:
        for key in sorted(grouped):
            cells = grouped[key]
            if not cells:
                continue
            for t0, t1, ch0, ch1 in _interval_rects(cells, ntime, nchan):
                suggestions.append(
                    {
                        "source": "auto",
                        "status": "suggested",
                        "reason": reason,
                        "antenna_a": key[0],
                        "antenna_b": key[1],
                        "t_start": t0,
                        "t_end": t1,
                        "ch_start": ch0,
                        "ch_end": ch1,
                    }
                )
    return suggestions
