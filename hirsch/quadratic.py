"""Gotoh's affine-gap global aligner with the full O(nm) DP matrix.

This module is the *oracle*.  It is the straightforward, obviously-correct
implementation: three full dynamic-programming layers, kept in memory, with a
traceback that walks backwards through them.  Everything in
:mod:`hirsch.linear` is validated against the scores and alignments produced
here, so this file is deliberately written for legibility over cleverness.

Three layers (Gotoh 1982, "An improved algorithm for matching biological
sequences", J. Mol. Biol. 162:705-708):

* ``M[i][j]`` - best score for ``x[:i]`` vs ``y[:j]`` whose last column is a
  substitution (``x[i-1]`` aligned to ``y[j-1]``).
* ``X[i][j]`` - best score whose last column is a gap in ``y``; it consumes
  ``x[i-1]``.  Call this a *vertical* step.
* ``Y[i][j]`` - best score whose last column is a gap in ``x``; it consumes
  ``y[j-1]``.  Call this a *horizontal* step.

Boundary gap charges (``tb`` / ``te``)
--------------------------------------
Both aligners accept two extra parameters, ``tb`` and ``te``, that override
the gap-open penalty charged for a vertical run sitting at the very start or
the very end of the alignment.  With ``tb = te = gap_open`` this is the plain
affine score.  The linear-space recursion needs ``tb`` or ``te`` set to ``0``
to express "the vertical gap you are about to open is really the continuation
of a gap that started outside this subproblem", which is the whole crux of the
gap-state handoff described in the README.  Only vertical runs need this,
because the divide-and-conquer splits on rows: a horizontal run can always be
kept wholly inside one subproblem.

If the entire alignment is one vertical run (``y`` empty) the run is
simultaneously leading and trailing, so it is charged ``min(tb, te)``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._ops import NEG_INF, Alignment, Op, render_ops
from .scoring import Scoring

__all__ = ["align", "align_score", "fill_matrices"]


def fill_matrices(
    xc: Sequence[int],
    yc: Sequence[int],
    sub: List[List[int]],
    gap_open: int,
    gap_extend: int,
    tb: int,
    te: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill the three full DP layers plus the trailing-gap column ``Xe``.

    Returns ``(M, X, Y, Xe)``.  ``Xe[i]`` is the best score for ``x[:i]`` vs
    all of ``y`` ending in a vertical run whose open penalty is ``te`` rather
    than ``gap_open``; it is only meaningful in the last column, because from
    any cell in the last column the only remaining moves are vertical.

    The row update is vectorised.  ``M`` and ``X`` at row ``i`` depend only on
    row ``i-1``, so they are plain numpy expressions.  ``Y`` at row ``i``
    depends on earlier cells of the *same* row, but unrolling the recurrence
    turns it into a prefix maximum::

        Y[i][j] = max_{k < j} ( max(M[i][k], X[i][k]) - gap_open - (j-k)*ge )
                = -gap_open - j*ge + max_{k < j} ( max(M[i][k], X[i][k]) + k*ge )

    which ``np.maximum.accumulate`` computes in one pass.
    """
    n, m = len(xc), len(yc)
    go, ge = gap_open, gap_extend

    M = np.full((n + 1, m + 1), NEG_INF, dtype=np.int64)
    X = np.full((n + 1, m + 1), NEG_INF, dtype=np.int64)
    Y = np.full((n + 1, m + 1), NEG_INF, dtype=np.int64)

    M[0, 0] = 0
    if n:
        rows = np.arange(1, n + 1, dtype=np.int64)
        X[1:, 0] = -(tb + rows * ge)
    if m:
        cols = np.arange(1, m + 1, dtype=np.int64)
        Y[0, 1:] = -(go + cols * ge)
        offsets = np.arange(m + 1, dtype=np.int64) * ge
        yarr = np.asarray(yc, dtype=np.int64)
        # sub_by_code[c] is the score row of x-code c against every y residue.
        sub_by_code = np.asarray(sub, dtype=np.int64)[:, yarr]

    for i in range(1, n + 1):
        prev_M = M[i - 1]
        prev_X = X[i - 1]
        prev_Y = Y[i - 1]
        if m:
            best_diag = np.maximum(np.maximum(prev_M[:-1], prev_X[:-1]), prev_Y[:-1])
            M[i, 1:] = best_diag + sub_by_code[xc[i - 1]]
            X[i, 1:] = np.maximum(
                np.maximum(prev_M[1:], prev_Y[1:]) - go - ge, prev_X[1:] - ge
            )
            base = np.maximum(M[i], X[i]) + offsets
            run = np.maximum.accumulate(base[:-1])
            Y[i, 1:] = run - go - offsets[1:]

    Xe = np.full(n + 1, NEG_INF, dtype=np.int64)
    if m:
        best = NEG_INF
        for i in range(1, n + 1):
            start = max(int(M[i - 1, m]), int(Y[i - 1, m])) - te
            best = max(best, start) - ge
            Xe[i] = best
    return M, X, Y, Xe


def _traceback(
    M: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Xe: np.ndarray,
    xc: Sequence[int],
    yc: Sequence[int],
    sub: List[List[int]],
    gap_open: int,
    gap_extend: int,
    tb: int,
    te: int,
) -> List[Op]:
    """Walk back through the filled layers, choosing predecessors by value.

    No pointer matrices are stored; at each cell we test which incoming edge
    reproduces the cell's value.  Candidates are tested in a fixed order, so
    the alignment returned for a given input is deterministic even when
    several co-optimal alignments exist.
    """
    n, m = len(xc), len(yc)
    go, ge = gap_open, gap_extend
    rev: List[Op] = []

    i, j = n, m
    final = max(int(M[n, m]), int(Y[n, m]), int(Xe[n]))
    if final == int(M[n, m]):
        state = "M"
    elif final == int(Xe[n]):
        state = "E"
    else:
        state = "Y"

    while i > 0 or j > 0:
        if state == "M":
            rev.append(("M", 1))
            target = int(M[i, j]) - sub[xc[i - 1]][yc[j - 1]]
            i -= 1
            j -= 1
            if i == 0 and j == 0:
                break
            if int(M[i, j]) == target:
                state = "M"
            elif int(X[i, j]) == target:
                state = "X"
            elif int(Y[i, j]) == target:
                state = "Y"
            else:  # pragma: no cover - defensive
                raise AssertionError("broken traceback at M(%d,%d)" % (i, j))
        elif state == "X":
            if j == 0:
                rev.append(("D", i))
                i = 0
                break
            value = int(X[i, j])
            rev.append(("D", 1))
            if int(X[i - 1, j]) - ge == value:
                i -= 1
                state = "X"
            elif int(M[i - 1, j]) - go - ge == value:
                i -= 1
                state = "M"
            elif int(Y[i - 1, j]) - go - ge == value:
                i -= 1
                state = "Y"
            else:  # pragma: no cover - defensive
                raise AssertionError("broken traceback at X(%d,%d)" % (i, j))
            if i == 0 and j == 0:
                break
        elif state == "E":
            # Trailing vertical run in the last column, open charged ``te``.
            value = int(Xe[i])
            rev.append(("D", 1))
            i -= 1
            if i == 0 and j == 0:
                break
            if int(Xe[i]) - ge == value:
                state = "E"
            elif int(M[i, j]) - te - ge == value:
                state = "M"
            elif int(Y[i, j]) - te - ge == value:
                state = "Y"
            else:  # pragma: no cover - defensive
                raise AssertionError("broken traceback at Xe(%d)" % i)
        else:  # state == "Y"
            if i == 0:
                rev.append(("I", j))
                j = 0
                break
            value = int(Y[i, j])
            rev.append(("I", 1))
            if int(Y[i, j - 1]) - ge == value:
                j -= 1
                state = "Y"
            elif int(M[i, j - 1]) - go - ge == value:
                j -= 1
                state = "M"
            elif int(X[i, j - 1]) - go - ge == value:
                j -= 1
                state = "X"
            else:  # pragma: no cover - defensive
                raise AssertionError("broken traceback at Y(%d,%d)" % (i, j))
            if i == 0 and j == 0:
                break

    rev.reverse()
    merged: List[Op] = []
    for op, count in rev:
        if merged and merged[-1][0] == op:
            merged[-1] = (op, merged[-1][1] + count)
        else:
            merged.append((op, count))
    return merged


def _boundary_defaults(scoring: Scoring, tb: Optional[int], te: Optional[int]):
    return (
        scoring.gap_open if tb is None else tb,
        scoring.gap_open if te is None else te,
    )


def align(
    x: str,
    y: str,
    scoring: Optional[Scoring] = None,
    tb: Optional[int] = None,
    te: Optional[int] = None,
) -> Alignment:
    """Optimal affine-gap global alignment using O(nm) memory."""
    scoring = scoring or Scoring()
    tb, te = _boundary_defaults(scoring, tb, te)
    n, m = len(x), len(y)
    ge = scoring.gap_extend

    if n == 0:
        score = -scoring.gap_cost(m)
        ops: List[Op] = [("I", m)] if m else []
        ax, ay = render_ops(x, y, ops)
        return Alignment(score, ax, ay)
    if m == 0:
        score = -(min(tb, te) + n * ge)
        ops = [("D", n)]
        ax, ay = render_ops(x, y, ops)
        return Alignment(score, ax, ay)

    xc, yc, sub = scoring.prepare(x, y)
    M, X, Y, Xe = fill_matrices(xc, yc, sub, scoring.gap_open, ge, tb, te)
    score = max(int(M[n, m]), int(Y[n, m]), int(Xe[n]))
    ops = _traceback(M, X, Y, Xe, xc, yc, sub, scoring.gap_open, ge, tb, te)
    ax, ay = render_ops(x, y, ops)
    return Alignment(score, ax, ay)


def align_score(
    x: str,
    y: str,
    scoring: Optional[Scoring] = None,
    tb: Optional[int] = None,
    te: Optional[int] = None,
) -> int:
    """Optimal score only, still O(nm) memory (this is the oracle, not the point)."""
    scoring = scoring or Scoring()
    tb, te = _boundary_defaults(scoring, tb, te)
    n, m = len(x), len(y)
    if n == 0:
        return -scoring.gap_cost(m)
    if m == 0:
        return -(min(tb, te) + n * scoring.gap_extend)
    xc, yc, sub = scoring.prepare(x, y)
    M, X, Y, Xe = fill_matrices(xc, yc, sub, scoring.gap_open, scoring.gap_extend, tb, te)
    return max(int(M[n, m]), int(Y[n, m]), int(Xe[n]))
