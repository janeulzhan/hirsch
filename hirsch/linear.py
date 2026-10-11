"""Myers-Miller: Gotoh's affine-gap recurrence in O(min(n, m)) working space.

This is Hirschberg's divide-and-conquer (1975) applied to Gotoh's three-layer
affine recurrence (1982), i.e. the algorithm of Myers & Miller, "Optimal
alignments in linear space", CABIOS 4(1):11-17 (1988).

The shape of it
---------------
To align ``x`` (length n) against ``y`` (length m), fix the split row
``r = n // 2``.  Run the three-layer DP forward over rows ``0..r`` keeping only
one row of each layer, and run it backward over rows ``n..r`` the same way.
That gives, for every column ``j``, six numbers::

    F_M[j], F_X[j], F_Y[j]   best score for x[:r] vs y[:j], by last edge type
    R_M[j], R_X[j], R_Y[j]   best score for x[r:] vs y[j:], by first edge type

where edge types are diagonal (``M``), vertical (``X``: consumes x, gap in y)
and horizontal (``Y``: consumes y, gap in x).  Pick the best column, recurse on
the two halves, concatenate.  Working space is one row of each layer, so
``O(m)``; swapping the arguments makes that ``O(min(n, m))``.

The gap-state handoff
---------------------
The naive combination takes ``max(F_*) + max(R_*)`` over columns.  That is
wrong, and it is wrong in a way that still looks plausible: the score it
reports is never *too high*, it is sometimes too *low*, and only on inputs
where the optimal path crosses the split row in the middle of a vertical gap.

Why.  Charge each gap run's open penalty once.  If the optimal path enters the
split node ``(r, j)`` on a vertical edge and leaves it on a vertical edge,
then that single gap run is split between the two halves, and ``F_X[j]`` and
``R_X[j]`` have each paid a gap-open for it.  The combination therefore
overcharges by exactly one ``gap_open``.  Worse, no other choice of split
column avoids it: ``(r, j)`` is the only node of row ``r`` on that path.

The fix has two halves.

1.  Score.  Evaluate a second candidate per column,
    ``F_X[j] + R_X[j] + gap_open``, which refunds the double-charged open.
    This value is the exact score of a real alignment, so it can never
    overshoot the optimum, and the spanning case is now reachable.  The
    reported optimum is the max over both candidate families and all columns.

2.  Traceback.  Having picked the spanning candidate, the recursion must force
    the left half to *end* in a vertical gap and the right half to *begin* in
    one; otherwise the two halves get stitched together into an alignment that
    does not have the score we just computed.  That is what the ``tb`` and
    ``te`` parameters do: they say "a vertical run at your boundary is the
    continuation of a run outside you, so charge it no gap-open".  The left
    half is solved with ``te = 0`` and the right half with ``tb = 0``.

    Forcing works because of a strictness rule.  The spanning candidate is
    only used when it is *strictly* greater than the best ordinary candidate.
    Expand that inequality and it gives ``F_X[j] + gap_open > F_M[j]`` and
    ``F_X[j] + gap_open > F_Y[j]``, which is precisely the statement that
    under ``te = 0`` the left half's unique optimal end state is the vertical
    gap.  The symmetric argument fixes the right half's start state.  On ties
    the ordinary candidate wins, so the strict inequality always holds when it
    is needed.

Horizontal runs never need this treatment.  The split is on rows, so a
horizontal run always has a leftmost node in row ``r``, and choosing that node
as the split keeps the run inside one half.  (A horizontal run straddling the
chosen column is still only ever *overcharged*, which the maximisation
discards.)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ._ops import NEG_INF, Alignment, Op, render_ops
from .quadratic import fill_matrices
from .quadratic import _traceback as _quadratic_traceback
from .scoring import Scoring

__all__ = ["align", "align_score", "split_statistics"]


class _Context:
    """Everything the recursion needs that does not change between calls.

    ``x`` codes stay a Python list because the recursion only ever reads single
    residues from it, while ``y`` codes live in a numpy array so that every
    subrange, forward or reversed, is a zero-copy view.  That is what keeps
    working space proportional to ``m`` instead of to ``m log n``.
    """

    __slots__ = ("xc", "yarr", "sub", "sub_rows", "go", "ge", "goe")

    def __init__(
        self, xc: np.ndarray, yc: np.ndarray, sub: List[List[int]], go: int, ge: int
    ):
        self.xc = xc
        self.yarr = np.asarray(yc)
        self.sub = sub
        self.sub_rows = [np.asarray(row, dtype=np.int64) for row in sub]
        self.go = go
        self.ge = ge
        self.goe = go + ge

    def y_codes(self, j0: int, step: int, length: int) -> np.ndarray:
        """A view of ``length`` y codes starting at ``j0`` with the given step."""
        if step == 1:
            return self.yarr[j0 : j0 + length]
        return self.yarr[j0 - length + 1 : j0 + 1][::-1]


def _layers(
    ctx: "_Context",
    x0: int,
    xstep: int,
    xlen: int,
    y0: int,
    ystep: int,
    ylen: int,
    tb: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row-by-row sweep of the three-layer DP, keeping only the last row.

    Returns the ``M``, ``X`` and ``Y`` layers at row ``xlen``, each of length
    ``ylen + 1``.  Space is ``O(ylen)``: a fixed number of row buffers, reused
    across rows, with no per-row allocation.

    ``tb`` is the gap-open penalty charged for a vertical run that starts at
    the very first column of the alignment (``j == 0``); every other vertical
    run is charged ``gap_open``.

    The row update is vectorised the same way as the quadratic oracle.  ``M``
    and ``X`` at row ``i`` depend only on row ``i - 1``, so they are plain
    numpy expressions.  ``Y`` depends on earlier cells of its own row, but
    unrolling that recurrence turns it into a prefix maximum, which
    ``np.maximum.accumulate`` evaluates in one pass.
    """
    m = ylen
    go, ge, goe, neg = ctx.go, ctx.ge, ctx.goe, NEG_INF
    ycodes = ctx.y_codes(y0, ystep, m)
    sub_rows = ctx.sub_rows
    xc = ctx.xc

    offsets = np.arange(m + 1, dtype=np.int64) * ge

    cur_m = np.full(m + 1, neg, dtype=np.int64)
    cur_x = np.full(m + 1, neg, dtype=np.int64)
    cur_y = np.full(m + 1, neg, dtype=np.int64)
    nxt_m = np.empty(m + 1, dtype=np.int64)
    nxt_x = np.empty(m + 1, dtype=np.int64)
    nxt_y = np.empty(m + 1, dtype=np.int64)
    scratch_a = np.empty(m, dtype=np.int64)
    scratch_b = np.empty(m, dtype=np.int64)
    base = np.empty(m + 1, dtype=np.int64)
    run = np.empty(m, dtype=np.int64)

    cur_m[0] = 0
    cur_y[1:] = -(go + offsets[1:])

    for i in range(1, xlen + 1):
        srow = sub_rows[xc[x0 + (i - 1) * xstep]]

        # Diagonal step into M: best of the three layers one row up and one
        # column left, plus the substitution score.
        np.maximum(cur_m[:-1], cur_x[:-1], out=scratch_a)
        np.maximum(scratch_a, cur_y[:-1], out=scratch_a)
        np.take(srow, ycodes, out=scratch_b)
        np.add(scratch_a, scratch_b, out=nxt_m[1:])
        nxt_m[0] = neg

        # Vertical step into X: same column, one row up.
        np.maximum(cur_m[1:], cur_y[1:], out=scratch_a)
        np.subtract(scratch_a, goe, out=scratch_a)
        np.subtract(cur_x[1:], ge, out=scratch_b)
        np.maximum(scratch_a, scratch_b, out=nxt_x[1:])
        nxt_x[0] = -(tb + i * ge)

        # Horizontal step into Y: unrolled into a prefix maximum over the
        # column where the horizontal run began.
        np.maximum(nxt_m, nxt_x, out=base)
        np.add(base, offsets, out=base)
        np.maximum.accumulate(base[:-1], out=run)
        np.subtract(run, offsets[1:], out=nxt_y[1:])
        np.subtract(nxt_y[1:], go, out=nxt_y[1:])
        nxt_y[0] = neg

        cur_m, nxt_m = nxt_m, cur_m
        cur_x, nxt_x = nxt_x, cur_x
        cur_y, nxt_y = nxt_y, cur_y

    return cur_m, cur_x, cur_y


def _base_case(
    ctx: "_Context", i1: int, i2: int, j1: int, j2: int, tb: int, te: int
) -> List[Op]:
    """Solve a subproblem with at most one row of ``x`` directly.

    With ``n <= 1`` the full Gotoh matrix is at most two rows, so deferring to
    the quadratic aligner here still costs only ``O(m)`` space.  This is where
    the recursion bottoms out.
    """
    n = i2 - i1
    m = j2 - j1
    if n == 0:
        return [("I", m)] if m else []
    if m == 0:
        return [("D", n)]
    xs = ctx.xc[i1:i2]
    ys = ctx.yarr[j1:j2]
    M, X, Y, Xe = fill_matrices(xs, ys, ctx.sub, ctx.go, ctx.ge, tb, te)
    return _quadratic_traceback(M, X, Y, Xe, xs, ys, ctx.sub, ctx.go, ctx.ge, tb, te)


def _base_case_score(
    ctx: "_Context", i1: int, i2: int, j1: int, j2: int, tb: int, te: int
) -> int:
    n = i2 - i1
    m = j2 - j1
    if n == 0:
        return -(ctx.go + m * ctx.ge) if m else 0
    if m == 0:
        return -(min(tb, te) + n * ctx.ge)
    xs = ctx.xc[i1:i2]
    ys = ctx.yarr[j1:j2]
    M, X, Y, Xe = fill_matrices(xs, ys, ctx.sub, ctx.go, ctx.ge, tb, te)
    return max(int(M[n, m]), int(Y[n, m]), int(Xe[n]))


def _recurse(
    ctx: "_Context",
    i1: int,
    i2: int,
    j1: int,
    j2: int,
    tb: int,
    te: int,
    out: List[Op],
    stats: Optional[dict] = None,
    score_box: Optional[List[Optional[int]]] = None,
) -> None:
    n = i2 - i1
    m = j2 - j1
    if n <= 1 or m == 0:
        if score_box is not None and score_box[0] is None:
            score_box[0] = _base_case_score(ctx, i1, i2, j1, j2, tb, te)
        out.extend(_base_case(ctx, i1, i2, j1, j2, tb, te))
        return

    r = n // 2
    go = ctx.go

    fm, fx, fy = _layers(ctx, i1, 1, r, j1, 1, m, tb)
    gm, gx, gy = _layers(ctx, i2 - 1, -1, n - r, j2 - 1, -1, m, te)

    # Reverse the half-row so that index jj means "column j1 + jj" in both.
    forward_best = np.maximum(np.maximum(fm, fx), fy)
    reverse_best = np.maximum(np.maximum(gm, gx), gy)[::-1]

    plain = forward_best + reverse_best
    plain_j = int(plain.argmax())
    best_plain = int(plain[plain_j])

    # The gap-spanning candidate: refund the single gap-open that both halves
    # charged for one vertical run straddling the split row.
    span = fx + gx[::-1] + go
    span_j = int(span.argmax())
    best_span = int(span[span_j])

    if score_box is not None and score_box[0] is None:
        score_box[0] = max(best_plain, best_span)

    if best_span > best_plain:
        # The optimal path crosses the split row inside a vertical gap. Waive
        # the gap-open charge on both sides of the join so that each half is
        # forced to agree on being inside that gap.
        jj = span_j
        left_te, right_tb = 0, 0
        if stats is not None:
            stats["spanning_splits"] += 1
    else:
        jj = plain_j
        left_te, right_tb = go, go
        if stats is not None:
            stats["plain_splits"] += 1

    del fm, fx, fy, gm, gx, gy, forward_best, reverse_best, plain, span

    _recurse(ctx, i1, i1 + r, j1, j1 + jj, tb, left_te, out, stats)
    _recurse(ctx, i1 + r, i2, j1 + jj, j2, right_tb, te, out, stats)


def _merge(ops: List[Op]) -> List[Op]:
    merged: List[Op] = []
    for op, count in ops:
        if count <= 0:
            continue
        if merged and merged[-1][0] == op:
            merged[-1] = (op, merged[-1][1] + count)
        else:
            merged.append((op, count))
    return merged


def _transpose(scoring: Scoring) -> Scoring:
    """Scoring with the substitution matrix transposed, for argument swapping."""
    if scoring.matrix is None:
        return scoring
    flipped = {(b, a): v for (a, b), v in scoring.matrix.items()}
    return Scoring(
        gap_open=scoring.gap_open,
        gap_extend=scoring.gap_extend,
        match=scoring.match,
        mismatch=scoring.mismatch,
        matrix=flipped,
        name=scoring.name,
    )


def align_score(x: str, y: str, scoring: Optional[Scoring] = None) -> int:
    """Optimal affine-gap score in O(min(n, m)) space, with no traceback.

    A single forward sweep is enough for the score alone; the divide and
    conquer is only needed to recover the alignment itself.
    """
    scoring = scoring or Scoring()
    if len(y) > len(x):
        return align_score(y, x, _transpose(scoring))
    n, m = len(x), len(y)
    if n == 0:
        return -scoring.gap_cost(m)
    if m == 0:
        return -scoring.gap_cost(n)
    xc, yc, sub = scoring.prepare(x, y)
    ctx = _Context(xc, yc, sub, scoring.gap_open, scoring.gap_extend)
    lm, lx, ly = _layers(ctx, 0, 1, n, 0, 1, m, scoring.gap_open)
    return int(max(lm[m], lx[m], ly[m]))


def align(
    x: str,
    y: str,
    scoring: Optional[Scoring] = None,
    _stats: Optional[dict] = None,
) -> Alignment:
    """Optimal affine-gap global alignment in O(min(n, m)) working space.

    The score reported is the one the linear-space recursion computes at its
    top-level split, not a re-scoring of the alignment that comes back; the
    test suite re-scores the alignment separately and checks the two agree.
    """
    scoring = scoring or Scoring()
    if len(y) > len(x):
        flipped = align(y, x, _transpose(scoring), _stats)
        return Alignment(flipped.score, flipped.aligned_y, flipped.aligned_x)

    n, m = len(x), len(y)
    if n == 0:
        ops: List[Op] = [("I", m)] if m else []
        ax, ay = render_ops(x, y, ops)
        return Alignment(-scoring.gap_cost(m), ax, ay)
    if m == 0:
        ops = [("D", n)]
        ax, ay = render_ops(x, y, ops)
        return Alignment(-scoring.gap_cost(n), ax, ay)

    xc, yc, sub = scoring.prepare(x, y)
    ctx = _Context(xc, yc, sub, scoring.gap_open, scoring.gap_extend)
    raw: List[Op] = []
    score_box: List[Optional[int]] = [None]
    _recurse(ctx, 0, n, 0, m, scoring.gap_open, scoring.gap_open, raw, _stats, score_box)
    ops = _merge(raw)
    ax, ay = render_ops(x, y, ops)
    assert score_box[0] is not None
    return Alignment(int(score_box[0]), ax, ay)


def split_statistics(x: str, y: str, scoring: Optional[Scoring] = None):
    """Align and report how many splits used the gap-spanning candidate.

    Exposed so the test suite can prove the gap-state handoff is actually
    being exercised rather than merely being present.
    """
    stats = {"plain_splits": 0, "spanning_splits": 0}
    result = align(x, y, scoring, _stats=stats)
    return result, stats
