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

from typing import List, Optional, Sequence, Tuple

from ._ops import NEG_INF, Alignment, Op, render_ops
from .quadratic import fill_matrices
from .quadratic import _traceback as _quadratic_traceback
from .scoring import Scoring

__all__ = ["align", "align_score", "split_statistics"]


def _layers(
    xc: Sequence[int],
    x0: int,
    xstep: int,
    xlen: int,
    yc: Sequence[int],
    y0: int,
    ystep: int,
    ylen: int,
    sub: List[List[int]],
    go: int,
    ge: int,
    tb: int,
) -> Tuple[List[int], List[int], List[int]]:
    """One row-by-row sweep of the three-layer DP, keeping only the last row.

    Addressing is by base/step/length so that neither the reverse sweep nor the
    recursion ever copies a subsequence; the only allocations that scale with
    the input are the six row buffers, which are reused across rows.

    ``tb`` is the gap-open penalty charged for a vertical run that starts at
    the very first column of the alignment (that is, at ``j == 0``).  Any other
    vertical run is charged ``go``.
    """
    m = ylen
    goe = go + ge
    neg = NEG_INF

    ycodes = [yc[y0 + l * ystep] for l in range(m)]

    cur_m = [neg] * (m + 1)
    cur_x = [neg] * (m + 1)
    cur_y = [neg] * (m + 1)
    nxt_m = [neg] * (m + 1)
    nxt_x = [neg] * (m + 1)
    nxt_y = [neg] * (m + 1)

    cur_m[0] = 0
    for j in range(1, m + 1):
        cur_y[j] = -(go + j * ge)

    for i in range(1, xlen + 1):
        srow = sub[xc[x0 + (i - 1) * xstep]]
        nxt_m[0] = neg
        nxt_x[0] = -(tb + i * ge)
        nxt_y[0] = neg
        for j in range(1, m + 1):
            # diagonal: best of the three layers one row up, one column left
            d = cur_m[j - 1]
            t = cur_x[j - 1]
            if t > d:
                d = t
            t = cur_y[j - 1]
            if t > d:
                d = t
            nxt_m[j] = d + srow[ycodes[j - 1]]

            # vertical: same column, one row up
            a = cur_m[j] - goe
            t = cur_y[j] - goe
            if t > a:
                a = t
            t = cur_x[j] - ge
            if t > a:
                a = t
            nxt_x[j] = a

            # horizontal: same row, one column left (already computed)
            b = nxt_m[j - 1] - goe
            t = nxt_x[j - 1] - goe
            if t > b:
                b = t
            t = nxt_y[j - 1] - ge
            if t > b:
                b = t
            nxt_y[j] = b

        cur_m, nxt_m = nxt_m, cur_m
        cur_x, nxt_x = nxt_x, cur_x
        cur_y, nxt_y = nxt_y, cur_y

    return cur_m, cur_x, cur_y


def _base_case(
    xc: Sequence[int],
    i1: int,
    i2: int,
    yc: Sequence[int],
    j1: int,
    j2: int,
    sub: List[List[int]],
    go: int,
    ge: int,
    tb: int,
    te: int,
) -> List[Op]:
    """Solve a subproblem with at most one row of ``x`` directly.

    With ``n <= 1`` the full Gotoh matrix is at most two rows, so running the
    quadratic aligner here still costs only ``O(m)`` space.  This is where the
    recursion bottoms out.
    """
    n = i2 - i1
    m = j2 - j1
    if n == 0:
        return [("I", m)] if m else []
    if m == 0:
        return [("D", n)]
    xs = list(xc[i1:i2])
    ys = list(yc[j1:j2])
    M, X, Y, Xe = fill_matrices(xs, ys, sub, go, ge, tb, te)
    return _quadratic_traceback(M, X, Y, Xe, xs, ys, sub, go, ge, tb, te)


def _recurse(
    xc: Sequence[int],
    i1: int,
    i2: int,
    yc: Sequence[int],
    j1: int,
    j2: int,
    sub: List[List[int]],
    go: int,
    ge: int,
    tb: int,
    te: int,
    out: List[Op],
    stats: Optional[dict] = None,
) -> None:
    n = i2 - i1
    m = j2 - j1
    if n <= 1 or m == 0:
        out.extend(_base_case(xc, i1, i2, yc, j1, j2, sub, go, ge, tb, te))
        return

    r = n // 2

    fm, fx, fy = _layers(xc, i1, 1, r, yc, j1, 1, m, sub, go, ge, tb)
    gm, gx, gy = _layers(xc, i2 - 1, -1, n - r, yc, j2 - 1, -1, m, sub, go, ge, te)

    best_plain = NEG_INF
    best_plain_j = 0
    best_span = NEG_INF
    best_span_j = 0
    for jj in range(m + 1):
        k = m - jj
        f = fm[jj]
        t = fx[jj]
        if t > f:
            f = t
        t = fy[jj]
        if t > f:
            f = t
        g = gm[k]
        t = gx[k]
        if t > g:
            g = t
        t = gy[k]
        if t > g:
            g = t
        plain = f + g
        if plain > best_plain:
            best_plain = plain
            best_plain_j = jj
        span = fx[jj] + gx[k] + go
        if span > best_span:
            best_span = span
            best_span_j = jj

    if best_span > best_plain:
        # The optimal path crosses the split row inside a vertical gap.
        # Hand the gap state across the boundary by waiving the gap-open
        # charge on both sides of the join.
        jj = best_span_j
        left_te, right_tb = 0, 0
        if stats is not None:
            stats["spanning_splits"] += 1
    else:
        jj = best_plain_j
        left_te, right_tb = go, go
        if stats is not None:
            stats["plain_splits"] += 1

    del fm, fx, fy, gm, gx, gy

    _recurse(xc, i1, i1 + r, yc, j1, j1 + jj, sub, go, ge, tb, left_te, out, stats)
    _recurse(xc, i1 + r, i2, yc, j1 + jj, j2, sub, go, ge, right_tb, te, out, stats)


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
    """Scoring with the substitution matrix transposed (for argument swapping)."""
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
    """Optimal affine-gap score using O(min(n, m)) space and no traceback."""
    scoring = scoring or Scoring()
    if len(y) > len(x):
        return align_score(y, x, _transpose(scoring))
    n, m = len(x), len(y)
    if n == 0:
        return -scoring.gap_cost(m)
    if m == 0:
        return -scoring.gap_cost(n)
    xc, yc, sub = scoring.prepare(x, y)
    go, ge = scoring.gap_open, scoring.gap_extend
    lm, lx, ly = _layers(xc, 0, 1, n, yc, 0, 1, m, sub, go, ge, go)
    return max(lm[m], lx[m], ly[m])


def align(
    x: str,
    y: str,
    scoring: Optional[Scoring] = None,
    _stats: Optional[dict] = None,
) -> Alignment:
    """Optimal affine-gap global alignment using O(min(n, m)) working space.

    The result is identical to :func:`hirsch.quadratic.align` in score, and the
    test suite asserts that on every pair it generates.
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
    go, ge = scoring.gap_open, scoring.gap_extend
    raw: List[Op] = []
    _recurse(xc, 0, n, yc, 0, m, sub, go, ge, go, go, raw, _stats)
    ops = _merge(raw)
    ax, ay = render_ops(x, y, ops)
    return Alignment(scoring.score_alignment(ax, ay), ax, ay)


def split_statistics(x: str, y: str, scoring: Optional[Scoring] = None):
    """Align and report how many splits used the gap-spanning candidate.

    Exposed so the test suite can prove that the gap-state handoff path is
    actually being exercised rather than merely being present.
    """
    stats = {"plain_splits": 0, "spanning_splits": 0}
    result = align(x, y, scoring, _stats=stats)
    return result, stats
