"""Exhaustive validation of the quadratic oracle on tiny inputs.

The oracle is what everything else is checked against, so it needs an
independent check of its own.  For sequences short enough to enumerate every
possible global alignment, we compare the DP score to the brute-force maximum
under a scorer written directly from the definition of the affine gap model
(including the ``tb`` / ``te`` boundary charges the linear-space recursion
relies on).
"""

from __future__ import annotations

import itertools
import random
from typing import Iterator, List, Optional, Tuple

import pytest

from hirsch.quadratic import align, align_score
from hirsch.scoring import Scoring

Ops = Tuple[str, ...]


def enumerate_alignments(n: int, m: int) -> Iterator[Ops]:
    """Every monotone path from (0,0) to (n,m) as a tuple of ops."""
    if n == 0 and m == 0:
        yield ()
        return
    if n:
        for rest in enumerate_alignments(n - 1, m):
            yield ("D",) + rest
    if m:
        for rest in enumerate_alignments(n, m - 1):
            yield ("I",) + rest
    if n and m:
        for rest in enumerate_alignments(n - 1, m - 1):
            yield ("M",) + rest


def runs(ops: Ops) -> List[Tuple[str, int, int]]:
    """Group ops into (op, start_index, length) runs."""
    out: List[Tuple[str, int, int]] = []
    for idx, (op, group) in enumerate(itertools.groupby(enumerate(ops), key=lambda p: p[1])):
        items = list(group)
        out.append((op, items[0][0], len(items)))
    return out


def model_score(
    x: str, y: str, ops: Ops, scoring: Scoring, tb: int, te: int
) -> int:
    """Score one alignment under the affine model with boundary overrides.

    Written straight from the definition: walk the runs, charge each gap run
    one open plus one extend per residue, and substitute the boundary open
    penalty for a vertical run that touches the start or the end.
    """
    total = 0
    i = j = 0
    total_len = len(ops)
    for op, start, length in runs(ops):
        if op == "M":
            for k in range(length):
                total += scoring.pair(x[i + k], y[j + k])
            i += length
            j += length
        elif op == "D":
            at_start = start == 0
            at_end = start + length == total_len
            if at_start and at_end:
                open_charge = min(tb, te)
            elif at_start:
                open_charge = tb
            elif at_end:
                open_charge = te
            else:
                open_charge = scoring.gap_open
            total -= open_charge + length * scoring.gap_extend
            i += length
        else:
            total -= scoring.gap_open + length * scoring.gap_extend
            j += length
    return total


def brute_force(x: str, y: str, scoring: Scoring, tb: int, te: int) -> int:
    return max(
        model_score(x, y, ops, scoring, tb, te)
        for ops in enumerate_alignments(len(x), len(y))
    )


SCHEMES = [
    Scoring(gap_open=0, gap_extend=1, match=1, mismatch=-1),
    Scoring(gap_open=2, gap_extend=1, match=2, mismatch=-1),
    Scoring(gap_open=5, gap_extend=2, match=3, mismatch=-2),
    Scoring(gap_open=11, gap_extend=1, match=1, mismatch=-4),
]


@pytest.mark.parametrize("scoring", SCHEMES, ids=lambda s: "go%d_ge%d" % (s.gap_open, s.gap_extend))
def test_oracle_matches_brute_force(scoring: Scoring) -> None:
    rng = random.Random(20261011)
    checked = 0
    for n in range(0, 5):
        for m in range(0, 5):
            for _ in range(3):
                x = "".join(rng.choice("AC") for _ in range(n))
                y = "".join(rng.choice("AC") for _ in range(m))
                for tb in (scoring.gap_open, 0):
                    for te in (scoring.gap_open, 0):
                        expected = brute_force(x, y, scoring, tb, te)
                        got = align_score(x, y, scoring, tb=tb, te=te)
                        assert got == expected, (x, y, tb, te, got, expected)
                        result = align(x, y, scoring, tb=tb, te=te)
                        assert result.score == expected
                        checked += 1
    assert checked > 200


@pytest.mark.parametrize("scoring", SCHEMES, ids=lambda s: "go%d_ge%d" % (s.gap_open, s.gap_extend))
def test_oracle_traceback_is_consistent(scoring: Scoring) -> None:
    """The returned alignment, re-scored independently, equals the DP score."""
    rng = random.Random(7)
    for _ in range(300):
        x = "".join(rng.choice("ACGT") for _ in range(rng.randint(0, 12)))
        y = "".join(rng.choice("ACGT") for _ in range(rng.randint(0, 12)))
        result = align(x, y, scoring)
        assert scoring.score_alignment(result.aligned_x, result.aligned_y) == result.score
        assert result.aligned_x.replace("-", "") == x
        assert result.aligned_y.replace("-", "") == y
