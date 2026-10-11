"""The differential property test: linear space must equal quadratic space.

This is the centrepiece of the suite.  For every generated pair it asserts
three things:

1. the linear-space score equals the quadratic oracle's score *exactly*
   (integers throughout, so there is no tolerance to hide behind);
2. the alignment the linear-space aligner returns, re-scored by
   :meth:`hirsch.scoring.Scoring.score_alignment` (which shares no code with
   either aligner), equals that same score;
3. the gapped strings degap back to the original inputs.

The score-only entry point is checked alongside the full-traceback one,
because they take different paths through the code.

The sweep is parameterised over three alphabet sizes and five gap settings
(see ``conftest.py``).  The default run is sized for CI; the much larger sweep
is marked ``slow`` and is skipped unless asked for.
"""

from __future__ import annotations

import random
from typing import Tuple

import pytest

from conftest import ALPHABETS, GAP_SETTINGS, random_sequence, scheme_id
from hirsch import linear, quadratic
from hirsch.scoring import Scoring

# Pairs per (alphabet, gap setting) combination.  There are 3 x 5 = 15
# combinations, so the totals are 15 * the numbers below.
CI_PAIRS_PER_COMBO = 400          # 6,000 pairs
SLOW_PAIRS_PER_COMBO = 2000       # 30,000 pairs


def check_pair(x: str, y: str, scoring: Scoring) -> None:
    expected = quadratic.align_score(x, y, scoring)

    assert linear.align_score(x, y, scoring) == expected, (
        "linear score-only disagrees with the oracle",
        x,
        y,
        scheme_id(scoring),
    )

    result = linear.align(x, y, scoring)
    assert result.score == expected, (
        "linear alignment score disagrees with the oracle",
        x,
        y,
        scheme_id(scoring),
    )
    assert scoring.score_alignment(result.aligned_x, result.aligned_y) == expected, (
        "re-scored alignment disagrees with its own reported score",
        result.aligned_x,
        result.aligned_y,
        scheme_id(scoring),
    )
    assert result.aligned_x.replace("-", "") == x
    assert result.aligned_y.replace("-", "") == y
    assert len(result.aligned_x) == len(result.aligned_y)


def sweep(
    alphabet: str, scoring: Scoring, pairs: int, seed: int, bounds: Tuple[int, int]
) -> int:
    rng = random.Random(seed)
    low, high = bounds
    for _ in range(pairs):
        x = random_sequence(rng, alphabet, low, high)
        y = random_sequence(rng, alphabet, low, high)
        check_pair(x, y, scoring)
    return pairs


@pytest.mark.parametrize("alphabet_name", sorted(ALPHABETS))
@pytest.mark.parametrize("scoring", GAP_SETTINGS, ids=scheme_id)
def test_differential_ci(alphabet_name: str, scoring: Scoring) -> None:
    """6,000 pairs in total across the 15 parameter combinations."""
    seed = hash((alphabet_name, scheme_id(scoring))) & 0xFFFF
    done = sweep(
        ALPHABETS[alphabet_name], scoring, CI_PAIRS_PER_COMBO, seed, (0, 45)
    )
    assert done == CI_PAIRS_PER_COMBO


@pytest.mark.slow
@pytest.mark.parametrize("alphabet_name", sorted(ALPHABETS))
@pytest.mark.parametrize("scoring", GAP_SETTINGS, ids=scheme_id)
def test_differential_slow(alphabet_name: str, scoring: Scoring) -> None:
    """30,000 pairs in total, with longer sequences.  Run locally."""
    seed = 0xC0FFEE ^ (hash((alphabet_name, scheme_id(scoring))) & 0xFFFF)
    done = sweep(
        ALPHABETS[alphabet_name], scoring, SLOW_PAIRS_PER_COMBO, seed, (0, 120)
    )
    assert done == SLOW_PAIRS_PER_COMBO


@pytest.mark.parametrize("scoring", GAP_SETTINGS, ids=scheme_id)
def test_differential_very_lopsided(scoring: Scoring) -> None:
    """One sequence far longer than the other, in both orders.

    Lopsided inputs are where the argument swap and the base case both get
    stressed, and where a mishandled boundary gap charge shows up fastest.
    """
    rng = random.Random(4242)
    for _ in range(40):
        short = random_sequence(rng, "ACGT", 0, 6)
        long = random_sequence(rng, "ACGT", 150, 260)
        check_pair(short, long, scoring)
        check_pair(long, short, scoring)


def test_differential_protein_matrix() -> None:
    """The same property, but with BLOSUM62 instead of flat match/mismatch."""
    from hirsch.scoring import blosum62

    rng = random.Random(19880101)
    for gap_open, gap_extend in [(11, 1), (10, 2), (7, 2), (0, 3)]:
        scoring = blosum62(gap_open, gap_extend)
        for _ in range(60):
            x = random_sequence(rng, ALPHABETS["protein"], 0, 70)
            y = random_sequence(rng, ALPHABETS["protein"], 0, 70)
            check_pair(x, y, scoring)
