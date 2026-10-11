"""Tests aimed squarely at the gap-state handoff at the split row.

The subtle failure mode in Myers-Miller is this: the forward and reverse half
matrices must agree at the split row not only on the *score* but on the *gap
state*.  If the recombination just takes the best score at the midpoint, then
whenever the optimal path crosses the split row in the middle of a vertical
gap, both halves separately pay that gap's open penalty.  The reported score
comes out too low and the assembled alignment is suboptimal.

These tests do three things:

* re-implement the naive recombination locally and show that it really does
  produce wrong answers (so the production fix is guarding a live bug, not a
  hypothetical one);
* confirm the gap-spanning code path is actually taken, using the split
  statistics the aligner exposes;
* check correctness on inputs constructed so the optimal path *must* cross
  the split row inside a long deletion.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import pytest

from conftest import ALPHABETS, GAP_SETTINGS, random_sequence, scheme_id
from hirsch import linear, quadratic
from hirsch._ops import NEG_INF
from hirsch.scoring import Scoring


def naive_split_score(x: str, y: str, scoring: Scoring) -> int:
    """Myers-Miller with the gap-state handoff left out.

    Identical to :func:`hirsch.linear.align_score` except that the split-row
    recombination only ever considers ``max(forward) + max(reverse)`` and the
    recursion never waives a boundary gap-open.  This is the version a careful
    but unlucky implementer writes, and it is what the production code is
    defending against.
    """
    xc, yc, sub = scoring.prepare(x, y)
    go, ge = scoring.gap_open, scoring.gap_extend
    ctx = linear._Context(xc, yc, sub, go, ge)

    def base(i1: int, i2: int, j1: int, j2: int, tb: int, te: int) -> int:
        n, m = i2 - i1, j2 - j1
        if n == 0:
            return -(go + m * ge) if m else 0
        if m == 0:
            return -(min(tb, te) + n * ge)
        M, X, Y, Xe = quadratic.fill_matrices(
            list(xc[i1:i2]), list(yc[j1:j2]), sub, go, ge, tb, te
        )
        return max(int(M[n, m]), int(Y[n, m]), int(Xe[n]))

    def rec(i1: int, i2: int, j1: int, j2: int, tb: int, te: int) -> int:
        n, m = i2 - i1, j2 - j1
        if n <= 1 or m == 0:
            return base(i1, i2, j1, j2, tb, te)
        r = n // 2
        fm, fx, fy = linear._layers(ctx, i1, 1, r, j1, 1, m, tb)
        gm, gx, gy = linear._layers(ctx, i2 - 1, -1, n - r, j2 - 1, -1, m, te)
        best, best_j = NEG_INF, 0
        for jj in range(m + 1):
            k = m - jj
            value = int(max(fm[jj], fx[jj], fy[jj])) + int(
                max(gm[k], gx[k], gy[k])
            )
            if value > best:
                best, best_j = value, jj
        return rec(i1, i1 + r, j1, j1 + best_j, tb, go) + rec(
            i1 + r, i2, j1 + best_j, j2, go, te
        )

    if not x:
        return -scoring.gap_cost(len(y))
    if not y:
        return -scoring.gap_cost(len(x))
    return rec(0, len(x), 0, len(y), go, go)


def find_spanning_cases(
    scoring: Scoring, wanted: int, seed: int, alphabet: str = "AC"
) -> List[Tuple[str, str]]:
    """Search random pairs for ones whose optimal path spans the split row."""
    rng = random.Random(seed)
    found: List[Tuple[str, str]] = []
    for _ in range(4000):
        if len(found) >= wanted:
            break
        x = random_sequence(rng, alphabet, 8, 34)
        y = random_sequence(rng, alphabet, 4, 20)
        _, stats = linear.split_statistics(x, y, scoring)
        if stats["spanning_splits"]:
            found.append((x, y))
    return found


def test_naive_recombination_is_actually_broken() -> None:
    """Sanity-check the premise: the naive version gets real inputs wrong.

    If this test ever starts failing it means the naive implementation above
    drifted into correctness, and the rest of this file stops proving
    anything.
    """
    scoring = Scoring(gap_open=2, gap_extend=1, match=2, mismatch=-1)
    cases = find_spanning_cases(scoring, 60, seed=99)
    assert cases, "no spanning cases were found to test with"
    wrong = 0
    for x, y in cases:
        correct = quadratic.align_score(x, y, scoring)
        if naive_split_score(x, y, scoring) != correct:
            wrong += 1
    assert wrong > 0, (
        "the naive recombination answered every spanning case correctly, "
        "so this test no longer demonstrates the bug it guards"
    )


def test_naive_recombination_only_ever_undershoots() -> None:
    """The naive bug is a silent *under*estimate, never an overestimate.

    Worth pinning down, because it is why the bug survives casual testing:
    nothing ever looks impossible, the scores are just quietly too low.
    """
    scoring = Scoring(gap_open=5, gap_extend=2, match=3, mismatch=-2)
    cases = find_spanning_cases(scoring, 40, seed=7)
    assert cases
    for x, y in cases:
        correct = quadratic.align_score(x, y, scoring)
        assert naive_split_score(x, y, scoring) <= correct


@pytest.mark.parametrize("scoring", GAP_SETTINGS, ids=scheme_id)
def test_spanning_cases_are_aligned_correctly(scoring: Scoring) -> None:
    """On pairs that do span the split row, the fixed version is exact."""
    if scoring.gap_open == 0:
        pytest.skip("with gap_open=0 there is no open penalty to double-charge")
    cases = find_spanning_cases(scoring, 40, seed=hash(scheme_id(scoring)) & 0xFFFF)
    assert cases, "no spanning cases found for %s" % scheme_id(scoring)
    for x, y in cases:
        expected = quadratic.align_score(x, y, scoring)
        result = linear.align(x, y, scoring)
        assert result.score == expected
        assert (
            scoring.score_alignment(result.aligned_x, result.aligned_y) == expected
        )


def test_handoff_path_is_exercised_often() -> None:
    """Report-worthy: the spanning path is common, not exotic.

    This is why the handoff cannot be dismissed as a corner case.
    """
    scoring = Scoring(gap_open=2, gap_extend=1, match=2, mismatch=-1)
    rng = random.Random(2026)
    spanning = 0
    trials = 300
    for _ in range(trials):
        x = random_sequence(rng, "AC", 10, 30)
        y = random_sequence(rng, "AC", 4, 20)
        _, stats = linear.split_statistics(x, y, scoring)
        if stats["spanning_splits"]:
            spanning += 1
    assert spanning > trials // 10, (
        "expected the gap-spanning split to be hit on a sizeable fraction of "
        "random pairs, saw %d/%d" % (spanning, trials)
    )


@pytest.mark.parametrize("insert_len", [3, 7, 16, 41, 64])
@pytest.mark.parametrize("flank", [4, 9, 25])
def test_deletion_straddling_the_split_row(insert_len: int, flank: int) -> None:
    """A constructed case: one long deletion centred on the split row.

    ``x`` is two conserved flanks with a block of unrelated residues wedged
    between them; ``y`` is just the two flanks.  With a cheap extend and an
    expensive open, the optimal alignment is a single deletion of the whole
    wedge.  The split row ``len(x) // 2`` lands inside that wedge whenever the
    wedge is long relative to the flanks, so the optimal path crosses the
    split row strictly inside a vertical gap.  There is no split column that
    avoids it.
    """
    scoring = Scoring(gap_open=30, gap_extend=1, match=5, mismatch=-4)
    flank_a = "ACGTACGTACGTACGTACGTACGTA"[:flank]
    flank_b = "TGCATGCATGCATGCATGCATGCAT"[:flank]
    wedge = "".join("WYFP"[i % 4] for i in range(insert_len))
    x = flank_a + wedge + flank_b
    y = flank_a + flank_b

    expected = quadratic.align_score(x, y, scoring)
    result, stats = linear.split_statistics(x, y, scoring)
    assert result.score == expected
    assert scoring.score_alignment(result.aligned_x, result.aligned_y) == expected

    # And the alignment really is one clean deletion of the wedge, not two
    # cheaper-looking halves stitched together at the split row.
    assert result.aligned_y.count("-") == insert_len
    assert "-" * insert_len in result.aligned_y
    assert stats["plain_splits"] + stats["spanning_splits"] > 0


def test_spanning_deletion_is_where_the_naive_version_fails() -> None:
    """The constructed straddling-deletion family breaks the naive version.

    Searching is not required here; the geometry is chosen so the failure is
    reproducible, which makes this the clearest single regression test for the
    handoff.
    """
    scoring = Scoring(gap_open=30, gap_extend=1, match=5, mismatch=-4)
    failures = 0
    for insert_len in (9, 17, 33, 65):
        x = "ACGTACGT" + "W" * insert_len + "TGCATGCA"
        y = "ACGTACGT" + "TGCATGCA"
        correct = quadratic.align_score(x, y, scoring)
        assert linear.align_score(x, y, scoring) == correct
        assert linear.align(x, y, scoring).score == correct
        if naive_split_score(x, y, scoring) != correct:
            failures += 1
    assert failures > 0


def test_how_often_the_naive_recombination_is_wrong() -> None:
    """Quantify the bug across the whole parameter grid, with exact counts.

    Everything here is seeded, so the counts are reproducible and are quoted
    in the README.  Two things are worth pinning:

    * at ``gap_open = 0`` the naive version is never wrong, because with no
      open penalty there is nothing to double-charge.  That is the theory
      making a falsifiable prediction, and it holds for all three alphabets.
    * at every affine setting it is wrong on roughly half of random pairs.
      This is not a rare corner case.
    """
    pairs_per_combo = 100
    wrong_total = 0
    spanning_total = 0
    checked = 0
    zero_open_wrong = 0

    for alphabet_name in sorted(ALPHABETS):
        for scoring in GAP_SETTINGS:
            rng = random.Random(4242)
            for _ in range(pairs_per_combo):
                x = random_sequence(rng, ALPHABETS[alphabet_name], 8, 40)
                y = random_sequence(rng, ALPHABETS[alphabet_name], 4, 30)
                correct = quadratic.align_score(x, y, scoring)
                result, stats = linear.split_statistics(x, y, scoring)

                # The fixed implementation is exact on every single pair.
                assert result.score == correct, (x, y, scheme_id(scoring))
                assert (
                    scoring.score_alignment(result.aligned_x, result.aligned_y)
                    == correct
                )

                checked += 1
                if stats["spanning_splits"]:
                    spanning_total += 1
                if naive_split_score(x, y, scoring) != correct:
                    wrong_total += 1
                    if scoring.gap_open == 0:
                        zero_open_wrong += 1

    assert checked == 1500
    assert spanning_total == 954
    assert wrong_total == 772
    assert zero_open_wrong == 0, (
        "with gap_open=0 there is no open penalty to double-charge, so the "
        "naive recombination must be exact"
    )
