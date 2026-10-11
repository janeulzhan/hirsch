"""Edge cases, in both aligners, under several gap settings."""

from __future__ import annotations

import pytest

from conftest import GAP_SETTINGS, scheme_id
from hirsch import linear, quadratic
from hirsch.scoring import Scoring, blosum62

ALIGNERS = [("quadratic", quadratic.align), ("linear", linear.align)]
ALIGNER_IDS = [name for name, _ in ALIGNERS]


def both(x, y, scoring):
    q = quadratic.align(x, y, scoring)
    l = linear.align(x, y, scoring)
    assert q.score == l.score
    assert linear.align_score(x, y, scoring) == q.score
    assert quadratic.align_score(x, y, scoring) == q.score
    for result in (q, l):
        assert len(result.aligned_x) == len(result.aligned_y)
        assert result.aligned_x.replace("-", "") == x
        assert result.aligned_y.replace("-", "") == y
        assert scoring.score_alignment(result.aligned_x, result.aligned_y) == result.score
    return q, l


@pytest.mark.parametrize("scoring", GAP_SETTINGS, ids=scheme_id)
def test_both_empty(scoring: Scoring) -> None:
    q, l = both("", "", scoring)
    assert q.score == 0
    assert q.aligned_x == "" and q.aligned_y == ""
    assert l.aligned_x == "" and l.aligned_y == ""


@pytest.mark.parametrize("scoring", GAP_SETTINGS, ids=scheme_id)
def test_one_empty(scoring: Scoring) -> None:
    for n in (1, 2, 5, 40):
        seq = "ACGT" * n
        seq = seq[:n]
        q, _ = both(seq, "", scoring)
        assert q.score == -(scoring.gap_open + len(seq) * scoring.gap_extend)
        assert q.aligned_y == "-" * len(seq)
        q, _ = both("", seq, scoring)
        assert q.score == -(scoring.gap_open + len(seq) * scoring.gap_extend)
        assert q.aligned_x == "-" * len(seq)


@pytest.mark.parametrize("scoring", GAP_SETTINGS, ids=scheme_id)
def test_length_one(scoring: Scoring) -> None:
    for x in ("A", "C"):
        for y in ("A", "C", "AC", "CA", "AAAAA"):
            both(x, y, scoring)
            both(y, x, scoring)


@pytest.mark.parametrize("scoring", GAP_SETTINGS, ids=scheme_id)
def test_identical_sequences(scoring: Scoring) -> None:
    for seq in ("A", "ACGT", "ACGT" * 30, "A" * 101):
        q, _ = both(seq, seq, scoring)
        assert q.score == len(seq) * scoring.match
        assert q.aligned_x == seq and q.aligned_y == seq


@pytest.mark.parametrize("scoring", GAP_SETTINGS, ids=scheme_id)
def test_disjoint_alphabets(scoring: Scoring) -> None:
    """No residue in common, so every column is a mismatch or a gap."""
    x = "AAAA" * 8
    y = "CCCC" * 8
    q, _ = both(x, y, scoring)
    all_mismatch = len(x) * scoring.mismatch
    two_gaps = -2 * (scoring.gap_open + len(x) * scoring.gap_extend)
    assert q.score == max(all_mismatch, two_gaps)


@pytest.mark.parametrize("scoring", GAP_SETTINGS, ids=scheme_id)
def test_all_gap_alignment(scoring: Scoring) -> None:
    """Make mismatching so expensive that gapping everything is optimal."""
    harsh = Scoring(
        gap_open=scoring.gap_open,
        gap_extend=scoring.gap_extend,
        match=1,
        mismatch=-10_000,
    )
    x = "A" * 24
    y = "C" * 24
    q, _ = both(x, y, harsh)
    assert "-" * 24 in q.aligned_x
    assert "-" * 24 in q.aligned_y
    assert q.score == -2 * (harsh.gap_open + 24 * harsh.gap_extend)


@pytest.mark.parametrize("scoring", GAP_SETTINGS, ids=scheme_id)
def test_one_much_longer_than_the_other(scoring: Scoring) -> None:
    both("ACGT", "ACGT" * 150, scoring)
    both("ACGT" * 150, "ACGT", scoring)
    both("A", "C" * 400, scoring)
    both("C" * 400, "A", scoring)


@pytest.mark.parametrize("scoring", GAP_SETTINGS, ids=scheme_id)
def test_single_residue_in_common(scoring: Scoring) -> None:
    both("AAAAAAAAAGAAAAAAAAA", "G", scoring)
    both("G", "AAAAAAAAAGAAAAAAAAA", scoring)


def test_blosum62_edge_cases() -> None:
    scoring = blosum62()
    both("", "", scoring)
    both("M", "", scoring)
    both("", "M", scoring)
    both("MVHLTPEEK", "MVHLTPEEK", scoring)
    both("W", "WWWWWWWWWW", scoring)


def test_unknown_residue_is_reported_clearly() -> None:
    scoring = blosum62()
    with pytest.raises(KeyError) as excinfo:
        quadratic.align("MVHL", "MVJL", scoring)
    assert "BLOSUM62" in str(excinfo.value)


def test_gap_gap_column_is_rejected() -> None:
    scoring = Scoring()
    with pytest.raises(ValueError):
        scoring.score_alignment("A-", "--")


def test_mismatched_alignment_lengths_rejected() -> None:
    with pytest.raises(ValueError):
        Scoring().score_alignment("AAA", "AA")
