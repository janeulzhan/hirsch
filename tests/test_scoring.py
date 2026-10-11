"""Scoring-scheme tests, including the embedded BLOSUM62 table."""

from __future__ import annotations

import pytest

from hirsch.scoring import BLOSUM62, Scoring, blosum62

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVBZX*"


def test_blosum62_is_complete_and_symmetric() -> None:
    assert len(BLOSUM62) == len(AMINO_ACIDS) ** 2
    for a in AMINO_ACIDS:
        for b in AMINO_ACIDS:
            assert BLOSUM62[(a, b)] == BLOSUM62[(b, a)], (a, b)


def test_blosum62_known_entries() -> None:
    """Spot-checks against the published NCBI table."""
    expected = {
        ("A", "A"): 4,
        ("W", "W"): 11,
        ("C", "C"): 9,
        ("H", "H"): 8,
        ("W", "C"): -2,
        ("A", "R"): -1,
        ("D", "E"): 2,
        ("I", "L"): 2,
        ("F", "Y"): 3,
        ("K", "R"): 2,
        ("G", "G"): 6,
        ("P", "P"): 7,
        ("*", "*"): 1,
        ("*", "A"): -4,
    }
    for pair, value in expected.items():
        assert BLOSUM62[pair] == value, pair


def test_blosum62_diagonal_is_never_worse_than_off_diagonal() -> None:
    """A residue should score at least as well against itself as against
    anything else, for every residue except the stop codon marker."""
    for a in AMINO_ACIDS.replace("*", "").replace("X", ""):
        own = BLOSUM62[(a, a)]
        for b in AMINO_ACIDS:
            if b != a:
                assert BLOSUM62[(a, b)] <= own, (a, b)


def test_gap_cost_arithmetic() -> None:
    scoring = Scoring(gap_open=11, gap_extend=1)
    assert scoring.gap_cost(0) == 0
    assert scoring.gap_cost(-3) == 0
    assert scoring.gap_cost(1) == 12
    assert scoring.gap_cost(5) == 16
    linear_gaps = Scoring(gap_open=0, gap_extend=4)
    assert linear_gaps.gap_cost(3) == 12


def test_negative_gap_penalties_rejected() -> None:
    with pytest.raises(ValueError):
        Scoring(gap_open=-1, gap_extend=1)
    with pytest.raises(ValueError):
        Scoring(gap_open=1, gap_extend=-1)


def test_non_integer_scores_rejected() -> None:
    with pytest.raises(ValueError):
        Scoring(gap_open=1, gap_extend=1, match=1.5)


def test_score_alignment_charges_each_run_once() -> None:
    scoring = Scoring(gap_open=10, gap_extend=1, match=2, mismatch=-1)
    # Two separate gap runs of 2, so two opens.
    assert scoring.score_alignment("AA--CC--", "AAGGCCGG") == 4 * 2 - 2 * (10 + 2)
    # One run of 4, so one open.
    assert scoring.score_alignment("AA----CC", "AAGGGGCC") == 4 * 2 - (10 + 4)


def test_score_alignment_handles_terminal_gaps() -> None:
    scoring = Scoring(gap_open=5, gap_extend=1, match=1, mismatch=-1)
    assert scoring.score_alignment("---ACGT", "GGGACGT") == 4 - (5 + 3)
    assert scoring.score_alignment("ACGT---", "ACGTGGG") == 4 - (5 + 3)


def test_prepare_covers_the_joint_alphabet() -> None:
    scoring = Scoring(match=3, mismatch=-2)
    xc, yc, sub = scoring.prepare("AC", "CG")
    assert len(sub) == 3  # A, C, G
    assert sub[xc[0]][yc[0]] == -2
    assert sub[xc[1]][yc[0]] == 3


def test_blosum62_factory_defaults() -> None:
    scoring = blosum62()
    assert scoring.gap_open == 11 and scoring.gap_extend == 1
    assert scoring.name == "BLOSUM62"
    assert scoring.pair("W", "W") == 11
