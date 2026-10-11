"""The same input must always produce the same alignment.

Co-optimal alignments are the norm, not the exception, so "optimal" alone
does not pin down the output.  Both aligners resolve ties by a fixed
predecessor preference order, which makes their output reproducible.  That
matters for the committed example output and for gating numbers in CI.
"""

from __future__ import annotations

import random
import subprocess
import sys

from conftest import GAP_SETTINGS, random_sequence, scheme_id
from hirsch import linear, quadratic
from hirsch.scoring import Scoring, blosum62


def test_repeated_calls_agree_within_a_process() -> None:
    rng = random.Random(31337)
    for scoring in GAP_SETTINGS:
        for _ in range(40):
            x = random_sequence(rng, "ACGT", 0, 60)
            y = random_sequence(rng, "ACGT", 0, 60)
            first = linear.align(x, y, scoring)
            for _ in range(3):
                again = linear.align(x, y, scoring)
                assert again == first, (x, y, scheme_id(scoring))
            q_first = quadratic.align(x, y, scoring)
            assert quadratic.align(x, y, scoring) == q_first


def test_alignment_is_stable_across_processes() -> None:
    """A fresh interpreter produces byte-identical output.

    Guards against anything leaking in from hash randomisation or from set or
    dict iteration order in the alphabet encoding.
    """
    code = (
        "from hirsch import linear;"
        "from hirsch.scoring import blosum62;"
        "a=linear.align('MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF',"
        "'MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDL',blosum62());"
        "print(a.score);print(a.aligned_x);print(a.aligned_y)"
    )
    runs = set()
    for _ in range(3):
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        runs.add(out.stdout)
    assert len(runs) == 1, "output differed between interpreter runs"


def test_tie_rich_input_is_deterministic() -> None:
    """Homopolymers give an enormous number of co-optimal alignments."""
    scoring = Scoring(gap_open=2, gap_extend=1, match=1, mismatch=-1)
    x = "A" * 40
    y = "A" * 25
    first = linear.align(x, y, scoring)
    for _ in range(5):
        assert linear.align(x, y, scoring) == first
    assert first.score == quadratic.align_score(x, y, scoring)


def test_score_only_and_full_alignment_agree() -> None:
    rng = random.Random(555)
    scoring = blosum62()
    for _ in range(60):
        x = random_sequence(rng, "ACDEFGHIKLMNPQRSTVWY", 0, 80)
        y = random_sequence(rng, "ACDEFGHIKLMNPQRSTVWY", 0, 80)
        assert linear.align_score(x, y, scoring) == linear.align(x, y, scoring).score
