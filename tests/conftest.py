"""Shared fixtures and scoring schemes for the test suite."""

from __future__ import annotations

import random
from typing import List

from hirsch.scoring import Scoring

# Three alphabets: binary (forces long gaps and many ties), DNA, and the
# twenty proteinogenic amino acids.
ALPHABETS = {
    "binary": "AC",
    "dna": "ACGT",
    "protein": "ACDEFGHIKLMNPQRSTVWY",
}

# Five gap settings, varying open and extend independently.  ``open=0`` is the
# degenerate linear-gap case, which must keep working; ``open=25`` makes gaps
# so expensive that the optimal path is pushed into long single runs, which is
# exactly where the gap-state handoff matters.
GAP_SETTINGS: List[Scoring] = [
    Scoring(gap_open=0, gap_extend=1, match=1, mismatch=-1),
    Scoring(gap_open=2, gap_extend=1, match=2, mismatch=-1),
    Scoring(gap_open=5, gap_extend=2, match=3, mismatch=-2),
    Scoring(gap_open=11, gap_extend=1, match=1, mismatch=-4),
    Scoring(gap_open=25, gap_extend=1, match=2, mismatch=-3),
]


def scheme_id(scoring: Scoring) -> str:
    return "o%d_e%d_m%d_x%d" % (
        scoring.gap_open,
        scoring.gap_extend,
        scoring.match,
        scoring.mismatch,
    )


def random_sequence(rng: random.Random, alphabet: str, low: int, high: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(low, high)))
