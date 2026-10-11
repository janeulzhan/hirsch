"""Scoring schemes for affine-gap global alignment.

Gap model
---------
A run of ``k`` consecutive gap characters costs ``gap_open + k * gap_extend``.
Both values are stored as non-negative penalties and are *subtracted* from the
alignment score, so a single-residue gap costs ``gap_open + gap_extend``.
This is the convention used by Gotoh (1982) and by Myers & Miller (1988).

Substitution scores are integers.  Keeping the whole score space integral is
deliberate: the linear-space aligner is validated against the quadratic
aligner with exact equality, and floating point would make that test
meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Scoring", "BLOSUM62", "blosum62"]


_BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYVBZX*"

# Standard NCBI BLOSUM62 (Henikoff & Henikoff, 1992).  Public domain.
_BLOSUM62_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4
-2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4
-1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
 0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4
-4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
"""


def _parse_blosum62() -> Dict[Tuple[str, str], int]:
    rows = [r for r in _BLOSUM62_ROWS.strip().splitlines() if r.strip()]
    if len(rows) != len(_BLOSUM62_ORDER):
        raise AssertionError("BLOSUM62 table is malformed")
    table: Dict[Tuple[str, str], int] = {}
    for a, row in zip(_BLOSUM62_ORDER, rows):
        values = [int(v) for v in row.split()]
        if len(values) != len(_BLOSUM62_ORDER):
            raise AssertionError("BLOSUM62 row %r is malformed" % a)
        for b, v in zip(_BLOSUM62_ORDER, values):
            table[(a, b)] = v
    return table


BLOSUM62: Dict[Tuple[str, str], int] = _parse_blosum62()


@dataclass(frozen=True)
class Scoring:
    """An affine gap scoring scheme.

    ``gap_open`` and ``gap_extend`` are non-negative penalties; a gap run of
    length ``k`` costs ``gap_open + k * gap_extend``.

    If ``matrix`` is ``None`` the scheme is a flat match/mismatch scheme over
    whatever alphabet the input sequences happen to use.  Otherwise ``matrix``
    maps residue pairs to integer scores (see :data:`BLOSUM62`).
    """

    gap_open: int = 10
    gap_extend: int = 1
    match: int = 1
    mismatch: int = -1
    matrix: Optional[Dict[Tuple[str, str], int]] = field(default=None, repr=False)
    name: str = "match/mismatch"

    def __post_init__(self) -> None:
        if self.gap_open < 0 or self.gap_extend < 0:
            raise ValueError("gap penalties must be non-negative")
        for value in (self.gap_open, self.gap_extend, self.match, self.mismatch):
            if int(value) != value:
                raise ValueError("scores must be integers")

    # -- public helpers ----------------------------------------------------

    def pair(self, a: str, b: str) -> int:
        """Score of aligning residue ``a`` against residue ``b``."""
        if self.matrix is None:
            return self.match if a == b else self.mismatch
        try:
            return self.matrix[(a, b)]
        except KeyError:
            raise KeyError(
                "scoring matrix %r has no entry for pair (%r, %r)" % (self.name, a, b)
            ) from None

    def gap_cost(self, length: int) -> int:
        """Penalty (a positive number) for a gap run of ``length`` residues."""
        if length <= 0:
            return 0
        return self.gap_open + length * self.gap_extend

    def prepare(
        self, x: Sequence[str], y: Sequence[str]
    ) -> Tuple[List[int], List[int], List[List[int]]]:
        """Encode both sequences to integer codes over their joint alphabet.

        Returns ``(x_codes, y_codes, sub)`` where ``sub[a][b]`` is the
        substitution score.  Using small integer codes plus a nested list of
        Python ints keeps the hot DP loop free of dictionary lookups and of
        numpy scalar boxing, both of which dominate runtime otherwise.
        """
        alphabet = sorted(set(x) | set(y))
        index = {ch: i for i, ch in enumerate(alphabet)}
        sub = [[self.pair(a, b) for b in alphabet] for a in alphabet]
        return [index[ch] for ch in x], [index[ch] for ch in y], sub

    def score_alignment(self, aligned_x: str, aligned_y: str) -> int:
        """Independently re-score an already-built alignment.

        This walks the two gapped strings and charges every gap run once.  It
        shares no code with either aligner, which is the point: the test suite
        uses it as a third opinion on any alignment the aligners produce.
        """
        if len(aligned_x) != len(aligned_y):
            raise ValueError("aligned strings must have equal length")
        total = 0
        run = 0
        run_in: Optional[str] = None
        for a, b in zip(aligned_x, aligned_y):
            if a == "-" and b == "-":
                raise ValueError("alignment contains a gap/gap column")
            if a == "-":
                side = "x"
            elif b == "-":
                side = "y"
            else:
                side = None
            if side != run_in:
                total -= self.gap_cost(run)
                run = 0
                run_in = side
            if side is None:
                total += self.pair(a, b)
            else:
                run += 1
        total -= self.gap_cost(run)
        return total


def blosum62(gap_open: int = 11, gap_extend: int = 1) -> Scoring:
    """BLOSUM62 with the NCBI BLAST default protein gap penalties."""
    return Scoring(
        gap_open=gap_open,
        gap_extend=gap_extend,
        matrix=BLOSUM62,
        name="BLOSUM62",
    )
