"""Internal alignment representation shared by both aligners.

An alignment is built as a list of runs ``(op, count)`` in left-to-right
order, where

* ``"M"`` consumes one residue from each sequence (a match or mismatch column),
* ``"D"`` consumes one residue from ``x`` and emits a gap in ``y``,
* ``"I"`` consumes one residue from ``y`` and emits a gap in ``x``.

Run-length encoding keeps peak memory of the *output* proportional to the
number of gap runs rather than to the alignment length while the recursion is
still unwinding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

__all__ = ["Alignment", "render_ops", "NEG_INF"]

# Sentinel for "unreachable".  Chosen so that repeated subtraction of gap
# penalties can never wrap into a reachable score, and so that it stays far
# inside int64 to keep the numpy matrices exact.
NEG_INF = -(1 << 50)

Op = Tuple[str, int]


@dataclass(frozen=True)
class Alignment:
    """A global alignment of two sequences.

    ``aligned_x`` and ``aligned_y`` have equal length and use ``-`` for gaps.
    ``score`` is the optimal affine-gap score under the scoring scheme used.
    """

    score: int
    aligned_x: str
    aligned_y: str

    def __len__(self) -> int:
        return len(self.aligned_x)

    @property
    def identities(self) -> int:
        return sum(1 for a, b in zip(self.aligned_x, self.aligned_y) if a == b)

    @property
    def gap_columns(self) -> int:
        return sum(1 for a, b in zip(self.aligned_x, self.aligned_y) if "-" in (a, b))

    def percent_identity(self) -> float:
        if not self.aligned_x:
            return 0.0
        return 100.0 * self.identities / len(self.aligned_x)


def render_ops(x: Sequence[str], y: Sequence[str], ops: List[Op]) -> Tuple[str, str]:
    """Expand run-length-encoded ops into the two gapped strings."""
    out_x: List[str] = []
    out_y: List[str] = []
    i = j = 0
    for op, count in ops:
        if count <= 0:
            continue
        if op == "M":
            out_x.append("".join(x[i : i + count]))
            out_y.append("".join(y[j : j + count]))
            i += count
            j += count
        elif op == "D":
            out_x.append("".join(x[i : i + count]))
            out_y.append("-" * count)
            i += count
        elif op == "I":
            out_x.append("-" * count)
            out_y.append("".join(y[j : j + count]))
            j += count
        else:  # pragma: no cover - defensive
            raise ValueError("unknown op %r" % (op,))
    if i != len(x) or j != len(y):  # pragma: no cover - defensive
        raise AssertionError(
            "ops consumed %d/%d of x and %d/%d of y" % (i, len(x), j, len(y))
        )
    return "".join(out_x), "".join(out_y)
