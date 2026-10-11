"""hirsch: optimal global sequence alignment with affine gaps, in linear space.

Two implementations of the same problem, kept side by side on purpose:

* :mod:`hirsch.quadratic` is Gotoh's three-layer affine-gap DP with the full
  O(nm) matrix.  Simple, obviously correct, and used as the oracle.
* :mod:`hirsch.linear` is the Myers-Miller algorithm: the same recurrence under
  Hirschberg's divide-and-conquer, using O(min(n, m)) working space.

References
----------
Hirschberg, D. S. (1975).  A linear space algorithm for computing maximal
common subsequences.  Communications of the ACM 18(6):341-343.

Gotoh, O. (1982).  An improved algorithm for matching biological sequences.
Journal of Molecular Biology 162(3):705-708.

Myers, E. W. and Miller, W. (1988).  Optimal alignments in linear space.
Computer Applications in the Biosciences 4(1):11-17.
"""

from ._ops import Alignment
from .scoring import BLOSUM62, Scoring, blosum62

__all__ = ["Alignment", "Scoring", "BLOSUM62", "blosum62", "linear", "quadratic"]
__version__ = "0.1.0"
