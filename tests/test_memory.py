"""The headline claim, as a test: peak memory growth.

``tracemalloc`` is used rather than RSS because it is deterministic: it counts
bytes the process asked for, not what the allocator or the OS happened to do
about it.  numpy registers its allocations with ``tracemalloc``, so both
implementations are measured with the same instrument (a 32 MB int64 array
shows up as 32,000,096 traced bytes on this machine).

The assertions are about *growth*, not about absolute byte counts, so they do
not depend on the machine.  Doubling n should roughly quadruple the oracle's
peak and roughly double the linear-space aligner's peak.
"""

from __future__ import annotations

import gc
import random
import tracemalloc
from typing import Callable, List, Tuple

import pytest

from hirsch import linear, quadratic
from hirsch.scoring import Scoring

SCORING = Scoring(gap_open=11, gap_extend=1, match=2, mismatch=-1)


def peak_bytes(fn: Callable[[], object]) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def make_pair(n: int, seed: int = 11) -> Tuple[str, str]:
    rng = random.Random(seed)
    alphabet = "ACGT"
    x = "".join(rng.choice(alphabet) for _ in range(n))
    y = "".join(rng.choice(alphabet) for _ in range(n))
    return x, y


SIZES = [100, 200, 400, 800]


def _peaks(aligner: Callable[..., object]) -> List[int]:
    peaks = []
    for n in SIZES:
        x, y = make_pair(n)
        peaks.append(peak_bytes(lambda: aligner(x, y, SCORING)))
    return peaks


def test_tracemalloc_sees_numpy_allocations() -> None:
    """Guard the premise of every other assertion in this file."""
    import numpy as np

    size = 4_000_000  # 4 million int64 = 32 MB
    peak = peak_bytes(lambda: np.zeros(size, dtype=np.int64))
    assert peak >= 8 * size, (
        "tracemalloc is not seeing numpy allocations on this build, so the "
        "memory comparison below would be meaningless"
    )


def test_quadratic_peak_grows_quadratically() -> None:
    peaks = _peaks(quadratic.align)
    ratios = [b / a for a, b in zip(peaks, peaks[1:])]
    assert all(r > 3.3 for r in ratios), (peaks, ratios)
    # Four doublings of n should be about a 64-fold growth in peak memory.
    assert peaks[-1] / peaks[0] > 40, (peaks,)


def test_linear_peak_grows_sub_quadratically() -> None:
    peaks = _peaks(linear.align)
    ratios = [b / a for a, b in zip(peaks, peaks[1:])]
    assert all(r < 2.6 for r in ratios), (peaks, ratios)
    assert peaks[-1] / peaks[0] < 12, (peaks,)


def marginal_bytes_per_residue(aligner, m: int, small: int, large: int) -> float:
    """Peak memory cost of one extra residue of the longer sequence.

    Holding ``y`` fixed and differencing two peaks cancels the fixed
    overhead, which is what makes the two implementations comparable and the
    result portable: it is a slope, not an absolute byte count.
    """
    y = make_pair(m, seed=5)[0]
    x_small = make_pair(small, seed=6)[0]
    x_large = make_pair(large, seed=6)[0]
    peak_small = peak_bytes(lambda: aligner(x_small, y, SCORING))
    peak_large = peak_bytes(lambda: aligner(x_large, y, SCORING))
    return (peak_large - peak_small) / (large - small)


def test_marginal_cost_of_a_longer_x_is_bytes_not_a_dp_row() -> None:
    """Hold ``y`` short and grow ``x``: what does each extra residue cost?

    For the oracle, one more residue of ``x`` means one more row of each of
    three int64 layers, so ``24 * (m + 1)`` bytes.  For the linear-space
    aligner the DP working set does not grow at all; the only growth is the
    encoded copy of ``x`` itself.

    These are the figures quoted in the README, so they are asserted here
    rather than merely bounded.  The oracle's slope is checked against the
    value its own data structure forces, which is a prediction no machine can
    disagree with.
    """
    m = 200
    small, large = 4_000, 20_000

    quadratic_marginal = marginal_bytes_per_residue(
        quadratic.align_score, m, small, large
    )
    predicted = 24 * (m + 1)
    assert abs(quadratic_marginal - predicted) / predicted < 0.02, (
        quadratic_marginal,
        predicted,
    )

    linear_marginal = marginal_bytes_per_residue(linear.align_score, m, small, large)
    # Encoding one residue costs one byte in the latin-1 buffer plus two in
    # the int16 code array; the rest is allocator granularity.
    assert 1 <= linear_marginal <= 10, linear_marginal

    assert quadratic_marginal / linear_marginal > 100, (
        quadratic_marginal,
        linear_marginal,
    )


def test_linear_uses_far_less_than_quadratic_at_the_same_size() -> None:
    """A direct head-to-head at the largest size both can comfortably run."""
    n = 800
    x, y = make_pair(n)
    quad_peak = peak_bytes(lambda: quadratic.align(x, y, SCORING))
    lin_peak = peak_bytes(lambda: linear.align(x, y, SCORING))
    assert quad_peak / lin_peak > 25, (quad_peak, lin_peak)
    # And the two still agree on the answer.
    assert quadratic.align_score(x, y, SCORING) == linear.align_score(x, y, SCORING)


def test_quadratic_peak_matches_three_int64_matrices() -> None:
    """Sanity-check that the oracle's peak is what the data structure implies.

    Three (n+1) x (m+1) int64 layers is 24*(n+1)*(m+1) bytes; anything far off
    that would mean the measurement is picking up something unintended.
    """
    n = 400
    x, y = make_pair(n)
    peak = peak_bytes(lambda: quadratic.align(x, y, SCORING))
    implied = 24 * (n + 1) * (n + 1)
    assert 0.9 * implied < peak < 1.4 * implied, (peak, implied)


@pytest.mark.parametrize("n", [1000, 4000])
def test_linear_peak_is_well_under_a_megabyte_per_thousand_residues(n: int) -> None:
    """A concrete, machine-independent-enough bound on the linear aligner.

    The working set is a constant number of int64 rows of length min(n, m) + 1,
    plus the output alignment. 1 KB per residue is a very loose ceiling on
    that and would be blown immediately by any accidental quadratic allocation.
    """
    x, y = make_pair(n)
    peak = peak_bytes(lambda: linear.align(x, y, SCORING))
    assert peak < 1024 * n, (n, peak)
