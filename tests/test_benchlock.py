"""Tests for the README number gate itself.

A gate that silently passes is worse than no gate, so benchlock gets its own
fixtures: a document where everything agrees, one that is stale but inside its
declared tolerance, one that has genuinely drifted, and several malformed
ones.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import benchlock

FIXTURES = Path(__file__).parent / "fixtures"
VALUES_JSON = FIXTURES / "benchlock_values.json"


def check(fixture: str, tmp_path: Path) -> int:
    """Copy the fixture so no test can ever modify the committed file."""
    target = tmp_path / fixture
    shutil.copy(FIXTURES / fixture, target)
    return benchlock.main(
        ["--readme", str(target), "--json", str(VALUES_JSON), "--check"]
    )


def test_fixture_values_are_loadable() -> None:
    payload = json.loads(VALUES_JSON.read_text())
    assert "values" in payload and payload["values"]


def test_matching_document_passes(tmp_path: Path) -> None:
    assert check("benchlock_ok.md", tmp_path) == 0


def test_stale_but_within_tolerance_passes(tmp_path: Path) -> None:
    """150,000 against a measured 146,216 is 2.6 percent, under the 10 percent
    declared in the marker, so it is allowed."""
    assert check("benchlock_in_tolerance.md", tmp_path) == 0


def test_drifted_number_fails(tmp_path: Path) -> None:
    assert check("benchlock_drifted.md", tmp_path) == 1


def test_exact_span_fails_on_a_single_digit(tmp_path: Path) -> None:
    """A span with no tol= must match exactly."""
    assert check("benchlock_exact_drift.md", tmp_path) == 1


def test_unclosed_marker_is_reported_as_malformed(tmp_path: Path) -> None:
    assert check("benchlock_malformed_unclosed.md", tmp_path) == 2


def test_timing_key_is_refused(tmp_path: Path) -> None:
    """A wall-clock key must be refused outright, not gated loosely.

    The README claims benchlock "refuses to gate a key outside those
    families". This is that claim, tested: the fixture asks to gate
    ``time.n1000_linear`` and benchlock must exit 2 (malformed) rather than
    0 or 1.
    """
    assert check("benchlock_malformed_key.md", tmp_path) == 2


@pytest.mark.parametrize(
    "key",
    [
        "time.n1000_linear",
        "wall.n5000_quadratic",
        "seconds.total",
        "mean_s.n100_linear",
        "stdev_s.n100_linear",
        "throughput.cells_per_second",
        "rss.n1000_linear",
    ],
)
def test_every_timing_shaped_key_family_is_refused(key: str, tmp_path: Path) -> None:
    """Not just the one spelling: nothing time-shaped may be gated.

    Checked directly against ``validate_markers`` so the test does not depend
    on a fixture file existing for each spelling.
    """
    document = "x <!--bench:%s tol=10%%-->1.0<!--/bench--> y" % key
    with pytest.raises(benchlock.MalformedMarkers) as excinfo:
        benchlock.validate_markers(document)
    assert key in str(excinfo.value)


def test_allowed_families_are_exactly_the_deterministic_ones() -> None:
    assert benchlock.ALLOWED_PREFIXES == (
        "mem.",
        "score.",
        "cells.",
        "ratio.",
        "pairs.",
        "marginal.",
    )
    for key in ("mem.x", "score.x", "cells.x", "ratio.x", "pairs.x", "marginal.x"):
        benchlock.validate_markers("<!--bench:%s-->1<!--/bench-->" % key)


def test_update_refuses_to_write_a_timing_key(tmp_path: Path) -> None:
    """The refusal must hold on the write path too, not only on --check."""
    target = tmp_path / "timing.md"
    target.write_text("<!--bench:time.n100_linear-->0.5<!--/bench-->")
    assert (
        benchlock.main(
            ["--readme", str(target), "--json", str(VALUES_JSON), "--update"]
        )
        == 2
    )
    assert target.read_text() == "<!--bench:time.n100_linear-->0.5<!--/bench-->"


def test_non_numeric_value_is_malformed(tmp_path: Path) -> None:
    assert check("benchlock_malformed_value.md", tmp_path) == 2


def test_unknown_key_fails_rather_than_passing_quietly(tmp_path: Path) -> None:
    assert check("benchlock_unknown_key.md", tmp_path) == 1


def test_update_rewrites_a_drifted_span(tmp_path: Path) -> None:
    target = tmp_path / "drifted.md"
    shutil.copy(FIXTURES / "benchlock_drifted.md", target)
    assert (
        benchlock.main(
            ["--readme", str(target), "--json", str(VALUES_JSON), "--update"]
        )
        == 0
    )
    assert "146,216" in target.read_text()
    assert (
        benchlock.main(["--readme", str(target), "--json", str(VALUES_JSON), "--check"])
        == 0
    )


def test_tolerance_parsing() -> None:
    assert benchlock.parse_tolerance("") == (0.0, False)
    assert benchlock.parse_tolerance(" tol=10%") == (10.0, True)
    assert benchlock.parse_tolerance(" tol=5") == (5.0, False)


def test_number_parsing_and_formatting() -> None:
    assert benchlock.parse_number(" 1,002,001 ") == 1002001
    assert benchlock.parse_number("165.3x") == 165.3
    assert benchlock.format_number(1002001) == "1,002,001"
    assert benchlock.format_number(165.3) == "165.3"
    with pytest.raises(benchlock.MalformedMarkers):
        benchlock.parse_number("   ")


def test_the_real_readme_is_gated(tmp_path: Path) -> None:
    """The project README must actually contain gated spans."""
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        pytest.skip("README not written yet")
    spans = benchlock.validate_markers(readme.read_text())
    assert len(spans) >= 6, "the README should gate its headline numbers"


def test_values_can_be_rebuilt_from_the_committed_csv() -> None:
    """The CI path: gate the README against committed data, measuring nothing.

    This is what makes the CI check exact and machine-independent, so it is
    worth asserting that the CSV really does reconstruct the same keys the
    benchmark emits.
    """
    results = REPO_ROOT / "bench" / "results.csv"
    if not results.exists():
        pytest.skip("benchmark results have not been generated")
    values = benchlock.load_values_from_csv(results)
    for key in (
        "mem.n5000_quadratic_bytes",
        "mem.n5000_linear_bytes",
        "cells.n20000",
        "score.n5000_linear",
        "ratio.n5000",
        "marginal.linear_bytes_per_residue",
        "marginal.quadratic_bytes_per_residue",
    ):
        assert key in values, key
    assert values["mem.n5000_quadratic_bytes"] > values["mem.n5000_linear_bytes"]


def test_the_real_readme_matches_the_committed_csv() -> None:
    """The exact check CI runs, run here too so it cannot rot unnoticed."""
    readme = REPO_ROOT / "README.md"
    results = REPO_ROOT / "bench" / "results.csv"
    if not (readme.exists() and results.exists()):
        pytest.skip("README or benchmark results not present")
    values = benchlock.load_values_from_csv(results)
    values.update(benchlock.extra_values())
    problems = benchlock.run_check(readme.read_text(), values)
    assert problems == []
