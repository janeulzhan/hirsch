#!/usr/bin/env python3
"""Keep the numbers in the README honest.

Every measured number in the README lives inside a marker span::

    <!--bench:mem.n5000_quadratic_bytes tol=10%-->571,428,792<!--/bench-->

``--update`` re-runs the benchmark and rewrites the spans.  ``--check`` leaves
the file alone and exits non-zero if a committed number has drifted past the
tolerance declared in its own marker, which is what CI runs.  A span with no
``tol=`` must match exactly.

Only deterministic quantities are allowed in these spans: peak memory from
``tracemalloc``, alignment scores, DP cell counts, and ratios derived from
those.  Wall-clock timings are hardware- and load-dependent, so they are kept
in a separate README table that is explicitly labelled as not CI-verified and
is not gated here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SPAN_RE = re.compile(
    r"<!--bench:(?P<key>[A-Za-z0-9_.]+)(?P<attrs>(?:\s+[a-z]+=[^\s>]+)*)\s*-->"
    r"(?P<value>.*?)"
    r"<!--/bench-->",
    re.DOTALL,
)
OPEN_RE = re.compile(r"<!--bench:[^>]*-->")
CLOSE_RE = re.compile(r"<!--/bench-->")
TOL_RE = re.compile(r"tol=(?P<num>[0-9]+(?:\.[0-9]+)?)(?P<pct>%?)")

# Keys outside these families are rejected: the point of the gate is that only
# deterministic quantities are locked.
ALLOWED_PREFIXES = ("mem.", "score.", "cells.", "ratio.", "pairs.", "marginal.")


class MalformedMarkers(Exception):
    """Raised when the marker spans in a document do not parse cleanly."""


def parse_tolerance(attrs: str) -> Tuple[float, bool]:
    """Return ``(amount, is_relative)``.  No ``tol=`` means exact."""
    match = TOL_RE.search(attrs)
    if not match:
        return 0.0, False
    return float(match.group("num")), bool(match.group("pct"))


def parse_number(text: str) -> float:
    cleaned = text.strip().replace(",", "").replace("x", "")
    if not cleaned:
        raise MalformedMarkers("empty value in a bench span")
    try:
        return float(cleaned)
    except ValueError as exc:
        raise MalformedMarkers("value %r is not a number" % text.strip()) from exc


def format_number(value: object) -> str:
    if isinstance(value, bool):
        raise MalformedMarkers("booleans are not measurable quantities")
    if isinstance(value, int):
        return "{:,}".format(value)
    if isinstance(value, float):
        if value == int(value):
            return "{:,}".format(int(value))
        return "{:,.1f}".format(value)
    raise MalformedMarkers("cannot format value %r" % (value,))


def validate_markers(text: str) -> List[Tuple[str, str, str]]:
    """Parse all spans, failing loudly on anything unbalanced or unknown."""
    spans = [(m.group("key"), m.group("attrs"), m.group("value")) for m in SPAN_RE.finditer(text)]
    opens = len(OPEN_RE.findall(text)) - len(CLOSE_RE.findall(text))
    # Every opener contributes one OPEN_RE hit; closers are counted separately.
    if len(OPEN_RE.findall(text)) != len(spans) or len(CLOSE_RE.findall(text)) != len(spans):
        raise MalformedMarkers(
            "unbalanced bench markers: %d openers, %d closers, %d complete spans"
            % (len(OPEN_RE.findall(text)), len(CLOSE_RE.findall(text)), len(spans))
        )
    del opens
    for key, _, _ in spans:
        if not key.startswith(ALLOWED_PREFIXES):
            raise MalformedMarkers(
                "key %r is not in an allowed family %s; only deterministic "
                "quantities may be gated" % (key, list(ALLOWED_PREFIXES))
            )
    return spans


def _coerce(row: Dict[str, str]) -> Dict[str, object]:
    out: Dict[str, object] = dict(row)
    for field in ("n", "len_x", "len_y", "dp_cells", "peak_bytes", "score"):
        value = row.get(field, "")
        if value != "":
            out[field] = int(value)
    return out


def load_values_from_csv(results_csv: Path) -> Dict[str, object]:
    """Rebuild the gated values from the committed raw CSV files.

    No measurement happens here, so this is exact and machine-independent.
    """
    from bench import benchmark  # noqa: PLC0415  (deliberately lazy)

    with results_csv.open(newline="") as handle:
        rows = [_coerce(row) for row in csv.DictReader(handle)]
    if not rows:
        raise MalformedMarkers("%s contains no rows" % results_csv)

    marginal: Dict[str, float] = {}
    marginal_path = results_csv.with_name("marginal.csv")
    if marginal_path.exists():
        with marginal_path.open(newline="") as handle:
            for record in csv.DictReader(handle):
                marginal[record["metric"]] = float(record["value"])

    report = {"rows": rows, "marginal": marginal}
    return dict(benchmark.flatten(report))


def load_values(
    json_path: Optional[Path],
    csv_path: Optional[Path] = None,
    timed_runs: int = 3,
) -> Dict[str, object]:
    if json_path is not None:
        payload = json.loads(json_path.read_text())
        return dict(payload["values"])
    if csv_path is not None:
        return load_values_from_csv(csv_path)
    from bench import benchmark  # noqa: PLC0415  (deliberately lazy)

    report = benchmark.run(timed_runs)
    return dict(benchmark.flatten(report))


def extra_values() -> Dict[str, object]:
    """Deterministic counts that do not come from the benchmark harness."""
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from test_differential import (  # noqa: PLC0415
        CI_PAIRS_PER_COMBO,
        SLOW_PAIRS_PER_COMBO,
    )
    from conftest import ALPHABETS, GAP_SETTINGS  # noqa: PLC0415

    combos = len(ALPHABETS) * len(GAP_SETTINGS)
    return {
        "pairs.combos": combos,
        "pairs.ci": combos * CI_PAIRS_PER_COMBO,
        "pairs.slow": combos * SLOW_PAIRS_PER_COMBO,
    }


def drift(committed: float, measured: float, amount: float, relative: bool) -> Optional[str]:
    if relative:
        allowed = abs(measured) * amount / 100.0
    else:
        allowed = amount
    if abs(committed - measured) <= allowed:
        return None
    return "committed %s, measured %s (allowed drift %s)" % (
        format_number(committed),
        format_number(measured),
        format_number(allowed) if allowed else "0",
    )


def run_check(text: str, values: Dict[str, object]) -> List[str]:
    problems: List[str] = []
    spans = validate_markers(text)
    if not spans:
        problems.append("no bench spans found; the gate would be vacuous")
    for key, attrs, raw in spans:
        if key not in values:
            problems.append("%s: no measured value for this key" % key)
            continue
        amount, relative = parse_tolerance(attrs)
        message = drift(parse_number(raw), float(values[key]), amount, relative)  # type: ignore[arg-type]
        if message:
            problems.append("%s: %s" % (key, message))
    return problems


def run_update(text: str, values: Dict[str, object]) -> Tuple[str, List[str]]:
    validate_markers(text)
    missing: List[str] = []

    def replace(match: "re.Match[str]") -> str:
        key = match.group("key")
        attrs = match.group("attrs")
        if key not in values:
            missing.append(key)
            return match.group(0)
        return "<!--bench:%s%s-->%s<!--/bench-->" % (
            key,
            attrs,
            format_number(values[key]),
        )

    return SPAN_RE.sub(replace, text), missing


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readme", type=Path, default=REPO_ROOT / "README.md", help="document to gate"
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="read values from this JSON file instead of running the benchmark",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="?",
        const=REPO_ROOT / "bench" / "results.csv",
        help="read values from the committed raw CSV instead of measuring; "
        "exact and machine-independent, and what CI uses",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify only, never write")
    mode.add_argument("--update", action="store_true", help="rewrite the spans in place")
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="timed benchmark runs per cell; the gated memory numbers do not "
        "depend on this, so CI passes --runs 1",
    )
    parser.add_argument(
        "--no-extras",
        action="store_true",
        help="skip the non-benchmark deterministic counts (test pair totals)",
    )
    args = parser.parse_args(argv)

    if not args.check and not args.update:
        args.check = True

    text = args.readme.read_text()
    try:
        values = load_values(args.json, args.csv, args.runs)
        if not args.no_extras:
            values.update(extra_values())

        if args.check:
            problems = run_check(text, values)
            if problems:
                print("benchlock: %d problem(s) in %s" % (len(problems), args.readme))
                for problem in problems:
                    print("  - %s" % problem)
                return 1
            print("benchlock: all gated numbers in %s match" % args.readme)
            return 0

        updated, missing = run_update(text, values)
        if missing:
            print("benchlock: no measured value for %s" % ", ".join(sorted(set(missing))))
            return 1
        if updated != text:
            args.readme.write_text(updated)
            print("benchlock: updated %s" % args.readme)
        else:
            print("benchlock: %s already up to date" % args.readme)
        return 0
    except MalformedMarkers as exc:
        print("benchlock: malformed markers in %s: %s" % (args.readme, exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
