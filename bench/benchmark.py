#!/usr/bin/env python3
"""Benchmark harness: peak memory and wall time for both aligners.

Peak memory comes from ``tracemalloc`` and is deterministic, so it is the
number this project actually gates in CI.  Wall time is reported too, but it
is hardware- and load-dependent and is never asserted on anywhere.

Usage::

    python3 bench/benchmark.py                 # human-readable table
    python3 bench/benchmark.py --csv bench/results.csv
    python3 bench/benchmark.py --json          # machine-readable, for benchlock

The quadratic aligner is skipped at sizes where three (n+1)x(m+1) int64
layers would not fit comfortably in memory; the skip is reported rather than
hidden.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from hirsch import linear, quadratic
from hirsch.scoring import Scoring

SIZES = [100, 1_000, 5_000, 20_000]
SCORING = Scoring(gap_open=11, gap_extend=1, match=2, mismatch=-1)

# Three int64 layers of (n+1)x(m+1) cells.  Above this many bytes the
# quadratic aligner is skipped rather than run.
QUADRATIC_BYTE_CEILING = 1_500_000_000

# Peak memory is always measured in a dedicated run that follows exactly one
# warmup call, no matter how many timed runs are requested. That keeps the
# gated memory numbers identical between a full local run and the fast run CI
# uses, while the timing samples stay configurable.
WARMUP_RUNS = 1
DEFAULT_TIMED_RUNS = 3


def hardware() -> Dict[str, str]:
    def sysctl(key: str) -> str:
        try:
            return subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return "unavailable"

    memsize = sysctl("hw.memsize")
    try:
        mem_human = "%.0f GiB" % (int(memsize) / (1024 ** 3))
    except ValueError:
        mem_human = memsize
    return {
        "cpu": sysctl("machdep.cpu.brand_string"),
        "memsize_bytes": memsize,
        "memsize": mem_human,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def make_pair(n: int, divergence: float = 0.12, seed: int = 20261011):
    """A related pair: y is x with substitutions and a few indels.

    Random unrelated sequences would be a slightly unfair benchmark, because
    the optimal alignment is then mostly gaps and the traceback is trivial.
    Making the pair homologous keeps the recursion doing real work.
    """
    rng = random.Random(seed)
    alphabet = "ACGT"
    x = [rng.choice(alphabet) for _ in range(n)]
    y: List[str] = []
    for ch in x:
        roll = rng.random()
        if roll < divergence * 0.7:
            y.append(rng.choice(alphabet))
        elif roll < divergence * 0.85:
            continue  # deletion
        elif roll < divergence:
            y.append(ch)
            y.append(rng.choice(alphabet))  # insertion
        else:
            y.append(ch)
    return "".join(x), "".join(y)


def measure(fn: Callable[[], object], timed_runs: int = DEFAULT_TIMED_RUNS) -> Dict[str, float]:
    """Warm up, then time several runs, then measure peak memory separately.

    Memory is measured in its own run because ``tracemalloc`` adds noticeable
    overhead to wall time and would contaminate the timing numbers.
    """
    for _ in range(WARMUP_RUNS):
        fn()

    times = []
    for _ in range(timed_runs):
        gc.collect()
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)

    gc.collect()
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return {
        "mean_s": statistics.fmean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "peak_bytes": float(peak),
        "score": float(getattr(result, "score", float("nan"))),
    }


MARGINAL_M = 200
MARGINAL_SMALL_N = 4_000
MARGINAL_LARGE_N = 20_000


def measure_marginal() -> Dict[str, float]:
    """What does one more residue of the *longer* sequence cost?

    Hold ``y`` short and fixed, grow ``x``, and take the difference in peak
    memory divided by the difference in length.  Subtracting two peaks cancels
    the fixed overhead, which is what makes this comparable between the two
    implementations.

    For the oracle the answer is forced by the data structure: one more
    residue of ``x`` is one more row of each of three int64 layers, so
    ``24 * (m + 1)`` bytes.  Reporting the measurement next to that prediction
    is the point; if they ever stop agreeing, something is wrong with either
    the implementation or the measurement.
    """
    m = MARGINAL_M
    y = make_pair(m, seed=777)[0]
    x_small = make_pair(MARGINAL_SMALL_N, seed=778)[0]
    x_large = make_pair(MARGINAL_LARGE_N, seed=778)[0]
    span = MARGINAL_LARGE_N - MARGINAL_SMALL_N

    out: Dict[str, float] = {}
    for name, fn in (
        ("linear", linear.align_score),
        ("quadratic", quadratic.align_score),
    ):
        small = measure(lambda: fn(x_small, y, SCORING), 1)["peak_bytes"]
        large = measure(lambda: fn(x_large, y, SCORING), 1)["peak_bytes"]
        out["%s_bytes_per_residue" % name] = round((large - small) / span, 2)
        out["%s_peak_small" % name] = small
        out["%s_peak_large" % name] = large
    out["predicted_quadratic_bytes_per_residue"] = float(24 * (m + 1))
    out["ratio"] = round(
        out["quadratic_bytes_per_residue"] / out["linear_bytes_per_residue"], 1
    )
    return out


def run(timed_runs: int = DEFAULT_TIMED_RUNS, sizes: Optional[List[int]] = None) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for n in sizes or SIZES:
        x, y = make_pair(n)
        cells = (len(x) + 1) * (len(y) + 1)
        quadratic_bytes = 24 * cells

        lin = measure(lambda: linear.align(x, y, SCORING), timed_runs)
        rows.append(
            {
                "n": n,
                "len_x": len(x),
                "len_y": len(y),
                "implementation": "linear",
                "dp_cells": cells,
                "peak_bytes": int(lin["peak_bytes"]),
                "mean_s": round(lin["mean_s"], 6),
                "stdev_s": round(lin["stdev_s"], 6),
                "score": int(lin["score"]),
                "skipped": "",
            }
        )

        if quadratic_bytes <= QUADRATIC_BYTE_CEILING:
            quad = measure(lambda: quadratic.align(x, y, SCORING), timed_runs)
            rows.append(
                {
                    "n": n,
                    "len_x": len(x),
                    "len_y": len(y),
                    "implementation": "quadratic",
                    "dp_cells": cells,
                    "peak_bytes": int(quad["peak_bytes"]),
                    "mean_s": round(quad["mean_s"], 6),
                    "stdev_s": round(quad["stdev_s"], 6),
                    "score": int(quad["score"]),
                    "skipped": "",
                }
            )
            assert int(quad["score"]) == int(lin["score"]), (
                "the two implementations disagreed at n=%d" % n
            )
        else:
            rows.append(
                {
                    "n": n,
                    "len_x": len(x),
                    "len_y": len(y),
                    "implementation": "quadratic",
                    "dp_cells": cells,
                    "peak_bytes": "",
                    "mean_s": "",
                    "stdev_s": "",
                    "score": "",
                    "skipped": "three int64 layers would need %.1f GB"
                    % (quadratic_bytes / 1e9),
                }
            )
    return {
        "hardware": hardware(),
        "rows": rows,
        "marginal": measure_marginal(),
    }


def flatten(report: Dict[str, object]) -> Dict[str, object]:
    """Dotted keys for benchlock, covering only deterministic quantities."""
    flat: Dict[str, object] = {}
    for row in report["rows"]:  # type: ignore[index]
        key = "%s%d_%s" % ("n", row["n"], row["implementation"])
        if row["peak_bytes"] != "":
            flat["mem.%s_bytes" % key] = int(row["peak_bytes"])
            flat["mem.%s_kib" % key] = round(int(row["peak_bytes"]) / 1024)
            flat["mem.%s_mib" % key] = round(int(row["peak_bytes"]) / (1024 * 1024), 1)
        if row["score"] != "":
            flat["score.%s" % key] = int(row["score"])
        flat["cells.n%d" % row["n"]] = int(row["dp_cells"])
    for name, value in report.get("marginal", {}).items():  # type: ignore[union-attr]
        flat["marginal.%s" % name] = value

    by = {
        (r["n"], r["implementation"]): r for r in report["rows"]  # type: ignore[index]
    }
    for n in sorted({int(r["n"]) for r in report["rows"]}):  # type: ignore[index]
        lin = by.get((n, "linear"))
        quad = by.get((n, "quadratic"))
        if lin and quad and quad["peak_bytes"] != "":
            flat["ratio.n%d" % n] = round(
                int(quad["peak_bytes"]) / int(lin["peak_bytes"]), 1
            )
    return flat


FIELDNAMES = [
    "n",
    "len_x",
    "len_y",
    "implementation",
    "dp_cells",
    "peak_bytes",
    "mean_s",
    "stdev_s",
    "score",
    "skipped",
]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="write raw results to this CSV path")
    parser.add_argument("--json", action="store_true", help="emit JSON on stdout")
    parser.add_argument(
        "--sizes",
        type=lambda v: [int(part) for part in v.split(",")],
        help="comma-separated sequence lengths to benchmark (default: %s)"
        % ",".join(str(n) for n in SIZES),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_TIMED_RUNS,
        help="timed runs per cell (does not affect the gated memory numbers)",
    )
    args = parser.parse_args(argv)

    report = run(args.runs, args.sizes)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in report["rows"]:  # type: ignore[index]
                writer.writerow(row)

        marginal_path = args.csv.with_name("marginal.csv")
        with marginal_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for key, value in sorted(report["marginal"].items()):  # type: ignore[index]
                writer.writerow([key, value])

    if args.json:
        json.dump({"hardware": report["hardware"], "values": flatten(report)}, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    hw = report["hardware"]
    print("hardware: %s, %s" % (hw["cpu"], hw["memsize"]))  # type: ignore[index]
    print("python %s, numpy %s" % (hw["python"], hw["numpy"]))  # type: ignore[index]
    print("%d warmup run, %d timed runs per cell" % (WARMUP_RUNS, args.runs))
    print()
    header = "%-7s %-11s %13s %12s %14s" % (
        "n",
        "impl",
        "peak",
        "peak/cell",
        "wall time",
    )
    print(header)
    print("-" * len(header))
    for row in report["rows"]:  # type: ignore[index]
        if row["skipped"]:
            print(
                "%-7d %-11s %13s %12s %14s"
                % (row["n"], row["implementation"], "skipped", "", row["skipped"])
            )
            continue
        peak = int(row["peak_bytes"])
        print(
            "%-7d %-11s %13s %12.1f %9.3f +/- %.3f s"
            % (
                row["n"],
                row["implementation"],
                human(peak),
                peak / int(row["dp_cells"]),
                row["mean_s"],
                row["stdev_s"],
            )
        )
    print()
    for key, value in sorted(flatten(report).items()):
        if key.startswith("ratio."):
            print("%s: %sx" % (key, value))

    marginal = report["marginal"]  # type: ignore[index]
    print()
    print(
        "marginal cost of one more residue of x, at m = %d (peaks at n = %d "
        "and n = %d):" % (MARGINAL_M, MARGINAL_SMALL_N, MARGINAL_LARGE_N)
    )
    print("  linear     %9.2f bytes/residue" % marginal["linear_bytes_per_residue"])
    print(
        "  quadratic  %9.2f bytes/residue (24*(m+1) predicts %.0f)"
        % (
            marginal["quadratic_bytes_per_residue"],
            marginal["predicted_quadratic_bytes_per_residue"],
        )
    )
    print("  ratio      %9.1fx" % marginal["ratio"])
    return 0


def human(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024 or unit == "GB":
            return "%.1f %s" % (num, unit) if unit != "B" else "%d B" % num
        num /= 1024.0
    return str(num)


if __name__ == "__main__":
    raise SystemExit(main())
