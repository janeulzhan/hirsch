"""Command line interface.

The headline use is two UniProt accessions::

    hirsch P69905 P68871

which aligns human haemoglobin subunit alpha against subunit beta: two
paralogues from the same ancient gene duplication, similar enough to align
end to end and different enough that the alignment has something to say.
"""

from __future__ import annotations

import argparse
import gc
import sys
import tracemalloc
from pathlib import Path
from typing import List, Optional, Tuple

from . import linear, quadratic
from ._ops import Alignment
from .fasta import Record, load_uniprot, read_fasta
from .scoring import Scoring, blosum62

BLOCK = 60

# Mid-lightness ANSI colours, legible on both light and dark terminals.
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
DIM = "\033[2m"
RESET = "\033[0m"

# Residue pairs scoring above this under the active matrix count as
# "conservative" substitutions and get their own annotation symbol.
CONSERVATIVE_THRESHOLD = 0


def build_scoring(args: argparse.Namespace) -> Scoring:
    if args.matrix == "blosum62":
        return blosum62(args.gap_open, args.gap_extend)
    return Scoring(
        gap_open=args.gap_open,
        gap_extend=args.gap_extend,
        match=args.match,
        mismatch=args.mismatch,
    )


def annotation(aligned_x: str, aligned_y: str, scoring: Scoring) -> str:
    """The middle line of a pairwise alignment block.

    ``|`` identity, ``+`` a positively scoring substitution, ``.`` a
    negatively scoring one, and a space for a gap column.
    """
    marks: List[str] = []
    for a, b in zip(aligned_x, aligned_y):
        if a == "-" or b == "-":
            marks.append(" ")
        elif a == b:
            marks.append("|")
        elif scoring.pair(a, b) > CONSERVATIVE_THRESHOLD:
            marks.append("+")
        else:
            marks.append(".")
    return "".join(marks)


def colourise(aligned_x: str, aligned_y: str, marks: str, use_colour: bool) -> Tuple[str, str, str]:
    if not use_colour:
        return aligned_x, aligned_y, marks

    def paint(seq: str) -> str:
        out: List[str] = []
        for ch, mark in zip(seq, marks):
            if ch == "-":
                out.append("%s%s%s" % (DIM, ch, RESET))
            elif mark == "|":
                out.append("%s%s%s" % (GREEN, ch, RESET))
            elif mark == "+":
                out.append("%s%s%s" % (BLUE, ch, RESET))
            else:
                out.append("%s%s%s" % (YELLOW, ch, RESET))
        return "".join(out)

    return paint(aligned_x), paint(aligned_y), marks


def format_alignment(
    result: Alignment,
    label_x: str,
    label_y: str,
    scoring: Scoring,
    use_colour: bool,
    block: int = BLOCK,
) -> str:
    marks = annotation(result.aligned_x, result.aligned_y, scoring)
    width = max(len(label_x), len(label_y), 9)
    total = len(result.aligned_x)
    lines: List[str] = []
    pos_x = pos_y = 0
    for start in range(0, total, block):
        stop = min(start + block, total)
        chunk_x = result.aligned_x[start:stop]
        chunk_y = result.aligned_y[start:stop]
        chunk_m = marks[start:stop]
        start_x = pos_x + 1
        start_y = pos_y + 1
        pos_x += len(chunk_x) - chunk_x.count("-")
        pos_y += len(chunk_y) - chunk_y.count("-")
        painted_x, painted_y, painted_m = colourise(
            chunk_x, chunk_y, chunk_m, use_colour
        )
        lines.append(
            "%-*s %5d %s %d" % (width, label_x, start_x, painted_x, pos_x)
        )
        lines.append("%-*s %5s %s" % (width, "", "", painted_m))
        lines.append(
            "%-*s %5d %s %d" % (width, label_y, start_y, painted_y, pos_y)
        )
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def summary(result: Alignment, scoring: Scoring, marks: str) -> List[str]:
    positives = marks.count("|") + marks.count("+")
    aligned_columns = len(result.aligned_x) - marks.count(" ")
    gap_runs = 0
    previous = None
    for a, b in zip(result.aligned_x, result.aligned_y):
        state = "x" if a == "-" else ("y" if b == "-" else None)
        if state is not None and state != previous:
            gap_runs += 1
        previous = state
    pct = (100.0 * result.identities / len(result.aligned_x)) if len(result) else 0.0
    pos_pct = (100.0 * positives / aligned_columns) if aligned_columns else 0.0
    return [
        "score            %d" % result.score,
        "alignment length %d columns" % len(result.aligned_x),
        "identities       %d (%.1f%%)" % (result.identities, pct),
        "positives        %d of %d aligned columns (%.1f%%)"
        % (positives, aligned_columns, pos_pct),
        "gap columns      %d in %d run(s)" % (result.gap_columns, gap_runs),
        "scoring          %s, gap open %d, gap extend %d"
        % (scoring.name, scoring.gap_open, scoring.gap_extend),
    ]


def resolve_inputs(args: argparse.Namespace) -> Tuple[Record, Record]:
    if args.seq:
        if len(args.seq) != 2:
            raise SystemExit("--seq takes exactly two sequences")
        return (
            Record("query", args.seq[0].upper()),
            Record("target", args.seq[1].upper()),
        )
    if args.file:
        if len(args.file) != 2:
            raise SystemExit("--file takes exactly two paths")
        return read_fasta(Path(args.file[0])), read_fasta(Path(args.file[1]))
    if len(args.accessions) != 2:
        raise SystemExit(
            "give two UniProt accessions, or use --seq / --file; "
            "try: hirsch P69905 P68871"
        )
    return (
        load_uniprot(args.accessions[0], offline=args.offline),
        load_uniprot(args.accessions[1], offline=args.offline),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hirsch",
        description="Optimal global alignment with affine gaps, in linear space.",
        epilog="example: hirsch P69905 P68871",
    )
    parser.add_argument(
        "accessions",
        nargs="*",
        default=[],
        help="two UniProtKB accessions, e.g. P69905 P68871",
    )
    parser.add_argument("--seq", nargs=2, metavar=("X", "Y"), help="two literal sequences")
    parser.add_argument("--file", nargs=2, metavar=("X", "Y"), help="two FASTA files")
    parser.add_argument(
        "--mode",
        choices=("linear", "quadratic"),
        default="linear",
        help="which aligner to use (default: linear)",
    )
    parser.add_argument(
        "--matrix",
        choices=("blosum62", "simple"),
        default="blosum62",
        help="substitution scores (default: blosum62)",
    )
    parser.add_argument("--gap-open", type=int, default=11, dest="gap_open")
    parser.add_argument("--gap-extend", type=int, default=1, dest="gap_extend")
    parser.add_argument("--match", type=int, default=2, help="only with --matrix simple")
    parser.add_argument(
        "--mismatch", type=int, default=-1, help="only with --matrix simple"
    )
    parser.add_argument("--no-color", action="store_true", help="plain text output")
    parser.add_argument(
        "--offline", action="store_true", help="use only cached FASTA records"
    )
    parser.add_argument(
        "--width", type=int, default=BLOCK, help="alignment block width (default: 60)"
    )
    args = parser.parse_args(argv)

    record_x, record_y = resolve_inputs(args)
    scoring = build_scoring(args)
    aligner = linear.align if args.mode == "linear" else quadratic.align

    use_colour = sys.stdout.isatty() and not args.no_color

    gc.collect()
    tracemalloc.start()
    try:
        result = aligner(record_x.sequence, record_y.sequence, scoring)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print("hirsch: optimal affine-gap global alignment in linear space")
    print()
    print("%-9s %s (%d residues)" % (record_x.name, record_x.description, len(record_x.sequence)))
    print("%-9s %s (%d residues)" % (record_y.name, record_y.description, len(record_y.sequence)))
    print()
    print(
        format_alignment(
            result, record_x.name, record_y.name, scoring, use_colour, args.width
        )
    )
    print()
    marks = annotation(result.aligned_x, result.aligned_y, scoring)
    for line in summary(result, scoring, marks):
        print(line)
    print("aligner          %s" % args.mode)
    print("peak memory      %d bytes (%.1f KB, tracemalloc)" % (peak, peak / 1024))
    print(
        "annotation       | identity, + positive substitution, . negative substitution"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
