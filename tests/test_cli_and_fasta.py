"""CLI and FASTA tests.

The two haemoglobin records are committed under ``tests/fixtures`` so that
everything here runs offline.  The one test that actually talks to UniProt is
marked ``network`` and is deselected by default.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hirsch import cli, linear, quadratic
from hirsch.fasta import Record, cache_path, load_uniprot, parse_fasta, read_fasta
from hirsch.scoring import blosum62

FIXTURES = Path(__file__).parent / "fixtures"
HBA = "P69905"
HBB = "P68871"


def test_fixtures_are_committed() -> None:
    for accession in (HBA, HBB):
        assert cache_path(accession).exists(), (
            "%s is not cached; the offline tests depend on it" % accession
        )


def test_parse_fasta_basics() -> None:
    records = parse_fasta(">one desc\nACGT\nACGT\n\n>two\nMMM\n")
    assert len(records) == 2
    assert records[0].sequence == "ACGTACGT"
    assert records[0].name == "one"
    assert records[0].description == "desc"
    assert records[1].sequence == "MMM"


def test_parse_fasta_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        parse_fasta("ACGT\n")
    with pytest.raises(ValueError):
        parse_fasta("\n\n")


def test_uniprot_header_naming() -> None:
    record = read_fasta(cache_path(HBA))
    assert record.name == "HBA_HUMAN"
    assert "Hemoglobin subunit alpha" in record.description
    assert len(record.sequence) == 142
    assert record.sequence.startswith("MVLSPADKTNVKAAWGKV")


def test_haemoglobin_sequences_are_the_expected_length() -> None:
    assert len(load_uniprot(HBA, offline=True).sequence) == 142
    assert len(load_uniprot(HBB, offline=True).sequence) == 147


def test_offline_mode_refuses_an_uncached_accession() -> None:
    with pytest.raises(RuntimeError):
        load_uniprot("Q00000ZZ", offline=True)


def test_haemoglobin_alignment_score_is_stable() -> None:
    """The headline demo, pinned.

    Both aligners must agree, and the alignment must re-score to the same
    value under an independent scorer.
    """
    scoring = blosum62()
    x = load_uniprot(HBA, offline=True).sequence
    y = load_uniprot(HBB, offline=True).sequence
    result = linear.align(x, y, scoring)
    assert result.score == 282
    assert quadratic.align_score(x, y, scoring) == 282
    assert scoring.score_alignment(result.aligned_x, result.aligned_y) == 282
    assert result.identities == 63
    assert len(result.aligned_x) == 149


def test_both_aligners_give_the_identical_haemoglobin_alignment() -> None:
    scoring = blosum62()
    x = load_uniprot(HBA, offline=True).sequence
    y = load_uniprot(HBB, offline=True).sequence
    assert linear.align(x, y, scoring) == quadratic.align(x, y, scoring)


def test_annotation_symbols() -> None:
    scoring = blosum62()
    marks = cli.annotation("MV-LS", "MVHLT", scoring)
    assert marks[0] == "|"
    assert marks[1] == "|"
    assert marks[2] == " "
    assert marks[3] == "|"
    assert marks[4] in "+."


def test_cli_runs_offline_and_reports_the_score(capsys) -> None:
    assert cli.main([HBA, HBB, "--offline", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "score            282" in out
    assert "HBA_HUMAN" in out and "HBB_HUMAN" in out
    assert "peak memory" in out
    assert "identities       63 (42.3%)" in out


def test_cli_literal_sequences(capsys) -> None:
    assert cli.main(["--seq", "ACGTACGT", "ACGTTACGT", "--matrix", "simple", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "score" in out


def test_cli_file_input(capsys) -> None:
    assert (
        cli.main(
            ["--file", str(cache_path(HBA)), str(cache_path(HBB)), "--no-color"]
        )
        == 0
    )
    assert "score            282" in capsys.readouterr().out


def test_cli_quadratic_mode_matches_linear_mode(capsys) -> None:
    cli.main([HBA, HBB, "--offline", "--no-color", "--mode", "linear"])
    lin = capsys.readouterr().out
    cli.main([HBA, HBB, "--offline", "--no-color", "--mode", "quadratic"])
    quad = capsys.readouterr().out

    def strip_variable(text: str) -> str:
        return "\n".join(
            line
            for line in text.splitlines()
            if not line.startswith(("aligner", "peak memory"))
        )

    assert strip_variable(lin) == strip_variable(quad)


def test_cli_reports_lower_peak_memory_for_the_linear_mode(capsys) -> None:
    def peak_of(mode: str) -> int:
        cli.main([HBA, HBB, "--offline", "--no-color", "--mode", mode])
        for line in capsys.readouterr().out.splitlines():
            if line.startswith("peak memory"):
                return int(line.split()[2])
        raise AssertionError("no peak memory line")

    assert peak_of("linear") < peak_of("quadratic")


def test_cli_rejects_wrong_argument_counts() -> None:
    with pytest.raises(SystemExit):
        cli.main([HBA, "--offline"])


def test_committed_example_output_is_current() -> None:
    """``docs/example.txt`` is shown in the README, so it must not go stale.

    Every line must match a fresh run byte for byte except the peak-memory
    line, which is the one genuinely machine-dependent figure in the output:
    the committed copy was measured on the hardware named in the README, and
    a different CPU or Python build will land a few bytes away.  The line is
    still checked for shape, just not for its exact value.
    """
    example = Path(__file__).parent.parent / "docs" / "example.txt"
    if not example.exists():
        pytest.skip("docs/example.txt has not been generated yet")
    fresh = subprocess.run(
        [sys.executable, "-m", "hirsch.cli", HBA, HBB, "--offline", "--no-color"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent.parent,
    ).stdout

    def split_off_memory(text: str):
        kept, memory = [], None
        for line in text.splitlines():
            if line.startswith("peak memory"):
                memory = line
            else:
                kept.append(line)
        return kept, memory

    fresh_lines, fresh_memory = split_off_memory(fresh)
    committed_lines, committed_memory = split_off_memory(example.read_text())

    assert fresh_lines == committed_lines
    assert fresh_memory is not None and committed_memory is not None
    assert fresh_memory.endswith("tracemalloc)")
    # It should still be in the same ballpark, which rules out an accidental
    # quadratic allocation sneaking into the linear path.
    assert int(fresh_memory.split()[2]) < 4 * int(committed_memory.split()[2])


@pytest.mark.network
def test_live_uniprot_fetch_matches_the_cached_copy() -> None:
    """Confirms the cached fixtures are faithful.  Needs the network."""
    from hirsch.fasta import fetch_uniprot

    for accession in (HBA, HBB):
        live = fetch_uniprot(accession)
        cached = read_fasta(cache_path(accession))
        assert live.sequence == cached.sequence
