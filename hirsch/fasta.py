"""Minimal FASTA reading and a keyless UniProt fetch.

The fetch shells out to ``curl`` rather than using ``urllib``.  That is not an
aesthetic choice: on a stock macOS Python without ``certifi`` installed,
``urllib`` raises ``CERTIFICATE_VERIFY_FAILED`` against the UniProt endpoint,
while ``curl`` uses the system trust store and just works.  Keeping the
dependency list at "numpy" is worth one subprocess call.

Fetched records are cached under ``tests/fixtures`` so the test suite and the
committed example output do not depend on the network.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

__all__ = ["Record", "parse_fasta", "read_fasta", "fetch_uniprot", "load_uniprot"]

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
ACCESSION_RE = re.compile(r"^[A-Za-z0-9]{3,12}$")


@dataclass(frozen=True)
class Record:
    """One FASTA record: its header line and its ungapped sequence."""

    header: str
    sequence: str

    @property
    def name(self) -> str:
        """A short label, e.g. ``HBA_HUMAN`` from a UniProt header."""
        parts = self.header.split("|")
        if len(parts) >= 3:
            return parts[2].split()[0]
        return self.header.split()[0] if self.header else "unnamed"

    @property
    def description(self) -> str:
        tail = self.header.split(None, 1)
        return tail[1] if len(tail) > 1 else ""


def parse_fasta(text: str) -> List[Record]:
    """Parse FASTA text.  Blank lines are ignored; whitespace is stripped."""
    records: List[Record] = []
    header: Optional[str] = None
    chunks: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append(Record(header, "".join(chunks)))
            header = line[1:].strip()
            chunks = []
        else:
            if header is None:
                raise ValueError("FASTA data before the first header line")
            chunks.append(line.replace(" ", ""))
    if header is not None:
        records.append(Record(header, "".join(chunks)))
    if not records:
        raise ValueError("no FASTA records found")
    return records


def read_fasta(path: Path) -> Record:
    """Read the first record from a FASTA file."""
    return parse_fasta(Path(path).read_text())[0]


def fetch_uniprot(accession: str, timeout: int = 30) -> Record:
    """Download one UniProtKB record as FASTA.  Needs the network."""
    if not ACCESSION_RE.match(accession):
        raise ValueError("%r does not look like a UniProt accession" % accession)
    url = UNIPROT_URL.format(accession=accession)
    try:
        completed = subprocess.run(
            ["curl", "--silent", "--show-error", "--fail", "--location", url],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on the host
        raise RuntimeError("curl is not available to fetch %s" % url) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "fetching %s failed: %s" % (url, (exc.stderr or "").strip())
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("fetching %s timed out" % url) from exc
    return parse_fasta(completed.stdout)[0]


def cache_path(accession: str) -> Path:
    return FIXTURE_DIR / ("%s.fasta" % accession)


def load_uniprot(accession: str, offline: bool = False) -> Record:
    """Return a UniProt record, preferring the committed cache.

    This is what the CLI and the tests use.  The cache is checked first so
    that a demo run is reproducible and does not hammer UniProt; set
    ``offline`` to refuse the network entirely.
    """
    path = cache_path(accession)
    if path.exists():
        return read_fasta(path)
    if offline:
        raise RuntimeError(
            "%s is not cached under %s and offline mode was requested"
            % (accession, FIXTURE_DIR)
        )
    record = fetch_uniprot(accession)
    try:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(">%s\n%s\n" % (record.header, record.sequence))
    except OSError:  # pragma: no cover - read-only checkouts
        pass
    return record
