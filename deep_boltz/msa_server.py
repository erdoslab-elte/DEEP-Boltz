"""Fetching an alignment from the MMseqs2 server that Boltz's own CLI uses.

The query sequence is uploaded to a public service (api.colabfold.com unless
told otherwise), so this is opt-in rather than a silent fallback: nothing here
runs unless the caller asks for it.  Alignments are cached by sequence, so a
rescan or a ddG sweep costs one round trip rather than one per run.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

DEFAULT_MSA_SERVER = "https://api.colabfold.com"
DEFAULT_CACHE_DIR = Path("~/.deep_boltz/msa_cache")

_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def _normalise(sequence: str) -> str:
    sequence = "".join(sequence.split()).upper()
    if not sequence:
        raise ValueError("cannot search for an empty sequence")
    unsupported = sorted(set(sequence) - _AMINO_ACIDS)
    if unsupported:
        raise ValueError(
            f"sequence contains unsupported amino acids: {unsupported}"
        )
    return sequence


def cached_msa_path(
    sequence: str,
    *,
    cache_dir: str | Path | None = None,
    server_url: str = DEFAULT_MSA_SERVER,
) -> Path:
    """Where `fetch_msa` keeps the alignment for this sequence.

    Keyed by server as well as sequence: two servers hold different databases,
    and a cache hit from the wrong one would be silently misleading.
    """
    directory = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    digest = hashlib.sha256(f"{server_url}\n{_normalise(sequence)}".encode()).hexdigest()
    return directory.expanduser() / f"{digest[:16]}.a3m"


def _query_sequence(a3m: str) -> str | None:
    """The first aligned row, which the server echoes back as the query."""

    for line in a3m.splitlines():
        line = line.replace("\x00", "").strip()  # noqa: PLW2901
        if not line or line.startswith(">") or line.startswith("#"):
            continue
        return line.replace("-", "").upper()
    return None


def fetch_msa(
    sequence: str,
    *,
    cache_dir: str | Path | None = None,
    server_url: str = DEFAULT_MSA_SERVER,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Search `server_url` for an alignment and return the cached a3m.

    Only the unpaired search is run: this potential scores one chain, and the
    featurizer keys the alignment to that chain alone, so the paired search
    Boltz runs for complexes would have nothing to pair against.
    """
    sequence = _normalise(sequence)
    path = cached_msa_path(sequence, cache_dir=cache_dir, server_url=server_url)
    if path.is_file() and not force:
        if not quiet:
            print(f"MSA                reusing cached search: {path}")
        return path

    from boltz.data.msa.mmseqs2 import run_mmseqs2

    if not quiet:
        print(
            f"MSA                searching {server_url} "
            f"({len(sequence)} residues uploaded); this usually takes a minute"
        )
    with tempfile.TemporaryDirectory() as tmp:
        a3m = run_mmseqs2(
            [sequence],
            Path(tmp) / "search",
            use_env=True,
            use_pairing=False,
            host_url=server_url,
        )[0]

    echoed = _query_sequence(a3m)
    if echoed is None:
        raise RuntimeError(f"{server_url} returned an empty alignment")
    if echoed != sequence:
        # Boltz would quietly swap in a dummy MSA instead of failing here.
        raise RuntimeError(
            "the server's alignment is for a different sequence than the one "
            f"queried ({echoed[:40]}... vs {sequence[:40]}...)"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    # Written via a temporary name so an interrupted run cannot leave a
    # truncated alignment behind that later runs would treat as a cache hit.
    partial = path.with_name(path.name + ".partial")
    partial.write_text(a3m)
    partial.replace(path)
    if not quiet:
        print(f"MSA                cached at {path}")
    return path
