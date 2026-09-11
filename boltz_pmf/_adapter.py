"""Adapters for the installed Boltz 2.2.1 package and structure parser."""

from __future__ import annotations

import contextlib
import gzip
import io
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class StructureBatch:
    """Boltz features plus the coordinates and scoring masks for one structure."""

    features: dict[str, Any]
    coords: Tensor
    resolved_mask: Tensor
    backbone_mask: Tensor
    padded_mask: Tensor
    chain_names: tuple[str, ...]
    num_unresolved_atoms: int


def load_boltz2_model(
    checkpoint: str | Path | None = None,
    *,
    device: str | torch.device = "cuda:0",
) -> torch.nn.Module:

    from boltz.main import (  # imported lazily so unit tests need only torch
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
    )
    from boltz.model.models.boltz2 import Boltz2

    checkpoint_path = (
        Path(checkpoint).expanduser()
        if checkpoint is not None
        else Path("~/.boltz/boltz2_conf.ckpt").expanduser()
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Boltz-2 checkpoint not found: {checkpoint_path}")

    steering = BoltzSteeringParams()
    steering.fk_steering = False
    steering.physical_guidance_update = False
    steering.contact_guidance_update = False
    predict_args = {
        "recycling_steps": 3,
        "sampling_steps": 5,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    model = Boltz2.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs(use_paired_feature=True)),
        steering_args=asdict(steering),
    )
    model.eval().to(torch.device(device))
    return model


def _protein_chains_from_structure(
    path: Path, selected_chain: str | None
) -> list[tuple[str, list[tuple[str, dict[str, tuple[np.ndarray, float]]]]]]:
    """Read observed protein residues without Boltz's PDB entity wrapper.

    The PDB wrapper in some Boltz 2.2.1 checkouts creates incomplete entity
    sequences for headerless AlphaFold PDBs. Reading coordinate records with
    Gemmi and building canonical Boltz atoms from sequence avoids that failure.
    """

    structure = gemmi.read_structure(str(path))
    structure.remove_waters()
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    if len(structure) == 0:
        raise ValueError(f"Structure has no models: {path}")

    parsed_chains = []
    for chain_index, source_chain in enumerate(structure[0]):
        chain_name = source_chain.name or chr(ord("A") + chain_index)
        if selected_chain is not None and chain_name != selected_chain:
            continue
        residues = []
        for residue in source_chain:
            residue_info = gemmi.find_tabulated_residue(residue.name)
            one_letter = residue_info.one_letter_code
            if not one_letter or len(one_letter) != 1 or not one_letter.isalpha():
                continue
            observed_atoms: dict[str, tuple[np.ndarray, float]] = {}
            for atom in residue:
                if atom.element.is_hydrogen:
                    continue
                observed_atoms[atom.name.strip()] = (
                    np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32),
                    float(atom.b_iso),
                )
            if observed_atoms:
                residues.append((one_letter.upper(), observed_atoms))
        if residues:
            parsed_chains.append((chain_name, residues))

    if selected_chain is not None and not parsed_chains:
        available = tuple((item.name or "<blank>") for item in structure[0])
        raise ValueError(
            f"Chain {selected_chain!r} not found or has no protein residues; "
            f"available chains: {available}"
        )
    if not parsed_chains:
        raise ValueError(f"No protein chains found in {path}")
    return parsed_chains


def chain_sequence(path: str | Path, chain: str | None = None) -> str:
    """The one-letter sequence of the chain an alignment would be keyed to.

    The featurizer keys an MSA to the first protein chain, so that is the chain
    a server search has to be run for.
    """
    structure_path = Path(path).expanduser().resolve()
    if not structure_path.is_file():
        raise FileNotFoundError(f"Structure not found: {structure_path}")
    _, residues = _protein_chains_from_structure(structure_path, chain)[0]
    return "".join(residue[0] for residue in residues)


def _target_from_observed_structure(
    path: Path,
    molecules: dict[str, Any],
    mol_dir: Path,
    chain: str | None,
    sequence_override: str | None = None,
) -> tuple[Any, tuple[str, ...]]:
    """Build canonical Boltz atoms from sequence, then insert observed coords."""

    from boltz.data.parse.schema import parse_boltz_schema

    source_chains = _protein_chains_from_structure(path, chain)
    source_sequences = [
        "".join(residue[0] for residue in residues)
        for _, residues in source_chains
    ]
    if sequence_override is not None:
        sequence_override = sequence_override.strip().upper()
        if len(source_chains) != 1:
            raise ValueError("sequence_override currently requires one protein chain")
        if len(sequence_override) != len(source_sequences[0]):
            raise ValueError(
                "sequence_override length does not match observed structure: "
                f"{len(sequence_override)} != {len(source_sequences[0])}"
            )
        if set(sequence_override) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError("sequence_override contains unsupported amino acids")
        source_sequences = [sequence_override]
    schema = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": chain_name,
                    "sequence": sequence,
                    "msa": "empty",
                }
            }
            for (chain_name, _), sequence in zip(source_chains, source_sequences)
        ],
    }
    with _quiet_empty_msa_notice():
        target = parse_boltz_schema(path.stem, schema, molecules, mol_dir, True)

    atoms = target.structure.atoms.copy()
    coords = target.structure.coords.copy()
    atoms["coords"] = 0.0
    atoms["is_present"] = False
    atoms["bfactor"] = 0.0
    atoms["plddt"] = 0.0
    coords["coords"] = 0.0

    if len(target.structure.chains) != len(source_chains):
        raise RuntimeError("Canonical chain count does not match observed chain count")
    for canonical_chain, (chain_name, observed_residues) in zip(
        target.structure.chains, source_chains
    ):
        res_start = int(canonical_chain["res_idx"])
        res_end = res_start + int(canonical_chain["res_num"])
        canonical_residues = target.structure.residues[res_start:res_end]
        if len(canonical_residues) != len(observed_residues):
            raise RuntimeError(
                f"Canonical/observed residue mismatch for chain {chain_name}: "
                f"{len(canonical_residues)} != {len(observed_residues)}"
            )
        for canonical_residue, (_, observed_atoms) in zip(
            canonical_residues, observed_residues
        ):
            atom_start = int(canonical_residue["atom_idx"])
            atom_end = atom_start + int(canonical_residue["atom_num"])
            for atom_index in range(atom_start, atom_end):
                atom_name = str(atoms[atom_index]["name"])
                observed = observed_atoms.get(atom_name)
                if observed is None:
                    continue
                position, bfactor = observed
                atoms[atom_index]["coords"] = position
                atoms[atom_index]["is_present"] = True
                atoms[atom_index]["bfactor"] = bfactor
                atoms[atom_index]["plddt"] = (
                    bfactor / 100.0 if bfactor > 1.0 else bfactor
                )
                coords[atom_index]["coords"] = position

    structure = replace(target.structure, atoms=atoms, coords=coords)
    record = replace(
        target.record,
        id=path.stem,
        structure=replace(target.record.structure, num_chains=len(source_chains)),
    )
    return replace(target, structure=structure, record=record), tuple(
        name for name, _ in source_chains
    )


def _target_from_sequence(
    sequence: str,
    molecules: dict[str, Any],
    mol_dir: Path,
    chain_name: str = "A",
) -> tuple[Any, tuple[str, ...]]:
    """Build a Boltz target from sequence alone, with no coordinates."""

    from boltz.data.parse.schema import parse_boltz_schema

    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("sequence is empty")
    unsupported = set(sequence) - set("ACDEFGHIKLMNPQRSTVWY")
    if unsupported:
        raise ValueError(
            f"sequence contains unsupported amino acids: {sorted(unsupported)}"
        )

    schema = {
        "version": 1,
        "sequences": [
            {"protein": {"id": chain_name, "sequence": sequence, "msa": "empty"}}
        ],
    }
    with _quiet_empty_msa_notice():
        target = parse_boltz_schema("sequence", schema, molecules, mol_dir, True)

    atoms = target.structure.atoms.copy()
    coords = target.structure.coords.copy()
    atoms["coords"] = 0.0
    # Marked present so the backbone mask covers the whole chain: the sampler
    # fills these positions, and a mask built from an unfolded placeholder would
    # otherwise exclude every atom it is about to write.
    atoms["is_present"] = True
    atoms["bfactor"] = 0.0
    atoms["plddt"] = 0.0
    coords["coords"] = 0.0

    structure = replace(target.structure, atoms=atoms, coords=coords)
    return replace(target, structure=structure), (chain_name,)


_EMPTY_MSA_NOTICE = "Found explicit empty MSA for some proteins"


@contextmanager
def _quiet_empty_msa_notice() -> Iterator[None]:
    """Swallow Boltz's single-sequence-mode notice while parsing a schema.

    Every schema built here declares ``"msa": "empty"``, because an alignment is
    keyed into the featurizer input afterwards rather than through the schema's
    MSA ids -- so the notice says nothing about whether this run actually has an
    MSA.  Anything else Boltz prints is passed through.
    """
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            yield
    finally:
        kept = [
            line
            for line in captured.getvalue().splitlines()
            if not line.startswith(_EMPTY_MSA_NOTICE)
        ]
        if kept:
            print("\n".join(kept))


def _parse_alignment(path: Path, max_seqs: int) -> Any:
    """Parse an a3m/a2m alignment into Boltz's MSA object.

    Boltz feeds every character of an alignment row straight into
    ``prot_letter_to_token``, so anything missing from that table is a KeyError
    rather than a skipped character: a trailing NUL from a truncated writer, an
    a2m ``.`` insertion gap, a ``*`` stop codon.  Clean the text up front and
    hand the parser a temporary copy.
    """
    from boltz.data import const
    from boltz.data.parse.a3m import parse_a3m

    known = set(const.prot_letter_to_token)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(str(path), "rt", errors="replace") as f:
        raw = f.read()

    lines: list[str] = []
    dropped: set[str] = set()
    replaced: set[str] = set()
    for line in raw.splitlines():
        # Tabs separate the annotation fields Boltz reads out of headers; every
        # other control or non-ASCII byte is noise.
        line = "".join(c for c in line if c == "\t" or (c.isascii() and c.isprintable()))
        if not line.strip():
            continue
        if line.startswith(">"):
            lines.append(line)
            continue
        row = []
        for c in line.strip():
            if c in known or c.islower():
                # Lowercase columns are insertions; the parser counts and skips them.
                row.append(c)
            elif c == ".":
                # a2m gap aligned to an insertion column: stands for no residue.
                dropped.add(c)
            else:
                replaced.add(c)
                row.append("X")
        if row:
            lines.append("".join(row))

    if dropped or replaced:
        detail = []
        if replaced:
            detail.append("read as X: " + " ".join(sorted(replaced)))
        if dropped:
            detail.append("ignored: " + " ".join(sorted(dropped)))
        warnings.warn(
            f"Unsupported characters in {path.name} ({'; '.join(detail)})",
            stacklevel=2,
        )

    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / "alignment.a3m"
        clean.write_text("\n".join(lines) + "\n")
        return parse_a3m(clean, taxonomy=None, max_seqs=max_seqs)


def _featurize_target(
    target: Any,
    chain_names: tuple[str, ...],
    molecules: dict[str, Any],
    molecule_dir: Path,
    *,
    seed: int,
    msa_path: str | Path | None,
    max_msa_seqs: int | None,
) -> StructureBatch:
    """Tokenize, featurize and collate a prepared target."""

    from boltz.data import const
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.module.inferencev2 import collate
    from boltz.data.mol import load_molecules
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Input

    parsed_msa: dict[int, Any] = {}
    if msa_path is not None:
        alignment = Path(msa_path).expanduser().resolve()
        if not alignment.is_file():
            raise FileNotFoundError(f"MSA not found: {alignment}")
        # Keyed by asym_id, which is how the featurizer looks alignments up.
        parsed_msa[0] = _parse_alignment(
            alignment,
            const.max_msa_seqs if max_msa_seqs is None else max_msa_seqs,
        )

    data = Input(
        structure=target.structure,
        msa=parsed_msa,
        record=target.record,
        residue_constraints=target.residue_constraints,
        templates=target.templates,
        extra_mols=target.extra_mols,
    )
    tokenized = Boltz2Tokenizer().tokenize(data)

    molecule_names = set(str(name) for name in tokenized.tokens["res_name"])
    molecules.update(target.extra_mols)
    missing = molecule_names - set(molecules)
    if missing:
        molecules.update(load_molecules(molecule_dir, missing))

    features = Boltz2Featurizer().process(
        tokenized,
        molecules=molecules,
        random=np.random.default_rng(seed),
        training=False,
        max_atoms=None,
        max_tokens=None,
        max_seqs=const.max_msa_seqs if max_msa_seqs is None else max_msa_seqs,
        pad_to_max_seqs=False,
        single_sequence_prop=0.0,
        compute_frames=True,
        compute_constraint_features=True,
    )
    features["record"] = target.record
    batch = collate([features])
    coords = batch["coords"][:, 0].float()
    padded_mask = batch["atom_pad_mask"].bool()
    resolved_mask = padded_mask & batch["atom_resolved_mask"].bool()
    backbone_mask = (
        batch["atom_backbone_feat"][..., 1:5].sum(dim=-1).bool()
        & resolved_mask
    )

    return StructureBatch(
        features=batch,
        coords=coords,
        resolved_mask=resolved_mask,
        backbone_mask=backbone_mask,
        padded_mask=padded_mask,
        chain_names=chain_names,
        num_unresolved_atoms=int((padded_mask & ~resolved_mask).sum().item()),
    )


def featurize_structure(
    path: str | Path,
    *,
    mol_dir: str | Path = "~/.boltz/mols",
    chain: str | None = None,
    use_assembly: bool = False,
    seed: int = 42,
    sequence_override: str | None = None,
    msa_path: str | Path | None = None,
    max_msa_seqs: int | None = None,
) -> StructureBatch:
    """Create Boltz-2 features while retaining supplied PDB/mmCIF coordinates.

    No MSA server is contacted; Boltz's dummy single-sequence MSA is used.

    ``msa_path`` supplies a real multiple sequence alignment (a3m or a2m; the
    parser only special-cases a ``.gz`` suffix).  Without it Boltz runs in
    single-sequence mode against a dummy MSA, which the model itself warns makes
    predictions suboptimal.  The alignment is keyed to the first protein chain,
    and its first sequence must match that chain's sequence.  ``max_msa_seqs``
    caps the depth; Boltz's own default is ``const.max_msa_seqs``.

    ``seed`` controls two independent sources of randomness
    """

    from boltz.data.mol import load_canonicals

    structure_path = Path(path).expanduser().resolve()
    if not structure_path.is_file():
        raise FileNotFoundError(f"Structure not found: {structure_path}")
    molecule_dir = Path(mol_dir).expanduser().resolve()
    if not molecule_dir.is_dir():
        raise FileNotFoundError(f"Boltz molecule directory not found: {molecule_dir}")
    if use_assembly:
        raise NotImplementedError(
            "Assembly expansion is not supported by the coordinate mapper; "
            "provide the desired biological assembly as the input structure."
        )
    if structure_path.suffix.lower() not in (".pdb", ".ent", ".cif", ".mmcif"):
        raise ValueError("Structure must be a .pdb, .ent, .cif, or .mmcif file")

    # Must precede featurization: the reference-conformer augmentation reads
    # torch's global RNG, not the generator handed to the featurizer below.
    torch.manual_seed(seed)

    molecules = load_canonicals(molecule_dir)
    target, chain_names = _target_from_observed_structure(
        structure_path,
        molecules,
        molecule_dir,
        chain,
        sequence_override=sequence_override,
    )
    return _featurize_target(
        target,
        chain_names,
        molecules,
        molecule_dir,
        seed=seed,
        msa_path=msa_path,
        max_msa_seqs=max_msa_seqs,
    )


def featurize_sequence(
    sequence: str,
    *,
    mol_dir: str | Path = "~/.boltz/mols",
    seed: int = 42,
    msa_path: str | Path | None = None,
    max_msa_seqs: int | None = None,
    chain_name: str = "A",
) -> StructureBatch:
    """Create Boltz-2 features for a sequence with no structure supplied.

    The returned batch carries the correct packed atom layout and masks but
    placeholder coordinates at the origin.  It is the input to the folding pass
    in :mod:`boltz_pmf.folding`.
    """

    from boltz.data.mol import load_canonicals

    molecule_dir = Path(mol_dir).expanduser().resolve()
    if not molecule_dir.is_dir():
        raise FileNotFoundError(f"Boltz molecule directory not found: {molecule_dir}")

    # As in `featurize_structure`: the conformer augmentation reads torch's
    # global RNG and has to be pinned before the featurizer runs.
    torch.manual_seed(seed)

    molecules = load_canonicals(molecule_dir)
    target, chain_names = _target_from_sequence(
        sequence, molecules, molecule_dir, chain_name
    )
    return _featurize_target(
        target,
        chain_names,
        molecules,
        molecule_dir,
        seed=seed,
        msa_path=msa_path,
        max_msa_seqs=max_msa_seqs,
    )
