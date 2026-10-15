"""Fold a sequence with Boltz-2 so the potential needs no input structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ._adapter import StructureBatch, featurize_sequence
from ._features import features_with_sequence

# Boltz's own prediction CLI default. Fewer steps give a rougher fold; the
# benchmark above was run at this value.
DEFAULT_SAMPLING_STEPS = 200


@dataclass
class FoldedBackbone:
    """A Boltz-2 prediction, in the packed atom layout that produced it."""

    batch: StructureBatch
    coords: Tensor
    """``[1, A, 3]``, aligned to ``batch.features`` -- not a re-read PDB."""
    sequence: str
    sampling_steps: int
    seed: int

    @property
    def backbone_coords(self) -> np.ndarray:
        """Scored N/CA/C/O positions, four per residue, in chain order."""

        selected = self.batch.backbone_mask[0].bool().cpu().numpy()
        return self.coords[0].cpu().numpy()[selected]


def fold_sequence(
    model: Any,
    scorer: Any,
    sequence: str,
    *,
    msa: str | Path | None = None,
    msa_depth: int | None = None,
    recycling_steps: int = 3,
    sampling_steps: int = DEFAULT_SAMPLING_STEPS,
    seed: int = 0,
    mol_dir: str | Path = "~/.boltz/mols",
) -> FoldedBackbone:
    """Predict a structure for ``sequence`` and return it with its features."""

    from boltz.main import BoltzSteeringParams
    from dataclasses import asdict

    sequence = sequence.strip().upper()
    batch = featurize_sequence(
        sequence, mol_dir=mol_dir, msa_path=msa, max_msa_seqs=msa_depth, seed=seed
    )

    steering = asdict(BoltzSteeringParams())
    steering["fk_steering"] = False
    steering["physical_guidance_update"] = False
    steering["contact_guidance_update"] = False

    # The full-atom layout, deliberately: see the module docstring.
    features = features_with_sequence(batch.features, sequence)
    context = scorer.prepare_context(features, recycling_steps=recycling_steps)
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.structure_module.sample(
                atom_mask=context["feats"]["atom_pad_mask"].float(),
                num_sampling_steps=sampling_steps,
                steering_args=steering,
                # `z_trunk` feeds the distogram head; the sampler passes its
                # context straight to the denoiser, which does not accept it.
                **{k: v for k, v in context.items() if k != "z_trunk"},
            )
        coords = out["sample_atom_coords"].detach()[:1].float().clone()
        del out
    finally:
        del context
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    return FoldedBackbone(
        batch=batch,
        coords=coords,
        sequence=sequence,
        sampling_steps=sampling_steps,
        seed=seed,
    )


def write_backbone_pdb(path: str | Path, folded: FoldedBackbone) -> Path:
    """Write the folded N/CA/C/O trace, so the fold can be inspected or reused."""

    from boltz.data import const

    coords = folded.backbone_coords
    sequence = folded.sequence
    if len(coords) != 4 * len(sequence):
        raise RuntimeError(
            f"expected four backbone atoms per residue, got {len(coords)} "
            f"for {len(sequence)} residues"
        )

    lines = [
        f"REMARK   1 Boltz-2 prediction, {folded.sampling_steps} sampling steps, "
        f"seed {folded.seed}",
        "REMARK   1 Backbone only; sidechains omitted (they do not enter the score).",
    ]
    serial = 1
    for residue, index in enumerate(range(0, len(coords), 4), start=1):
        res_name = const.prot_letter_to_token.get(sequence[residue - 1], "UNK")
        for offset, name in enumerate(("N", "CA", "C", "O")):
            x, y, z = coords[index + offset]
            # Column-exact PDB. The fields are fixed-width and every reader
            # slices rather than splits, so a single missing space silently
            # shifts the coordinates into the wrong columns instead of failing:
            #   7-11 serial | 13-16 name | 17 altLoc | 18-20 resName
            #   22 chainID  | 23-26 resSeq | 27 iCode | 31-54 xyz
            #   55-60 occupancy | 61-66 tempFactor | 77-78 element
            # The atom name is right-shifted by one because these are all
            # one-letter elements, which is where it belongs in a 4-column field.
            lines.append(
                f"ATOM  {serial:>5}  {name:<3} {res_name:>3} A{residue:>4}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {name[0]:>2}"
            )
            serial += 1
    lines.append("END")

    destination = Path(path).expanduser()
    if destination.parent != Path(""):
        destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    return destination
