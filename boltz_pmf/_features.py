"""Fixed-backbone sequence conditioning for mutation-effect evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def sequence_token_ids(sequence: str, *, device: torch.device) -> Tensor:
    """Return Boltz token IDs for a canonical one-letter protein sequence."""

    from boltz.data import const

    sequence = sequence.strip().upper()
    unsupported = set(sequence) - AMINO_ACIDS
    if unsupported:
        raise ValueError(f"Unsupported amino acids in sequence: {sorted(unsupported)}")
    return torch.tensor(
        [const.token_ids[const.prot_letter_to_token[aa]] for aa in sequence],
        device=device,
        dtype=torch.long,
    )


def fixed_backbone_features(
    features: Mapping[str, Any],
    backbone_mask: Tensor,
) -> dict[str, Any]:
    """Return a shallow feature copy whose active atom set is fixed backbone."""

    if backbone_mask.ndim == 1:
        backbone_mask = backbone_mask.unsqueeze(0)
    atom_pad_mask = features.get("atom_pad_mask")
    if not isinstance(atom_pad_mask, Tensor):
        raise KeyError("features must contain tensor atom_pad_mask")
    if backbone_mask.shape != atom_pad_mask.shape:
        raise ValueError(
            f"backbone mask shape {tuple(backbone_mask.shape)} does not match "
            f"atom_pad_mask {tuple(atom_pad_mask.shape)}"
        )
    output = dict(features)
    output["atom_pad_mask"] = backbone_mask.to(
        device=atom_pad_mask.device, dtype=atom_pad_mask.dtype
    )
    output["atom_resolved_mask"] = backbone_mask.to(
        device=atom_pad_mask.device, dtype=torch.bool
    )
    return output


def features_with_sequence(
    fixed_features: Mapping[str, Any],
    sequence: str,
) -> dict[str, Any]:
    """Replace protein sequence features while preserving the WT atom layout."""

    from boltz.data import const

    res_type = fixed_features.get("res_type")
    msa = fixed_features.get("msa")
    profile = fixed_features.get("profile")
    token_pad_mask = fixed_features.get("token_pad_mask")
    if not all(isinstance(value, Tensor) for value in (res_type, msa, profile, token_pad_mask)):
        raise KeyError("features must contain res_type, msa, profile, and token_pad_mask")
    if res_type.shape[0] != 1 or msa.shape[0] != 1:
        raise ValueError("mutation features currently require batch size one")
    active_tokens = int(token_pad_mask.bool().sum().item())
    sequence = sequence.strip().upper()
    if len(sequence) != active_tokens:
        raise ValueError(
            f"Mutant sequence length {len(sequence)} does not match "
            f"the {active_tokens} active structure tokens"
        )
    ids = sequence_token_ids(sequence, device=res_type.device)
    one_hot = torch.nn.functional.one_hot(ids, num_classes=const.num_tokens)

    output = dict(fixed_features)
    mutated_res_type = res_type.clone()
    mutated_res_type[0, :active_tokens] = one_hot.to(res_type.dtype)
    output["res_type"] = mutated_res_type

    mutated_msa = msa.clone()
    mutated_msa[0, 0, :active_tokens] = ids.to(msa.device)
    output["msa"] = mutated_msa

    if msa.shape[1] > 1:
        output["profile"] = profile
    else:
        mutated_profile = profile.clone()
        mutated_profile[0, :active_tokens] = one_hot.to(profile.dtype)
        output["profile"] = mutated_profile
    return output


SEQUENCE_FEATURES = ("res_type", "msa", "profile")


def features_with_sequences(
    fixed_features: Mapping[str, Any],
    sequences: Sequence[str],
) -> dict[str, Any]:
    """Batch several mutant sequences against one shared WT atom layout."""

    if not sequences:
        raise ValueError("sequences must not be empty")
    batch = len(sequences)
    per_sequence = [
        features_with_sequence(fixed_features, sequence) for sequence in sequences
    ]

    output: dict[str, Any] = {}
    for key, value in fixed_features.items():
        if not isinstance(value, Tensor):
            output[key] = value
        elif key in SEQUENCE_FEATURES:
            output[key] = torch.cat(
                [features[key] for features in per_sequence], dim=0
            )
        elif value.ndim >= 1 and value.shape[0] == 1:
            output[key] = value.repeat(batch, *([1] * (value.ndim - 1)))
        else:
            output[key] = value
    return output
