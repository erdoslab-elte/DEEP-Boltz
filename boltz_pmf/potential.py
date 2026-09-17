"""The distogram potential: a log-odds over the native contact shell."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

# Boltz-2's distogram: 64 bins, 63 boundaries evenly spaced over 2-22 Angstrom.
N_BINS = 64
BIN_MIN, BIN_MAX = 2.0, 22.0
# The terminal bin is unbounded and has no defined volume. It is truncated at
# this radius for the volume correction
FAR_RADIUS = 45.0

# ddG is summed only over residue pairs within this distance of a mutated residue.
# Over the full 117,811 substitutions of ProteinGym Stability
# it raises mean per-assay Spearman from 0.6574 (every pair) to 0.7197.
# `None` restores the untruncated sum.
DEFAULT_RADIUS = 6.0


def bin_log_volumes(far_radius: float = FAR_RADIUS) -> np.ndarray:
    """log V for each of the 64 bins, in cubic Angstrom.

    Each resolved bin is a spherical shell, 4 pi r^2 dr, evaluated at the bin
    centre.
    """

    edges = np.linspace(BIN_MIN, BIN_MAX, N_BINS - 1)
    width = edges[1] - edges[0]
    centres = np.concatenate([edges[:1], 0.5 * (edges[:-1] + edges[1:])])
    volumes = 4.0 * np.pi * centres**2 * width
    far = (4.0 / 3.0) * np.pi * (far_radius**3 - BIN_MAX**3)
    return np.concatenate([np.log(volumes), [np.log(far)]])


@dataclass(frozen=True)
class PotentialConfig:
    """Everything that defines the score."""

    weighting: str = "entropy_p"

    # ddG only: `energy()` scores a whole structure and has no mutated site to
    # measure from, so it is always the untruncated sum.
    radius: float | None = DEFAULT_RADIUS

    volume_correction: bool = True
    far_radius: float = FAR_RADIUS
    recycling_steps: int = 3
    msa_depth: int = 256


def pair_weights(stats: dict[str, np.ndarray], weighting: str) -> np.ndarray:
    """Per-pair weights, all derived from the reference sequence and structure."""

    pair_mask = stats["mask"]
    entropy = np.exp(-stats["entropy"]) * pair_mask
    if weighting == "entropy":
        return entropy
    if weighting == "entropy_p":
        return entropy * np.exp(stats["log_prob"]) * pair_mask
    if weighting == "flat":
        return pair_mask.astype(float)
    raise ValueError(f"unknown weighting {weighting!r}")


def pair_potential(
    log_probs: Tensor,
    bins: Tensor,
    pair_mask: Tensor,
    *,
    log_volume: Tensor | None = None,
) -> Tensor:
    """Per-pair log-odds, ``[N, N]``.

    ``log_probs`` is the model's ``[N, N, 64]`` prediction, ``bins`` the bin each
    pair actually occupies in the supplied structure.
    """

    observed = torch.gather(log_probs, -1, bins.unsqueeze(-1)).squeeze(-1)
    far = log_probs[..., -1]
    if log_volume is not None:
        lv = log_volume.to(device=log_probs.device, dtype=log_probs.dtype)
        observed = observed - lv[bins]
        far = far - lv[-1]
    return (observed - far) * pair_mask


def site_pair_mask(distances: np.ndarray, sites: "tuple[int, ...]",
                   radius: float) -> np.ndarray:
    """Which pairs are inside `radius` of a mutated residue, ``[N, N]``."""

    if not sites:
        raise ValueError("no mutated sites to measure from")
    near = distances[list(sites)].min(axis=0)
    return np.minimum(near[:, None], near[None, :]) <= radius


def residue_profile(pair_terms: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-residue decomposition: the sum over each residue's partners."""

    off_diagonal = ~np.eye(pair_terms.shape[0], dtype=bool)
    return (pair_terms * weights * off_diagonal).sum(axis=1)
