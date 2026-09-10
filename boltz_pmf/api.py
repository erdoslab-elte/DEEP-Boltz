"""
`energy` scores one structure. `ddg` scores substitutions against a fixed
backbone: the wild-type coordinates are reused for every mutant and only the
sequence conditioning changes, which is what makes the comparison a difference of
one system rather than a comparison between two predicted structures.

That backbone can come from either of two places. Pass `structure=` to use
coordinates you supply. Pass `sequence=` and Boltz-2 folds the wild type itself,
so no external structure is needed and the pipeline contains one model
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import mutations as mut
from ._adapter import featurize_structure, load_boltz2_model
from ._features import features_with_sequence, features_with_sequences, fixed_backbone_features
from ._trunk import Boltz2EnergyScorer
from .calibration import Calibration
from .folding import DEFAULT_SAMPLING_STEPS, fold_sequence
from .potential import (
    PotentialConfig, bin_log_volumes, pair_potential, pair_weights, residue_profile)


@dataclass
class EnergyResult:
    nats: float
    kcal_per_mol: float
    n_residues: int
    n_contributing_pairs: int
    sequence: str
    per_residue_nats: np.ndarray = field(repr=False)


@dataclass
class DdgResult:
    label: str
    delta_nats: float
    ddg_kcal_per_mol: float
    mutations: tuple[str, ...]


class PMFScorer:
    """Holds the model so a batch of mutants costs one load, not many."""

    def __init__(
        self,
        structure: str | Path | None = None,
        *,
        sequence: str | None = None,
        chain: str | None = None,
        config: PotentialConfig | None = None,
        msa: str | Path | None = None,
        device: str = "cuda:0",
        checkpoint: str | Path | None = None,
        calibration: Calibration | None = None,
        sampling_steps: int = DEFAULT_SAMPLING_STEPS,
        fold_seed: int = 0,
    ) -> None:
        if (structure is None) == (sequence is None):
            raise ValueError(
                "supply exactly one of `structure` (score the coordinates you "
                "have) or `sequence` (let Boltz-2 fold the wild type itself)"
            )
        self.config = config or PotentialConfig()
        self.calibration = calibration or Calibration()
        self.device = device
        torch.set_float32_matmul_precision("highest")
        self.model = load_boltz2_model(checkpoint, device=device)
        self._scorer = Boltz2EnergyScorer(self.model)

        depth = self.config.msa_depth if msa is not None else None
        if sequence is not None:
            self.folded = fold_sequence(
                self.model, self._scorer, sequence, msa=msa, msa_depth=depth,
                recycling_steps=self.config.recycling_steps,
                sampling_steps=sampling_steps, seed=fold_seed)
            self.batch = self.folded.batch
            coords = self.folded.coords
        else:
            self.folded = None
            self.batch = featurize_structure(
                Path(structure), chain=chain, msa_path=msa, max_msa_seqs=depth)
            coords = self.batch.coords
        self._fixed = fixed_backbone_features(self.batch.features, self.batch.backbone_mask)
        self._mask = self.batch.backbone_mask
        self._coords = coords if coords.ndim == 3 else coords.unsqueeze(0)
        self._log_volume = (
            torch.from_numpy(bin_log_volumes(self.config.far_radius)).float()
            if self.config.volume_correction else None)

        # The reference pass: fixes the weights and the wild-type sequence.
        self.sequence = self._read_sequence()
        if self.folded is not None and self.sequence != self.folded.sequence:
            raise RuntimeError(
                "featurized sequence does not match the sequence folded; "
                "refusing to score a structure that is not this sequence's"
            )
        self._reference = self._score(self.sequence, reference=True)

    # -- internals ---------------------------------------------------------

    def _read_sequence(self) -> str:
        from boltz.data import const

        context = self._scorer.prepare_context(self._fixed, recycling_steps=1)
        try:
            res_type = context["feats"]["res_type"][0].argmax(-1).cpu().numpy()
            n_tok = int(context["feats"]["token_pad_mask"][0].bool().sum().item())
        finally:
            del context
            gc.collect()
        lookup = {v: k for k, v in const.token_ids.items()}
        letters = []
        for token in res_type[:n_tok]:
            name = lookup.get(int(token), "UNK")
            letters.append(const.prot_token_to_letter.get(name, "X"))
        return "".join(letters)

    def _contexts(self, sequences: list[str]):
        feats = (features_with_sequence(self._fixed, sequences[0]) if len(sequences) == 1
                 else features_with_sequences(self._fixed, sequences))
        return self._scorer.prepare_context(feats, recycling_steps=self.config.recycling_steps)

    def _score(self, sequence_or_list, *, reference: bool = False):
        sequences = ([sequence_or_list] if isinstance(sequence_or_list, str)
                     else list(sequence_or_list))
        context = self._contexts(sequences)
        try:
            module = self.model.distogram_module
            z = context["z_trunk"]
            with torch.no_grad():
                log_probs = torch.log_softmax(module(z).float(), dim=-1)
                if log_probs.ndim == 5:
                    log_probs = log_probs[:, :, :, 0, :]
            n_tok = int(context["feats"]["token_pad_mask"][0].bool().sum().item())
            centres = self._scorer._token_centres(
                self._coords.to(z.device),
                (self._mask if self._mask.ndim == 2 else self._mask.unsqueeze(0)).to(z.device),
                context)
            boundaries = torch.linspace(2.0, 22.0, 63, device=z.device, dtype=centres.dtype)
            distances = torch.cdist(centres, centres)
            bins = (distances.unsqueeze(-1) > boundaries).sum(dim=-1).long()
            if bins.shape[0] == 1 and log_probs.shape[0] > 1:
                bins = bins.expand(log_probs.shape[0], -1, -1)
            token_mask = context["feats"]["token_disto_mask"].to(z.device).float()
            if token_mask.shape[0] == 1 and log_probs.shape[0] > 1:
                token_mask = token_mask.expand(log_probs.shape[0], -1)
            pair_mask = token_mask[:, :, None] * token_mask[:, None, :]
            terms = pair_potential(log_probs, bins, pair_mask,
                                   log_volume=self._log_volume).cpu().numpy()
            if reference:
                stats = {k: v.cpu().numpy()[0] for k, v in
                         self._scorer.distogram_pair_stats(context).items()}
                if self.folded is not None:
                    # `stats["log_prob"]` is read off `disto_target`, which the
                    # featurizer computed from the placeholder at the origin, not
                    # from the folded backbone. Rebuild it at the coordinates
                    # actually being scored. `entropy` and `mask` are functions of
                    # the sequence alone and stay valid either way.
                    observed = torch.gather(
                        log_probs, -1, bins.unsqueeze(-1)).squeeze(-1)
                    stats["log_prob"] = observed[0].cpu().numpy()
                self._weights = pair_weights(stats, self.config.weighting)[:n_tok, :n_tok]
                self._n_tok = n_tok
            return terms[:, :n_tok, :n_tok]
        finally:
            del context
            gc.collect()
            torch.cuda.empty_cache()

    def _total(self, terms: np.ndarray) -> float:
        off = ~np.eye(terms.shape[0], dtype=bool)
        return float((terms * self._weights * off).sum())

    # -- public ------------------------------------------------------------

    def energy(self) -> EnergyResult:
        terms = self._reference[0]
        nats = self._total(terms)
        off = ~np.eye(terms.shape[0], dtype=bool)
        live = int(((np.abs(terms) > 1e-9) & off).sum() // 2)
        return EnergyResult(
            nats=nats,
            kcal_per_mol=self.calibration.energy(nats, self._n_tok),
            n_residues=self._n_tok,
            n_contributing_pairs=live,
            sequence=self.sequence,
            per_residue_nats=residue_profile(terms, self._weights),
        )

    def ddg(self, tokens: list[str], *, max_order: int = 2,
            batch_size: int = 8) -> list[DdgResult]:
        parsed = [mut.parse(t) for t in tokens]
        for m in parsed:
            mut.validate(m, self.sequence)
        combos = mut.combinations(parsed, max_order=max_order)
        wt_nats = self._total(self._reference[0])

        results: list[DdgResult] = []
        for start in range(0, len(combos), batch_size):
            window = combos[start : start + batch_size]
            seqs = [mut.apply(self.sequence, c) for c in window]
            terms = self._score(seqs)
            for i, combo in enumerate(window):
                delta = self._total(terms[i]) - wt_nats
                results.append(DdgResult(
                    label="+".join(str(m) for m in combo),
                    delta_nats=delta,
                    ddg_kcal_per_mol=self.calibration.ddg(delta, self._n_tok),
                    mutations=tuple(str(m) for m in combo),
                ))
        return results
