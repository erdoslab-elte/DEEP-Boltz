"""Boltz-2 trunk evaluation, and the distogram statistics read off it.

`prepare_context` runs the trunk once for one sequence (input embedder,
recycling, MSA module, pairformer) and returns the pair representation
`z_trunk` that the distogram head reads.

`distogram_pair_stats` reads the head, and `_token_centres` reconstructs the
CB positions the head bins against. 
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class Boltz2EnergyScorer:
    """Run the frozen Boltz-2 trunk and read its distogram head.

    Parameters
    ----------
    model:
        A loaded ``boltz.model.models.boltz2.Boltz2`` instance.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.structure = model.structure_module

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        """Device holding the frozen structure model."""

        try:
            return next(self.structure.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def prepare_context(
        self, feats: Mapping[str, Any], recycling_steps: int = 3
    ) -> dict[str, Any]:
        """Run Boltz-2's coordinate-independent trunk once for one sequence.
        """

        required = (
            "input_embedder",
            "s_init",
            "z_init_1",
            "z_init_2",
            "rel_pos",
            "token_bonds",
            "contact_conditioning",
            "msa_module",
            "pairformer_module",
            "diffusion_conditioning",
        )
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise TypeError(
                "prepare_context requires a Boltz2 model; missing attributes: "
                + ", ".join(missing)
            )
        if recycling_steps < 0:
            raise ValueError("recycling_steps cannot be negative")

        device_feats: dict[str, Any] = {
            key: value.to(self.device) if isinstance(value, Tensor) else value
            for key, value in feats.items()
        }

        with torch.no_grad():
            s_inputs = self.model.input_embedder(device_feats)
            s_init = self.model.s_init(s_inputs)
            z_init = (
                self.model.z_init_1(s_inputs)[:, :, None]
                + self.model.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = self.model.rel_pos(device_feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + self.model.token_bonds(
                device_feats["token_bonds"].float()
            )
            if getattr(self.model, "bond_type_feature", False):
                z_init = z_init + self.model.token_bonds_type(
                    device_feats["type_bonds"].long()
                )
            z_init = z_init + self.model.contact_conditioning(device_feats)

            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)
            mask = device_feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            for _ in range(recycling_steps + 1):
                s = s_init + self.model.s_recycle(self.model.s_norm(s))
                z = z_init + self.model.z_recycle(self.model.z_norm(z))

                if getattr(self.model, "use_templates", False):
                    template_module = self.model.template_module
                    if getattr(self.model, "is_template_compiled", False):
                        template_module = template_module._orig_mod
                    z = z + template_module(
                        z,
                        device_feats,
                        pair_mask,
                        use_kernels=getattr(self.model, "use_kernels", False),
                    )

                msa_module = self.model.msa_module
                if getattr(self.model, "is_msa_compiled", False):
                    msa_module = msa_module._orig_mod
                z = z + msa_module(
                    z,
                    s_inputs,
                    device_feats,
                    use_kernels=getattr(self.model, "use_kernels", False),
                )

                pairformer_module = self.model.pairformer_module
                if getattr(self.model, "is_pairformer_compiled", False):
                    pairformer_module = pairformer_module._orig_mod
                s, z = pairformer_module(
                    s,
                    z,
                    mask=mask,
                    pair_mask=pair_mask,
                    use_kernels=getattr(self.model, "use_kernels", False),
                )

            q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                self.model.diffusion_conditioning(
                    s_trunk=s,
                    z_trunk=z,
                    relative_position_encoding=relative_position_encoding,
                    feats=device_feats,
                )
            )

        return {
            "s_inputs": s_inputs.float().detach(),
            "s_trunk": s.float().detach(),
            # Retained for the distogram head, which is a linear read-out of the
            # pair representation the trunk has already produced.
            "z_trunk": z.float().detach(),
            "feats": device_feats,
            "multiplicity": 1,
            "diffusion_conditioning": {
                "q": q.detach(),
                "c": c.detach(),
                "to_keys": to_keys,
                "atom_enc_bias": atom_enc_bias.detach(),
                "atom_dec_bias": atom_dec_bias.detach(),
                "token_trans_bias": token_trans_bias.detach(),
            },
        }

    def _token_centres(self, x: Tensor, mask: Tensor, context: Mapping[str, Any]) -> Tensor:
        """Virtual distogram centres for coordinates ``x``, shape ``[B, N, 3]``.
        """

        from boltz.data import const

        scored = torch.nonzero(mask[0]).flatten()
        if scored.numel() % 4:
            raise ValueError(
                "expected four scored atoms per token in N, CA, C, O order; "
                f"got {scored.numel()}"
            )
        idx = scored.view(-1, 4)
        n, ca, c = x[:, idx[:, 0]], x[:, idx[:, 1]], x[:, idx[:, 2]]
        b_vec, c_vec = ca - n, c - ca
        a_vec = torch.cross(b_vec, c_vec, dim=-1)
        cb = -0.58273431 * a_vec + 0.56802827 * b_vec - 0.54067466 * c_vec + ca

        # Glycine has no CB, so its centre is CA -- and which residues are
        # glycine is a property of *each* sequence in the batch. Reading the mask
        # from row 0 and applying it to the rest silently mis-centres every
        # mutant whose glycine pattern differs from the first one's, which for a
        # batch containing any substitution to or from glycine is all of them.
        res_type = context["feats"]["res_type"]
        gly = res_type.argmax(-1) == const.token_ids[const.prot_letter_to_token["G"]]
        gly = gly[:, : cb.shape[1]].to(cb.device)
        if gly.shape[0] == 1 and cb.shape[0] > 1:
            gly = gly.expand(cb.shape[0], -1)
        return torch.where(gly[:, :, None], ca, cb)

    def distogram_pair_stats(
        self, context: Mapping[str, Any], log_volume: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Per-pair distogram quantities, all ``[B, N, N]``.

        - ``log_prob``: log p assigned to the bin the structure actually
          occupies, the term :meth:`distogram_log_likelihood` sums.
        - ``entropy``: Shannon entropy of the full 64-bin prediction for that
          pair. Low entropy means the model is confident about the distance
        - ``mode_prob``: the peak of the prediction, ``max_b p(b | s)``.
        - ``mode_distance``: the distance at that peak
        - ``mask``: valid-pair mask.
        """

        module = getattr(self.model, "distogram_module", None)
        if module is None:
            raise TypeError("model has no distogram_module")
        z = context.get("z_trunk")
        if not isinstance(z, Tensor):
            raise KeyError("context lacks z_trunk; rebuild it with prepare_context")
        feats = context["feats"]
        target = feats["disto_target"].to(z.device)
        mask = feats["token_disto_mask"].to(z.device).float()

        with torch.no_grad():
            log_probs = torch.log_softmax(module(z).float(), dim=-1)
            if target.ndim == log_probs.ndim:
                observed = (target.float() * log_probs).sum(dim=(-1, -2))
                probs = log_probs.exp().squeeze(-2)
                logs = log_probs.squeeze(-2)
            else:
                observed = (target.float().unsqueeze(-2) * log_probs).sum(dim=(-1, -2))
                probs = log_probs.exp().squeeze(-2)
                logs = log_probs.squeeze(-2)
            if log_volume is not None:
                # Divide by bin volume and renormalise, so `logs` describes a
                # radial density rather than a histogram. `target` is one-hot
                # over bins, so contracting against it reads the observed bin's
                # value under the corrected distribution.
                lv = log_volume.to(device=probs.device, dtype=probs.dtype)
                logs = logs - lv
                logs = logs - torch.logsumexp(logs, dim=-1, keepdim=True)
                probs = logs.exp()
                one_hot = target.float()
                while one_hot.ndim > logs.ndim:
                    one_hot = one_hot.squeeze(-2)
                observed = (one_hot * logs).sum(dim=-1)
            entropy = -(probs * logs).sum(dim=-1)
            mode_prob, mode_bin = probs.max(dim=-1)
            edges = torch.linspace(2.0, 22.0, 63, device=z.device, dtype=probs.dtype)
            centres = torch.cat(
                [edges[:1], 0.5 * (edges[:-1] + edges[1:]), edges[-1:]]
            )
            mode_distance = centres[mode_bin.clamp(0, centres.numel() - 1)]
            batch = z.shape[0]
            if mask.shape[0] == 1 and batch > 1:
                mask = mask.expand(batch, -1)
            pair_mask = mask[:, :, None] * mask[:, None, :]
        return {
            "log_prob": observed,
            "entropy": entropy,
            "mode_prob": mode_prob,
            "mode_distance": mode_distance,
            "mask": pair_mask,
        }

