"""Command line: an energy for a structure, or ddG for substitutions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="boltz-pmf",
        description="Statistical-potential energy from a frozen Boltz-2 checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  boltz-pmf structure.pdb
  boltz-pmf structure.pdb --msa alignment.a3m
  boltz-pmf structure.pdb --mutations G13C A20V T45S
  boltz-pmf structure.pdb --mutations G13C A20V --json out.json

  boltz-pmf --sequence MKVLA... --mutations G13C A20V
  boltz-pmf --fasta target.fasta --msa alignment.a3m --mutations G13C

A structure is optional. With --sequence or --fasta, Boltz-2 folds the wild type
itself and scores mutations against its own prediction.

With --mutations the program reports ddG for every single substitution and every
pair of them (positions must differ). Positive ddG means stabilising.
""")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("structure", type=Path, nargs="?", default=None,
                        help="PDB or mmCIF file to score.")
    source.add_argument("--sequence", default=None, metavar="MKVLA...",
                        help="One-letter protein sequence. Boltz-2 folds it and "
                             "scores against its own prediction; no structure "
                             "file is needed.")
    source.add_argument("--fasta", type=Path, default=None,
                        help="FASTA file whose first record is the sequence to "
                             "fold. Equivalent to --sequence.")
    p.add_argument("--mutations", nargs="+", metavar="G13C",
                   help="Substitutions in the form G13C, 1-based over the scored "
                        "chain. Every single and every pair is reported.")
    p.add_argument("--chain", default=None,
                   help="Chain to score; default is the first protein chain.")
    p.add_argument("--msa", type=Path, default=None,
                   help="Alignment (a3m/a2m) for the scored chain. Strongly "
                        "recommended: worth +0.19 Spearman on ddG. Without it "
                        "the model runs single-sequence.")
    p.add_argument("--msa-depth", type=int, default=256,
                   help="Alignment depth cap (default 256, the selected value).")
    p.add_argument("--max-order", type=int, default=2, choices=(1, 2),
                   help="Highest combination order to report (default 2).")
    p.add_argument("--no-volume-correction", action="store_true",
                   help="Disable the radial-volume term. It cannot change ddG, "
                        "and it is what makes the score work on decoys.")
    p.add_argument("--weighting", default="entropy_p",
                   choices=("entropy_p", "entropy", "flat"),
                   help="Use `entropy` when the structure is the thing being "
                        "judged rather than an input, e.g. decoy ranking.")
    p.add_argument("--sampling-steps", type=int, default=200,
                   help="Reverse-diffusion steps when folding (default 200, "
                        "Boltz's own default). Ignored with a structure file.")
    p.add_argument("--fold-seed", type=int, default=0,
                   help="Seed for the folding pass. Sampling is stochastic, so "
                        "this pins which structure is predicted.")
    p.add_argument("--save-folded", type=Path, default=None,
                   help="Write the folded backbone (N/CA/C/O) to this PDB.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--per-residue", type=Path, default=None,
                   help="Write the per-residue energy decomposition to a CSV.")
    p.add_argument("--json", type=Path, default=None, help="Write results as JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.structure is not None and not args.structure.is_file():
        print(f"no such file: {args.structure}", file=sys.stderr)
        return 2

    sequence = args.sequence
    if args.fasta is not None:
        if not args.fasta.is_file():
            print(f"no such file: {args.fasta}", file=sys.stderr)
            return 2
        records = [
            line.strip() for line in args.fasta.read_text().splitlines()
            if line.strip() and not line.startswith(">")
        ]
        if not records:
            print(f"no sequence found in {args.fasta}", file=sys.stderr)
            return 2
        sequence = "".join(records)
    if sequence is not None:
        sequence = "".join(sequence.split()).upper()

    from .api import PMFScorer
    from .calibration import LOAO_RMSE, NULL_RMSE
    from .potential import PotentialConfig

    config = PotentialConfig(
        weighting=args.weighting,
        volume_correction=not args.no_volume_correction,
        msa_depth=args.msa_depth,
    )
    try:
        scorer = PMFScorer(args.structure, sequence=sequence, chain=args.chain,
                           config=config, msa=args.msa, device=args.device,
                           checkpoint=args.checkpoint,
                           sampling_steps=args.sampling_steps,
                           fold_seed=args.fold_seed)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    energy = scorer.energy()

    if args.save_folded:
        if scorer.folded is None:
            print("--save-folded needs --sequence or --fasta; there is nothing "
                  "to save when the structure was supplied", file=sys.stderr)
            return 2
        from .folding import write_backbone_pdb
        written = write_backbone_pdb(args.save_folded, scorer.folded)
        print(f"folded backbone -> {written}")

    payload: dict = {
        "structure": str(args.structure) if args.structure else None,
        "backbone_source": "supplied" if scorer.folded is None else "boltz2_folded",
        "n_residues": energy.n_residues,
        "sequence": energy.sequence,
        "energy_nats": energy.nats,
        "energy_kcal_per_mol": energy.kcal_per_mol,
        "contributing_pairs": energy.n_contributing_pairs,
        "alignment": str(args.msa) if args.msa else None,
        "volume_correction": config.volume_correction,
        "weighting": config.weighting,
    }
    if scorer.folded is not None:
        payload["sampling_steps"] = scorer.folded.sampling_steps
        payload["fold_seed"] = scorer.folded.seed

    origin = (str(args.structure) if scorer.folded is None else
              f"Boltz-2 prediction ({scorer.folded.sampling_steps} steps, "
              f"seed {scorer.folded.seed})")
    print(f"backbone           {origin}")
    print(f"chain length       {energy.n_residues} residues")
    print(f"contributing pairs {energy.n_contributing_pairs} "
          f"(pairs beyond 22 A cancel exactly and are excluded)")
    print(f"alignment          {args.msa if args.msa else 'none (single-sequence)'}")
    print()
    print(f"energy             {energy.kcal_per_mol:+10.3f} kcal/mol"
          f"   ({energy.nats:+.1f} nats)")
    print("                   an interaction pseudo-energy, NOT a folding free")
    print("                   energy; only differences are validated.")

    if args.mutations:
        try:
            results = scorer.ddg(args.mutations, max_order=args.max_order,
                                 batch_size=args.batch_size)
        except ValueError as error:
            print(f"\nerror: {error}", file=sys.stderr)
            print(f"the scored chain begins {energy.sequence[:30]}...",
                  file=sys.stderr)
            return 1
        singles = [r for r in results if len(r.mutations) == 1]
        pairs = [r for r in results if len(r.mutations) == 2]
        print(f"\n{'mutation':<20}{'ddG (kcal/mol)':>16}{'delta (nats)':>15}")
        print("-" * 51)
        for r in singles:
            print(f"{r.label:<20}{r.ddg_kcal_per_mol:>16.3f}{r.delta_nats:>15.1f}")
        if pairs:
            print(f"{'-- pairs --':<20}")
            for r in pairs:
                print(f"{r.label:<20}{r.ddg_kcal_per_mol:>16.3f}{r.delta_nats:>15.1f}")
        print()
        print("positive ddG = stabilising. Leave-one-assay-out accuracy on")
        print(f"ProteinGym Stability: RMSE {LOAO_RMSE:.2f} kcal/mol against a")
        print(f"{NULL_RMSE:.2f} null, per-assay Spearman 0.657. The ranking is")
        print("more trustworthy than the magnitude.")
        payload["ddg"] = [
            {"label": r.label, "mutations": list(r.mutations),
             "ddg_kcal_per_mol": r.ddg_kcal_per_mol, "delta_nats": r.delta_nats}
            for r in results
        ]

    if args.per_residue:
        args.per_residue.parent.mkdir(parents=True, exist_ok=True)
        with args.per_residue.open("w") as handle:
            handle.write("position,residue,energy_nats\n")
            for i, v in enumerate(energy.per_residue_nats, start=1):
                handle.write(f"{i},{energy.sequence[i - 1]},{v:.6f}\n")
        print(f"\nper-residue decomposition -> {args.per_residue}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"json -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
