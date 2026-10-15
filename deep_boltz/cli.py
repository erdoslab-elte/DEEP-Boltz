"""Command line: an energy for a structure, or ddG for substitutions."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

from .msa_server import DEFAULT_CACHE_DIR, DEFAULT_MSA_SERVER


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deep-boltz",
        description="Statistical-potential energy from a frozen Boltz-2 checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  deep-boltz structure.pdb --mutations G13C A20V T45S
  deep-boltz structure.pdb --msa alignment.a3m --mutations G13C A20V
  deep-boltz structure.pdb --mutations 13C 20V --json out.json

  deep-boltz structure.pdb --mutations G13C A20V:T45S     # exactly these two
  deep-boltz structure.pdb --mutations-file panel.txt     # one mutant per line

  deep-boltz --sequence MKVLA... --mutations G13C A20V
  deep-boltz --fasta target.fasta --msa alignment.a3m --mutations G13C

DEEP-Boltz reports ddG for every single substitution and every 
pair of them (positions must differ). Positive ddGmeans stabilising.

The wild-type letter is optional: 13C and G13C mean the same thing, and when it
is given it is checked against the scored chain.

A structure is optional. With --sequence or --fasta, Boltz-2 folds the wild type
itself and scores mutations against its own prediction.
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
                   help="Mutants to score, as G13C or 13C (1-based over the "
                        "scored chain; the wild-type letter is optional and is "
                        "checked against the chain when given). Join two with a "
                        "colon, G13C:A20V, to score that specific double. If "
                        "every entry is a single substitution, every pair of "
                        "them is scored too; if any entry is a double, only the "
                        "listed mutants are scored.")
    p.add_argument("--mutations-file", type=Path, metavar="FILE",
                   help="Read mutants from a file instead, one per line, in the "
                        "same notation. Blank lines and lines beginning with # "
                        "are ignored. Use this for a defined panel rather than "
                        "listing hundreds of mutants on the command line.")
    p.add_argument("--chain", default=None,
                   help="Chain to score; default is the first protein chain.")
    p.add_argument("--msa", type=Path, default=None,
                   help="Alignment (a3m/a2m) for the scored chain. Strongly "
                        "recommended: worth +0.19 Spearman on ddG. Without it "
                        "the model runs single-sequence.")
    p.add_argument("--use-msa-server", action="store_true",
                   help="Search an MMseqs2 server for the alignment instead of "
                        "supplying one. Uploads the query sequence to the "
                        "server; results are cached by sequence.")
    p.add_argument("--msa-server-url", default=DEFAULT_MSA_SERVER,
                   help=f"MSA server to search (default {DEFAULT_MSA_SERVER}).")
    p.add_argument("--msa-cache", type=Path, default=None,
                   help=f"Where searched alignments are kept (default "
                        f"{DEFAULT_CACHE_DIR}).")
    p.add_argument("--msa-depth", type=int, default=256,
                   help="Alignment depth cap (default 256, the selected value).")
    p.add_argument("--radius", default="6", metavar="ANGSTROM",
                   help="ddG is summed only over residue pairs within this "
                        "distance of a mutated residue (default 6, the selected "
                        "value: 0.7197 mean per-assay Spearman against 0.6574 "
                        "for the untruncated sum). Pass `none` for every pair. "
                        "Only radii with a fitted kcal/mol calibration are "
                        "accepted: 4, 6, 8, 10, 12, 16, 22, none.")
    p.add_argument("--max-order", type=int, default=2, choices=(1, 2),
                   help="When entries are enumerated, the highest order to "
                        "report: 1 for singles only, 2 (default) to add every "
                        "pair. Ignored when the entries name doubles directly.")
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

    # One of the two mutant sources is required; argparse cannot express that
    # across two flags, so it is checked here.
    if not args.mutations and args.mutations_file is None:
        print("error: one of --mutations or --mutations-file is required; this "
              "is a ddG predictor and there is nothing to score without one",
              file=sys.stderr)
        return 2
    if args.mutations and args.mutations_file is not None:
        print("error: pass either --mutations or --mutations-file, not both",
              file=sys.stderr)
        return 2
    mutations = list(args.mutations or [])
    if args.mutations_file is not None:
        if not args.mutations_file.is_file():
            print(f"no such file: {args.mutations_file}", file=sys.stderr)
            return 2
        for line in args.mutations_file.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                mutations.extend(line.split())
        if not mutations:
            print(f"no mutants found in {args.mutations_file}", file=sys.stderr)
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

    from . import mutations as _mut
    from .api import DEEPScorer, NoMSAWarning
    from .calibration import Calibration
    from .potential import PotentialConfig

    if args.radius.strip().lower() in ("none", "all", "inf"):
        radius = None
    else:
        try:
            radius = float(args.radius)
        except ValueError:
            print(f"error: --radius must be a number or `none`, not "
                  f"{args.radius!r}", file=sys.stderr)
            return 2
    try:
        Calibration.for_radius(radius)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    config = PotentialConfig(
        weighting=args.weighting,
        radius=radius,
        volume_correction=not args.no_volume_correction,
        msa_depth=args.msa_depth,
    )
    try:
        # The summary below carries this warning in its own format.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NoMSAWarning)
            scorer = DEEPScorer(args.structure, sequence=sequence, chain=args.chain,
                               config=config, msa=args.msa,
                               use_msa_server=args.use_msa_server,
                               msa_server_url=args.msa_server_url,
                               msa_cache=args.msa_cache, device=args.device,
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
        "alignment": str(scorer.msa_path) if scorer.msa_path else None,
        "alignment_source": ("msa server" if args.use_msa_server else
                             "supplied" if args.msa else None),
        "volume_correction": config.volume_correction,
        "weighting": config.weighting,
        "ddg_radius_angstrom": config.radius,
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
    alignment = scorer.msa_path
    print(f"alignment          {alignment if alignment else 'none (single-sequence)'}"
          f"{' (from the MSA server)' if args.use_msa_server else ''}")
    if alignment is None:
        print("                   WARNING: without an alignment the trunk sees")
        print("                   no coevolution signal, and this energy (and")
        print("                   any ddG below) is significantly less")
        print("                   accurate. Pass --msa alignment.a3m.")
    print()
    print(f"energy             {energy.kcal_per_mol:+10.3f} kcal/mol"
          f"   ({energy.nats:+.1f} nats)")
    print(f"ddG radius         "
          + ("every pair (untruncated)" if config.radius is None
             else f"{config.radius:g} A around each mutated residue"))

    try:
        results = scorer.ddg(mutations, max_order=args.max_order,
                             batch_size=args.batch_size)
    except ValueError as error:
        print(f"\nerror: {error}", file=sys.stderr)
        print(f"the scored chain begins {energy.sequence[:30]}...",
              file=sys.stderr)
        return 1
    singles = [r for r in results if len(r.mutations) == 1]
    pairs = [r for r in results if len(r.mutations) == 2]
    explicit = any(_mut.is_mutant_spec(m) for m in mutations)
    print(f"mutants            {len(results)} scored "
          + ("as listed" if explicit
             else f"({len(singles)} single, {len(pairs)} enumerated pairs)"))
    print(f"\n{'mutation':<20}{'ddG (kcal/mol)':>16}{'delta (nats)':>15}"
          f"{'pairs':>8}")
    print("-" * 59)
    for r in singles:
        print(f"{r.label:<20}{r.ddg_kcal_per_mol:>16.3f}{r.delta_nats:>15.1f}"
              f"{r.n_pairs:>8}")
    if pairs:
        print(f"{'-- pairs --':<20}")
        for r in pairs:
            print(f"{r.label:<20}{r.ddg_kcal_per_mol:>16.3f}"
                  f"{r.delta_nats:>15.1f}{r.n_pairs:>8}")
    print()
    print("positive ddG = stabilising.")
    payload["ddg"] = [
        {"label": r.label, "mutations": list(r.mutations),
         "ddg_kcal_per_mol": r.ddg_kcal_per_mol, "delta_nats": r.delta_nats,
         "n_pairs": r.n_pairs}
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
