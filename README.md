# Boltz-PMF

A zero-shot statistical potential from the **Boltz-2** structure prediction method.

Boltz-PMF gives accurate estimates for protein stability using free energy calculation.

## Basic usage

An alignment matters a lot here, and the easiest way to get one is to let
Boltz-PMF search for it — no alignment file needed:

```bash
boltz-pmf structure.pdb --use-msa-server --mutations G13C A20V T45S
```

If you already have an alignment, pass it instead and nothing leaves your
machine:

```bash
boltz-pmf structure.pdb --msa alignment.a3m --mutations G13C A20V T45S
```

**A structure is optional.** Boltz-PMF can also generate the reference wild type structure, 
then scores mutations against its own prediction.

```bash
boltz-pmf --sequence MKVLATDLA... --msa alignment.a3m --mutations G13C A20V T45S
```

On ProteinGym Stability the two routes are equivalent: **0.6596** mean per-assay
Spearman folding against **0.6574** on the AlphaFold2 backbone the benchmark
ships.

---

## What is Boltz-PMF?

Boltz-PMF converts the structural information learned by Boltz-2 into an effective, sequence-dependent energy function.

Boltz-2's distogram head predicts, for every pair of residues, a probability distribution over their spatial separation given the protein sequence. Through the Boltzmann relation, a probability distribution along a coordinate can be interpreted as an effective free-energy profile. We therefore treat the predicted distograms as defining an **implied potential of mean force (PMF)** over residue-residue distances.

This potential summarizes the interaction preferences learned by Boltz-2 from protein structures, with effects such as solvent and other unobserved degrees of freedom implicitly integrated out.

For a structure with residue-pair distances $d_{ij}$, Boltz-PMF calculates the score as

$$
E = \sum_{ij} w_{ij}\left[\log p\left(\mathrm{bin}(d_{ij}) \mid s\right)-\log p\left(\mathrm{far} \mid s\right)\right]
$$

Here:

- $p(\mathrm{bin}(d_{ij}) \mid s)$ is the probability assigned by Boltz-2 to the observed distance between residues $i$ and $j$.
- $p(\mathrm{far} \mid s)$ is the probability assigned to the distogram's non-contact, or **far**, bin.
- $w_{ij}$ controls the contribution of each residue pair.
- $s$ denotes the protein sequence.

### Reference-state normalization

The far-distance bin provides a natural reference state. Rather than using the absolute probability of a particular residue-pair distance, Boltz-PMF measures how strongly Boltz-2 favors that distance relative to the residues being far apart:

$$
\log \left(\frac{p\left(\mathrm{bin}(d_{ij}) \mid s\right)}{p\left(\mathrm{far} \mid s\right)}\right)
$$

The resulting score can therefore be interpreted as a **sequence-dependent effective interaction potential**.

### Relation to the Boltz-2 diffusion process

This reference-state interpretation is particularly relevant because Boltz-2 is a diffusion model. It was not trained to fold proteins starting from physically realistic unfolded conformations. Instead, structure generation begins from a highly noised, approximately Gaussian state.

Consequently, the model does not contain an explicit physical unfolded-state ensemble that can be used directly as a reference state. Normalization by the far-distance bin provides a simple alternative: it removes the background probability of two residues being effectively non-interacting and isolates the model's preference for bringing them to a particular distance.

In this sense, the far-bin normalization plays a role analogous to the **reference state in conventional statistical potentials**, while the resulting Boltz-PMF score captures the sequence-dependent structural preferences encoded by Boltz-2.

## Installation

Boltz-2 must already be installed and its weights available; this package reads
that checkpoint and does not ship one.

```bash
git clone <url> boltz_pmf
cd boltz_pmf
pip install -e .
```

Requires Python ≥ 3.10, `torch`, `numpy`, `gemmi`, `pyyaml`, `scipy`, and a GPU
with enough memory for the target (roughly N² in chain length).

## Usage

### Letting the MSA server find the alignment

`--use-msa-server` searches an MMseqs2 server for the alignment, the same way
`boltz predict --use_msa_server` does. The sequence of the scored chain is read
straight out of the structure, so there is nothing else to prepare:

```bash
boltz-pmf structure.pdb --use-msa-server
```

```
MSA                searching https://api.colabfold.com (40 residues uploaded); this usually takes a minute
COMPLETE: 100%|██████████| 150/150 [elapsed: 00:01 remaining: 00:00]
MSA                cached at ~/.boltz/pmf_msa/121b21e7e8ef2bf9.a3m
backbone           structure.pdb
chain length       40 residues
contributing pairs 475 (pairs beyond 22 A cancel exactly and are excluded)
alignment          ~/.boltz/pmf_msa/121b21e7e8ef2bf9.a3m (from the MSA server)

energy                 +6.548 kcal/mol   (+1414.3 nats)
```

**The query sequence is uploaded** to `api.colabfold.com`, a public service, so
the flag is opt-in and never a fallback. Point `--msa-server-url` at your own
MMseqs2 server if the sequence cannot leave your network.

The result is cached under `~/.boltz/pmf_msa` (`--msa-cache` to move it), keyed
by sequence and server, so a rescan or a ΔΔG sweep searches once and every later
run is offline:

```
MSA                reusing cached search: ~/.boltz/pmf_msa/121b21e7e8ef2bf9.a3m
```

Alignments from different sources are not interchangeable at the last decimal —
the same chain scored `+6.548` from the server and `+6.503` from a hand-built
alignment, a 0.7% difference, well inside the ±1.05 kcal/mol calibration RMSE
but not zero. Keep one alignment source across any series you intend to compare.

### An energy for a structure

```bash
boltz-pmf structure.pdb
```

```
structure          structure.pdb
chain length       47 residues
contributing pairs 1033 (pairs beyond 22 A cancel exactly and are excluded)
alignment          none (single-sequence)
                   WARNING: without an alignment the trunk sees
                   no coevolution signal, and this energy (and
                   any ddG below) is significantly less
                   accurate. Pass --msa alignment.a3m.

energy                 +9.247 kcal/mol   (+2347.0 nats)
```

### Without a structure

```bash
boltz-pmf --sequence MKVLATDLA... --mutations A10C F17R --save-folded wt.pdb
boltz-pmf --fasta target.fasta --msa alignment.a3m --mutations A10C
```

Boltz-2 folds the wild type once and every mutant is scored against that fixed
prediction, exactly as against a supplied backbone. `--save-folded` writes the
predicted N/CA/C/O trace so you can see what was scored.

This measures as free: over all 64 ProteinGym Tsuboyama assays the folded
backbone gives 0.6596 / 0.6631 / 0.4423 (all / single / double) against 0.6574 /
0.6609 / 0.4419 supplied.

Sampling is stochastic, so `--fold-seed` pins which structure is predicted, and
`--sampling-steps` trades time for fold quality (default 200, Boltz's own).

### ΔΔG for substitutions

```bash
boltz-pmf structure.pdb --msa alignment.a3m --mutations A10C F17R T38V
```

```
mutation              ddG (kcal/mol)   delta (nats)
---------------------------------------------------
A10C                          -0.983          -28.8
F17R                          -1.226          -90.3
T38V                          -0.996          -32.0
-- pairs --
A10C+F17R                     -1.367         -126.2
A10C+T38V                     -1.143          -69.2
F17R+T38V                     -1.403         -135.1
```

Every single substitution and every pair is reported (`--max-order 1` for singles
only). **Positive ΔΔG means stabilising.**

### Options worth knowing

| flag | why |
| --- | --- |
| `--msa alignment.a3m` | **Strongly recommended.** Without it the model runs single-sequence. |
| `--use-msa-server` | No alignment at hand? Search one on an MMseqs2 server, the way `boltz predict --use_msa_server` does. Uploads the query sequence to `api.colabfold.com` (`--msa-server-url` to point elsewhere), and caches the result under `~/.boltz/pmf_msa` (`--msa-cache`) so later runs are offline. |
| `--no-volume-correction` | Disables the radial term. |
| `--sequence` / `--fasta` | Fold the wild type instead of supplying coordinates. Mutually exclusive with a structure file. |
| `--sampling-steps 200` | Reverse-diffusion steps when folding. Ignored with a structure file. |
| `--fold-seed 0` | Pins the sampled structure; folding is stochastic. |
| `--save-folded wt.pdb` | Write the folded backbone that was scored. |
| `--per-residue out.csv` | Per-residue decomposition of the energy. |
| `--json out.json` | Machine-readable output. |

### As a library

```python
from boltz_pmf import PMFScorer

scorer = PMFScorer("structure.pdb", msa="alignment.a3m", device="cuda:0")
# ... or let the MSA server find one (uploads the sequence):
scorer = PMFScorer("structure.pdb", use_msa_server=True, device="cuda:0")
print(scorer.energy().kcal_per_mol)
for r in scorer.ddg(["G13C", "A20V"]):
    print(r.label, r.ddg_kcal_per_mol)
```

Or with no structure at all:

```python
scorer = PMFScorer(sequence="MKVLATDLA...", msa="alignment.a3m", device="cuda:0")
scorer.folded.backbone_coords        # what Boltz predicted, N/CA/C/O per residue
for r in scorer.ddg(["G13C", "A20V"]):
    print(r.label, r.ddg_kcal_per_mol)
```

Exactly one of `structure` or `sequence` must be given. The model is loaded once,
so a batch of mutants costs one load.
