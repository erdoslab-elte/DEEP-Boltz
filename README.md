# Boltz-PMF

A zero-shot statistical potential from the **Boltz-2** structure prediction method.

Boltz-PMF gives accurate estimates for protein stability using free energy calculation.

## Basic usage

An alignment matters a lot here, and the easiest way to get one is to let
Boltz-PMF search for it — no alignment file needed:

```bash
boltz-pmf structure.pdb --use-msa-server --mutations G13C A20V T45S
```

The wild-type letter
is optional — `13C` and `G13C` mean the same thing, and when you give it, it is
checked against the scored chain.

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

On ProteinGym Stability, ΔΔG reaches **0.7197** mean per-assay Spearman over all
117,811 substitutions. The two backbone routes are equivalent: folding the wild
type scores **0.6596** against **0.6574** supplied, measured on the untruncated
sum.

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

### Locality: ΔΔG is summed near the mutation

A substitution is a local perturbation, but the sum above runs over every pair in
the chain. Restricting a ΔΔG to the pairs within **6 Å** of a mutated residue significantly improves accuracy.

| ΔΔG summed over | mean per-assay Spearman |
| --- | --- |
| pairs within 4 Å of the mutation | 0.6679 |
| **pairs within 6 Å** (default) | **0.7197** |
| pairs within 8 Å | 0.7179 |
| pairs within 12 Å | 0.6966 |
| every pair | 0.6574 |

Distances come from the wild type and stay frozen across mutants.
`--radius none` restores the untruncated sum.

The wild type's own **energy**, printed above every ΔΔG table, has no mutated site
to measure from, so it is always the untruncated sum and `--radius` does not
affect it.

### Units: the kcal/mol calibration

ΔΔG is reported in kcal/mol through a linear calibration fitted on the same 64
ProteinGym Tsuboyama assays (117,811 substitutions), leave-one-assay-out so no
protein contributes to its own calibration:

    ddG [kcal/mol] = slope * (dE / N) + intercept

The coefficients depend on the radius, and on how many substitutions are being scored at once. 
A double mutant perturbs a larger
region and is on average more destabilising, and one line cannot express both:

| at 6 Å | slope | intercept | LOAO RMSE | null |
| --- | --- | --- | --- | --- |
| one substitution | 0.62551 | −0.41609 | **0.836** | 1.018 |
| two substitutions | 0.18467 | −1.63024 | **0.989** | 1.030 |
| both, one shared line | — | — | 1.024 | 1.194 |

Orders above two have no fitted scale and raise rather than return a
plausible-looking number; `delta_nats` still ranks them.

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
boltz-pmf structure.pdb --use-msa-server --mutations G13C
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
ddG radius         6 A around each mutated residue
...
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
alignment, a 0.7% difference, well inside the ±0.84 kcal/mol calibration RMSE
but not zero. Keep one alignment source across any series you intend to compare.

### The energy of the scored chain

Every run prints the wild type's own energy above the ΔΔG table. It is the
untruncated sum over every pair, it is unaffected by `--radius`, and it is **not
a stability prediction** — the score carries no absolute signal that transfers
between different proteins, so compare it only between conformations of one
chain.

```bash
boltz-pmf structure.pdb --mutations G13C
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
`--sampling-steps` trades time for fold quality (default 200, Boltz's own). Keep
the default if you lower nothing else: the ΔΔG radius is measured on the backbone
being scored, so an under-sampled fold is not merely less accurate — it is less
compact, and fewer pairs fall inside 6 Å. At 200 steps the folded backbone
selects the same neighbourhoods as the deposited one (340/178/340 pairs against
340/220/340 for the three substitutions below); at 10 steps it collapses to
91.

### ΔΔG for substitutions

```bash
boltz-pmf structure.pdb --msa alignment.a3m --mutations 10C F17R 38V
```

```
ddG radius         6 A around each mutated residue

mutation              ddG (kcal/mol)   delta (nats)   pairs
-----------------------------------------------------------
A10C                          -0.592          -13.2     340
F17R                          -0.904          -36.7     220
T38V                          -0.725          -23.2     340
-- pairs --
A10C+F17R                     -1.930          -76.4     486
A10C+T38V                     -1.829          -50.7     585
F17R+T38V                     -2.005          -95.5     520
```

Every single substitution and every pair is reported (`--max-order 1` for singles
only). **Positive ΔΔG means stabilising.** `pairs` is how many weighted residue
pairs fell inside the radius; a double mutant takes the union of both
neighbourhoods, which is why its count exceeds either single's. The labels are
the resolved full form, so `10C` is echoed back as `A10C` and the output always
says which residue was replaced.

### Options worth knowing

| flag | why |
| --- | --- |
| `--mutations G13C 20V` | **Required.** `G13C` or `20V`; the wild-type letter is optional and is checked when present. |
| `--msa alignment.a3m` | **Strongly recommended.** Without it the model runs single-sequence. |
| `--use-msa-server` | No alignment at hand? Search one on an MMseqs2 server, the way `boltz predict --use_msa_server` does. Uploads the query sequence to `api.colabfold.com` (`--msa-server-url` to point elsewhere), and caches the result under `~/.boltz/pmf_msa` (`--msa-cache`) so later runs are offline. |
| `--radius 6` | Å around each mutated residue that ΔΔG is summed over. The default is the selected value; `--radius none` gives the untruncated sum. Only radii with a fitted kcal/mol calibration are accepted (4, 6, 8, 10, 12, 16, 22, none). |
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

The radius lives on `PotentialConfig`, and the kcal/mol calibration follows it:

```python
from boltz_pmf import PMFScorer, PotentialConfig

scorer = PMFScorer("structure.pdb", msa="alignment.a3m",
                   config=PotentialConfig(radius=None))   # untruncated
for r in scorer.ddg(["G13C"]):
    print(r.label, r.ddg_kcal_per_mol, r.n_pairs)
```

Each radius has its own slope, because truncating the sum changes the scale of
`delta_nats` by more than a factor of two. `Calibration.for_radius` raises on a
radius nothing was fitted for rather than returning a plausible-looking wrong
number; pass `calibration=` explicitly if you have fitted your own.
