## What is DEEP-Boltz?

DEEP-Boltz converts the structural information learned by Boltz-2 into an effective, sequence-dependent energy function.

Boltz-2's distogram head predicts, for every pair of residues, a probability distribution over their spatial separation given the protein sequence. Through the Boltzmann relation, a probability distribution along a coordinate can be interpreted as an effective free-energy profile. We therefore treat the predicted distograms as defining an implied potential of mean force over residue-residue distances.

This potential summarizes the interaction preferences learned by Boltz-2 from protein structures, with effects such as solvent and other unobserved degrees of freedom implicitly integrated out.

For a structure with residue-pair distances $d_{ij}$, DEEP-Boltz calculates the score as

$$
E = \sum_{ij} w_{ij}\left[\log p\left(\mathrm{bin}(d_{ij}) \mid s\right)-\log p\left(\mathrm{far} \mid s\right)\right]
$$

Here:

- $p(\mathrm{bin}(d_{ij}) \mid s)$ is the probability assigned by Boltz-2 to the observed distance between residues $i$ and $j$.
- $p(\mathrm{far} \mid s)$ is the probability assigned to the distogram's non-contact, or **far**, bin.
- $w_{ij}$ controls the contribution of each residue pair.
- $s$ denotes the protein sequence.

### Reference-state normalization

The far-distance bin provides a natural reference state. Rather than using the absolute probability of a particular residue-pair distance, DEEP-Boltz measures how strongly Boltz-2 favors that distance relative to the residues being far apart:

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

The wild type's own energy, printed above every ΔΔG table, has no mutated site
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
