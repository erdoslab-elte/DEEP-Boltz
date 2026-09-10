"""Converting the potentia into kcal/mol.

It was fitted on the 64 Tsuboyama-2023 stability assays of ProteinGym v1.3
(117,811 substitutions), whose labels are ML-fitted ddG values already in
kcal/mol, and validated leave-one-assay-out so no protein contributes to its own
calibration.

    ddG [kcal/mol] = 0.18518 * (dE / N) - 0.8702
"""

from __future__ import annotations

from dataclasses import dataclass

# Fitted on ProteinGym v1.3 Stability, leave-one-assay-out validated.
SLOPE = 0.18518          # kcal/mol per (nat / residue)
INTERCEPT = -0.8702      # kcal/mol
LOAO_RMSE = 1.052        # kcal/mol
NULL_RMSE = 1.194        # kcal/mol, predicting the dataset mean
PER_ASSAY_SPEARMAN = 0.6571


@dataclass(frozen=True)
class Calibration:
    slope: float = SLOPE
    intercept: float = INTERCEPT

    def ddg(self, delta_nats: float, n_residues: int) -> float:
        """Change in the potential -> ddG in kcal/mol, positive = stabilising."""

        if n_residues <= 0:
            raise ValueError("n_residues must be positive")
        return self.slope * (delta_nats / n_residues) + self.intercept

    def energy(self, nats: float, n_residues: int) -> float:
        """An absolute potential on the same kcal/mol scale.
        """

        if n_residues <= 0:
            raise ValueError("n_residues must be positive")
        return self.slope * (nats / n_residues)
