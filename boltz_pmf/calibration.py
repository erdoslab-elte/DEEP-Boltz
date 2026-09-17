"""Converting the potential into kcal/mol.

Fitted on the 64 Tsuboyama-2023 stability assays of ProteinGym v1.3 (117,811
substitutions), whose labels are ML-fitted ddG values already in kcal/mol, and
validated leave-one-assay-out so no protein contributes to its own calibration.

    ddG [kcal/mol] = slope * (dE / N) + intercept

**The coefficients depend on two things.**

*The spatial radius.* Truncating the sum at 6 A of a mutated residue leaves far
fewer pairs in it, so `dE` is a different quantity on a different scale; the
all-pair slope would be wrong by roughly a factor of three.

*How many substitutions.* A double mutant perturbs a larger region and is on
average more destabilising, and one line cannot express both. Fitting the two
orders separately drops leave-one-assay-out RMSE at 6 A from **1.024 to 0.906**
kcal/mol, against a null of 1.194 -- a bigger gain than any other change to the
calibration, including the choice of normaliser (dE, dE/N and dE/sqrt(N) agree to
0.003 kcal/mol, so the shipped dE/N is kept).

Every row below comes from the same 117,811 substitutions under the same
protocol, so neither knob can silently invalidate the conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

NULL_RMSE = 1.194        # kcal/mol, predicting the dataset mean
SINGLE_NULL_RMSE = 1.018
DOUBLE_NULL_RMSE = 1.030

# An absolute energy is a sum over every pair, is not a ddG, and is not refitted
# per order: `energy()` keeps the untruncated pooled slope this package shipped
# with, whatever radius ddG is using. `DG_STABILITY.md` establishes that the
# score carries no absolute cross-protein stability signal, so this number orders
# conformations of one chain and is not a stability prediction.
ENERGY_SLOPE = 0.18518


@dataclass(frozen=True)
class Fit:
    """One (slope, intercept) line and the RMSE it achieves out of fold."""

    slope: float
    intercept: float
    loao_rmse: float


# radius (A) -> (order-1 fit, order-2 fit), mean per-assay Spearman.
_FITS: dict[float | None, tuple[Fit, Fit, float]] = {
    4.0:  (Fit(2.13689, -0.39043, 0.8331), Fit(0.60441, -1.59669, 0.9855), 0.6679),
    6.0:  (Fit(0.62551, -0.41609, 0.8356), Fit(0.18467, -1.63024, 0.9890), 0.7197),
    8.0:  (Fit(0.41270, -0.42737, 0.8400), Fit(0.13551, -1.64396, 0.9901), 0.7179),
    10.0: (Fit(0.32229, -0.43736, 0.8486), Fit(0.11557, -1.64768, 0.9901), 0.7090),
    12.0: (Fit(0.27099, -0.44331, 0.8557), Fit(0.10127, -1.65598, 0.9925), 0.6966),
    16.0: (Fit(0.23084, -0.45275, 0.8690), Fit(0.09219, -1.66177, 0.9964), 0.6762),
    22.0: (Fit(0.21822, -0.45577, 0.8760), Fit(0.09112, -1.66176, 0.9967), 0.6615),
    None: (Fit(0.21699, -0.45529, 0.8770), Fit(0.09106, -1.66189, 0.9967), 0.6574),
}

# The production formulation, kept as module constants for readability.
SLOPE = _FITS[6.0][0].slope
INTERCEPT = _FITS[6.0][0].intercept
LOAO_RMSE = 0.9064       # both orders together, at 6 A
PER_ASSAY_SPEARMAN = _FITS[6.0][2]


@dataclass(frozen=True)
class Calibration:
    """The kcal/mol conversion for one radius, one line per mutation order."""

    single: Fit = _FITS[6.0][0]
    double: Fit = _FITS[6.0][1]
    radius: float | None = 6.0
    per_assay_spearman: float = PER_ASSAY_SPEARMAN

    @classmethod
    def for_radius(cls, radius: float | None) -> "Calibration":
        """The calibration fitted for this radius.

        Raises rather than falling back, because a kcal/mol number produced with
        the wrong slope looks exactly like a right one.
        """

        key = None if radius is None else float(radius)
        if key not in _FITS:
            fitted = ", ".join("none" if r is None else f"{r:g}" for r in _FITS)
            raise ValueError(
                f"no kcal/mol calibration was fitted for radius {radius!r}; "
                f"fitted radii are {fitted}. Use one of those, or pass an "
                f"explicit `calibration=Calibration(single=Fit(...), "
                f"double=Fit(...))` if you have fitted your own."
            )
        single, double, rho = _FITS[key]
        return cls(single=single, double=double, radius=key,
                   per_assay_spearman=rho)

    def fit_for(self, n_mutations: int) -> Fit:
        if n_mutations == 1:
            return self.single
        if n_mutations == 2:
            return self.double
        raise ValueError(
            f"no calibration was fitted for {n_mutations} simultaneous "
            f"substitutions, so orders above 2 have no kcal/mol scale."
            f"The raw `delta_nats` is still meaningful for ranking."
        )

    def ddg(self, delta_nats: float, n_residues: int,
            n_mutations: int = 1) -> float:
        """Change in the potential -> ddG in kcal/mol, positive = stabilising."""

        if n_residues <= 0:
            raise ValueError("n_residues must be positive")
        fit = self.fit_for(n_mutations)
        return fit.slope * (delta_nats / n_residues) + fit.intercept


def energy_kcal(nats: float, n_residues: int) -> float:
    """An absolute potential on a kcal/mol scale.

    Not a stability prediction and not a ddG: see `ENERGY_SLOPE`.
    """

    if n_residues <= 0:
        raise ValueError("n_residues must be positive")
    return ENERGY_SLOPE * (nats / n_residues)
