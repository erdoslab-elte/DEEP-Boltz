"""Parsing and applying substitutions given as ``G13C``."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

MUTATION = re.compile(r"^([A-Z])(\d+)([A-Z])$")
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Mutation:
    wild_type: str
    position: int  # 1-based
    mutant: str

    def __str__(self) -> str:
        return f"{self.wild_type}{self.position}{self.mutant}"


def parse(token: str) -> Mutation:
    match = MUTATION.match(token.strip().upper())
    if not match:
        raise ValueError(
            f"cannot parse mutation {token!r}; expected a form like G13C")
    wt, position, mut = match.group(1), int(match.group(2)), match.group(3)
    for letter in (wt, mut):
        if letter not in AMINO_ACIDS:
            raise ValueError(f"{token!r}: {letter!r} is not one of the 20 amino acids")
    if position < 1:
        raise ValueError(f"{token!r}: positions are 1-based")
    return Mutation(wt, position, mut)


def validate(mutation: Mutation, sequence: str) -> None:
    if mutation.position > len(sequence):
        raise ValueError(
            f"{mutation}: position {mutation.position} is past the end of a "
            f"{len(sequence)}-residue chain")
    actual = sequence[mutation.position - 1]
    if actual != mutation.wild_type:
        raise ValueError(
            f"{mutation}: the structure has {actual} at position "
            f"{mutation.position}, not {mutation.wild_type}. Positions are "
            "1-based over the scored chain's own sequence.")


def apply(sequence: str, mutations: tuple[Mutation, ...]) -> str:
    seen: dict[int, Mutation] = {}
    for mutation in mutations:
        if mutation.position in seen:
            raise ValueError(
                f"two substitutions at position {mutation.position}: "
                f"{seen[mutation.position]} and {mutation}")
        seen[mutation.position] = mutation
    letters = list(sequence)
    for mutation in mutations:
        letters[mutation.position - 1] = mutation.mutant
    return "".join(letters)


def combinations(mutations: list[Mutation], max_order: int = 2):
    """Every single, then every pair, in the order a reader expects them."""

    out: list[tuple[Mutation, ...]] = []
    for order in range(1, max_order + 1):
        for combo in itertools.combinations(mutations, order):
            positions = [m.position for m in combo]
            if len(set(positions)) == len(positions):
                out.append(combo)
    return out
