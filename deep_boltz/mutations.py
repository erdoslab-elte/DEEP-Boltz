"""Parsing and applying substitutions given as ``G13C`` or ``13C``.

The wild-type letter is optional. When it is present it is checked against the
scored chain and a mismatch is an error, which is worth having: it catches a
numbering offset immediately rather than silently scoring the wrong residue. When
it is absent it is read from the chain, and every label the program prints is the
resolved full form, so the output always says which residue was replaced.

A *mutant* is one or more substitutions applied together. It may be written as a
list of tokens, ``["G13C", "A20V"]``, or joined into one string with ``:`` or
``+``, as ``"G13C:A20V"`` -- the colon form is what ProteinGym and most ddG
benchmarks use, so a mutant list can be fed in as it comes.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

MUTATION = re.compile(r"^([A-Z])?(\d+)([A-Z])$")
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
SEPARATORS = (":", "+")


@dataclass(frozen=True)
class Mutation:
    wild_type: str | None  # None until `resolve` reads it off the chain
    position: int          # 1-based
    mutant: str

    def __str__(self) -> str:
        return f"{self.wild_type or ''}{self.position}{self.mutant}"


def parse(token: str) -> Mutation:
    match = MUTATION.match(token.strip().upper())
    if not match:
        raise ValueError(
            f"cannot parse mutation {token!r}; expected a form like G13C, or "
            f"13C to let the wild-type residue be read off the chain")
    wt, position, mutant = match.group(1), int(match.group(2)), match.group(3)
    for letter in (wt, mutant):
        if letter is not None and letter not in AMINO_ACIDS:
            raise ValueError(f"{token!r}: {letter!r} is not one of the 20 amino acids")
    if position < 1:
        raise ValueError(f"{token!r}: positions are 1-based")
    return Mutation(wt, position, mutant)


def resolve(mutation: Mutation, sequence: str) -> Mutation:
    """Check the position, and fill in the wild-type residue if it was omitted.

    Returns the mutation with `wild_type` set, so everything downstream can print
    the unambiguous full form regardless of how it was written.
    """

    if mutation.position > len(sequence):
        raise ValueError(
            f"{mutation}: position {mutation.position} is past the end of a "
            f"{len(sequence)}-residue chain")
    actual = sequence[mutation.position - 1]
    if mutation.wild_type is None:
        return Mutation(actual, mutation.position, mutation.mutant)
    if actual != mutation.wild_type:
        raise ValueError(
            f"{mutation}: the structure has {actual} at position "
            f"{mutation.position}, not {mutation.wild_type}. Positions are "
            f"1-based over the scored chain's own sequence. Write "
            f"{mutation.position}{mutation.mutant} to take the wild-type "
            f"residue from the chain instead.")
    return mutation


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


def is_mutant_spec(item) -> bool:
    """Does `item` name a mutant of more than one substitution?

    This is what distinguishes "score exactly these mutants" from "enumerate
    combinations of these substitutions". A caller who spells out a double has
    said which doubles they want.
    """

    if isinstance(item, (list, tuple, set)):
        return True
    return any(sep in str(item) for sep in SEPARATORS)


def parse_mutant(spec, sequence: str) -> tuple[Mutation, ...]:
    """One mutant -> its substitutions, resolved against `sequence`.

    Accepts ``"G13C"``, ``"G13C:A20V"``, ``"G13C+A20V"`` or ``["G13C", "A20V"]``.
    """

    if isinstance(spec, (list, tuple)):
        tokens = [str(t).strip() for t in spec]
    else:
        text = str(spec).strip()
        for sep in SEPARATORS[1:]:
            text = text.replace(sep, SEPARATORS[0])
        tokens = [t for t in text.split(SEPARATORS[0]) if t.strip()]
    if not tokens:
        raise ValueError(f"empty mutant {spec!r}")

    out = tuple(resolve(parse(t), sequence) for t in tokens)
    seen: dict[int, Mutation] = {}
    for m in out:
        if m.position in seen:
            raise ValueError(
                f"mutant {spec!r} has two substitutions at position "
                f"{m.position}: {seen[m.position]} and {m}")
        seen[m.position] = m
    return out


def combinations(mutations: list[Mutation], max_order: int = 2):
    """Every single, then every pair, in the order a reader expects them."""

    out: list[tuple[Mutation, ...]] = []
    for order in range(1, max_order + 1):
        for combo in itertools.combinations(mutations, order):
            positions = [m.position for m in combo]
            if len(set(positions)) == len(positions):
                out.append(combo)
    return out
