#!/usr/bin/env python3
"""Data loading and negative sampling for implication training.

The central assumption here is the one stated for `implication_graph.json`:

    every ordered pair present in the graph is an implication;
    every ordered pair absent from the graph is a non-implication.

The dataset therefore samples:

    positive examples: edges from implication_graph.json
    negative examples: random ordered equation pairs not in that edge set

Equation ids in the graph are strings such as "Equation37". The equation text
files are line-based, so line 1 corresponds to Equation1, line 2 to Equation2,
and so on.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from torch.utils.data import Dataset

from postfix_equation_tensor import EquationTensor, compile_postfix_equation


DEFAULT_EQUATIONS = Path("equations_postfix.txt")
DEFAULT_GRAPH = Path("implication_graph.json")

EQUATION_ID_RE = re.compile(r"^Equation(\d+)$")


@dataclass(frozen=True)
class EquationTensorData:
    """Compiled equation tensors loaded from `equations_postfix.txt`."""

    tensors: list[EquationTensor]
    max_variables: int


@dataclass(frozen=True)
class ImplicationPairData:
    """Positive implication pairs and membership set.

    `positive_pair_codes` stores pairs compactly as integers:

        code = lhs_index * num_equations + rhs_index

    where `lhs_index` and `rhs_index` are zero-based equation indices.
    The set contains the same codes and is used to test whether a randomly
    sampled ordered pair is absent from the implication graph.
    """

    num_equations: int
    positive_pair_codes: array
    positive_pair_set: set[int]
    skipped_missing_equation_edges: int
    skipped_duplicate_edges: int


def parse_equation_id(name: str) -> int:
    """Convert "Equation37" to integer id 37."""
    match = EQUATION_ID_RE.fullmatch(name)
    if not match:
        raise ValueError(f"Invalid equation id: {name!r}")
    return int(match.group(1))


def pair_code(lhs_index: int, rhs_index: int, num_equations: int) -> int:
    """Encode a zero-based ordered pair as one integer."""
    return lhs_index * num_equations + rhs_index


def decode_pair_code(code: int, num_equations: int) -> tuple[int, int]:
    """Decode an integer pair code into zero-based `(lhs_index, rhs_index)`."""
    return divmod(code, num_equations)


def load_equation_tensors(path: Path = DEFAULT_EQUATIONS) -> EquationTensorData:
    """Load postfix equations and compile them into `EquationTensor` objects."""

    tensors: list[EquationTensor] = []
    max_variables = 0

    with path.open() as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                tensor = compile_postfix_equation(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc

            tensors.append(tensor)
            max_variables = max(max_variables, len(tensor.variable_to_id))

    if not tensors:
        raise ValueError(f"No equations found in {path}")

    return EquationTensorData(tensors=tensors, max_variables=max_variables)


def iter_implication_edges(graph_path: Path) -> Iterable[dict[str, str]]:
    """Yield implication edges from the graph JSON file.

    This intentionally uses the standard library JSON parser for simplicity.
    For very large graphs, a streaming JSON parser would reduce peak memory
    usage, but would add an extra dependency or more parsing code.
    """

    with graph_path.open() as f:
        data = json.load(f)

    implications = data.get("implications")
    if not isinstance(implications, list):
        raise ValueError(f"{graph_path} does not contain an 'implications' list")

    yield from implications


def load_implication_pair_data(
    graph_path: Path = DEFAULT_GRAPH,
    num_equations: int | None = None,
) -> ImplicationPairData:
    """Load positive implication edges that have equation text available.

    The graph currently contains a few equation ids that do not have matching
    lines in `equations_postfix.txt`. Those edges cannot be used by the encoder,
    so this loader skips them and reports the count.
    """

    if num_equations is None:
        num_equations = sum(1 for _ in DEFAULT_EQUATIONS.open())

    positive_pair_codes = array("Q")
    positive_pair_set: set[int] = set()
    skipped_missing = 0
    skipped_duplicate = 0

    for edge in iter_implication_edges(graph_path):
        lhs_id = parse_equation_id(edge["lhs"])
        rhs_id = parse_equation_id(edge["rhs"])

        if not (1 <= lhs_id <= num_equations and 1 <= rhs_id <= num_equations):
            skipped_missing += 1
            continue

        code = pair_code(lhs_id - 1, rhs_id - 1, num_equations)
        if code in positive_pair_set:
            skipped_duplicate += 1
            continue

        positive_pair_set.add(code)
        positive_pair_codes.append(code)

    if not positive_pair_codes:
        raise ValueError(f"No usable implication edges found in {graph_path}")

    return ImplicationPairData(
        num_equations=num_equations,
        positive_pair_codes=positive_pair_codes,
        positive_pair_set=positive_pair_set,
        skipped_missing_equation_edges=skipped_missing,
        skipped_duplicate_edges=skipped_duplicate,
    )


class ImplicationPairDataset(Dataset):
    """Sample ordered implication/non-implication training pairs.

    The dataset length is `samples_per_epoch`. It does not materialize all
    negative examples, because there can be many absent ordered pairs. Instead,
    each negative item is sampled on demand by rejection sampling until it finds
    a pair not present in `positive_pair_set`.
    """

    def __init__(
        self,
        pair_data: ImplicationPairData,
        samples_per_epoch: int,
        negative_ratio: int = 1,
        seed: int = 0,
    ) -> None:
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if negative_ratio < 0:
            raise ValueError("negative_ratio must be nonnegative")
        if (
            negative_ratio > 0
            and len(pair_data.positive_pair_set)
            >= pair_data.num_equations * pair_data.num_equations
        ):
            raise ValueError(
                "negative sampling requested, but every ordered pair is positive"
            )

        self.pair_data = pair_data
        self.samples_per_epoch = samples_per_epoch
        self.negative_ratio = negative_ratio
        self.seed = seed

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[int, int, float]:
        rng = random.Random(self.seed + index)

        # With negative_ratio = 1, even slots are positive and odd slots are
        # negative. With negative_ratio = 3, one out of every four slots is
        # positive and the other three are negative.
        cycle_length = self.negative_ratio + 1
        use_positive = self.negative_ratio == 0 or index % cycle_length == 0

        if use_positive:
            return (*self._sample_positive(rng), 1.0)

        return (*self._sample_negative(rng), 0.0)

    def _sample_positive(self, rng: random.Random) -> tuple[int, int]:
        code = self.pair_data.positive_pair_codes[
            rng.randrange(len(self.pair_data.positive_pair_codes))
        ]
        return decode_pair_code(code, self.pair_data.num_equations)

    def _sample_negative(self, rng: random.Random) -> tuple[int, int]:
        num_equations = self.pair_data.num_equations
        total_ordered_pairs = num_equations * num_equations

        while True:
            code = rng.randrange(total_ordered_pairs)
            if code not in self.pair_data.positive_pair_set:
                return decode_pair_code(code, num_equations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load implication graph data and print sampling statistics."
    )
    parser.add_argument("--equations", type=Path, default=DEFAULT_EQUATIONS)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--samples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    equation_data = load_equation_tensors(args.equations)
    pair_data = load_implication_pair_data(args.graph, len(equation_data.tensors))
    dataset = ImplicationPairDataset(pair_data, samples_per_epoch=args.samples)

    print(f"equations: {len(equation_data.tensors)}")
    print(f"max local variables: {equation_data.max_variables}")
    print(f"positive implication pairs: {len(pair_data.positive_pair_codes)}")
    print(f"skipped graph-only edges: {pair_data.skipped_missing_equation_edges}")
    print(f"skipped duplicate edges: {pair_data.skipped_duplicate_edges}")
    print("sampled pairs:")
    for index in range(len(dataset)):
        lhs_index, rhs_index, label = dataset[index]
        print(
            f"  Equation{lhs_index + 1} -> Equation{rhs_index + 1}: "
            f"label={int(label)}"
        )


if __name__ == "__main__":
    main()
