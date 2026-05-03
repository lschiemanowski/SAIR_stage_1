#!/usr/bin/env python3
"""Construct the full implication graph from explicitly known implications.

Input
-----
The input JSON file must have this shape:

    {
      "implications": [
        {"lhs": "Equation2", "rhs": "Equation3"},
        ...
      ]
    }

Each edge means `lhs implies rhs`. The script computes the transitive closure
of those edges, so if EquationA implies EquationB and EquationB implies
EquationC, the output will include EquationA implies EquationC.

Output
------
The output JSON file has the same top-level shape, but its `implications` array
contains every non-reflexive implication reachable by following one or more
explicit implication edges.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT = Path("explicit_implications.json")
DEFAULT_EQUATIONS = Path("equations_postfix.txt")
DEFAULT_OUTPUT = Path("implication_graph.json")

# Equation identifiers are usually named "Equation<number>". Sorting by the
# numeric suffix makes the output deterministic and easier to compare with
# existing graph files. The fallback below keeps the script usable if a future
# input contains a differently named node.
EQUATION_RE = re.compile(r"^Equation(\d+)$")


def equation_sort_key(name: str) -> tuple[int, int | str]:
    """Sort Equation<number> names numerically, with other names last."""
    match = EQUATION_RE.match(name)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def equation_number(name: str) -> int:
    """Return the numeric suffix from an Equation<number> identifier."""
    match = EQUATION_RE.match(name)
    if not match:
        raise ValueError(f"Invalid equation identifier: {name!r}")
    return int(match.group(1))


def count_equations(path: Path) -> int:
    """Count non-empty equation lines in the equation catalog."""
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def load_implications(path: Path) -> list[dict[str, str]]:
    """Load and validate the explicit implication list from `path`."""
    with path.open() as f:
        data = json.load(f)

    implications = data.get("implications")
    if not isinstance(implications, list):
        raise ValueError(f"{path} does not contain an 'implications' list")

    for index, edge in enumerate(implications):
        if (
            not isinstance(edge, dict)
            or not isinstance(edge.get("lhs"), str)
            or not isinstance(edge.get("rhs"), str)
        ):
            raise ValueError(f"Invalid implication edge at index {index}: {edge!r}")

    return implications


def filter_implications_to_catalog(
    implications: Iterable[dict[str, str]],
    equation_count: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Keep only implications whose endpoints exist in the equation catalog.

    The equation catalog is line-based: line 1 is Equation1, line 2 is
    Equation2, and so on. Any implication involving Equation0 or an equation id
    greater than `equation_count` cannot be encoded by the model and is skipped.
    """
    kept: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    for edge in implications:
        lhs_number = equation_number(edge["lhs"])
        rhs_number = equation_number(edge["rhs"])
        if 1 <= lhs_number <= equation_count and 1 <= rhs_number <= equation_count:
            kept.append(edge)
        else:
            skipped.append(edge)

    return kept, skipped


def build_transitive_closure(
    implications: Iterable[dict[str, str]],
) -> tuple[list[str], list[int]]:
    """Return `(nodes, reachability)` for the transitive implication closure.

    `nodes[i]` is the equation name assigned to index `i`.

    `reachability[i]` is an integer bitset. Bit `j` is set when `nodes[i]`
    implies `nodes[j]`, either explicitly or through a chain of implications.
    Python integers are arbitrary precision, so each row of the reachability
    matrix can be stored compactly as one integer instead of a Python set.
    """
    # Collect every equation that appears on either side of an implication.
    # Isolated equations cannot be inferred from this input format, so they are
    # not represented in the output graph.
    nodes = sorted(
        {node for edge in implications for node in (edge["lhs"], edge["rhs"])},
        key=equation_sort_key,
    )
    node_index = {node: index for index, node in enumerate(nodes)}

    # `reachability[lhs_index]` starts as the direct outgoing edges for that
    # left-hand side. Setting bit `rhs_index` records `lhs -> rhs`.
    reachability = [0] * len(nodes)

    for edge in implications:
        lhs_index = node_index[edge["lhs"]]
        rhs_index = node_index[edge["rhs"]]
        reachability[lhs_index] |= 1 << rhs_index

    # Compute transitive closure with Warshall's algorithm, using bitsets for
    # rows. If `lhs` can reach `intermediate`, then everything reachable from
    # `intermediate` is also reachable from `lhs`.
    for intermediate_index in range(len(nodes)):
        intermediate_bit = 1 << intermediate_index
        intermediate_reachability = reachability[intermediate_index]
        for lhs_index, lhs_reachability in enumerate(reachability):
            if lhs_reachability & intermediate_bit:
                reachability[lhs_index] = lhs_reachability | intermediate_reachability

    return nodes, reachability


def iter_reachable_indices(bits: int) -> Iterable[int]:
    """Yield the set-bit positions in `bits` from lowest to highest."""
    while bits:
        # Extract and remove the lowest set bit. This avoids scanning every
        # possible node index when a row has relatively few reachable nodes.
        lowest_bit = bits & -bits
        yield lowest_bit.bit_length() - 1
        bits ^= lowest_bit


def write_implication_graph(
    nodes: list[str], reachability: list[int], output_path: Path
) -> int:
    """Stream the full implication graph to JSON and return its edge count."""
    edge_count = 0
    first = True

    # The output can be hundreds of megabytes, so avoid materializing the full
    # list of edge dictionaries in memory. Instead, write one JSON object at a
    # time while maintaining valid comma placement.
    with output_path.open("w") as f:
        f.write('{"implications":[')
        for lhs_index, lhs in enumerate(nodes):
            # Cycles in the implication graph make a node reachable from itself.
            # The reference full implication graph omits those reflexive
            # `EquationX -> EquationX` edges, so clear the self bit before
            # writing each row.
            reachable = reachability[lhs_index] & ~(1 << lhs_index)
            for rhs_index in iter_reachable_indices(reachable):
                if first:
                    first = False
                else:
                    f.write(",")
                json.dump(
                    {"rhs": nodes[rhs_index], "lhs": lhs},
                    f,
                    separators=(",", ":"),
                )
                edge_count += 1
        f.write("]}\n")

    return edge_count


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Load explicit_implications.json and construct the full transitive "
            "implication graph."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input explicit implications JSON file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--equations",
        type=Path,
        default=DEFAULT_EQUATIONS,
        help=(
            "Equation catalog used to reject implication endpoints outside "
            f"Equation1..EquationN. Default: {DEFAULT_EQUATIONS}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file for the full implication graph. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    """Load explicit implications, compute closure, and write the graph."""
    args = parse_args()

    implications = load_implications(args.input)
    loaded_implication_count = len(implications)
    equation_count = count_equations(args.equations)
    implications, skipped_implications = filter_implications_to_catalog(
        implications,
        equation_count,
    )

    nodes, reachability = build_transitive_closure(implications)
    edge_count = write_implication_graph(nodes, reachability, args.output)

    print(f"Loaded explicit implications: {loaded_implication_count}")
    print(f"Kept explicit implications: {len(implications)}")
    print(f"Equation catalog entries: {equation_count}")
    print(f"Skipped out-of-catalog explicit implications: {len(skipped_implications)}")
    print(f"Implication nodes: {len(nodes)}")
    print(f"Wrote full implication graph: {args.output}")
    print(f"Full implication edge count: {edge_count}")


if __name__ == "__main__":
    main()
