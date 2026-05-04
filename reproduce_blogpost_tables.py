#!/usr/bin/env python3
"""Write compact Markdown tables for the blogpost oracle comparisons.

This is a thin wrapper around `evaluate_oracle_on_splits.py`. It is useful when
there are several oracle artifacts to compare on the same family of SAIR splits:

    python3 reproduce_blogpost_tables.py \
      --oracle collapsed=relation_fingerprint_runs/<run>/collapsed_k4_summary.json \
      --syntax \
      --splits SAIR_challenge_dataset/data \
      --output-md oracle_tables.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_oracle_on_splits import (
    ClusterOracle,
    SyntaxOracle,
    evaluate_oracle,
    parse_labeled_summary,
    write_markdown,
)
from oracle_utils import expand_split_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate blogpost-style Markdown tables for oracle artifacts."
    )
    parser.add_argument(
        "--oracle",
        action="append",
        default=[],
        type=parse_labeled_summary,
        metavar="LABEL=PATH",
        help="Cluster-oracle summary JSON. May be repeated.",
    )
    parser.add_argument("--syntax", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        type=Path,
        required=True,
        help="JSONL split files or directories containing JSONL files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Global threshold for every oracle. Omit to use saved thresholds.",
    )
    parser.add_argument("--output-md", type=Path, default=Path("blogpost_oracle_tables.md"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--top-error-blocks", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_paths = expand_split_paths(args.splits)

    oracles = [
        ClusterOracle(label, path, threshold=args.threshold)
        for label, path in args.oracle
    ]
    if args.syntax:
        oracles.append(SyntaxOracle(0.5 if args.threshold is None else args.threshold))
    if not oracles:
        raise ValueError("Pass at least one --oracle or --syntax")

    results = [
        evaluate_oracle(oracle, split_paths, top_error_blocks=args.top_error_blocks)
        for oracle in oracles
    ]
    write_markdown(args.output_md, results)
    print(f"wrote {args.output_md}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"results": results}, indent=2))
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
