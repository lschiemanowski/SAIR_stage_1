#!/usr/bin/env python3
"""Evaluate a block oracle on SAIR-style JSONL splits.

The script supports two oracle sources:

1. a learned-cluster artifact such as `relation_fingerprint_runs/<run>/summary.json`
2. the hand-written syntax oracle in `equation_type_heuristics.py`

Both are evaluated the same way: each equation is assigned a type, the
source/target type pair gives a block rate, and rates above one global threshold
are predicted as implications.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from equation_type_heuristics import (
    TYPE_ORDER,
    classify_coarse_type,
    implication_rate_from_types,
)
from oracle_utils import (
    answer_to_bool,
    compute_metrics,
    equation_id_to_index,
    expand_split_paths,
    iter_jsonl,
    load_cluster_oracle_summary,
    metrics_to_dict,
    percent,
)


@dataclass(frozen=True)
class Prediction:
    truth: bool
    prediction: bool
    score: float
    source_type: str
    target_type: str


class Oracle(Protocol):
    name: str
    threshold: float
    type_labels: list[str]

    def predict_record(self, record: dict) -> Prediction:
        ...


class ClusterOracle:
    def __init__(self, name: str, summary_path: Path, threshold: float | None) -> None:
        data = load_cluster_oracle_summary(summary_path)
        self.name = name
        self.summary_path = summary_path
        self.cluster_of_equation = data.cluster_of_equation
        self.block_rates = data.block_rates
        self.threshold = data.threshold if threshold is None else threshold
        self.type_labels = data.type_labels

    def predict_record(self, record: dict) -> Prediction:
        source_index = equation_id_to_index(record["eq1_id"])
        target_index = equation_id_to_index(record["eq2_id"])
        source_cluster = self.cluster_of_equation[source_index]
        target_cluster = self.cluster_of_equation[target_index]
        score = self.block_rates[source_cluster][target_cluster]
        return Prediction(
            truth=answer_to_bool(record["answer"]),
            prediction=score > self.threshold,
            score=score,
            source_type=self.type_labels[source_cluster],
            target_type=self.type_labels[target_cluster],
        )


class SyntaxOracle:
    def __init__(self, threshold: float) -> None:
        self.name = "syntax"
        self.threshold = threshold
        self.type_labels = list(TYPE_ORDER)

    def predict_record(self, record: dict) -> Prediction:
        source_type = classify_coarse_type(record["equation1"])
        target_type = classify_coarse_type(record["equation2"])
        score = implication_rate_from_types(source_type, target_type)
        return Prediction(
            truth=answer_to_bool(record["answer"]),
            prediction=score > self.threshold,
            score=score,
            source_type=source_type,
            target_type=target_type,
        )


def parse_labeled_summary(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        if not label:
            raise argparse.ArgumentTypeError("oracle label may not be empty")
        return label, Path(path)
    path = Path(value)
    return path.stem, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate learned-cluster or syntax block oracles on JSONL splits."
    )
    parser.add_argument(
        "--oracle-summary",
        action="append",
        default=[],
        type=parse_labeled_summary,
        metavar="LABEL=PATH",
        help=(
            "Cluster-oracle summary JSON. May be repeated. If LABEL= is omitted, "
            "the file stem is used."
        ),
    )
    parser.add_argument(
        "--syntax",
        action="store_true",
        help="Also evaluate the hand-written S/2/4/5 syntax oracle.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=Path,
        required=True,
        help="JSONL split files or directories containing JSONL split files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Global threshold used for every oracle. If omitted, cluster summaries "
            "use their saved threshold and the syntax oracle uses 0.5."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument(
        "--top-error-blocks",
        type=int,
        default=8,
        help="Number of source/target type-pair error blocks to report per split.",
    )
    return parser.parse_args()


def block_count_template() -> dict[str, int]:
    return {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "total": 0,
        "errors": 0,
    }


def update_block_counts(counts: dict[str, int], prediction: Prediction) -> None:
    counts["total"] += 1
    if prediction.truth and prediction.prediction:
        counts["tp"] += 1
    elif not prediction.truth and not prediction.prediction:
        counts["tn"] += 1
    elif not prediction.truth and prediction.prediction:
        counts["fp"] += 1
        counts["errors"] += 1
    else:
        counts["fn"] += 1
        counts["errors"] += 1


def evaluate_split(path: Path, oracle: Oracle, top_error_blocks: int) -> dict:
    truths: list[bool] = []
    predictions: list[bool] = []
    block_counts: dict[tuple[str, str], dict[str, int]] = {}

    for record in iter_jsonl(path):
        prediction = oracle.predict_record(record)
        truths.append(prediction.truth)
        predictions.append(prediction.prediction)
        key = (prediction.source_type, prediction.target_type)
        update_block_counts(block_counts.setdefault(key, block_count_template()), prediction)

    metrics = compute_metrics(truths, predictions)
    sorted_error_blocks = sorted(
        (
            {
                "source_type": source_type,
                "target_type": target_type,
                **counts,
            }
            for (source_type, target_type), counts in block_counts.items()
            if counts["errors"] > 0
        ),
        key=lambda block: (-block["errors"], -block["total"], block["source_type"], block["target_type"]),
    )

    return {
        "split": path.stem,
        "path": str(path),
        "metrics": metrics_to_dict(metrics),
        "top_error_blocks": sorted_error_blocks[:top_error_blocks],
    }


def evaluate_oracle(oracle: Oracle, split_paths: list[Path], top_error_blocks: int) -> dict:
    split_results = [
        evaluate_split(path, oracle, top_error_blocks=top_error_blocks)
        for path in split_paths
    ]
    return {
        "oracle": oracle.name,
        "threshold": oracle.threshold,
        "type_labels": oracle.type_labels,
        "splits": split_results,
    }


def build_oracles(args: argparse.Namespace) -> list[Oracle]:
    oracles: list[Oracle] = []
    for label, path in args.oracle_summary:
        oracles.append(ClusterOracle(label, path, threshold=args.threshold))

    if args.syntax:
        oracles.append(SyntaxOracle(0.5 if args.threshold is None else args.threshold))

    if not oracles:
        raise ValueError("Pass at least one --oracle-summary or --syntax")
    return oracles


def metric_row(oracle_name: str, split_result: dict) -> list[str]:
    metrics = split_result["metrics"]
    return [
        oracle_name,
        split_result["split"],
        percent(metrics["accuracy"]),
        percent(metrics["balanced_accuracy"]),
        percent(metrics["precision"]),
        percent(metrics["recall"]),
        percent(metrics["f1"]),
        str(metrics["tp"]),
        str(metrics["tn"]),
        str(metrics["fp"]),
        str(metrics["fn"]),
    ]


def write_markdown(path: Path, results: list[dict]) -> None:
    lines = [
        "# Oracle Evaluation",
        "",
        "| Oracle | Split | Accuracy | Balanced Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        for split_result in result["splits"]:
            lines.append("| " + " | ".join(metric_row(result["oracle"], split_result)) + " |")

    lines.append("")
    lines.append("## Largest Error Blocks")
    for result in results:
        lines.append("")
        lines.append(f"### {result['oracle']}")
        for split_result in result["splits"]:
            lines.append("")
            lines.append(f"#### {split_result['split']}")
            blocks = split_result["top_error_blocks"]
            if not blocks:
                lines.append("No errors.")
                continue
            lines.append("")
            lines.append("| Source Type | Target Type | Errors | Total | FP | FN |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
            for block in blocks:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(block["source_type"]),
                            str(block["target_type"]),
                            str(block["errors"]),
                            str(block["total"]),
                            str(block["fp"]),
                            str(block["fn"]),
                        ]
                    )
                    + " |"
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def print_summary(results: list[dict]) -> None:
    header = [
        "oracle",
        "split",
        "acc",
        "bal_acc",
        "precision",
        "recall",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    print("\t".join(header))
    for result in results:
        for split_result in result["splits"]:
            print("\t".join(metric_row(result["oracle"], split_result)))


def main() -> None:
    args = parse_args()
    split_paths = expand_split_paths(args.splits)
    oracles = build_oracles(args)
    results = [
        evaluate_oracle(oracle, split_paths, top_error_blocks=args.top_error_blocks)
        for oracle in oracles
    ]

    print_summary(results)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"results": results}, indent=2))
        print(f"wrote {args.output_json}")

    if args.output_md is not None:
        write_markdown(args.output_md, results)
        print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
