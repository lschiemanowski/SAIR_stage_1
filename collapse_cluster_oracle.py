#!/usr/bin/env python3
"""Collapse a learned k x k cluster oracle to a smaller block oracle.

The input is a summary JSON from `cluster_relation_fingerprints.py` or a
compatible cluster-oracle artifact. The output has the same basic shape as the input:
an equation-to-cluster lookup, block counts, block rates, thresholded
predictions, and full-graph block metrics.

Two collapse modes are supported:

1. explicit mapping, for reproducing a known collapse
2. greedy agglomerative search, for asking whether a smaller oracle can keep
   most of the full-graph block performance
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from oracle_utils import (
    block_metrics,
    load_cluster_oracle_summary,
    metrics_to_dict,
    percent,
    threshold_rates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse a learned cluster oracle to fewer equation types."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--target-clusters", type=int, default=4)
    parser.add_argument(
        "--mapping",
        default=None,
        help=(
            "Explicit collapse, for example 'S:1,3,6,7,8;2:2;4:4;5:5'. "
            "When omitted, a greedy merge search is used."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["accuracy", "balanced_accuracy", "f1"],
        default="balanced_accuracy",
        help="Full-graph block metric optimized by the greedy collapse.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Global block-rate threshold. Defaults to the threshold saved in the summary.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def aggregate_counts(
    old_positive: list[list[int]],
    old_total: list[list[int]],
    old_to_new: list[int],
    new_count: int,
) -> tuple[list[list[int]], list[list[int]]]:
    positive = [[0 for _ in range(new_count)] for _ in range(new_count)]
    total = [[0 for _ in range(new_count)] for _ in range(new_count)]

    for old_source, (pos_row, total_row) in enumerate(zip(old_positive, old_total, strict=True)):
        new_source = old_to_new[old_source]
        for old_target, (pos_value, total_value) in enumerate(zip(pos_row, total_row, strict=True)):
            new_target = old_to_new[old_target]
            positive[new_source][new_target] += int(pos_value)
            total[new_source][new_target] += int(total_value)
    return positive, total


def rates_from_counts(
    positive: list[list[int]],
    total: list[list[int]],
) -> list[list[float]]:
    rates: list[list[float]] = []
    for pos_row, total_row in zip(positive, total, strict=True):
        rate_row: list[float] = []
        for pos_value, total_value in zip(pos_row, total_row, strict=True):
            rate_row.append(pos_value / total_value if total_value else 0.0)
        rates.append(rate_row)
    return rates


def mapping_from_groups(groups: list[set[int]], old_count: int) -> list[int]:
    mapping = [-1] * old_count
    for new_index, group in enumerate(groups):
        for old_index in group:
            mapping[old_index] = new_index
    if any(value < 0 for value in mapping):
        raise ValueError("collapse groups do not cover every old cluster")
    return mapping


def evaluate_groups(
    groups: list[set[int]],
    old_positive: list[list[int]],
    old_total: list[list[int]],
    threshold: float,
):
    mapping = mapping_from_groups(groups, len(old_positive))
    positive, total = aggregate_counts(old_positive, old_total, mapping, len(groups))
    rates = rates_from_counts(positive, total)
    predictions = threshold_rates(rates, threshold)
    metrics = block_metrics(positive, total, predictions)
    return mapping, positive, total, rates, predictions, metrics


def parse_explicit_mapping(text: str, old_count: int) -> tuple[list[set[int]], list[str]]:
    groups: list[set[int]] = []
    labels: list[str] = []

    def parse_member(piece: str) -> int:
        value = int(piece.strip())
        # The blogpost labels the saved 8-cluster artifact as 1..8, with raw
        # artifact cluster 0 displayed as cluster 8. Accept that convention for
        # the common 8-cluster collapse while keeping other cluster counts raw.
        if old_count == 8 and value == 8:
            return 0
        return value

    for raw_group in text.split(";"):
        group = raw_group.strip()
        if not group:
            continue
        if ":" not in group:
            raise ValueError(f"mapping group must be LABEL:ids, got {group!r}")
        label, members_text = group.split(":", 1)
        members = {parse_member(piece) for piece in members_text.split(",") if piece.strip()}
        if not members:
            raise ValueError(f"mapping group has no members: {group!r}")
        labels.append(label.strip())
        groups.append(members)

    seen: set[int] = set()
    for group in groups:
        overlap = seen.intersection(group)
        if overlap:
            raise ValueError(f"old clusters occur in more than one group: {sorted(overlap)}")
        seen.update(group)

    expected = set(range(old_count))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"mapping does not cover old clusters; missing={missing}, extra={extra}")

    return groups, labels


def greedy_collapse(
    old_positive: list[list[int]],
    old_total: list[list[int]],
    old_labels: list[str],
    target_count: int,
    threshold: float,
    metric_name: str,
) -> tuple[list[set[int]], list[str], list[dict]]:
    groups = [{index} for index in range(len(old_positive))]
    labels = list(old_labels)
    history: list[dict] = []

    while len(groups) > target_count:
        best_candidate = None
        for left, right in combinations(range(len(groups)), 2):
            candidate_groups = [
                set(group)
                for index, group in enumerate(groups)
                if index not in {left, right}
            ]
            candidate_groups.append(groups[left] | groups[right])
            _, _, _, _, _, metrics = evaluate_groups(
                candidate_groups,
                old_positive,
                old_total,
                threshold,
            )
            score = getattr(metrics, metric_name)
            tie_break = (metrics.accuracy, metrics.balanced_accuracy, -left, -right)
            candidate = (score, tie_break, left, right, candidate_groups, metrics)
            if best_candidate is None or candidate[:2] > best_candidate[:2]:
                best_candidate = candidate

        assert best_candidate is not None
        _, _, left, right, groups, metrics = best_candidate
        merged_label = "+".join(sorted([labels[left], labels[right]]))
        labels = [
            label
            for index, label in enumerate(labels)
            if index not in {left, right}
        ] + [merged_label]
        history.append(
            {
                "merged_left": left,
                "merged_right": right,
                "merged_label": merged_label,
                "remaining_clusters": len(groups),
                "metrics": metrics_to_dict(metrics),
            }
        )

    ordered = sorted(zip(labels, groups, strict=True), key=lambda item: min(item[1]))
    labels = [label for label, _ in ordered]
    groups = [group for _, group in ordered]
    return groups, labels, history


def output_paths(args: argparse.Namespace, final_cluster_count: int) -> tuple[Path, Path]:
    if args.output_json is None:
        suffix = f"collapsed_k{final_cluster_count}_summary.json"
        json_path = args.summary.with_name(suffix)
    else:
        json_path = args.output_json

    if args.output_md is None:
        md_path = json_path.with_suffix(".md")
    else:
        md_path = args.output_md
    return json_path, md_path


def write_markdown(
    path: Path,
    result: dict,
) -> None:
    metrics = result["metrics"]
    lines = [
        "# Collapsed Cluster Oracle",
        "",
        f"- source summary: `{result['source_summary']}`",
        f"- old clusters: {result['collapse']['old_num_clusters']}",
        f"- new clusters: {result['num_clusters']}",
        f"- threshold: {result['oracle_threshold']}",
        f"- accuracy: {percent(metrics['accuracy'])}",
        f"- balanced accuracy: {percent(metrics['balanced_accuracy'])}",
        f"- f1: {percent(metrics['f1'])}",
        "",
        "## Groups",
        "",
        "| New Type | Old Clusters | Size |",
        "| --- | --- | ---: |",
    ]
    for group in result["collapse"]["groups"]:
        old_clusters = ", ".join(str(value) for value in group["old_clusters"])
        lines.append(f"| {group['label']} | {old_clusters} | {group['size']} |")

    lines.extend(
        [
            "",
            "## Block Rates",
            "",
            "| Source \\ Target | "
            + " | ".join(result["type_labels"])
            + " |",
            "| --- | " + " | ".join("---:" for _ in result["type_labels"]) + " |",
        ]
    )
    for label, row in zip(result["type_labels"], result["block_rates"], strict=True):
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{100.0 * value:.1f}%" for value in row)
            + " |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    data = load_cluster_oracle_summary(args.summary)
    old_count = data.num_clusters
    threshold = data.threshold if args.threshold is None else args.threshold

    if data.total_counts == [[0 for _ in range(old_count)] for _ in range(old_count)]:
        raise ValueError(
            "The summary does not contain block counts. Collapse needs counts, "
            "not just rates."
        )

    if args.mapping is not None:
        groups, labels = parse_explicit_mapping(args.mapping, old_count)
        history: list[dict] = []
        method = "explicit"
    else:
        if not (1 < args.target_clusters < old_count):
            raise ValueError("--target-clusters must be between 2 and old cluster count - 1")
        groups, labels, history = greedy_collapse(
            data.positive_counts,
            data.total_counts,
            data.type_labels,
            args.target_clusters,
            threshold,
            args.metric,
        )
        method = "greedy"

    old_to_new, positive, total, rates, predictions, metrics = evaluate_groups(
        groups,
        data.positive_counts,
        data.total_counts,
        threshold,
    )

    cluster_of_equation = [old_to_new[old_cluster] for old_cluster in data.cluster_of_equation]
    cluster_sizes = [
        sum(1 for value in cluster_of_equation if value == new_cluster)
        for new_cluster in range(len(groups))
    ]
    group_payload = [
        {
            "label": label,
            "old_clusters": sorted(group),
            "size": cluster_sizes[index],
        }
        for index, (label, group) in enumerate(zip(labels, groups, strict=True))
    ]

    result = {
        "source_summary": str(args.summary),
        "num_clusters": len(groups),
        "type_labels": labels,
        "cluster_of_equation": cluster_of_equation,
        "cluster_sizes": cluster_sizes,
        "oracle_threshold": threshold,
        "block_positive_counts": positive,
        "block_total_counts": total,
        "block_rates": rates,
        "predicted_positive_blocks": predictions,
        "metrics": metrics_to_dict(metrics),
        "collapse": {
            "method": method,
            "optimized_metric": args.metric if method == "greedy" else None,
            "old_num_clusters": old_count,
            "new_num_clusters": len(groups),
            "old_to_new_cluster": old_to_new,
            "labels": labels,
            "groups": group_payload,
            "greedy_history": history,
        },
    }

    json_path, md_path = output_paths(args, final_cluster_count=len(groups))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2))
    write_markdown(md_path, result)

    print(
        f"collapsed {old_count} -> {len(groups)} clusters: "
        f"accuracy={percent(metrics.accuracy)} "
        f"balanced_accuracy={percent(metrics.balanced_accuracy)} "
        f"f1={percent(metrics.f1)}"
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
