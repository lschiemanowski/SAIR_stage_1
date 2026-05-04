#!/usr/bin/env python3
"""Small shared helpers for block-oracle analysis.

The analysis scripts in this repository all use the same simple convention:

    score(source, target) = empirical implication rate of their type block
    prediction           = score > threshold

This file keeps the bookkeeping code in one place so the individual scripts can
focus on the experiment they are meant to explain.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


EQUATION_ID_RE = re.compile(r"^Equation(\d+)$")


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int
    total: int


@dataclass(frozen=True)
class ClusterOracleData:
    """Cluster lookup and block-rate table loaded from an oracle artifact."""

    cluster_of_equation: list[int]
    block_rates: list[list[float]]
    positive_counts: list[list[int]]
    total_counts: list[list[int]]
    threshold: float
    type_labels: list[str]
    raw_payload: dict[str, Any]

    @property
    def num_clusters(self) -> int:
        return len(self.block_rates)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Metrics:
    total = tp + tn + fp + fn
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return Metrics(
        accuracy=safe_divide(tp + tn, total),
        balanced_accuracy=0.5 * (recall + specificity),
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        total=total,
    )


def compute_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Metrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    tp = tn = fp = fn = 0
    for truth, prediction in zip(y_true, y_pred, strict=True):
        if truth and prediction:
            tp += 1
        elif not truth and not prediction:
            tn += 1
        elif not truth and prediction:
            fp += 1
        else:
            fn += 1
    return metrics_from_counts(tp=tp, tn=tn, fp=fp, fn=fn)


def metrics_to_dict(metrics: Metrics) -> dict[str, float | int]:
    return asdict(metrics)


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def answer_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in {0, 1}:
            return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "1"}:
            return True
        if lowered in {"false", "f", "no", "0"}:
            return False
    raise ValueError(f"Cannot interpret answer as boolean: {value!r}")


def equation_id_to_index(value: Any) -> int:
    """Convert a 1-based equation id to a 0-based equation index."""

    if isinstance(value, int):
        return value - 1
    if isinstance(value, str):
        match = EQUATION_ID_RE.fullmatch(value)
        if match:
            return int(match.group(1)) - 1
        if value.isdigit():
            return int(value) - 1
    raise ValueError(f"Cannot interpret equation id: {value!r}")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield record


def expand_split_paths(paths: Sequence[Path]) -> list[Path]:
    """Expand files and directories into a sorted list of JSONL split files."""

    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.jsonl")))
        else:
            expanded.append(path)

    missing = [path for path in expanded if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing split files: {missing_text}")
    return expanded


def _matrix_from_payload(payload: dict[str, Any], top_key: str, oracle_key: str):
    if top_key in payload:
        return payload[top_key]
    oracle = payload.get("oracle")
    if isinstance(oracle, dict) and oracle_key in oracle:
        return oracle[oracle_key]
    return None


def _load_cluster_lookup(payload: dict[str, Any]) -> list[int]:
    if "cluster_of_equation" in payload:
        return [int(value) for value in payload["cluster_of_equation"]]

    clustering = payload.get("clustering")
    if isinstance(clustering, dict) and "equation_cluster_lookup" in clustering:
        return [int(value) for value in clustering["equation_cluster_lookup"]]

    raise ValueError(
        "Oracle summary must contain either 'cluster_of_equation' or "
        "'clustering.equation_cluster_lookup'"
    )


def _load_block_rates(payload: dict[str, Any]) -> list[list[float]]:
    rates = _matrix_from_payload(payload, "block_rates", "rate_table")
    if rates is None:
        raise ValueError(
            "Oracle summary must contain either 'block_rates' or "
            "'oracle.rate_table'"
        )
    return [[float(value) for value in row] for row in rates]


def _load_counts(
    payload: dict[str, Any],
    rates: list[list[float]],
    threshold: float,
) -> tuple[list[list[int]], list[list[int]]]:
    positive = _matrix_from_payload(payload, "block_positive_counts", "positive_counts")
    total = _matrix_from_payload(payload, "block_total_counts", "total_counts")
    negative = _matrix_from_payload(payload, "block_negative_counts", "negative_counts")

    if positive is not None and total is not None:
        return (
            [[int(value) for value in row] for row in positive],
            [[int(value) for value in row] for row in total],
        )

    if positive is not None and negative is not None:
        positive_int = [[int(value) for value in row] for row in positive]
        negative_int = [[int(value) for value in row] for row in negative]
        total_int = [
            [pos + neg for pos, neg in zip(pos_row, neg_row, strict=True)]
            for pos_row, neg_row in zip(positive_int, negative_int, strict=True)
        ]
        return positive_int, total_int

    # Some legacy artifacts only keep rates. The evaluator can still run on
    # external splits, but full-graph collapse metrics are not available.
    size = len(rates)
    zero_counts = [[0 for _ in range(size)] for _ in range(size)]
    return zero_counts, zero_counts


def _load_threshold(payload: dict[str, Any], default: float) -> float:
    if "oracle_threshold" in payload:
        return float(payload["oracle_threshold"])
    oracle = payload.get("oracle")
    if isinstance(oracle, dict) and "threshold" in oracle:
        return float(oracle["threshold"])
    return default


def _load_labels(payload: dict[str, Any], num_clusters: int) -> list[str]:
    if "type_labels" in payload:
        labels = [str(value) for value in payload["type_labels"]]
        if len(labels) == num_clusters:
            return labels

    collapse = payload.get("collapse")
    if isinstance(collapse, dict) and "labels" in collapse:
        labels = [str(value) for value in collapse["labels"]]
        if len(labels) == num_clusters:
            return labels

    return [str(index) for index in range(num_clusters)]


def load_cluster_oracle_summary(path: Path, default_threshold: float = 0.5) -> ClusterOracleData:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")

    cluster_of_equation = _load_cluster_lookup(payload)
    block_rates = _load_block_rates(payload)
    threshold = _load_threshold(payload, default_threshold)
    positive_counts, total_counts = _load_counts(payload, block_rates, threshold)
    labels = _load_labels(payload, len(block_rates))

    return ClusterOracleData(
        cluster_of_equation=cluster_of_equation,
        block_rates=block_rates,
        positive_counts=positive_counts,
        total_counts=total_counts,
        threshold=threshold,
        type_labels=labels,
        raw_payload=payload,
    )


def block_metrics(
    positive_counts: list[list[int]],
    total_counts: list[list[int]],
    predicted_positive: list[list[bool]],
) -> Metrics:
    tp = tn = fp = fn = 0
    for pos_row, total_row, pred_row in zip(
        positive_counts,
        total_counts,
        predicted_positive,
        strict=True,
    ):
        for positive, total, prediction in zip(pos_row, total_row, pred_row, strict=True):
            negative = total - positive
            if prediction:
                tp += positive
                fp += negative
            else:
                fn += positive
                tn += negative
    return metrics_from_counts(tp=tp, tn=tn, fp=fp, fn=fn)


def threshold_rates(rates: list[list[float]], threshold: float) -> list[list[bool]]:
    return [[float(value) > threshold for value in row] for row in rates]
