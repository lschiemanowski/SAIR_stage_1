#!/usr/bin/env python3
"""Cluster equations by learned relation fingerprints.

The relation model predicts logits for ordered pairs of equations. To turn that
pair model into equation clusters, this script gives each equation a fingerprint
of how it relates to a fixed set of anchor equations:

    fp(E) = [logit(E -> A_1), ..., logit(E -> A_m),
             logit(A_1 -> E), ..., logit(A_m -> E)].

K-means on these fingerprints gives equation clusters. After
clustering, the script builds the induced k x k oracle: for every source
cluster and target cluster, it records the empirical implication rate in the
full graph and reports thresholded oracle metrics.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from implication_dataset import (
    DEFAULT_EQUATIONS,
    DEFAULT_GRAPH,
    decode_pair_code,
    load_equation_tensors,
    load_implication_pair_data,
)
from train_relation_model import ImplicationRelationModel, choose_device

try:
    from sklearn.cluster import KMeans
except ImportError:  # pragma: no cover - optional analysis dependency
    KMeans = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional convenience dependency
    tqdm = None


DEFAULT_OUTPUT_ROOT = Path("relation_fingerprint_runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster equations using relation-model fingerprints."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--equations", type=Path, default=DEFAULT_EQUATIONS)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=0,
        help="Use all equations as anchors when 0; otherwise sample this many.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--row-batch-size", type=int, default=32)
    parser.add_argument("--pair-batch-size", type=int, default=65536)
    parser.add_argument("--oracle-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--save-fingerprints", action="store_true")
    parser.add_argument("--no-progress", action="store_true")

    # These are only used for older checkpoints that did not save model_config.
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--equation-dim", type=int, default=None)
    parser.add_argument("--composition-hidden-dim", type=int, default=None)
    parser.add_argument("--relation-hidden-dim", type=int, default=128)
    parser.add_argument("--relation-bottleneck-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def progress(iterable, enabled: bool, desc: str):
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc, leave=False)
    return iterable


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def model_config_from_checkpoint(checkpoint: dict, args: argparse.Namespace) -> dict:
    saved_config = checkpoint.get("model_config") or {}
    saved_args = checkpoint.get("args") or {}

    def value(name: str, fallback):
        if name in saved_config and saved_config[name] is not None:
            return saved_config[name]
        if name in saved_args and saved_args[name] is not None:
            return saved_args[name]
        return fallback

    feature_dim = int(value("feature_dim", args.feature_dim))
    equation_dim = value("equation_dim", args.equation_dim)
    if equation_dim is None:
        equation_dim = feature_dim

    return {
        "feature_dim": feature_dim,
        "equation_dim": int(equation_dim),
        "composition_hidden_dim": value(
            "composition_hidden_dim",
            args.composition_hidden_dim,
        ),
        "relation_hidden_dim": int(
            value("relation_hidden_dim", args.relation_hidden_dim)
        ),
        "relation_bottleneck_dim": int(
            value("relation_bottleneck_dim", args.relation_bottleneck_dim)
        ),
        "dropout": float(value("dropout", args.dropout)),
    }


def build_model(
    checkpoint: dict,
    args: argparse.Namespace,
    equation_data,
    device: torch.device,
) -> ImplicationRelationModel:
    config = model_config_from_checkpoint(checkpoint, args)
    model = ImplicationRelationModel(
        equation_tensors=equation_data.tensors,
        max_variables=equation_data.max_variables,
        **config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


@torch.no_grad()
def encode_all_equations(
    model: ImplicationRelationModel,
    batch_size: int,
    device: torch.device,
    show_progress: bool,
) -> torch.Tensor:
    embeddings: list[torch.Tensor] = []
    ranges = range(0, model.equation_encoder.num_equations, batch_size)

    for start in progress(ranges, enabled=show_progress, desc="encode equations"):
        end = min(start + batch_size, model.equation_encoder.num_equations)
        indices = torch.arange(start, end, dtype=torch.long, device=device)
        embeddings.append(model.encode_equations(indices))

    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def score_embedding_grid(
    model: ImplicationRelationModel,
    left_embeddings: torch.Tensor,
    right_embeddings: torch.Tensor,
    pair_batch_size: int,
) -> torch.Tensor:
    """Score every ordered pair in left_embeddings x right_embeddings."""

    num_left = left_embeddings.size(0)
    num_right = right_embeddings.size(0)
    total_pairs = num_left * num_right
    logits = torch.empty(total_pairs, dtype=left_embeddings.dtype, device=left_embeddings.device)

    for start in range(0, total_pairs, pair_batch_size):
        end = min(start + pair_batch_size, total_pairs)
        flat_positions = torch.arange(start, end, dtype=torch.long, device=left_embeddings.device)
        left_indices = flat_positions // num_right
        right_indices = flat_positions % num_right
        logits[start:end] = model.relation_head(
            left_embeddings[left_indices],
            right_embeddings[right_indices],
        )

    return logits.reshape(num_left, num_right)


@torch.no_grad()
def build_relation_fingerprints(
    model: ImplicationRelationModel,
    equation_embeddings: torch.Tensor,
    anchor_indices: np.ndarray,
    row_batch_size: int,
    pair_batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    num_equations = equation_embeddings.size(0)
    num_anchors = int(anchor_indices.size)
    fingerprints = np.empty((num_equations, 2 * num_anchors), dtype=np.float32)

    anchor_tensor = torch.tensor(anchor_indices, dtype=torch.long, device=equation_embeddings.device)
    anchor_embeddings = equation_embeddings[anchor_tensor]
    ranges = range(0, num_equations, row_batch_size)

    for start in progress(ranges, enabled=show_progress, desc="relation fingerprints"):
        end = min(start + row_batch_size, num_equations)
        row_embeddings = equation_embeddings[start:end]

        forward_logits = score_embedding_grid(
            model=model,
            left_embeddings=row_embeddings,
            right_embeddings=anchor_embeddings,
            pair_batch_size=pair_batch_size,
        )
        backward_logits = score_embedding_grid(
            model=model,
            left_embeddings=anchor_embeddings,
            right_embeddings=row_embeddings,
            pair_batch_size=pair_batch_size,
        ).transpose(0, 1)

        fingerprints[start:end, :num_anchors] = forward_logits.cpu().numpy()
        fingerprints[start:end, num_anchors:] = backward_logits.cpu().numpy()

    return fingerprints


def choose_anchor_indices(
    num_equations: int,
    max_anchors: int,
    seed: int,
) -> np.ndarray:
    if max_anchors <= 0 or max_anchors >= num_equations:
        return np.arange(num_equations, dtype=np.int64)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_equations, size=max_anchors, replace=False))


def compute_block_oracle(
    clusters: np.ndarray,
    pair_data,
    threshold: float,
) -> dict:
    num_clusters = int(clusters.max()) + 1
    cluster_sizes = np.bincount(clusters, minlength=num_clusters).astype(np.int64)
    total_counts = np.outer(cluster_sizes, cluster_sizes).astype(np.int64)
    total_counts[np.arange(num_clusters), np.arange(num_clusters)] -= cluster_sizes

    positive_counts = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    for code in pair_data.positive_pair_codes:
        lhs_index, rhs_index = decode_pair_code(code, pair_data.num_equations)
        if lhs_index == rhs_index:
            continue
        positive_counts[clusters[lhs_index], clusters[rhs_index]] += 1

    rates = np.divide(
        positive_counts,
        total_counts,
        out=np.zeros_like(positive_counts, dtype=np.float64),
        where=total_counts > 0,
    )
    predicted_positive = rates > threshold
    metrics = metrics_from_blocks(
        positive_counts=positive_counts,
        total_counts=total_counts,
        predicted_positive=predicted_positive,
    )

    return {
        "cluster_sizes": cluster_sizes,
        "positive_counts": positive_counts,
        "total_counts": total_counts,
        "rates": rates,
        "predicted_positive": predicted_positive,
        "metrics": metrics,
    }


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def metrics_from_blocks(
    positive_counts: np.ndarray,
    total_counts: np.ndarray,
    predicted_positive: np.ndarray,
) -> dict[str, float | int]:
    negative_counts = total_counts - positive_counts

    true_positive = int(positive_counts[predicted_positive].sum())
    false_positive = int(negative_counts[predicted_positive].sum())
    false_negative = int(positive_counts[~predicted_positive].sum())
    true_negative = int(negative_counts[~predicted_positive].sum())

    total = true_positive + false_positive + false_negative + true_negative
    positive_total = true_positive + false_negative
    negative_total = true_negative + false_positive

    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, positive_total)
    specificity = safe_divide(true_negative, negative_total)

    return {
        "accuracy": safe_divide(true_positive + true_negative, total),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "total_pairs": total,
    }


def make_output_dir(path: Path | None) -> Path:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        return path

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = DEFAULT_OUTPUT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    model_config: dict,
    anchor_indices: np.ndarray,
    clusters: np.ndarray,
    oracle: dict,
) -> None:
    rates = oracle["rates"]
    metrics = oracle["metrics"]
    cluster_sizes = oracle["cluster_sizes"]

    summary = {
        "checkpoint": str(args.checkpoint),
        "equations": str(args.equations),
        "graph": str(args.graph),
        "model_config": model_config,
        "num_clusters": int(args.num_clusters),
        "num_anchors": int(anchor_indices.size),
        "anchor_indices": anchor_indices.tolist(),
        "cluster_of_equation": clusters.astype(int).tolist(),
        "cluster_sizes": cluster_sizes.astype(int).tolist(),
        "oracle_threshold": float(args.oracle_threshold),
        "block_positive_counts": oracle["positive_counts"].astype(int).tolist(),
        "block_total_counts": oracle["total_counts"].astype(int).tolist(),
        "block_rates": rates.tolist(),
        "predicted_positive_blocks": oracle["predicted_positive"].astype(bool).tolist(),
        "metrics": metrics,
    }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (output_dir / "report.md").open("w") as f:
        f.write("# Relation Fingerprint Clustering\n\n")
        f.write(f"- checkpoint: `{args.checkpoint}`\n")
        f.write(f"- clusters: {args.num_clusters}\n")
        f.write(f"- anchors: {anchor_indices.size}\n")
        f.write(f"- threshold: {args.oracle_threshold}\n")
        f.write(f"- balanced accuracy: {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"- accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"- f1: {metrics['f1']:.4f}\n\n")
        f.write("## Cluster Sizes\n\n")
        for index, size in enumerate(cluster_sizes.tolist()):
            f.write(f"- cluster {index}: {size}\n")
        f.write("\n## Block Rates\n\n")
        f.write("| from \\ to | " + " | ".join(str(i) for i in range(args.num_clusters)) + " |\n")
        f.write("| --- | " + " | ".join("---" for _ in range(args.num_clusters)) + " |\n")
        for row_index in range(args.num_clusters):
            values = " | ".join(f"{rates[row_index, col]:.4f}" for col in range(args.num_clusters))
            f.write(f"| {row_index} | {values} |\n")


def main() -> None:
    args = parse_args()
    if args.num_clusters <= 1:
        raise ValueError("--num-clusters must be greater than 1")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive")
    if args.row_batch_size <= 0:
        raise ValueError("--row-batch-size must be positive")
    if args.pair_batch_size <= 0:
        raise ValueError("--pair-batch-size must be positive")
    if KMeans is None:
        raise RuntimeError("scikit-learn is required for KMeans clustering")

    show_progress = not args.no_progress
    device = choose_device(args.device)

    print(f"device: {device}")
    print(f"loading equations from {args.equations}")
    equation_data = load_equation_tensors(args.equations)
    print(f"loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)
    model_config = model_config_from_checkpoint(checkpoint, args)
    model = build_model(checkpoint, args, equation_data, device)
    model.eval()

    print(f"model_config={model_config}")
    print(f"loading implication graph from {args.graph}")
    pair_data = load_implication_pair_data(args.graph, len(equation_data.tensors))

    anchor_indices = choose_anchor_indices(
        num_equations=len(equation_data.tensors),
        max_anchors=args.max_anchors,
        seed=args.seed,
    )
    print(f"anchors={anchor_indices.size}")

    equation_embeddings = encode_all_equations(
        model=model,
        batch_size=args.embedding_batch_size,
        device=device,
        show_progress=show_progress,
    )
    fingerprints = build_relation_fingerprints(
        model=model,
        equation_embeddings=equation_embeddings,
        anchor_indices=anchor_indices,
        row_batch_size=args.row_batch_size,
        pair_batch_size=args.pair_batch_size,
        show_progress=show_progress,
    )

    print("running k-means")
    kmeans = KMeans(
        n_clusters=args.num_clusters,
        n_init=args.n_init,
        random_state=args.seed,
    )
    clusters = kmeans.fit_predict(fingerprints).astype(np.int64)
    oracle = compute_block_oracle(
        clusters=clusters,
        pair_data=pair_data,
        threshold=args.oracle_threshold,
    )

    output_dir = make_output_dir(args.output_dir)
    if args.save_fingerprints:
        np.save(output_dir / "fingerprints.npy", fingerprints)
    write_report(
        output_dir=output_dir,
        args=args,
        model_config=model_config,
        anchor_indices=anchor_indices,
        clusters=clusters,
        oracle=oracle,
    )

    metrics = oracle["metrics"]
    print(
        "oracle: "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )
    print(f"wrote: {output_dir}")


if __name__ == "__main__":
    main()
