#!/usr/bin/env python3
"""Plot the same 2D equation projection with different cluster labels.

The blogpost needs three closely comparable pictures:

1. equations colored by the learned 8 relation-fingerprint clusters
2. the same points colored by the 4-type coarsening of those clusters
3. the same points colored by the hand-written syntax classifier

The key point is that all three figures use identical coordinates. Only the
colors change, so the reader can see what is preserved and what is lost when
going from learned clusters to a coarsening and then to syntax rules.

The default input convention is the one produced by this repository:

    relation_fingerprint_runs/<run>/summary.json
    relation_fingerprint_runs/<run>/fingerprints.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collapse_cluster_oracle import parse_explicit_mapping
from equation_type_heuristics import TYPE_ORDER, classify_coarse_type
from oracle_utils import load_cluster_oracle_summary


DEFAULT_EQUATIONS = Path("equations.txt")
DEFAULT_COLLAPSE = "S:1,3,6,7,8;2:2;4:4;5:5"

TYPE_COLORS = {
    "S": "#1f77b4",
    "2": "#ff7f0e",
    "4": "#2ca02c",
    "5": "#d62728",
}


def display_cluster_label(cluster_id: int) -> str:
    """Display old artifact cluster 0 as cluster 8.

    The saved cluster artifacts are zero-based. The blogpost uses one-based
    labels, with raw cluster 0 renamed to 8 and clusters 1..7 left as they were.
    """

    return "8" if cluster_id == 0 else str(cluster_id)


def display_cluster_order(cluster_ids: np.ndarray) -> list[str]:
    raw_order = sorted(int(value) for value in np.unique(cluster_ids))
    return [display_cluster_label(value) for value in raw_order if value != 0] + [
        display_cluster_label(value) for value in raw_order if value == 0
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw PCA scatter plots for learned, coarsened, and syntax clusters."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Cluster summary JSON written by cluster_relation_fingerprints.py.",
    )
    parser.add_argument(
        "--fingerprints",
        type=Path,
        default=None,
        help=(
            "fingerprints.npy written by cluster_relation_fingerprints.py "
            "--save-fingerprints. Defaults to fingerprints.npy next to --summary."
        ),
    )
    parser.add_argument(
        "--projection-cache",
        type=Path,
        default=None,
        help=(
            "Optional local torch cache containing equation_coords. When omitted, "
            "coordinates are computed by PCA from --fingerprints."
        ),
    )
    parser.add_argument("--equations", type=Path, default=DEFAULT_EQUATIONS)
    parser.add_argument(
        "--collapse-mapping",
        default=DEFAULT_COLLAPSE,
        help="Mapping from learned cluster ids to coarsened labels.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("blogpost_cluster_pca"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["svg", "png", "pdf"],
        default=["svg", "png"],
    )
    parser.add_argument("--point-size", type=float, default=12.0)
    parser.add_argument("--alpha", type=float, default=0.62)
    parser.add_argument(
        "--markdown-snippet",
        type=Path,
        default=None,
        help="Optional Markdown snippet listing the generated figures.",
    )
    return parser.parse_args()


def pca_from_fingerprints(path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    fingerprints = np.load(path)
    if fingerprints.ndim != 2:
        raise ValueError(f"{path} must contain a 2D fingerprint array")

    features = fingerprints.astype(np.float64, copy=False)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    scaled = (features - mean) / std

    _, singular_values, right_vectors = np.linalg.svd(scaled, full_matrices=False)
    coords = scaled @ right_vectors[:2].T
    variances = singular_values**2
    explained = variances[:2] / variances.sum()
    return coords, (float(explained[0]), float(explained[1]))


def load_projection(path: Path) -> tuple[np.ndarray, tuple[float | None, float | None]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "equation_coords" not in payload:
        raise ValueError(f"{path} does not contain equation_coords")

    coords = np.asarray(payload["equation_coords"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{path}: equation_coords must have shape [n, 2]")

    variance = payload.get("equation_explained_variance")
    if variance is None:
        return coords, (None, None)
    variance_array = np.asarray(variance, dtype=np.float64)
    return coords, (float(variance_array[0]), float(variance_array[1]))


def load_learned_clusters(path: Path) -> np.ndarray:
    summary = load_cluster_oracle_summary(path)
    return np.asarray(summary.cluster_of_equation, dtype=np.int64)


def load_equations(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.strip() for line in handle if line.strip()]


def coarsen_labels(
    learned_clusters: np.ndarray,
    mapping_text: str,
) -> tuple[np.ndarray, list[str]]:
    old_count = int(learned_clusters.max()) + 1
    groups, labels = parse_explicit_mapping(mapping_text, old_count)
    old_to_new_label: dict[int, str] = {}
    for label, group in zip(labels, groups, strict=True):
        for old_cluster in group:
            old_to_new_label[old_cluster] = label
    return np.asarray(
        [old_to_new_label[int(cluster)] for cluster in learned_clusters],
        dtype=object,
    ), labels


def syntax_labels(equations: list[str]) -> np.ndarray:
    return np.asarray([classify_coarse_type(equation) for equation in equations], dtype=object)


def axis_label(axis_name: str, variance: float | None) -> str:
    if variance is None:
        return axis_name
    return f"{axis_name} ({100.0 * variance:.1f}% variance)"


def output_path(prefix: Path, suffix: str, extension: str) -> Path:
    return prefix.with_name(f"{prefix.name}_{suffix}.{extension}")


def save_figure(fig, prefix: Path, suffix: str, formats: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for extension in formats:
        path = output_path(prefix, suffix, extension)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_one(
    coords: np.ndarray,
    labels: np.ndarray,
    label_order: list[str],
    color_map: dict[str, str],
    title: str,
    variance: tuple[float | None, float | None],
    point_size: float,
    alpha: float,
):
    fig, ax = plt.subplots(figsize=(8.0, 6.3))
    draw_scatter(
        ax,
        coords,
        labels,
        label_order,
        color_map,
        point_size,
        alpha,
    )
    finish_axes(ax, title, variance)
    fig.tight_layout()
    return fig


def draw_scatter(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    label_order: list[str],
    color_map: dict[str, str],
    point_size: float,
    alpha: float,
) -> None:
    for label in label_order:
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            color=color_map[label],
            label=f"{label} ({int(mask.sum())})",
            linewidths=0,
        )


def finish_axes(ax, title: str, variance: tuple[float | None, float | None]) -> None:
    ax.set_title(title, pad=10)
    ax.set_xlabel(axis_label("PCA 1", variance[0]))
    ax.set_ylabel(axis_label("PCA 2", variance[1]))
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        title="Cluster",
    )


def learned_color_map(cluster_ids: np.ndarray) -> dict[str, str]:
    cmap = plt.get_cmap("tab10")
    return {
        display_cluster_label(int(cluster)): cmap(int(cluster) % 10)
        for cluster in sorted(np.unique(cluster_ids))
    }


def plot_combined(
    coords: np.ndarray,
    learned: np.ndarray,
    coarsened: np.ndarray,
    syntax: np.ndarray,
    learned_order: list[str],
    coarse_order: list[str],
    learned_colors: dict[str, str],
    coarse_colors: dict[str, str],
    variance: tuple[float | None, float | None],
    point_size: float,
    alpha: float,
):
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.4), sharex=True, sharey=True)
    panels = [
        ("Learned 8 clusters", learned, learned_order, learned_colors),
        ("4-type coarsening", coarsened, coarse_order, coarse_colors),
        ("Syntax-rule types", syntax, list(TYPE_ORDER), coarse_colors),
    ]
    for ax, (title, labels, order, colors) in zip(axes, panels, strict=True):
        draw_scatter(ax, coords, labels, order, colors, point_size, alpha)
        ax.set_title(title, pad=10)
        ax.set_xlabel(axis_label("PCA 1", variance[0]))
        ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)
        ax.legend(frameon=False, fontsize=8, loc="upper right", title="Cluster")
    axes[0].set_ylabel(axis_label("PCA 2", variance[1]))
    fig.tight_layout()
    return fig


def write_markdown_snippet(path: Path, written: dict[str, list[Path]]) -> None:
    lines = ["# Cluster PCA Figures", ""]
    for title, paths in written.items():
        svg_paths = [item for item in paths if item.suffix == ".svg"]
        display_path = svg_paths[0] if svg_paths else paths[0]
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"![{title}]({display_path})")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.projection_cache is not None:
        coords, variance = load_projection(args.projection_cache)
    else:
        fingerprints = args.fingerprints
        if fingerprints is None:
            fingerprints = args.summary.parent / "fingerprints.npy"
        coords, variance = pca_from_fingerprints(fingerprints)

    learned_int = load_learned_clusters(args.summary)
    learned = np.asarray(
        [display_cluster_label(int(value)) for value in learned_int],
        dtype=object,
    )
    coarsened, coarse_order = coarsen_labels(learned_int, args.collapse_mapping)
    syntax = syntax_labels(load_equations(args.equations))

    if not (len(coords) == len(learned) == len(syntax)):
        raise ValueError(
            "coordinate, learned-cluster, and syntax-label lengths differ: "
            f"{len(coords)}, {len(learned)}, {len(syntax)}"
        )

    learned_order = display_cluster_order(learned_int)
    learned_colors = learned_color_map(learned_int)
    coarse_colors = {label: TYPE_COLORS[label] for label in TYPE_ORDER}

    written: dict[str, list[Path]] = {}
    fig = plot_one(
        coords,
        learned,
        learned_order,
        learned_colors,
        "Learned 8 relation-fingerprint clusters",
        variance,
        args.point_size,
        args.alpha,
    )
    written["Learned 8 Clusters"] = save_figure(fig, args.output_prefix, "learned8", args.formats)

    fig = plot_one(
        coords,
        coarsened,
        coarse_order,
        coarse_colors,
        "Same projection after 4-type coarsening",
        variance,
        args.point_size,
        args.alpha,
    )
    written["4-Type Coarsening"] = save_figure(fig, args.output_prefix, "coarsened4", args.formats)

    fig = plot_one(
        coords,
        syntax,
        list(TYPE_ORDER),
        coarse_colors,
        "Same projection colored by syntax rules",
        variance,
        args.point_size,
        args.alpha,
    )
    written["Syntax Types"] = save_figure(fig, args.output_prefix, "syntax4", args.formats)

    fig = plot_combined(
        coords,
        learned,
        coarsened,
        syntax,
        learned_order,
        coarse_order,
        learned_colors,
        coarse_colors,
        variance,
        args.point_size,
        args.alpha,
    )
    written["Three Views"] = save_figure(fig, args.output_prefix, "three_views", args.formats)

    if args.markdown_snippet is not None:
        write_markdown_snippet(args.markdown_snippet, written)
        print(f"wrote {args.markdown_snippet}")

    for title, paths in written.items():
        print(f"{title}:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
