#!/usr/bin/env python3
"""Train the equation encoder plus relation head on the implication graph.

This is an intentionally straightforward end-to-end trainer:

    postfix equation -> equation tensor -> equation embedding
    ordered pair of equation embeddings -> implication logit

Positive examples are sampled from `implication_graph.json`. Negative examples
are sampled as ordered equation pairs absent from that graph, using the
assumption that the implication graph is complete.

A compiled syntax encoder produces one embedding per equation, and an ordered
MLP with a small bottleneck scores pairs of equation embeddings. The
implementation is still pedagogical: every stage is a named module, and the data
flow stays close to the mathematical description.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from compiled_equation_encoder import CompiledEquationEncoder
from implication_dataset import (
    DEFAULT_EQUATIONS,
    DEFAULT_GRAPH,
    ImplicationPairDataset,
    load_equation_tensors,
    load_implication_pair_data,
)
from postfix_equation_tensor import EquationTensor
from relation_head import RelationHead

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional convenience dependency
    tqdm = None


DEFAULT_CHECKPOINT = Path("relation_model.pt")


class ImplicationRelationModel(nn.Module):
    """End-to-end implication model.

    The equation encoder is symmetric in the two sides of one equation: it
    aggregates the two root states with `[hL + hR ; |hL - hR|] -> MLP`. The
    relation head is directional in the two equations, because implication is
    directional.
    """

    def __init__(
        self,
        equation_tensors: list[EquationTensor],
        max_variables: int,
        feature_dim: int,
        equation_dim: int,
        relation_hidden_dim: int | None = None,
        relation_bottleneck_dim: int | None = None,
        composition_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.equation_encoder = CompiledEquationEncoder(
            equation_tensors=equation_tensors,
            max_variables=max_variables,
            feature_dim=feature_dim,
            equation_dim=equation_dim,
            composition_hidden_dim=composition_hidden_dim,
        )
        self.relation_head = RelationHead(
            equation_dim=equation_dim,
            hidden_dim=relation_hidden_dim,
            bottleneck_dim=relation_bottleneck_dim,
            dropout=dropout,
        )

    def encode_equations(self, indices: torch.Tensor) -> torch.Tensor:
        """Return equation embeddings for equation indices."""

        return self.equation_encoder.encode_indices(indices)

    def encode_pair_embeddings(
        self,
        lhs_indices: torch.Tensor,
        rhs_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the equations appearing in one ordered-pair batch."""

        combined_indices = torch.cat([lhs_indices, rhs_indices], dim=0)
        combined_embeddings = self.equation_encoder.encode_indices(combined_indices)
        split = lhs_indices.size(0)
        return combined_embeddings[:split], combined_embeddings[split:]

    def relation_embedding(
        self,
        lhs_indices: torch.Tensor,
        rhs_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return the learned relation bottleneck for equation-index pairs."""

        lhs_embeddings, rhs_embeddings = self.encode_pair_embeddings(
            lhs_indices,
            rhs_indices,
        )
        return self.relation_head.relation_embedding(lhs_embeddings, rhs_embeddings)

    def forward(self, lhs_indices: torch.Tensor, rhs_indices: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of ordered equation-index pairs."""

        lhs_embeddings, rhs_embeddings = self.encode_pair_embeddings(
            lhs_indices,
            rhs_indices,
        )
        return self.relation_head(lhs_embeddings, rhs_embeddings)


def choose_device(device_name: str) -> torch.device:
    """Choose a PyTorch device from a CLI value."""

    if device_name != "auto":
        return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def progress_iterator(iterable, enabled: bool, desc: str):
    """Wrap an iterable in tqdm when available and requested."""

    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc, leave=False)
    return iterable


def train_one_epoch(
    model: ImplicationRelationModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    show_progress: bool = False,
    epoch: int | None = None,
) -> tuple[float, float]:
    """Train for one epoch and return `(average_loss, accuracy)`."""

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    iterator = progress_iterator(
        loader,
        enabled=show_progress,
        desc=f"train epoch {epoch:03d}" if epoch is not None else "train",
    )

    for lhs_indices, rhs_indices, labels in iterator:
        lhs_indices = lhs_indices.to(device=device, dtype=torch.long)
        rhs_indices = rhs_indices.to(device=device, dtype=torch.long)
        labels = labels.to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)

        logits = model(lhs_indices, rhs_indices)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = (logits >= 0).to(labels.dtype)
            total_correct += int((predictions == labels).sum().item())
            total_examples += int(labels.numel())
            total_loss += float(loss.detach()) * int(labels.numel())

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=total_loss / total_examples,
                acc=total_correct / total_examples,
            )

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(
    model: ImplicationRelationModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    show_progress: bool = False,
    epoch: int | None = None,
) -> tuple[float, float]:
    """Evaluate and return `(average_loss, accuracy)`."""

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    iterator = progress_iterator(
        loader,
        enabled=show_progress,
        desc=f"val epoch {epoch:03d}" if epoch is not None else "val",
    )

    for lhs_indices, rhs_indices, labels in iterator:
        lhs_indices = lhs_indices.to(device=device, dtype=torch.long)
        rhs_indices = rhs_indices.to(device=device, dtype=torch.long)
        labels = labels.to(device=device, dtype=torch.float32)

        logits = model(lhs_indices, rhs_indices)
        loss = loss_fn(logits, labels)

        predictions = (logits >= 0).to(labels.dtype)
        total_correct += int((predictions == labels).sum().item())
        total_examples += int(labels.numel())
        total_loss += float(loss.detach()) * int(labels.numel())

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=total_loss / total_examples,
                acc=total_correct / total_examples,
            )

    return total_loss / total_examples, total_correct / total_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an implication relation model from implication_graph.json."
    )
    parser.add_argument("--equations", type=Path, default=DEFAULT_EQUATIONS)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-epoch", type=int, default=8192)
    parser.add_argument("--validation-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--equation-dim", type=int, default=None)
    parser.add_argument("--composition-hidden-dim", type=int, default=None)
    parser.add_argument("--relation-hidden-dim", type=int, default=128)
    parser.add_argument("--relation-bottleneck-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run training without writing a checkpoint.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars even when tqdm is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.feature_dim <= 0:
        raise ValueError("--feature-dim must be positive")
    if args.equation_dim is not None and args.equation_dim <= 0:
        raise ValueError("--equation-dim must be positive")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    equation_dim = args.equation_dim or args.feature_dim
    show_progress = not args.no_progress

    print(f"device: {device}")
    print(f"loading equations from {args.equations}")
    equation_data = load_equation_tensors(args.equations)

    print(f"loading implication graph from {args.graph}")
    pair_data = load_implication_pair_data(args.graph, len(equation_data.tensors))

    print(f"equations: {len(equation_data.tensors)}")
    print(f"max local variables: {equation_data.max_variables}")
    print(f"positive implication pairs: {len(pair_data.positive_pair_codes)}")
    print(f"skipped graph-only edges: {pair_data.skipped_missing_equation_edges}")
    print(f"skipped duplicate edges: {pair_data.skipped_duplicate_edges}")
    print(
        "model: "
        f"feature_dim={args.feature_dim} "
        f"equation_dim={equation_dim} "
        "aggregation=sum_abs_mlp "
        f"relation_hidden_dim={args.relation_hidden_dim} "
        f"relation_bottleneck_dim={args.relation_bottleneck_dim}"
    )

    train_dataset = ImplicationPairDataset(
        pair_data=pair_data,
        samples_per_epoch=args.samples_per_epoch,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    validation_dataset = ImplicationPairDataset(
        pair_data=pair_data,
        samples_per_epoch=args.validation_samples,
        negative_ratio=args.negative_ratio,
        seed=args.seed + 10_000_000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ImplicationRelationModel(
        equation_tensors=equation_data.tensors,
        max_variables=equation_data.max_variables,
        feature_dim=args.feature_dim,
        equation_dim=equation_dim,
        relation_hidden_dim=args.relation_hidden_dim,
        relation_bottleneck_dim=args.relation_bottleneck_dim,
        composition_hidden_dim=args.composition_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            show_progress=show_progress,
            epoch=epoch,
        )
        validation_loss, validation_accuracy = evaluate(
            model=model,
            loader=validation_loader,
            loss_fn=loss_fn,
            device=device,
            show_progress=show_progress,
            epoch=epoch,
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
            f"val loss {validation_loss:.4f} acc {validation_accuracy:.4f}"
        )

    if not args.no_save:
        model_config = {
            "feature_dim": args.feature_dim,
            "equation_dim": equation_dim,
            "composition_hidden_dim": args.composition_hidden_dim,
            "relation_hidden_dim": args.relation_hidden_dim,
            "relation_bottleneck_dim": args.relation_bottleneck_dim,
            "dropout": args.dropout,
        }
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "model_config": model_config,
            "max_variables": equation_data.max_variables,
            "num_equations": len(equation_data.tensors),
        }
        torch.save(checkpoint, args.checkpoint)
        print(f"wrote checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
