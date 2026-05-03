#!/usr/bin/env python3
"""Ordered relation head for implication prediction.

An equation encoder maps each equation E to an embedding e(E). The relation
head receives two equation embeddings and predicts whether the first equation
implies the second:

    E_left  -> E_right

This order matters. Even if the equation encoder is symmetric in the two sides
of a single equation, implication between two equations is not symmetric.
"""

from __future__ import annotations

import torch
from torch import nn


class RelationHead(nn.Module):
    """MLP scorer for ordered pairs of equation embeddings.

    For embeddings `a` and `b`, the head builds the ordered pair feature

        [a ; b ; a - b ; a * b]

    where `*` is coordinatewise multiplication. The concatenation order and
    the signed difference make the feature directional, so the score for
    `(a, b)` can differ from the score for `(b, a)`.
    """

    def __init__(
        self,
        equation_dim: int,
        hidden_dim: int | None = None,
        bottleneck_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if equation_dim <= 0:
            raise ValueError("equation_dim must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        self.equation_dim = equation_dim

        pair_feature_dim = 4 * equation_dim
        hidden = hidden_dim or 2 * equation_dim
        bottleneck = bottleneck_dim or hidden

        self.network = nn.Sequential(
            nn.Linear(pair_feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 1),
        )

    def pair_features(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return ordered pair features for `left` and `right`.

        `left` and `right` may be single vectors with shape `(equation_dim,)` or
        batched tensors with shape `(..., equation_dim)`.
        """

        if left.shape[-1] != self.equation_dim:
            raise ValueError(
                f"left has last dimension {left.shape[-1]}, "
                f"expected {self.equation_dim}"
            )
        if right.shape[-1] != self.equation_dim:
            raise ValueError(
                f"right has last dimension {right.shape[-1]}, "
                f"expected {self.equation_dim}"
            )

        return torch.cat([left, right, left - right, left * right], dim=-1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return implication logits for ordered pairs `(left, right)`.

        The returned logits have shape `()` for one pair or `(...)` for a batch.
        They are intended for `torch.nn.BCEWithLogitsLoss`.
        """

        features = self.pair_features(left, right)
        return self.network(features).squeeze(-1)


if __name__ == "__main__":
    head = RelationHead(equation_dim=8)
    left = torch.randn(4, 8)
    right = torch.randn(4, 8)
    logits = head(left, right)

    print(f"left shape: {tuple(left.shape)}")
    print(f"right shape: {tuple(right.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
