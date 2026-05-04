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

        self.hidden_dim = hidden_dim or 128
        self.bottleneck_dim = bottleneck_dim or self.hidden_dim

        pair_feature_dim = 4 * equation_dim
        self.relation_encoder = nn.Sequential(
            nn.Linear(pair_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Use a simple initialization convention for the MLP layers."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def relation_embedding(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return the learned bottleneck vector for ordered pairs.

        These bottleneck vectors are useful for inspection, but the clustering
        workflow usually fingerprints equations by the relation logits they
        receive against many anchor equations.
        """

        features = self.pair_features(left, right)
        return self.relation_encoder(features)

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

        relation = self.relation_embedding(left, right)
        return self.classifier(relation).squeeze(-1)


if __name__ == "__main__":
    head = RelationHead(equation_dim=8)
    left = torch.randn(4, 8)
    right = torch.randn(4, 8)
    logits = head(left, right)

    print(f"left shape: {tuple(left.shape)}")
    print(f"right shape: {tuple(right.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
